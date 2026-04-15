from pathlib import Path

import pandas as pd
import yaml
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

from .binarization import Binner


def _association_kwargs(cfg):
    q = cfg.get("quality") or {}
    assoc = q.get("association") or {}
    top_k = int(assoc.get("top_k", 5))
    raw_max = assoc.get("max_rule_features")
    if raw_max is not None:
        max_rule_features = int(raw_max)
    else:
        max_rule_features = top_k
    return {
        "n_bins": int(assoc.get("n_bins", 10)),
        "min_support": float(assoc.get("min_support", 0.1)),
        "min_confidence": float(assoc.get("min_confidence", 0.6)),
        "rule_confidence_threshold": float(assoc.get("rule_confidence_threshold", 0.8)),
        "rule_lift_threshold": float(assoc.get("rule_lift_threshold", 1.2)),
        "top_k": top_k,
        "max_rule_features": max_rule_features,
    }


def augment_row_from_specs(row, rule_feature_specs, binner, num_columns, cat_columns):
    """
    Add binary columns: 1 iff all antecedent bin items for that rule are present in the row.
    ``rule_feature_specs`` comes from the quality report.
    """
    if not rule_feature_specs:
        return {}
    items = set(binner.binarize_row(row, num_columns, cat_columns))
    out = {}
    for spec in rule_feature_specs:
        ant = frozenset(spec["antecedent_items"])
        out[spec["feature_name"]] = int(ant <= items)
    return out


def augment_batch_from_specs(batch, rule_feature_specs, binner, num_columns, cat_columns):
    return [
        {**row, **augment_row_from_specs(row, rule_feature_specs, binner, num_columns, cat_columns)}
        for row in batch
    ]


def binner_and_columns_from_stats(cfg, stats, n_bins, missing_values=(None, "")):
    num_stats = stats["numeric_features"]
    cat_stats = stats["categorical_features"]
    num_columns = list(num_stats.keys())
    cat_columns = list(cat_stats.keys())
    binner = Binner(n_bins, missing_values)
    binner.fit(num_stats)
    return binner, num_columns, cat_columns


def max_rule_features_from_cfg(cfg):
    assoc = (cfg.get("quality") or {}).get("association") or {}
    v = assoc.get("max_rule_features")
    if v is not None:
        return int(v)
    return int(assoc.get("top_k", 5))


def load_rule_feature_specs(quality_path, max_n=None):
    path = Path(quality_path)
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    specs = doc.get("rule_feature_specs") or []
    if max_n is not None:
        specs = specs[: int(max_n)]
    return specs


class AssociationRulesAnalyzer:
    def __init__(
        self,
        num_stats,
        cat_stats,
        *,
        n_bins: int = 10,
        missing_values=(None, ""),
        min_confidence: float = 0.6,
        min_support: float = 0.1,
        rule_confidence_threshold: float = 0.8,
        rule_lift_threshold: float = 1.2,
        top_k: int = 5,
        max_rule_features: int = 5,
    ):
        self.n_bins = n_bins
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.top_k = top_k
        self.max_rule_features = max(0, int(max_rule_features))
        self.missing_values = missing_values
        self.num_stats = num_stats
        self.cat_stats = cat_stats
        self.rule_confidence_threshold = rule_confidence_threshold
        self.rule_lift_threshold = rule_lift_threshold

        self.num_columns = list(num_stats.keys())
        self.cat_columns = list(cat_stats.keys())

        self.binner = Binner(n_bins, missing_values)
        self.transactions = []
        self._rule_antecedents = []

    @classmethod
    def from_config(
        cls,
        cfg,
        stats=None,
        *,
        config_path=None,
    ):
        """
        Pass ``stats`` pre-loaded, or ``config_path`` so statistics are read from
        ``data_storage.statistics_path``.
        """
        if stats is None:
            if config_path is None:
                raise ValueError("Either stats or config_path must be provided")
            root = Path(config_path).resolve().parent
            path = root / cfg["data_storage"]["statistics_path"]
            with path.open("r", encoding="utf-8") as f:
                stats = yaml.safe_load(f)

        kw = _association_kwargs(cfg)
        return cls(
            stats["numeric_features"],
            stats["categorical_features"],
            missing_values=(None, ""),
            **kw,
        ).fit()

    def fit(self):
        self.binner.fit(self.num_stats)
        return self

    def build_transactions(self, batch):
        transactions = []
        for row in batch:
            items = self.binner.binarize_row(row, self.num_columns, self.cat_columns)
            transactions.append(items)
        return transactions

    def update(self, batch) -> None:
        self.transactions.extend(self.build_transactions(batch))

    def _mine_rules_dataframe(self) -> pd.DataFrame:
        if not self.transactions:
            return pd.DataFrame()

        te = TransactionEncoder()
        te_array = te.fit(self.transactions).transform(self.transactions)
        df = pd.DataFrame(te_array, columns=te.columns_)

        freq_items = apriori(df, min_support=self.min_support, use_colnames=True)
        if freq_items.empty:
            return pd.DataFrame()

        rules = association_rules(
            freq_items, metric="confidence", min_threshold=self.min_confidence
        )
        return rules

    def select_top_k_rules(self, rules: pd.DataFrame) -> pd.DataFrame:
        if rules is None or rules.empty:
            return pd.DataFrame()
        rules_sorted = rules.sort_values(
            by=["confidence", "lift"],
            ascending=False,
        )
        return rules_sorted.head(self.top_k)

    def analyze_rules(self, rules):
        if rules is None or rules.empty:
            return {
                "association_rules": [],
                "insights": [],
                "rule_feature_specs": [],
            }

        formatted_rules = []
        insights = []
        rule_feature_specs = []
        self._rule_antecedents = []
        feat_idx = 0

        for i, (_, r) in enumerate(rules.iterrows()):
            antecedents = list(r["antecedents"])
            consequents = list(r["consequents"])

            rule_dict = {
                "rule": f"{antecedents} -> {consequents}",
                "confidence": float(r["confidence"]),
                "lift": float(r["lift"]),
                "support": float(r["support"]),
            }
            formatted_rules.append(rule_dict)

            ant = r["antecedents"]
            if feat_idx < self.max_rule_features:
                self._rule_antecedents.append(ant)
                rule_feature_specs.append(
                    {
                        "feature_name": f"ar_rule_{feat_idx}",
                        "antecedent_items": sorted(ant),
                    }
                )
                feat_idx += 1

            if r["confidence"] > self.rule_confidence_threshold:
                insights.append(
                    f"Strong rule: {antecedents} -> {consequents} "
                    f"(conf={round(r['confidence'], 3)})"
                )
            elif r["lift"] > self.rule_lift_threshold:
                insights.append(
                    f"Interesting dependency: {antecedents} -> {consequents} "
                    f"(lift={round(r['lift'], 3)})"
                )

        return {
            "association_rules": formatted_rules,
            "insights": insights,
            "rule_feature_specs": rule_feature_specs,
        }

    def augment_row(self, row):
        """Binary features ar_rule_*: 1 iff row's bin items contain the rule antecedent set."""
        if not self._rule_antecedents:
            return {}
        items = set(self.binner.binarize_row(row, self.num_columns, self.cat_columns))
        out = {}
        for i, ant in enumerate(self._rule_antecedents):
            out[f"ar_rule_{i}"] = int(ant <= items)
        return out

    def augment_batch(self, batch):
        return [{**row, **self.augment_row(row)} for row in batch]

    def mine_and_report(self):
        rules = self._mine_rules_dataframe()
        top = self.select_top_k_rules(rules)
        return self.analyze_rules(top)

    def report(self):
        return self.mine_and_report()
