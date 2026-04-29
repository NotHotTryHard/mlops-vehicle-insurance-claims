import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
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
        "min_lift": float(assoc.get("min_lift", 1.02)),
        "top_k": top_k,
        "max_rule_features": max_rule_features,
        "binning": str(assoc.get("binning", "quantile")).lower(),
        "exclude_target": bool(assoc.get("exclude_target", True)),
        "exclude_from_association": list(assoc.get("exclude_from_association") or []),
        "sample_cap_per_column": int(assoc.get("sample_cap_per_column", 25_000)),
        "max_levels_per_categorical": int(assoc.get("max_levels_per_categorical", 40)),
        "max_transactions": int(assoc.get("max_transactions", 60_000)),
    }


def collapse_categorical_items(
    items: Iterable[str],
    cat_keep: dict[str, set[str]],
) -> list[str]:
    """Map rare category levels to COL = __OTHER__ using top-value sets per column."""
    out = []
    for it in items:
        if " = " in it and not it.strip().startswith("("):
            col, _, val = it.partition(" = ")
            if col in cat_keep and val not in cat_keep[col]:
                it = f"{col} = __OTHER__"
        out.append(it)
    return out


def load_association_projection(quality_path: Path) -> dict | None:
    path = Path(quality_path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    proj = doc.get("association_projection")
    return proj if isinstance(proj, dict) else None


def augment_row_from_specs(
    row,
    rule_feature_specs,
    binner,
    num_columns,
    cat_columns,
    *,
    association_projection=None,
):
    """
    Add binary columns: 1 iff all antecedent bin items for that rule are present in the row.
    ``rule_feature_specs`` comes from the quality report.
    """
    if not rule_feature_specs:
        return {}
    raw = binner.binarize_row(row, num_columns, cat_columns)
    if association_projection:
        ck = association_projection.get("categorical_top_values") or {}
        cat_keep = {k: set(v) for k, v in ck.items()}
        raw = collapse_categorical_items(raw, cat_keep)
    items = set(raw)
    out = {}
    for spec in rule_feature_specs:
        ant = frozenset(spec["antecedent_items"])
        out[spec["feature_name"]] = int(ant <= items)
    return out


def augment_batch_from_specs(
    batch,
    rule_feature_specs,
    binner,
    num_columns,
    cat_columns,
    *,
    association_projection=None,
):
    return [
        {
            **row,
            **augment_row_from_specs(
                row,
                rule_feature_specs,
                binner,
                num_columns,
                cat_columns,
                association_projection=association_projection,
            ),
        }
        for row in batch
    ]


def load_association_binning(quality_path: Path) -> dict | None:
    path = Path(quality_path)
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        doc = yaml.safe_load(f) or {}
    binning = doc.get("association_binning")
    return binning if isinstance(binning, dict) else None


def binner_and_columns_from_stats(
    cfg,
    stats,
    n_bins,
    missing_values=(None, ""),
    association_binning=None,
):
    num_stats = stats["numeric_features"]
    cat_stats = stats["categorical_features"]
    cat_columns = list(cat_stats.keys())
    binner = Binner(n_bins, missing_values)

    if association_binning and association_binning.get("numeric_edges"):
        for col, edges in association_binning["numeric_edges"].items():
            binner.numeric_binner.bins[col] = np.asarray(edges, dtype=np.float64)
        num_columns = list(
            association_binning.get("numeric_columns") or binner.numeric_binner.bins.keys()
        )
    else:
        binner.fit_equal_width(num_stats)
        num_columns = list(num_stats.keys())

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
        min_lift: float = 1.02,
        top_k: int = 5,
        max_rule_features: int = 5,
        binning: str = "quantile",
        exclude_target: bool = True,
        exclude_from_association=None,
        sample_cap_per_column: int = 25000,
        max_levels_per_categorical: int = 40,
        max_transactions: int = 60000,
        target_column: str | None = None,
    ):
        self.n_bins = n_bins
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.max_levels_per_categorical = max(2, max_levels_per_categorical)
        self.top_k = top_k
        self.max_rule_features = max(0, int(max_rule_features))
        self.min_lift = min_lift
        self.missing_values = missing_values
        self.num_stats = num_stats
        self.cat_stats = cat_stats
        self.rule_confidence_threshold = rule_confidence_threshold
        self.rule_lift_threshold = rule_lift_threshold
        self.binning = binning if binning in ("quantile", "equal_width") else "quantile"
        self.exclude_target = exclude_target
        self.target_column = target_column
        self.sample_cap_per_column = max(1000, sample_cap_per_column)
        self.max_transactions = max(5000, int(max_transactions))

        exclude = set(exclude_from_association or [])
        if exclude_target and target_column:
            exclude.add(target_column)
        self._exclude_columns = exclude

        self.cat_columns = list(cat_stats.keys())
        self._association_numeric_columns = [
            c for c in num_stats.keys() if c not in self._exclude_columns
        ]

        self.binner = Binner(n_bins, missing_values)
        self.transactions = []
        self._rule_antecedents = []
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._cat_keep: dict[str, set[str]] = {}
        self._allowed_items: set[str] = set()
        self._projection_export: dict = {}

    @classmethod
    def from_config(
        cls,
        cfg,
        stats=None,
        *,
        config_path=None,
    ):
        if stats is None:
            if config_path is None:
                raise ValueError("Either stats or config_path must be provided")
            root = Path(config_path).resolve().parent
            path = root / cfg["data_storage"]["statistics_path"]
            with path.open("r", encoding="utf-8") as f:
                stats = yaml.safe_load(f)

        kw = _association_kwargs(cfg)
        target_column = (cfg.get("columns") or {}).get("target")
        return cls(
            stats["numeric_features"],
            stats["categorical_features"],
            missing_values=(None, ""),
            target_column=target_column,
            **kw,
        )

    def finalize_bins(self) -> None:
        """Call after optional sampling (quantile) or immediately for equal_width."""
        if self.binning == "quantile":
            self.binner.fit_quantile_per_column(
                dict(self._samples),
                self.n_bins,
                self._association_numeric_columns,
                self.num_stats,
            )
        else:
            sub_stats = {k: self.num_stats[k] for k in self._association_numeric_columns}
            self.binner.fit_equal_width(sub_stats)

    def accumulate_samples(self, batch) -> None:
        """Reservoir-style cap per column for quantile bin fitting."""
        for row in batch:
            for column in self._association_numeric_columns:
                raw = row.get(column)
                if raw in self.missing_values:
                    continue
                try:
                    value = float(raw)
                except (TypeError, ValueError):
                    continue
                bucket = self._samples[column]
                bucket.append(value)
                if len(bucket) > self.sample_cap_per_column:
                    bucket.pop(random.randint(0, len(bucket) - 1))

    def build_transactions(self, batch):
        transactions = []
        for row in batch:
            items = self.binner.binarize_row(
                row, self._association_numeric_columns, self.cat_columns
            )
            transactions.append(items)
        return transactions

    def update(self, batch) -> None:
        self.transactions.extend(self.build_transactions(batch))

    def _compress_transactions_for_apriori(self) -> None:
        """
        Reduce item vocabulary before Apriori (avoids 100k+ columns / OOM).

        1. Per categorical column, keep top max_levels_per_categorical levels;
           map the rest to COL = __OTHER__.
        2. Drop any item with global count < min_support * N transactions.
        """
        n = len(self.transactions)
        if n == 0:
            self._cat_keep = {}
            self._allowed_items = set()
            self._projection_export = {}
            return

        col_vals: dict[str, Counter] = defaultdict(Counter)
        for t in self.transactions:
            for it in t:
                if " = " in it and not it.strip().startswith("("):
                    col, _, val = it.partition(" = ")
                    if col in self.cat_columns:
                        col_vals[col][val] += 1

        max_cat = self.max_levels_per_categorical
        self._cat_keep = {
            col: {v for v, _ in ctr.most_common(max_cat)}
            for col, ctr in col_vals.items()
        }

        self.transactions = [
            collapse_categorical_items(t, self._cat_keep) for t in self.transactions
        ]

        freq: Counter = Counter()
        for t in self.transactions:
            for it in t:
                freq[it] += 1

        min_cnt = max(1, int(self.min_support * n))
        self._allowed_items = {it for it, c in freq.items() if c >= min_cnt}

        self.transactions = [
            [it for it in t if it in self._allowed_items] for t in self.transactions
        ]
        self.transactions = [t for t in self.transactions if t]

        self._projection_export = {
            "categorical_top_values": {
                col: sorted(vals) for col, vals in self._cat_keep.items()
            },
            "min_support_used": self.min_support,
            "min_item_count_floor": min_cnt,
        }

    def association_binning_export(self) -> dict:
        edges = {
            k: self.binner.numeric_binner.bins[k].astype(float).tolist()
            for k in self._association_numeric_columns
            if k in self.binner.numeric_binner.bins
        }
        return {
            "mode": self.binning,
            "numeric_columns": list(self._association_numeric_columns),
            "numeric_edges": edges,
            "categorical_columns": list(self.cat_columns),
        }

    def _mine_rules_dataframe(self) -> pd.DataFrame:
        if not self.transactions:
            return pd.DataFrame()

        nuniq = len({it for t in self.transactions for it in t})
        if nuniq > 2_500:
            raise ValueError(
                f"Association mining: {nuniq} distinct items after compression — "
                "raise quality.association.min_support, lower n_bins / "
                "max_levels_per_categorical, or reduce max_transactions."
            )

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
        rules = rules[rules["lift"] >= self.min_lift].copy()
        if rules.empty:
            return pd.DataFrame()
        rules_sorted = rules.sort_values(
            by=["lift", "confidence"],
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

        for _, r in rules.iterrows():
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
        raw = self.binner.binarize_row(
            row, self._association_numeric_columns, self.cat_columns
        )
        if self._cat_keep:
            raw = collapse_categorical_items(raw, self._cat_keep)
        if self._allowed_items:
            raw = [it for it in raw if it in self._allowed_items]
        items = set(raw)
        out = {}
        for i, ant in enumerate(self._rule_antecedents):
            out[f"ar_rule_{i}"] = int(ant <= items)
        return out

    def augment_batch(self, batch):
        return [{**row, **self.augment_row(row)} for row in batch]

    def mine_and_report(self):
        self._compress_transactions_for_apriori()
        n_tx = len(self.transactions)
        if n_tx > self.max_transactions:
            self.transactions = random.sample(self.transactions, self.max_transactions)
            print(
                f"Association: subsampled transactions {n_tx} -> {self.max_transactions} "
                f"(quality.association.max_transactions)."
            )
        rules = self._mine_rules_dataframe()
        top = self.select_top_k_rules(rules)
        report = self.analyze_rules(top)
        report["association_binning"] = self.association_binning_export()
        report["association_projection"] = dict(self._projection_export)
        return report

    def report(self):
        return self.mine_and_report()


def run_association_passes(
    cfg: dict,
    analyzer: AssociationRulesAnalyzer,
    show_progress: bool,
) -> None:
    """One pass (equal_width) or two passes (quantile: sample edges, then transactions)."""
    batch_size = int(cfg["batch"]["size"])
    from src.data.database.db_stream import db_stream
    from tqdm import tqdm

    if analyzer.binning == "quantile":
        stream = db_stream(batch_size=batch_size)
        if show_progress:
            stream = tqdm(stream, desc="Sampling for quantile bins...")
        for batch in stream:
            analyzer.accumulate_samples(batch)
        analyzer.finalize_bins()
        stream = db_stream(batch_size=batch_size)
        if show_progress:
            stream = tqdm(stream, desc="Association transactions...")
        for batch in stream:
            analyzer.update(batch)
    else:
        analyzer.finalize_bins()
        stream = db_stream(batch_size=batch_size)
        if show_progress:
            stream = tqdm(stream, desc="Finding association rules...")
        for batch in stream:
            analyzer.update(batch)
