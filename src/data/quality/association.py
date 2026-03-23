from .stats import DataStatsAnalyzer
from .binarization import Binner
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

class AssociationRulesAnalyzer:
    def __init__(
        self,
        num_stats,
        cat_stats,
        n_bins=10,
        missing_values = (None, ""),
        min_confidence=0.6,
        min_support=0.1,
        rule_confidence_threshold=0.8,
        rule_lift_threshold=1.2,
        top_k=5,
    ):
        self.n_bins = n_bins
        self.min_confidence = min_confidence
        self.min_support = min_support
        self.top_k= top_k
        self.missing_values = missing_values
        self.num_stats = num_stats
        self.cat_stats = cat_stats
        self.rule_confidence_threshold = rule_confidence_threshold
        self.rule_lift_threshold = rule_lift_threshold

        self.num_columns = list(num_stats.keys())
        self.cat_columns = list(cat_stats.keys())

        self.binner = Binner(n_bins, missing_values)
        self.transactions = []

    def fit(self):
        self.binner.fit(self.num_stats)
        return self

    def binarize_row(self, row):
        items = []

        for col, val in row.items():
            if val is None:
                continue

            items.append(f"{col}={val}")
        return items
    
    def build_transactions(self, batch):
        transactions = []
        for row in batch:
            items = self.binner.binarize_row(row, self.num_columns, self.cat_columns)
            transactions.append(items)
        return transactions

    def update(self, batch):
        self.transactions.extend(self.build_transactions(batch))

    def find_association_rules(self):
        # One-hot encoding
        te = TransactionEncoder()
        te_array = te.fit(self.transactions).transform(self.transactions)

        df = pd.DataFrame(te_array, columns=te.columns_)
        freq_items = apriori(df, min_support=self.min_support, use_colnames=True)
        rules = association_rules(freq_items, metric="confidence", min_threshold=self.min_confidence)
        return rules

    def select_top_k_rules(self, rules):
        rules_sorted = rules.sort_values(
            by=["confidence", "lift"],
            ascending=False,
        )
        return rules_sorted.head(self.top_k)

    def get_final_rules(self):
        rules = self.find_association_rules()
        top_rules = self.select_top_k_rules(rules)
        return top_rules

    def analyze_rules(self, rules):
        if rules is None or rules.empty:
            report = {
                "association_rules": [],
                "insights": []
            }
            return

        formatted_rules = []
        insights = []

        for _, row in rules.iterrows():
            antecedents = list(row["antecedents"])
            consequents = list(row["consequents"])

            rule_dict = {
                "rule": f"{antecedents} -> {consequents}",
                "confidence": float(row["confidence"]),
                "lift": float(row["lift"]),
                "support": float(row["support"]),
            }

            formatted_rules.append(rule_dict)

            if row["confidence"] > self.rule_confidence_threshold:
                insights.append(
                    f"Strong rule: {antecedents} -> {consequents} (conf={round(row['confidence'], 3)})"
                )
            elif row["lift"] > self.rule_lift_threshold:
                insights.append(
                    f"Interesting dependency: {antecedents} -> {consequents} (lift={round(row['lift'], 3)})"
                    )
        report = {
            "association_rules": formatted_rules,
            "insights": insights
        }
        return report

    def report(self):
        return self.analyze_rules(self.get_final_rules())
