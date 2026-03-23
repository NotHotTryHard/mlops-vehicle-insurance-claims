from .association import AssociationRulesAnalyzer
from src.data.utils import load_config
import yaml
from tqdm import tqdm
from src.data.database import db_stream

def find_association_rules(config_path="config.yaml"):
    cfg = load_config(config_path)
    with open(cfg["data_storage"]["statistics_path"], "r") as f:
        stats = yaml.safe_load(f)
    num_stats, cat_stats = stats["numeric_features"], stats["categorical_features"]
    analyzer = AssociationRulesAnalyzer(
        num_stats,
        cat_stats,
    ).fit()
    print("Finding association rules...")
    for batch in tqdm(db_stream(batch_size=cfg["batch"]["size"])):
        analyzer.update(batch)
        analyzer.binner.parse_rule(batch, "['13.0 <= EFFECTIVE_YR < 23.0)', '0.0 <= CARRYING_CAPACITY < 100000.0)'] -> ['0.0 <= INSURED_VALUE < 25000000.0)']")
    return analyzer


class QualityChecker:
    def __init__(
        self,
        stats=None,
        rules_report=None,
        result_path="",
        thresholds={
            "missing_frequency": 0.3,
            "nonvalid_frequency": 0.2,
        },
    ):
        self.rules = rules_report
        self.num_stats = stats["numeric_features"]
        self.cat_stats = stats["categorical_features"]
        self.thresholds = thresholds
        self.result_path = result_path
        self.report = {}
        self.warnings = []

    def analyze_thresholds(self, stats):
        for key, value in stats.items():
            for name, threshold in self.thresholds.items():
                if name in value and value[name] > threshold:
                    self.warnings.append(
                        f"Column {key} has {name} greater than appropriate threshold {threshold}."
                    )
    
    def analyze_quality(self):
        print("\nAnalyzing data quality...\n")
        self.analyze_thresholds(self.num_stats)
        self.analyze_thresholds(self.cat_stats)
        if len(self.warnings):
            print("WARNING: some requirements were not satisfied.")
        else:
            print("OK: Data satisfies all conditions.")
        
        self.report = {**self.rules, **{"warnings": self.warnings}}

    def save_report(self):
        with open(self.result_path, "w") as f:
            yaml.dump(self.report, f, allow_unicode=True, sort_keys=False)
        print(f"Full quality report was saved in {self.result_path}")


if __name__ == "__main__":
    analyzer = find_association_rules()
    cfg = load_config("config.yaml")
    with open(cfg["data_storage"]["statistics_path"], "r") as f:
        stats = yaml.safe_load(f)
    quality_checker = QualityChecker(stats, analyzer.report(), result_path=cfg["data_storage"]["quality_path"])
    quality_checker.analyze_quality()
    import pdb; pdb.set_trace()
    quality_checker.save_report()