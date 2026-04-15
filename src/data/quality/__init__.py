import importlib

from .association import AssociationRulesAnalyzer
from .binarization import Binner, NumericBinner
from .stats import DataStatsGlobalAnalyzer

__all__ = [
    "DataStatsGlobalAnalyzer",
    "AssociationRulesAnalyzer",
    "Binner",
    "NumericBinner",
    "QualityChecker",
    "DataCleaner",
    "find_association_rules",
    "load_statistics_bundle",
    "run_association_rules",
    "run_full_quality_pipeline",
    "run_cleaning_summary",
    "stream_cleaned_batches",
    "augment_row_from_specs",
    "load_rule_feature_specs",
    "max_rule_features_from_cfg",
]

_LAZY_FROM_QUALITY_REPORT = frozenset(
    {
        "QualityChecker",
        "find_association_rules",
        "load_statistics_bundle",
        "run_association_rules",
        "run_full_quality_pipeline",
    }
)

_LAZY_FROM_CLEAN = frozenset(
    {
        "DataCleaner",
        "run_cleaning_summary",
        "stream_cleaned_batches",
        "augment_row_from_specs",
        "load_rule_feature_specs",
        "max_rule_features_from_cfg",
    }
)


def __getattr__(name: str):
    if name in _LAZY_FROM_QUALITY_REPORT:
        mod = importlib.import_module(".quality_report", __name__)
        return getattr(mod, name)
    if name in _LAZY_FROM_CLEAN:
        mod = importlib.import_module(".clean", __name__)
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
