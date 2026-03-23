from data.collection import stream_batches
from .quality import DataQualityChecker
from data.utils import load_config
from data.collection import db_stream
import argparse
from tqdm import tqdm

class DataCleaner:
    def __init__(self, quality_checker=None, missing_freq_threshold=0, missing_values=(None, "")):
        self.qc = quality_checker
        self.num_stats = quality_checker.num_stats
        self.cat_stats = quality_checker.cat_stats
        self.missing_values = missing_values
        self.rules = quality_checker.rules

        self.missing_freq_threshold = missing_freq_threshold

    def is_bad_column(self, column):
        if column in self.num_stats:
            stats = self.num_stats[column]
            return stats["missing_frequency"] > self.missing_freq_threshold

        if column in self.cat_stats:
            stats = self.cat_stats[column]
            return stats["missing_frequency"] > self.missing_freq_threshold
        # column is automatically good if not required for training (not in features)
        return False

    def is_bad_row(self, row):
        for column, value in row.items():
            if self.is_bad_column(column):
                return True

            if value in self.missing_values:
                return True

            if column in self.num_stats:
                try:
                    val = float(val)
                except:
                    return True
        return False

    def clean_batch(self, batch):
        cleaned_batch = []

        for row in batch:
            if not self.is_bad_row(row):
                cleaned_batch.append((idx, row))
        return cleaned_batch


def load_clean()

def db_stream_cleaned(
    config_path="config.yaml",
    quality_checker=None,
    batch_size=4,
):
    cfg = load_config(config_path)
    cleaner = DataCleaner(quality_checker, missing_freq_threshold=0.8)
    for batch in tqdm(db_stream(config_path=config_path, batch_size=4)):
        cleaned_batch = cleaner.clean_batch(batch)
        if len(cleaned_batch):
            yield cleaned_batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    quality_checker = db_quality_analyze()
    db_stream_cleaned(args.config, quality_checker)
