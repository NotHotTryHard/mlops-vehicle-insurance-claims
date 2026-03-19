from datetime import datetime

class DataMeta:
    def __init__(
        self,
        dt_col="INSR_BEGIN", 
        dt_fmt="%d-%b-%y",
        missing_values=[None, ""],
    ):
        self.dt_col = dt_col
        self.dt_fmt = dt_fmt
        self.missing_values = missing_values

        self._initialize_stats()
    
    def _initialize_stats(self):
        self.total_rows = 0
        self.total_missing = 0
        self.min_date = None
        self.max_date = None
        self.loaded_at = datetime.now().isoformat()

    def update(self, batch):
        self.total_rows += len(batch)
        self.total_missing += sum(
            1 for _, row in batch for v in row.values() if v in self.missing_values
        )

        dates = [
            datetime.strptime(row.get(self.dt_col), self.dt_fmt) for _, row in batch if row.get(self.dt_col) not in self.missing_values
        ]

        if dates:
            batch_min = min(dates)
            batch_max = max(dates)
            self.min_date = batch_min if self.min_date is None else min(self.min_date, batch_min)
            self.max_date = batch_max if self.max_date is None else max(self.max_date, batch_max)

    def to_dict(self):
        return {
            "total_rows": self.total_rows,
            "total_missing": self.total_missing,
            "min_date": self.min_date.isoformat(),
            "max_date": self.max_date.isoformat(),
            "loaded_at": self.loaded_at,
        }
    
    def __str__(self):
        report = "\nData Meta Parameters\n"
        for key, value in self.to_dict().items():
            report += f"{key}: {value}\n"
        return report
