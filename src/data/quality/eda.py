import warnings
from pathlib import Path

import pandas as pd

from src.data.database.db_stream import db_stream
from src.data.utils import get_all_features, load_config
from tqdm import tqdm


def eda_column_names(cfg: dict) -> list[str]:
    feature_cols = get_all_features(cfg)
    target = cfg["columns"]["target"]
    dt_col = cfg["columns"].get("datetime")
    columns = list(feature_cols) + [target]
    if dt_col:
        columns.append(dt_col)
    return columns


def _coerce_numeric_columns_for_eda(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    target = cfg["columns"]["target"]
    dt_col = cfg["columns"].get("datetime")
    numeric_names = set(cfg["columns"]["features"]["numeric"] + [target])
    for c in df.columns:
        if dt_col and c == dt_col:
            continue
        if c in numeric_names:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def build_dataframe_from_rows(cfg: dict, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return _coerce_numeric_columns_for_eda(df, cfg)


def load_eda_rows_from_db(config_path: str = "config.yaml") -> list[dict]:
    cfg = load_config(Path(config_path).resolve())
    quality_section = cfg.get("quality") or {}
    eda_section = quality_section.get("eda") or {}
    max_rows = int(eda_section.get("max_rows", 30000))
    columns = eda_column_names(cfg)
    rows = []
    batch_size = min(5000, max_rows)
    for batch in tqdm(db_stream(batch_size=batch_size), desc="Constructing EDA report..."):
        for row in batch:
            rows.append({c: row[c] for c in columns if c in row})
            if len(rows) >= max_rows:
                return rows
    return rows


def _fallback_html_report(df: pd.DataFrame, title: str) -> str:
    desc = df.describe(include="all").to_html()
    num = df.select_dtypes(include=["number"])
    if num.shape[1] > 1:
        corr = num.corr().round(4).to_html()
    else:
        corr = "<p>No correlation (insufficient numeric columns).</p>"
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>{title}</title></head>
<body><h1>{title}</h1><h2>Describe</h2>{desc}<h2>Correlation (numeric)</h2>{corr}</body></html>"""


def _write_eda_profile(cfg: dict, df: pd.DataFrame, config_path: str) -> str | None:
    quality_section = cfg.get("quality") or {}
    eda_section = quality_section.get("eda") or {}
    out_rel = eda_section.get("report_path", "session/reports/eda_profile.html")
    out_path = Path(config_path).resolve().parent / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    title = eda_section.get("title", "Automatic EDA")
    minimal = bool(eda_section.get("minimal_profile", True))

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from ydata_profiling import ProfileReport

            profile = ProfileReport(df, title=title, minimal=minimal)
            profile.to_file(out_path)
    except Exception as e:
        html = _fallback_html_report(df, title)
        out_path.write_text(html, encoding="utf-8")
        print(
            f"ydata-profiling skipped ({type(e).__name__}: {e}); "
            "wrote fallback HTML (describe + corr)."
        )

    print(f"EDA report written to {out_path}")
    return str(out_path)


def run_automatic_eda(config_path: str, rows: list[dict]) -> str | None:
    cfg = load_config(Path(config_path).resolve())
    df = build_dataframe_from_rows(cfg, rows)
    if df.empty:
        print("No rows loaded for EDA.")
        return None
    return _write_eda_profile(cfg, df, config_path)


if __name__ == "__main__":
    import sys

    cfg = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    run_automatic_eda(cfg, load_eda_rows_from_db(cfg))
