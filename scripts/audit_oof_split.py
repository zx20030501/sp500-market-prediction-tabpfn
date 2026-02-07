import argparse
from pathlib import Path
import pandas as pd

FORWARD_KEYWORDS = ("forward", "future", "lead", "next")


def _coerce_series(series: pd.Series):
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float), "numeric"
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(float), "numeric"
    dt = pd.to_datetime(series, errors="coerce")
    if dt.notna().all():
        return dt, "datetime"
    return series, "string"


def _compute_len(start, end, kind):
    if kind == "numeric":
        return int(end - start + 1)
    if kind == "datetime":
        return int((end - start).days + 1)
    return None


def _compute_gap(train_end, valid_start, kind):
    if kind == "numeric":
        return int(valid_start - train_end - 1)
    if kind == "datetime":
        return int((valid_start - train_end).days - 1)
    return None


def _print_fold_report(df: pd.DataFrame) -> None:
    required = {"fold", "train_start", "train_end", "valid_start", "valid_end"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    train_start, kind = _coerce_series(df["train_start"])
    train_end, _ = _coerce_series(df["train_end"])
    valid_start, _ = _coerce_series(df["valid_start"])
    valid_end, _ = _coerce_series(df["valid_end"])

    print("=== Fold Split Audit ===")
    print(f"date_type: {kind}")
    for idx, row in df.iterrows():
        fold = row["fold"]
        ts = train_start.iloc[idx]
        te = train_end.iloc[idx]
        vs = valid_start.iloc[idx]
        ve = valid_end.iloc[idx]
        ok = te < vs if kind in ("numeric", "datetime") else "N/A"
        train_len = _compute_len(ts, te, kind)
        valid_len = _compute_len(vs, ve, kind)
        gap = _compute_gap(te, vs, kind)
        purge = row.get("purge_days", "")
        embargo = row.get("embargo_days", "")

        print(f"fold={fold} train=[{ts}..{te}] valid=[{vs}..{ve}] train_end<valid_start={ok}")
        if train_len is not None:
            print(f"  train_len={train_len} valid_len={valid_len} gap={gap}")
        if purge != "" or embargo != "":
            print(f"  purge_days={purge} embargo_days={embargo}")


def _scan_forward_columns(train_cols, test_cols):
    def _flag(cols):
        flagged = []
        for c in cols:
            cl = c.lower()
            if cl.startswith("lagged_"):
                continue
            if any(k in cl for k in FORWARD_KEYWORDS):
                flagged.append(c)
        return sorted(set(flagged))

    flagged = sorted(set(_flag(train_cols) + _flag(test_cols)))
    print("=== Forward-looking Column Scan (name-based) ===")
    print("keywords:", ", ".join(FORWARD_KEYWORDS))
    if flagged:
        for c in flagged:
            print("-", c)
    else:
        print("- none")


def _single_split_report(data_root: Path, date_col: str) -> None:
    train = pd.read_csv(data_root / "train.csv")
    test = pd.read_csv(data_root / "test.csv")
    test_start = int(test[date_col].min())
    overlap_mask = train[date_col] >= test_start
    overlap_dates = sorted(train.loc[overlap_mask, date_col].unique().tolist())
    cleaned_train = train.loc[~overlap_mask]
    train_end = int(cleaned_train[date_col].max())

    print("=== Split Summary (single holdout) ===")
    print(f"date_col: {date_col}")
    print(f"test_start: {test_start}")
    print(f"train_end: {train_end}")
    print(f"train_rows: {len(cleaned_train)}")
    print(f"test_rows: {len(test)}")
    print(f"overlap_dates_in_raw_train: {len(overlap_dates)}")
    print(f"train_end < test_start: {train_end < test_start}")
    if overlap_dates:
        print("overlap_dates_sample:", overlap_dates[:5])

    _scan_forward_columns(train.columns, test.columns)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit OOF/time split for leakage.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to dataset root containing train.csv/test.csv")
    parser.add_argument("--folds-csv", type=str, default=None, help="CSV with fold split ranges")
    parser.add_argument("--date-col", type=str, default="date_id", help="Date column name")
    args = parser.parse_args()

    if args.folds_csv:
        df = pd.read_csv(args.folds_csv)
        _print_fold_report(df)
    elif args.data_root:
        _single_split_report(Path(args.data_root), args.date_col)
    else:
        raise SystemExit("Provide --folds-csv or --data-root")


if __name__ == "__main__":
    main()
