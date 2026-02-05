"""
Exploratory data analysis utilities for the Hull TabPFN project.

This script generates:
1) Descriptive stats (numeric/categorical)
2) Missing-value and data-quality reports
3) Correlations with target
4) PCA loadings and explained variance
5) A combined key-feature ranking
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_DEFAULT = "forward_returns"


@dataclass
class DatasetBundle:
    train: pd.DataFrame
    test: pd.DataFrame


def resolve_default_data_root() -> Path:
    env_root = os.getenv("HULL_DATA_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    kaggle_root = Path("/kaggle/input/hull-tactical-market-prediction")
    if kaggle_root.exists():
        return kaggle_root
    for candidate in [
        REPO_ROOT / "data" / "hull-tactical-market-prediction",
        REPO_ROOT / "input" / "hull-tactical-market-prediction",
    ]:
        if candidate.exists():
            return candidate
    return REPO_ROOT / "data" / "hull-tactical-market-prediction"


def load_bundle(data_root: Path) -> DatasetBundle:
    train_path = data_root / "train.csv"
    test_path = data_root / "test.csv"
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found: {train_path}")
    if not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    return DatasetBundle(train=train, test=test)


def describe_dataframe(df: pd.DataFrame, name: str, out_dir: Path) -> None:
    """Save descriptive stats for numeric and categorical columns."""
    numeric_report = df.describe(include=[np.number]).T
    numeric_report.to_csv(out_dir / f"{name}_numeric_stats.csv")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) > 0:
        categorical_report = df[cat_cols].describe().T
        if not categorical_report.empty:
            categorical_report.to_csv(out_dir / f"{name}_categorical_stats.csv")


def report_missing_values(train: pd.DataFrame, out_dir: Path) -> None:
    missing = train.isna().mean().rename("missing_ratio").sort_values(ascending=False)
    missing.to_csv(out_dir / "missing_ratio.csv", header=True)


def data_quality(train: pd.DataFrame, out_dir: Path) -> None:
    """Generate data-quality metrics."""
    metrics: List[Dict[str, object]] = []
    metrics.append({"metric": "rows", "value": len(train)})
    metrics.append({"metric": "columns", "value": train.shape[1]})
    duplicate_rows = train.duplicated().sum()
    metrics.append({"metric": "duplicate_rows", "value": int(duplicate_rows)})

    constant_cols = [c for c in train.columns if train[c].nunique(dropna=False) <= 1]
    metrics.append({"metric": "constant_columns", "value": len(constant_cols)})
    high_missing = train.isna().mean() >= 0.5
    metrics.append({"metric": "cols_missing_over_50pct", "value": int(high_missing.sum())})

    report = pd.DataFrame(metrics)
    report.to_csv(out_dir / "data_quality_overview.csv", index=False)

    details = pd.DataFrame({
        "column": train.columns,
        "dtype": train.dtypes.astype(str),
        "nunique": train.nunique(dropna=False),
        "missing_ratio": train.isna().mean(),
    })
    details.to_csv(out_dir / "data_quality_details.csv", index=False)


def correlation_analysis(train: pd.DataFrame, target_col: str, out_dir: Path) -> Tuple[pd.Series, pd.DataFrame]:
    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric columns available for correlation analysis.")
    corr_matrix = train[numeric_cols].corr()
    corr_matrix.to_csv(out_dir / "correlation_matrix.csv")

    target_corr = corr_matrix[target_col].dropna().sort_values(key=lambda s: -s.abs())
    target_corr.to_csv(out_dir / "target_correlations.csv", header=["correlation"])
    return target_corr, corr_matrix


def run_pca(train: pd.DataFrame, target_col: str, out_dir: Path, max_components: int = 20) -> Tuple[np.ndarray, List[str]]:
    numeric_df = train.select_dtypes(include=[np.number]).copy()
    features = [c for c in numeric_df.columns if c != target_col]
    work = numeric_df[features].fillna(0.0)
    if work.empty:
        raise ValueError("No numeric features to run PCA.")

    scaler = StandardScaler()
    scaled = scaler.fit_transform(work)
    n_components = min(max_components, scaled.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    _ = pca.fit_transform(scaled)
    loadings = pca.components_

    explained = pd.DataFrame({
        "component": [f"PC{i + 1}" for i in range(len(pca.explained_variance_ratio_))],
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "cumulative_ratio": np.cumsum(pca.explained_variance_ratio_),
    })
    explained.to_csv(out_dir / "pca_explained_variance.csv", index=False)

    loadings_df = pd.DataFrame(loadings.T, index=features, columns=explained["component"])
    loadings_df.to_csv(out_dir / "pca_loadings.csv")

    summary_rows: List[Dict[str, object]] = []
    for component in loadings_df.columns:
        top_features = loadings_df[component].abs().sort_values(ascending=False).head(10)
        summary_rows.append({
            "component": component,
            "top_features": ", ".join(top_features.index.tolist()),
        })
    pd.DataFrame(summary_rows).to_csv(out_dir / "pca_top_features.csv", index=False)
    return loadings, features


def identify_key_features(
    target_corr: pd.Series,
    pca_loadings: np.ndarray,
    feature_names: List[str],
    top_k: int,
    out_dir: Path,
) -> pd.DataFrame:
    """Combine target correlation and PCA loadings to rank features."""
    corr_score = target_corr.reindex(feature_names).abs().fillna(0.0)
    corr_norm = corr_score / (corr_score.max() or 1.0)

    if pca_loadings.size == 0:
        loading_norm = pd.Series(0.0, index=feature_names)
    else:
        leading_components = pca_loadings[: min(5, pca_loadings.shape[0])]
        loading_score = np.abs(leading_components).sum(axis=0)
        loading_norm = pd.Series(loading_score / (loading_score.max() or 1.0), index=feature_names)

    combined = 0.6 * corr_norm + 0.4 * loading_norm
    ranking = (
        pd.DataFrame({
            "feature": feature_names,
            "corr_score": corr_norm.values,
            "pca_score": loading_norm.loc[feature_names].values,
            "combined_score": combined.loc[feature_names].values,
        })
        .sort_values("combined_score", ascending=False)
        .head(top_k)
    )
    ranking.to_csv(out_dir / "key_features.csv", index=False)
    return ranking


def main() -> None:
    parser = argparse.ArgumentParser(description="TabPFN EDA helper.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=resolve_default_data_root(),
        help="Directory containing train.csv/test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eda_outputs"),
        help="Directory to write reports.",
    )
    parser.add_argument(
        "--target-col",
        type=str,
        default=TARGET_DEFAULT,
        help="Target column name.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=40,
        help="Number of key features to output.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Loading data from {args.data_root}")
    bundle = load_bundle(args.data_root)

    print("[INFO] Generating descriptive statistics...")
    describe_dataframe(bundle.train, "train", args.output_dir)
    describe_dataframe(bundle.test, "test", args.output_dir)
    report_missing_values(bundle.train, args.output_dir)
    data_quality(bundle.train, args.output_dir)

    print("[INFO] Running correlation analysis...")
    target_corr, _ = correlation_analysis(bundle.train, args.target_col, args.output_dir)

    print("[INFO] Running PCA...")
    loadings, feature_names = run_pca(bundle.train, args.target_col, args.output_dir)

    print("[INFO] Ranking key features...")
    ranking = identify_key_features(target_corr, loadings, feature_names, args.top_k, args.output_dir)
    print(ranking.head(10))

    print(f"[DONE] Reports saved to: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
