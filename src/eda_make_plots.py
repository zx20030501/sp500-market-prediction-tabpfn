from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_COL = "forward_returns"
SAMPLE_ROWS = 200_000


def resolve_data_root() -> Path:
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


def main() -> None:
    sns.set_theme(style="whitegrid")

    work_dir = Path(__file__).resolve().parent
    data_root = resolve_data_root()
    out_dir = work_dir / ".." / "assets" / "eda"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = data_root / "train.csv"
    print("[INFO] Train path:", train_path)
    train_df = pd.read_csv(train_path)

    if len(train_df) > SAMPLE_ROWS:
        eda_df = train_df.sample(SAMPLE_ROWS, random_state=42)
        print(f"[INFO] Using sampled {len(eda_df)} rows for plots (from {len(train_df)} total).")
    else:
        eda_df = train_df.copy()
        print(f"[INFO] Using full train ({len(eda_df)} rows) for plots.")

    # 1) Missing ratio top 30
    missing_ratio = train_df.isna().mean().sort_values(ascending=False)
    top_missing = missing_ratio.head(30)
    plt.figure(figsize=(14, 6))
    sns.barplot(x=top_missing.index, y=top_missing.values, palette="Reds")
    plt.xticks(rotation=90)
    plt.ylabel("Missing Ratio")
    plt.title("Top 30 Missing Ratio Columns")
    plt.tight_layout()
    plt.savefig(out_dir / "missing_top30.png", dpi=150)
    plt.close()
    print("[INFO] Saved:", out_dir / "missing_top30.png")

    # 2) Correlation heatmap (top 25)
    numeric_cols = eda_df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in numeric_cols:
        corr_matrix = eda_df[numeric_cols].corr()
        target_corr = corr_matrix[TARGET_COL].dropna().sort_values(key=lambda s: -s.abs())
        top_feats = target_corr.head(25).index.tolist()
        plt.figure(figsize=(14, 10))
        sns.heatmap(corr_matrix.loc[top_feats, top_feats], cmap="coolwarm", center=0, annot=False)
        plt.title("Correlation Heatmap (Top 25 incl. Target)")
        plt.tight_layout()
        plt.savefig(out_dir / "corr_top25_heatmap.png", dpi=150)
        plt.close()
        print("[INFO] Saved:", out_dir / "corr_top25_heatmap.png")
    else:
        print(f"[WARN] {TARGET_COL} not in numeric columns; skip correlation heatmap.")

    # 3) PCA explained variance
    feature_cols = [c for c in numeric_cols if c != TARGET_COL]
    if feature_cols:
        pca_df = eda_df[feature_cols].fillna(0.0)
        scaler = StandardScaler()
        scaled = scaler.fit_transform(pca_df)
        n_components = min(20, scaled.shape[1])
        pca = PCA(n_components=n_components, random_state=42)
        pca.fit(scaled)

        explained_ratio = pca.explained_variance_ratio_
        cumulative = explained_ratio.cumsum()

        idx = np.arange(n_components)
        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax1.bar(idx, explained_ratio, color="steelblue", label="Variance Ratio")
        ax1.set_xlabel("Principal Component")
        ax1.set_ylabel("Explained Variance Ratio")

        ax2 = ax1.twinx()
        ax2.plot(idx, cumulative, marker="o", color="orange", label="Cumulative")
        ax2.set_ylabel("Cumulative Ratio")

        ax1.set_xticks(idx)
        ax1.set_xticklabels([f"PC{i + 1}" for i in idx], rotation=45)
        fig.suptitle("PCA Explained Variance")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc="upper right")

        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(out_dir / "pca_explained_variance.png", dpi=150)
        plt.close(fig)
        print("[INFO] Saved:", out_dir / "pca_explained_variance.png")
    else:
        print("[WARN] No feature columns for PCA plots.")

    print("[DONE] EDA plots saved to", out_dir)


if __name__ == "__main__":
    main()
