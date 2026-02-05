from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tabpfn import TabPFNRegressor

REPO_ROOT = Path(__file__).resolve().parents[1]


def resolve_data_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
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
    raise FileNotFoundError(
        "Dataset not found. Set --data-root or HULL_DATA_ROOT, or place the data in "
        "data/hull-tactical-market-prediction or input/hull-tactical-market-prediction."
    )


def resolve_hf_home(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env_home = os.getenv("HF_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return REPO_ROOT / ".hf_cache"


def resolve_ckpt_path(explicit: str | None, hf_home: Path) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.exists() else path
    candidate = hf_home / "tabpfn-v2-regressor.ckpt"
    return candidate if candidate.exists() else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal TabPFN baseline for the Hull competition.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to hull-tactical-market-prediction.")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for submissions.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to tabpfn-v2-regressor.ckpt.")
    parser.add_argument("--hf-home", type=str, default=None, help="Override HF_HOME cache directory.")
    parser.add_argument("--offline", action="store_true", help="Force HuggingFace offline mode.")
    parser.add_argument("--n-estimators", type=int, default=2, help="TabPFN n_estimators.")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu. Default: auto.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = resolve_data_root(args.data_root)
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    hf_home = resolve_hf_home(args.hf_home)
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"

    ckpt_path = resolve_ckpt_path(args.ckpt_path, hf_home)
    if ckpt_path is not None and ckpt_path.exists():
        print(f"Using ckpt: {ckpt_path}")
    else:
        print("No local ckpt found. TabPFN will download if online.")

    train = pd.read_csv(data_root / "train.csv")
    test = pd.read_csv(data_root / "test.csv")

    # Clean data leak: remove any train rows overlapping test dates.
    test_start = int(test["date_id"].min())
    train = train.loc[train["date_id"] < test_start].reset_index(drop=True)

    # Lag features.
    lag_cols = ["forward_returns", "market_forward_excess_returns", "risk_free_rate"]
    for col in lag_cols:
        train[f"lagged_{col}"] = train[col].shift(1)
    train = train.dropna().reset_index(drop=True)
    for col in lag_cols:
        if col in test.columns:
            test[f"lagged_{col}"] = test[col].shift(1)

    # Feature selection: remove forward-looking columns.
    exclude = {"forward_returns", "risk_free_rate", "market_forward_excess_returns"}
    forward_kw = ("forward", "future", "lead", "next")
    safe_prefix = ("lagged_",)

    def is_forward(name: str) -> bool:
        lower = name.lower()
        if any(lower.startswith(prefix) for prefix in safe_prefix):
            return False
        return any(keyword in lower for keyword in forward_kw)

    feature_cols = [
        col
        for col in train.columns
        if col in test.columns and col not in exclude and not is_forward(col)
    ]
    feature_cols.sort()

    x_train = train[feature_cols].fillna(0)
    y_train = train["forward_returns"].to_numpy()
    x_test = test[feature_cols].fillna(0)

    print("train shape", x_train.shape, "test shape", x_test.shape)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_kwargs = {
        "n_estimators": args.n_estimators,
        "device": device,
        "ignore_pretraining_limits": True,
    }
    if ckpt_path is not None:
        model_kwargs["model_path"] = str(ckpt_path)

    model = TabPFNRegressor(**model_kwargs)
    model.fit(x_train, y_train)
    preds = model.predict(x_test)

    if "row_id" in test.columns:
        ids = test["row_id"].to_numpy()
        id_col = "row_id"
    elif "id" in test.columns:
        ids = test["id"].to_numpy()
        id_col = "id"
    else:
        ids = np.arange(len(preds))
        id_col = "row_id"

    submission = pd.DataFrame({id_col: ids, "prediction": preds})
    submission.to_csv(out_dir / "submission.csv", index=False)
    submission.to_parquet(out_dir / "submission.parquet", index=False)
    print(submission.head())
    print("Saved submission", submission.shape)


if __name__ == "__main__":
    main()
