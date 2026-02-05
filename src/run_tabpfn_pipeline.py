from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple
from warnings import warn

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import Bounds, minimize
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


@dataclass
class Paths:
    data_root: Path
    train_csv: Path
    test_csv: Path
    out_dir: Path
    model_dir: Path
    submission_csv: Path
    submission_parquet: Path
    metadata_path: Path


def resolve_paths(data_root: Path, out_dir: Path, verbose: bool = True) -> Paths:
    model_dir = out_dir / "artifacts" / "tabpfn_model"
    paths = Paths(
        data_root=data_root,
        train_csv=data_root / "train.csv",
        test_csv=data_root / "test.csv",
        out_dir=out_dir,
        model_dir=model_dir,
        submission_csv=out_dir / "submission.csv",
        submission_parquet=out_dir / "submission.parquet",
        metadata_path=model_dir / "metadata.json",
    )
    if verbose:
        print("=== Paths ===")
        for key, value in paths.__dict__.items():
            print(f"{key:18s}: {value}")
    return paths


@dataclass
class DataBundle:
    train: pd.DataFrame
    test: pd.DataFrame
    overlap_dates: List[int]


def load_data(paths: Paths) -> DataBundle:
    train_df = pd.read_csv(paths.train_csv)
    test_df = pd.read_csv(paths.test_csv)
    test_start = int(test_df["date_id"].min())
    leak_mask = train_df["date_id"] >= test_start
    overlap_dates = sorted(train_df.loc[leak_mask, "date_id"].unique().tolist())
    cleaned_train = train_df.loc[~leak_mask].sort_values("date_id").reset_index(drop=True)
    cleaned_test = test_df.sort_values("date_id").reset_index(drop=True)
    print(f"Train rows: {len(cleaned_train)} | Test rows: {len(cleaned_test)} | Overlap dates: {len(overlap_dates)}")
    return DataBundle(train=cleaned_train, test=cleaned_test, overlap_dates=overlap_dates)


TARGET_COL = "forward_returns"
LAG_SOURCE_COLUMNS: Tuple[str, ...] = (
    "forward_returns",
    "market_forward_excess_returns",
    "risk_free_rate",
)
FORWARD_KEYWORDS: Tuple[str, ...] = ("forward", "future", "lead", "next")
SAFE_FORWARD_PREFIXES: Tuple[str, ...] = ("lagged_",)
EXCLUDE_COLUMNS: Tuple[str, ...] = (
    "forward_returns",
    "risk_free_rate",
    "market_forward_excess_returns",
)


def _is_forward_looking(name: str) -> bool:
    lower = name.lower()
    if any(lower.startswith(prefix) for prefix in SAFE_FORWARD_PREFIXES):
        return False
    return any(keyword in lower for keyword in FORWARD_KEYWORDS)


@dataclass
class FeatureBundle:
    feature_columns: List[str]
    train_features: pd.DataFrame
    test_features: pd.DataFrame
    train_with_target: pd.DataFrame
    calibration_frame: pd.DataFrame


def build_features(bundle: DataBundle) -> FeatureBundle:
    train_df = bundle.train.copy()
    test_df = bundle.test.copy()
    for col in LAG_SOURCE_COLUMNS:
        train_df[f"lagged_{col}"] = train_df[col].shift(1)
    lag_columns = [f"lagged_{col}" for col in LAG_SOURCE_COLUMNS]
    train_df = train_df.dropna(subset=lag_columns).reset_index(drop=True)

    base_exclude = set(EXCLUDE_COLUMNS)
    feature_columns = [
        col
        for col in train_df.columns
        if col in test_df.columns and col not in base_exclude and not _is_forward_looking(col)
    ]
    feature_columns.sort()

    train_features = train_df.reindex(columns=feature_columns, fill_value=np.nan).fillna(0)
    test_features = test_df.reindex(columns=feature_columns, fill_value=np.nan).fillna(0)
    train_with_target = train_features.copy()
    train_with_target[TARGET_COL] = train_df[TARGET_COL].to_numpy(dtype=np.float64)
    calibration_frame = train_df[["date_id", "forward_returns", "risk_free_rate"]].copy()

    print(f"Feature columns: {len(feature_columns)}")
    return FeatureBundle(feature_columns, train_features, test_features, train_with_target, calibration_frame)


def prepare_features(df: pd.DataFrame, feature_columns: Sequence[str]) -> pd.DataFrame:
    return df.reindex(columns=feature_columns, fill_value=np.nan).fillna(0)


@dataclass
class CalibrationResult:
    scale: float
    shift: float
    raw_sharpe: float
    adjusted_sharpe: float

    @property
    def params(self) -> Tuple[float, float]:
        return (self.scale, self.shift)


def adjusted_sharpe(
    solution: pd.DataFrame,
    positions: np.ndarray,
    min_position: float,
    max_position: float,
) -> float:
    if solution.empty:
        return 0.0
    df = solution[["forward_returns", "risk_free_rate"]].copy()
    clipped_positions = np.clip(np.asarray(positions, dtype=np.float64), min_position, max_position)
    df["position"] = clipped_positions

    strategy_returns = df["risk_free_rate"] * (1 - df["position"]) + df["position"] * df["forward_returns"]
    excess = strategy_returns - df["risk_free_rate"]
    if len(excess) == 0:
        return 0.0

    cum_excess = float(np.prod(1 + excess))
    mean_excess = cum_excess ** (1 / len(df)) - 1
    std_excess = float(strategy_returns.std(ddof=0))
    if std_excess == 0:
        return 0.0

    annual_days = 252
    sharpe = mean_excess / std_excess * np.sqrt(annual_days)
    market_excess = df["forward_returns"] - df["risk_free_rate"]
    market_cum = float(np.prod(1 + market_excess))
    market_mean = market_cum ** (1 / len(df)) - 1 if len(df) else 0.0
    market_std = float(market_excess.std(ddof=0))
    market_vol = market_std * np.sqrt(annual_days) * 100
    strat_vol = std_excess * np.sqrt(annual_days) * 100

    excess_vol_penalty = 1 + max(0, strat_vol / market_vol - 1.2) if market_vol > 0 else 1
    return_gap = max(0, (market_mean - mean_excess) * 100 * annual_days)
    return_penalty = 1 + (return_gap ** 2) / 100

    score = sharpe / (excess_vol_penalty * return_penalty)
    return float(min(score, 1_000_000))


def apply_calibration(values: np.ndarray, params: Tuple[float, float], min_position: float, max_position: float) -> np.ndarray:
    scale, shift = params
    arr = np.asarray(values, dtype=np.float64)
    return np.clip(arr * scale + shift, min_position, max_position)


def calibrate_predictions(
    train_frame: pd.DataFrame,
    base_predictions: pd.Series,
    window: int,
    bounds: Bounds,
    min_position: float,
    max_position: float,
) -> CalibrationResult:
    if base_predictions is None or len(base_predictions) == 0:
        return CalibrationResult(scale=1.0, shift=0.0, raw_sharpe=0.0, adjusted_sharpe=0.0)

    work = train_frame[["date_id", "forward_returns", "risk_free_rate"]].copy()
    work["model_pred"] = base_predictions.values
    work = work.sort_values("date_id")
    if len(work) > window:
        work = work.tail(window)
    work = work.set_index("date_id")

    def objective(params: np.ndarray) -> float:
        adjusted = apply_calibration(work["model_pred"].values, params, min_position, max_position)
        return -adjusted_sharpe(work[["forward_returns", "risk_free_rate"]], adjusted, min_position, max_position)

    initial = np.array([1.0, 0.0])
    try:
        result = minimize(
            objective,
            x0=initial,
            method="Powell",
            bounds=bounds,
            options={"xtol": 1e-4, "ftol": 1e-4, "maxiter": 500},
        )
        if not result.success:
            warn(f"Calibration did not converge: {result.message}")
            scale, shift = initial
        else:
            scale, shift = result.x
    except Exception as exc:
        warn(f"Calibration failed, using identity transform: {exc}")
        scale, shift = initial

    adjusted = apply_calibration(work["model_pred"].values, (scale, shift), min_position, max_position)
    raw_score = adjusted_sharpe(work[["forward_returns", "risk_free_rate"]], work["model_pred"].values, min_position, max_position)
    adj_score = adjusted_sharpe(work[["forward_returns", "risk_free_rate"]], adjusted, min_position, max_position)
    return CalibrationResult(scale=float(scale), shift=float(shift), raw_sharpe=raw_score, adjusted_sharpe=adj_score)


class TabPFNService:
    def __init__(
        self,
        paths: Paths,
        seed: int = 42,
        min_position: float = 0.0,
        max_position: float = 2.0,
        calibration_window: int = 180,
        max_training_rows: Optional[int] = None,
        ckpt_path: Optional[Path] = None,
    ) -> None:
        self.paths = paths
        self.seed = seed
        self.min_position = min_position
        self.max_position = max_position
        self.calibration_window = calibration_window
        self.max_training_rows = max_training_rows
        self.ckpt_path = ckpt_path
        self.model = None
        self.feature_columns: List[str] = []
        self.calibration: Optional[CalibrationResult] = None
        self.model_source: str = "unknown"

    @property
    def device(self) -> str:
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, features: FeatureBundle) -> None:
        self.paths.model_dir.mkdir(parents=True, exist_ok=True)
        model_kwargs = {
            "device": self.device,
            "random_state": self.seed,
            "ignore_pretraining_limits": True,
        }
        if self.ckpt_path is not None:
            model_kwargs["model_path"] = str(self.ckpt_path)
        model = TabPFNRegressor(**model_kwargs)

        train_x = features.train_features
        train_y = features.train_with_target[TARGET_COL]
        original_rows = len(train_x)
        if self.max_training_rows is not None and original_rows > self.max_training_rows:
            rng = np.random.default_rng(self.seed)
            idx = rng.choice(original_rows, self.max_training_rows, replace=False)
            train_x = train_x.iloc[idx].reset_index(drop=True)
            train_y = train_y.iloc[idx].reset_index(drop=True)
            print(f"TabPFN fit: downsampled from {original_rows} -> {len(train_x)} rows")

        model.fit(train_x, train_y)
        joblib.dump({"model": model, "feature_columns": features.feature_columns}, self.paths.model_dir / "tabpfn_model.joblib")
        self.model = model
        self.feature_columns = list(features.feature_columns)
        self.model_source = "trained"

    def load(self) -> bool:
        model_path = self.paths.model_dir / "tabpfn_model.joblib"
        if not model_path.exists():
            return False
        saved = joblib.load(model_path)
        self.model = saved["model"]
        self.feature_columns = list(saved["feature_columns"])
        self.model_source = "pretrained"
        return True

    def predict_raw(self, df: pd.DataFrame) -> pd.Series:
        feats = prepare_features(df, self.feature_columns)
        preds = self.model.predict(feats)
        return pd.Series(preds, index=feats.index, name=TARGET_COL)

    def ensure_calibration(self, features: FeatureBundle) -> None:
        bounds = Bounds([0.8, -0.5], [1.2, 0.5])
        train_preds = self.predict_raw(features.train_features)
        self.calibration = calibrate_predictions(
            features.calibration_frame,
            train_preds,
            window=self.calibration_window,
            bounds=bounds,
            min_position=self.min_position,
            max_position=self.max_position,
        )
        payload = {
            "source": self.model_source,
            "feature_columns": self.feature_columns,
            "calibration": {
                "scale": self.calibration.scale,
                "shift": self.calibration.shift,
                "raw_sharpe": self.calibration.raw_sharpe,
                "adjusted_sharpe": self.calibration.adjusted_sharpe,
                "window": self.calibration_window,
            },
        }
        self.paths.model_dir.mkdir(parents=True, exist_ok=True)
        self.paths.metadata_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        raw = self.predict_raw(df)
        if self.calibration is None:
            return raw
        adjusted = apply_calibration(raw.values, self.calibration.params, self.min_position, self.max_position)
        return pd.Series(adjusted, index=raw.index, name=raw.name)


def run_pipeline(
    data_root: Path,
    out_dir: Path,
    ckpt_path: Optional[Path],
    train_if_needed: bool = True,
    max_training_rows: Optional[int] = None,
) -> pd.DataFrame:
    paths = resolve_paths(data_root, out_dir, verbose=True)
    bundle = load_data(paths)
    features = build_features(bundle)

    service = TabPFNService(
        paths,
        max_training_rows=max_training_rows,
        ckpt_path=ckpt_path,
    )

    loaded = service.load()
    if not loaded:
        if not train_if_needed:
            raise RuntimeError("No pretrained TabPFN model found and training is disabled.")
        print("No saved model found -> training TabPFN ...")
        service.fit(features)
    else:
        print(f"Loaded pretrained model from {paths.model_dir}")

    service.ensure_calibration(features)
    print("Calibration scale/shift:", service.calibration.scale, service.calibration.shift)

    test_preds = service.predict(features.test_features)
    if "row_id" in bundle.test.columns:
        ids = bundle.test["row_id"].to_numpy()
        row_col = "row_id"
    elif "id" in bundle.test.columns:
        ids = bundle.test["id"].to_numpy()
        row_col = "id"
    else:
        ids = np.arange(len(test_preds))
        row_col = "row_id"

    submission = pd.DataFrame({row_col: ids, "prediction": test_preds})
    submission.to_parquet(paths.submission_parquet, index=False)
    submission.to_csv(paths.submission_csv, index=False)
    print("Saved submission:", submission.shape, "->", paths.submission_csv)
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabPFN pipeline with calibration for Hull.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to hull-tactical-market-prediction.")
    parser.add_argument("--out-dir", type=str, default=".", help="Output directory for submissions and artifacts.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to tabpfn-v2-regressor.ckpt.")
    parser.add_argument("--hf-home", type=str, default=None, help="Override HF_HOME cache directory.")
    parser.add_argument("--offline", action="store_true", help="Force HuggingFace offline mode.")
    parser.add_argument("--no-train", action="store_true", help="Skip training if a model is not found.")
    parser.add_argument("--max-training-rows", type=int, default=None, help="Downsample training rows for speed.")
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

    _ = run_pipeline(
        data_root=data_root,
        out_dir=out_dir,
        ckpt_path=ckpt_path,
        train_if_needed=not args.no_train,
        max_training_rows=args.max_training_rows,
    )


if __name__ == "__main__":
    main()
