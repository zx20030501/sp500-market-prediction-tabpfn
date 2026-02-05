from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Optional

# Allow running from repo root.
REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT / "src"))

from run_tabpfn_pipeline import resolve_data_root, resolve_hf_home, resolve_ckpt_path, run_pipeline  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local scoring helper (Adjusted Sharpe).")
    parser.add_argument("--data-root", type=str, default=None, help="Path to hull-tactical-market-prediction.")
    parser.add_argument("--out-dir", type=str, default="outputs_local_score", help="Output directory.")
    parser.add_argument("--ckpt-path", type=str, default=None, help="Path to tabpfn-v2-regressor.ckpt.")
    parser.add_argument("--hf-home", type=str, default=None, help="Override HF_HOME cache directory.")
    parser.add_argument("--offline", action="store_true", help="Force HuggingFace offline mode.")
    parser.add_argument("--max-training-rows", type=int, default=None, help="Downsample training rows for speed.")
    parser.add_argument("--no-train", action="store_true", help="Skip training if a model is not found.")
    return parser.parse_args()


def load_metrics(out_dir: Path) -> Optional[dict]:
    metadata_path = out_dir / "artifacts" / "tabpfn_model" / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text(encoding="utf-8"))


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

    run_pipeline(
        data_root=data_root,
        out_dir=out_dir,
        ckpt_path=ckpt_path,
        train_if_needed=not args.no_train,
        max_training_rows=args.max_training_rows,
    )

    metrics = load_metrics(out_dir)
    if metrics is None:
        print("[WARN] metadata.json not found. No local score available.")
        return

    calib = metrics.get("calibration", {})
    print("=== Local Score (Calibration Window) ===")
    print(f"raw_sharpe: {calib.get('raw_sharpe')}")
    print(f"adjusted_sharpe: {calib.get('adjusted_sharpe')}")
    print(f"scale: {calib.get('scale')}")
    print(f"shift: {calib.get('shift')}")
    print(f"window: {calib.get('window')}")


if __name__ == "__main__":
    main()
