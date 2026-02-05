import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Patch a Kaggle notebook to cap TabPFN training rows.")
    parser.add_argument(
        "--notebook",
        type=str,
        default="kaggle-submit-tabtfn.ipynb",
        help="Path to the notebook to patch.",
    )
    parser.add_argument(
        "--max-training-rows",
        type=int,
        default=20000,
        help="Maximum rows used for TabPFN fit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.notebook)
    nb = json.loads(path.read_text(encoding="utf-8"))
    modified = False

    def find_cell(containing: str):
        for cell in nb.get("cells", []):
            if cell.get("cell_type") != "code":
                continue
            if containing in "".join(cell.get("source", [])):
                return cell
        return None

    service_cell = find_cell("class TabPFNService")
    if service_cell is None:
        raise SystemExit("service cell not found")
    service_lines = service_cell["source"]

    if not any("import numpy as np" in line for line in service_lines):
        for idx, line in enumerate(service_lines):
            if "from scipy.optimize import Bounds" in line:
                service_lines.insert(idx + 1, "import numpy as np\n")
                modified = True
                break

    for idx, line in enumerate(service_lines):
        if line.strip().startswith("def __init__("):
            new_line = line.replace(
                "calibration_window: int = 180)",
                f"calibration_window: int = 180, max_training_rows: Optional[int] = {args.max_training_rows})",
            )
            if new_line != line:
                service_lines[idx] = new_line
                modified = True
            init_indent = line[: len(line) - len(line.lstrip(" "))]
            break
    else:
        raise SystemExit("__init__ not found")

    inserted_attr = False
    for idx, line in enumerate(service_lines):
        if "self.calibration_window" in line:
            service_lines.insert(
                idx + 1,
                f"{init_indent}        self.max_training_rows = max_training_rows  # limit training samples for runtime control\n",
            )
            modified = True
            inserted_attr = True
            break
    if not inserted_attr:
        raise SystemExit("calibration_window assignment not found")

    train_block = [
        "        train_X = features.train_features\n",
        "        train_y = features.train_with_target[TARGET_COL]\n",
        "        original_rows = len(train_X)\n",
        "        if self.max_training_rows is not None and original_rows > self.max_training_rows:\n",
        "            rng = np.random.default_rng(self.seed)\n",
        "            idx = rng.choice(original_rows, self.max_training_rows, replace=False)\n",
        "            train_X = train_X.iloc[idx].reset_index(drop=True)\n",
        "            train_y = train_y.iloc[idx].reset_index(drop=True)\n",
        "            print(f\"TabPFN fit: downsampled from {original_rows} -> {len(train_X)} rows for faster training\")\n",
        "        model.fit(train_X, train_y)\n",
    ]
    fit_line_idx = None
    for i, line in enumerate(service_lines):
        if "model.fit(" in line and "train_features" in line:
            fit_line_idx = i
            break
    if fit_line_idx is None:
        raise SystemExit("original model.fit line not found")
    del service_lines[fit_line_idx]
    for line in reversed(train_block):
        service_lines.insert(fit_line_idx, line)
    modified = True

    run_cell = find_cell("def run_pipeline")
    if run_cell is None:
        raise SystemExit("run cell not found")
    run_lines = run_cell["source"]
    for idx, line in enumerate(run_lines):
        if line.startswith("def run_pipeline("):
            run_lines[idx] = (
                "def run_pipeline(train_if_needed: bool = True, max_training_rows: Optional[int] = "
                f"{args.max_training_rows}):\n"
            )
            run_lines.insert(idx + 1, "    \"\"\"Run the pipeline with an optional training-row cap for faster experiments.\"\"\"\n")
            modified = True
            break
    else:
        raise SystemExit("run_pipeline definition not found")

    for idx, line in enumerate(run_lines):
        if "TabPFNService(" in line:
            run_lines[idx] = "    service = TabPFNService(paths, max_training_rows=max_training_rows)\n"
            modified = True
            break

    if modified:
        path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Patched notebook: {path}")
    else:
        print("No changes made.")


if __name__ == "__main__":
    main()
