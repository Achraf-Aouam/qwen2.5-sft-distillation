"""Build GT-accuracy comparison plots across rank runs and a soft-label run."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


METRIC_ORDER = ["order", "vehicle", "invoice", "overall_weighted"]
METRIC_COLUMN = {metric: f"gt_acc/{metric}" for metric in METRIC_ORDER}


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Plot inline-eval GT accuracy comparisons (rank runs + soft-label run)."
    )
    parser.add_argument(
        "--r32-history",
        type=Path,
        default=repo_root / "inline_eval" / "inline_eval_r32" / "history.csv",
        help="CSV history path for rank-32 run.",
    )
    parser.add_argument(
        "--r64-history",
        type=Path,
        default=repo_root / "inline_eval" / "inline_eval_r64" / "history.csv",
        help="CSV history path for rank-64 run.",
    )
    parser.add_argument(
        "--r128-history",
        type=Path,
        default=repo_root / "inline_eval" / "inline_eval_r128" / "history.csv",
        help="CSV history path for rank-128 run.",
    )
    parser.add_argument(
        "--softlabel-history",
        type=Path,
        default=repo_root / "inline_eval" / "inline_eval_fourth_kd" / "history.csv",
        help="CSV history path for the soft-label run.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=repo_root / "inline_eval" / "plots_gt_acc",
        help="Directory where comparison PNGs will be saved.",
    )
    return parser.parse_args()


def _to_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df[column], errors="coerce")


def load_rank_series(
    history_path: Path, metric: str
) -> Tuple[List[float], List[float]]:
    metric_column = METRIC_COLUMN[metric]
    df = pd.read_csv(history_path)

    if "step" not in df.columns:
        raise ValueError(f"Missing 'step' column in {history_path}")
    if metric_column not in df.columns:
        raise ValueError(f"Missing '{metric_column}' column in {history_path}")

    slim = pd.DataFrame(
        {
            "step": _to_numeric_series(df, "step"),
            "value": _to_numeric_series(df, metric_column),
        }
    ).dropna(subset=["step", "value"])

    slim = slim.sort_values("step", kind="mergesort")
    return slim["step"].tolist(), slim["value"].tolist()


def build_plots(
    rank_histories: Dict[str, Path],
    softlabel_history: Path,
    out_dir: Path,
) -> List[Path]:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install it with: pip install matplotlib"
        ) from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: List[Path] = []

    if not softlabel_history.exists():
        raise FileNotFoundError(f"Soft-label history not found: {softlabel_history}")

    for metric in METRIC_ORDER:
        plt.figure(figsize=(11, 6))

        for label, history_path in rank_histories.items():
            steps, values = load_rank_series(history_path, metric)
            plt.plot(steps, values, marker="o", linewidth=2, label=label)

        soft_steps, soft_values = load_rank_series(softlabel_history, metric)
        plt.plot(
            soft_steps,
            soft_values,
            marker="o",
            linewidth=2,
            linestyle="--",
            label="soft label",
        )

        plt.title(f"gt_acc/{metric} over steps")
        plt.xlabel("step")
        plt.ylabel("accuracy (%)")
        plt.grid(True, alpha=0.35)
        plt.legend()

        out_path = out_dir / f"gt_acc_{metric}_compare_r32_r64_r128.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()

        saved_paths.append(out_path)

    return saved_paths


def main() -> int:
    args = parse_args()

    rank_histories = {
        "rank 32": args.r32_history,
        "rank 64": args.r64_history,
        "rank 128": args.r128_history,
    }

    saved_paths = build_plots(
        rank_histories=rank_histories,
        softlabel_history=args.softlabel_history,
        out_dir=args.out_dir,
    )

    print("Saved plots:")
    for path in saved_paths:
        print(f"- {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
