#!/usr/bin/env python3
"""
Plot merged BOLT training metrics from two log files.

Merge rule:
- run_419 epoch 0..19   -> merged index 0..19
- run_420 epoch 0..14   -> merged index 20..34

It generates one figure per metric:
  mse, avg, |gamma|, |gamma|max, r, dW_rel, H_attn, p_max
"""

import argparse
import os
import re
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


# Support both UTF-8 gamma and mojibake gamma from terminal logs.
GAMMA_TOKEN = r"(?:γ|Î³)"
NUM = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"

POSTFIX_PATTERN = re.compile(
    rf"\[Epoch\s+(\d+)\]:.*?"
    rf"mse=({NUM}),\s*avg=({NUM}),\s*\|{GAMMA_TOKEN}\|=({NUM}),\s*"
    rf"\|{GAMMA_TOKEN}\|max=({NUM}),\s*r=({NUM}),\s*dW_rel=({NUM}),\s*"
    rf"H_attn=({NUM}),\s*p_max=({NUM})",
    re.DOTALL,
)


def parse_epoch_metrics(log_path: str) -> Dict[int, Dict[str, float]]:
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    metrics_by_epoch: Dict[int, Dict[str, float]] = {}
    for m in POSTFIX_PATTERN.finditer(text):
        epoch = int(m.group(1))
        metrics_by_epoch[epoch] = {
            "mse": float(m.group(2)),
            "avg": float(m.group(3)),
            "gamma_abs": float(m.group(4)),
            "gamma_abs_max": float(m.group(5)),
            "r": float(m.group(6)),
            "dW_rel": float(m.group(7)),
            "H_attn": float(m.group(8)),
            "p_max": float(m.group(9)),
        }
    return metrics_by_epoch


def build_merged_series(
    run419_metrics: Dict[int, Dict[str, float]],
    run420_metrics: Dict[int, Dict[str, float]],
) -> Tuple[List[int], Dict[str, List[float]]]:
    merged_idx: List[int] = []
    merged_values: Dict[str, List[float]] = {
        "mse": [],
        "avg": [],
        "gamma_abs": [],
        "gamma_abs_max": [],
        "r": [],
        "dW_rel": [],
        "H_attn": [],
        "p_max": [],
    }

    # 419: epoch 0..19 -> idx 0..19
    for ep in range(0, 20):
        if ep not in run419_metrics:
            print(f"[WARN] run419 missing epoch {ep}, skip")
            continue
        merged_idx.append(ep)
        row = run419_metrics[ep]
        for k in merged_values:
            merged_values[k].append(row[k])

    # 420: epoch 0..14 -> idx 20..34
    for ep in range(0, 50):
        if ep not in run420_metrics:
            print(f"[WARN] run420 missing epoch {ep}, skip")
            continue
        merged_idx.append(ep + 20)
        row = run420_metrics[ep]
        for k in merged_values:
            merged_values[k].append(row[k])

    return merged_idx, merged_values


def plot_one(
    x: List[int],
    y: List[float],
    ylabel: str,
    title: str,
    save_path: str,
) -> None:
    plt.figure(figsize=(10, 4.5))
    plt.plot(x, y, marker="o", linewidth=1.8, markersize=4)
    plt.axvline(19.5, linestyle="--", linewidth=1.0, color="gray", alpha=0.8)
    plt.text(1, max(y), "run419", fontsize=9, alpha=0.8, va="top")
    plt.text(22, max(y), "run420", fontsize=9, alpha=0.8, va="top")
    plt.xlabel("Merged index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot merged BOLT metrics (419 + 420).")
    parser.add_argument(
        "--log419",
        type=str,
        default="/root/autodl-fs/output/bolt_ref_attn2_419/train_0419_2252.log",
        help="Path to run 419 log",
    )
    parser.add_argument(
        "--log420",
        type=str,
        default="/root/autodl-fs/output/bolt_ref_attn2_420_resume_from_419/train_0420_0952.log",
        help="Path to run 420 log",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/root/autodl-fs/output/bolt_ref_attn2_420_resume_from_419/plots_merged_419_420",
        help="Output directory for figures",
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    run419 = parse_epoch_metrics(args.log419)
    run420 = parse_epoch_metrics(args.log420)
    print(f"[INFO] parsed run419 epochs: {sorted(run419.keys())[:3]} ... {sorted(run419.keys())[-3:]}")
    print(f"[INFO] parsed run420 epochs: {sorted(run420.keys())[:3]} ... {sorted(run420.keys())[-3:]}")

    x, vals = build_merged_series(run419, run420)
    print(f"[INFO] merged points: {len(x)} (expected up to 35)")

    specs = [
        ("mse", "mse", "mse vs merged index", "metric_mse.png"),
        ("avg", "avg", "avg vs merged index", "metric_avg.png"),
        ("gamma_abs", "|gamma|", "|gamma| vs merged index", "metric_gamma_abs.png"),
        ("gamma_abs_max", "|gamma|max", "|gamma|max vs merged index", "metric_gamma_abs_max.png"),
        ("r", "r", "r vs merged index", "metric_r.png"),
        ("dW_rel", "dW_rel", "dW_rel vs merged index", "metric_dW_rel.png"),
        ("H_attn", "H_attn", "H_attn vs merged index", "metric_H_attn.png"),
        ("p_max", "p_max", "p_max vs merged index", "metric_p_max.png"),
    ]

    for key, ylabel, title, fname in specs:
        save_path = os.path.join(args.outdir, fname)
        plot_one(x, vals[key], ylabel=ylabel, title=title, save_path=save_path)
        print(f"[SAVE] {save_path}")


if __name__ == "__main__":
    main()
