#!/usr/bin/env python3
"""
对比不同 bolt_min_chunk_distance 下 alpha/power sweep 的控制变量结果，输出 4 张图。

默认输入:
- output_bolt_sweep_train_offline_md3/viz_data
- output_bolt_sweep_train_offline_md2/viz_data

输出:
1) combo_similarity_md3.png
2) combo_similarity_md2.png
3) selected_count_delta_md2_minus_md3.png
4) chunk3_chunk7_indices_compare.png
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_matrix_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    tags = rows[0][1:]
    mat = []
    for r in rows[1:]:
        mat.append([float(x) if x else np.nan for x in r[1:]])
    return tags, np.array(mat, dtype=np.float32)


def read_wide_csv(path):
    with open(path, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    combos = [k for k in rows[0].keys() if k not in ("sample_path", "current_chunk_idx")]
    data = {}
    for r in rows:
        cidx = int(r["current_chunk_idx"])
        data[cidx] = {k: (r[k] or "").strip() for k in combos}
    return combos, data


def count_selected(cell: str):
    if not cell:
        return 0
    return len([x for x in cell.split(",") if x != ""])


def plot_similarity(tags, mat, title, out_png):
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(tags)))
    ax.set_yticks(np.arange(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tags, fontsize=9)
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label="Similarity")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_count_delta(combos, wide3, wide2, out_png):
    chunk_ids = sorted(set(wide3.keys()) & set(wide2.keys()))
    delta = np.zeros((len(chunk_ids), len(combos)), dtype=np.float32)
    for i, cidx in enumerate(chunk_ids):
        for j, combo in enumerate(combos):
            c3 = count_selected(wide3[cidx].get(combo, ""))
            c2 = count_selected(wide2[cidx].get(combo, ""))
            delta[i, j] = c2 - c3

    vmax = max(1.0, float(np.max(np.abs(delta))))
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(delta, cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(combos)))
    ax.set_yticks(np.arange(len(chunk_ids)))
    ax.set_xticklabels(combos, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"chunk_{x}" for x in chunk_ids], fontsize=9)
    ax.set_title("Selected Count Delta (min_dist=2 minus min_dist=3)")
    for i in range(delta.shape[0]):
        for j in range(delta.shape[1]):
            ax.text(j, i, f"{delta[i, j]:.0f}", ha="center", va="center", color="black", fontsize=8)
    fig.colorbar(im, ax=ax, label="count delta")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_chunk3_chunk7_table(combos, wide3, wide2, out_png):
    target_chunks = [3, 7]
    rows = []
    for combo in combos:
        row = [combo]
        for cidx in target_chunks:
            row.append(wide3.get(cidx, {}).get(combo, "-") or "-")
            row.append(wide2.get(cidx, {}).get(combo, "-") or "-")
        rows.append(row)

    headers = ["combo", "c3@md3", "c3@md2", "c7@md3", "c7@md2"]
    n_rows = len(rows) + 1
    n_cols = len(headers)
    fig_w = max(12, n_cols * 2.2)
    fig_h = max(6, n_rows * 0.55)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    for x in range(n_cols + 1):
        ax.plot([x, x], [0, n_rows], color="#bbbbbb", linewidth=0.9)
    for y in range(n_rows + 1):
        ax.plot([0, n_cols], [y, y], color="#bbbbbb", linewidth=0.9)

    for j, h in enumerate(headers):
        ax.text(j + 0.5, n_rows - 0.5, h, ha="center", va="center", fontsize=10, fontweight="bold")

    for i, row in enumerate(rows, start=1):
        y = n_rows - i - 0.5
        for j, cell in enumerate(row):
            ax.text(j + 0.5, y, cell, ha="center", va="center", fontsize=9)

    ax.set_title("Chunk 3/7 Selected Indices under min_dist 3 vs 2", pad=12)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot min_chunk_distance control-variable comparison")
    parser.add_argument("--sweep_root_md3", default="output_bolt_sweep_train_offline_md3")
    parser.add_argument("--sweep_root_md2", default="output_bolt_sweep_train_offline_md2")
    parser.add_argument("--out_dir", default="output_bolt_sweep_compare_md2_vs_md3")
    args = parser.parse_args()

    viz3 = os.path.join(args.sweep_root_md3, "viz_data")
    viz2 = os.path.join(args.sweep_root_md2, "viz_data")
    ensure_dir(args.out_dir)

    tags3, mat3 = read_matrix_csv(os.path.join(viz3, "combo_similarity_matrix.csv"))
    tags2, mat2 = read_matrix_csv(os.path.join(viz2, "combo_similarity_matrix.csv"))
    if tags3 != tags2:
        raise ValueError("Combo tags mismatch between md3 and md2 runs")
    combos = tags3

    _, wide3 = read_wide_csv(os.path.join(viz3, "selection_wide.csv"))
    _, wide2 = read_wide_csv(os.path.join(viz2, "selection_wide.csv"))

    out1 = os.path.join(args.out_dir, "01_combo_similarity_md3.png")
    out2 = os.path.join(args.out_dir, "02_combo_similarity_md2.png")
    out3 = os.path.join(args.out_dir, "03_selected_count_delta_md2_minus_md3.png")
    out4 = os.path.join(args.out_dir, "04_chunk3_chunk7_indices_compare.png")

    plot_similarity(combos, mat3, "Combo Similarity (min_chunk_distance=3)", out1)
    plot_similarity(combos, mat2, "Combo Similarity (min_chunk_distance=2)", out2)
    plot_count_delta(combos, wide3, wide2, out3)
    plot_chunk3_chunk7_table(combos, wide3, wide2, out4)

    print("=" * 80)
    print("Generated 4 comparison figures:")
    print(f"- {out1}")
    print(f"- {out2}")
    print(f"- {out3}")
    print(f"- {out4}")
    print("=" * 80)


if __name__ == "__main__":
    main()
