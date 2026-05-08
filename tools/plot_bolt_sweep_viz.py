#!/usr/bin/env python3
"""
根据 prepare_bolt_sweep_viz_data.py 输出的 CSV 画图。
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_similarity_matrix(path):
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    headers = rows[0][1:]
    data = []
    for r in rows[1:]:
        vals = [float(x) if x != "" else np.nan for x in r[1:]]
        data.append(vals)
    return headers, np.array(data, dtype=np.float32)


def plot_combo_similarity(matrix_csv, out_png):
    tags, mat = read_similarity_matrix(matrix_csv)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(len(tags)))
    ax.set_yticks(np.arange(len(tags)))
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(tags, fontsize=9)
    ax.set_title("Combo Similarity (Jaccard Mean)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Similarity")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def build_chunk_combo_matrix(selection_wide_csv):
    rows = read_csv_rows(selection_wide_csv)
    if not rows:
        raise RuntimeError("selection_wide.csv is empty")
    combo_tags = [k for k in rows[0].keys() if k not in ("sample_path", "current_chunk_idx")]

    chunk_ids = []
    count_mat = []
    text_mat = []
    for r in rows:
        chunk_idx = int(r["current_chunk_idx"])
        chunk_ids.append(chunk_idx)
        count_row = []
        text_row = []
        for tag in combo_tags:
            cell = (r[tag] or "").strip()
            if cell == "":
                count_row.append(0)
                text_row.append("-")
            else:
                n = len([x for x in cell.split(",") if x != ""])
                count_row.append(n)
                text_row.append(cell)
        count_mat.append(count_row)
        text_mat.append(text_row)
    return combo_tags, chunk_ids, np.array(count_mat, dtype=np.int32), text_mat


def plot_selected_count_heatmap(selection_wide_csv, out_png):
    combo_tags, chunk_ids, count_mat, _ = build_chunk_combo_matrix(selection_wide_csv)
    fig, ax = plt.subplots(figsize=(12, 6))
    vmax = max(1, int(np.max(count_mat)))
    im = ax.imshow(count_mat, cmap="YlOrRd", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(len(combo_tags)))
    ax.set_yticks(np.arange(len(chunk_ids)))
    ax.set_xticklabels(combo_tags, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels([f"chunk_{c}" for c in chunk_ids], fontsize=9)
    ax.set_title("Selected Count per Chunk and Combo")
    for i in range(count_mat.shape[0]):
        for j in range(count_mat.shape[1]):
            ax.text(j, i, str(count_mat[i, j]), ha="center", va="center", color="black", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("selected_count")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_selected_indices_grid(selection_wide_csv, out_png):
    combo_tags, chunk_ids, _, text_mat = build_chunk_combo_matrix(selection_wide_csv)
    n_rows = len(chunk_ids) + 1
    n_cols = len(combo_tags) + 1

    fig_w = max(10, n_cols * 1.6)
    fig_h = max(4, n_rows * 0.7)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, n_cols)
    ax.set_ylim(0, n_rows)
    ax.axis("off")

    for x in range(n_cols + 1):
        ax.plot([x, x], [0, n_rows], color="#cccccc", linewidth=0.8)
    for y in range(n_rows + 1):
        ax.plot([0, n_cols], [y, y], color="#cccccc", linewidth=0.8)

    ax.text(0.5, n_rows - 0.5, "chunk", ha="center", va="center", fontsize=10, fontweight="bold")
    for j, tag in enumerate(combo_tags, start=1):
        ax.text(j + 0.5, n_rows - 0.5, tag, ha="center", va="center", fontsize=9, rotation=20)

    for i, chunk_idx in enumerate(chunk_ids, start=1):
        y = n_rows - i - 0.5
        ax.text(0.5, y, f"chunk_{chunk_idx}", ha="center", va="center", fontsize=9)
        for j, txt in enumerate(text_mat[i - 1], start=1):
            ax.text(j + 0.5, y, txt if txt else "-", ha="center", va="center", fontsize=8)

    ax.set_title("Selected Indices Grid", pad=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot visualization figures for BOLT sweep")
    parser.add_argument("--sweep_root", default="output_bolt_sweep_train_offline")
    parser.add_argument("--viz_dir", default=None, help="Directory containing viz CSVs")
    parser.add_argument("--plot_dir", default=None, help="Output plot directory")
    args = parser.parse_args()

    viz_dir = args.viz_dir or os.path.join(args.sweep_root, "viz_data")
    plot_dir = args.plot_dir or os.path.join(viz_dir, "plots")
    ensure_dir(plot_dir)

    matrix_csv = os.path.join(viz_dir, "combo_similarity_matrix.csv")
    wide_csv = os.path.join(viz_dir, "selection_wide.csv")

    out1 = os.path.join(plot_dir, "combo_similarity_heatmap.png")
    out2 = os.path.join(plot_dir, "chunk_selected_count_heatmap.png")
    out3 = os.path.join(plot_dir, "chunk_selected_indices_grid.png")

    plot_combo_similarity(matrix_csv, out1)
    plot_selected_count_heatmap(wide_csv, out2)
    plot_selected_indices_grid(wide_csv, out3)

    print("=" * 80)
    print("Plots generated:")
    print(f"- {out1}")
    print(f"- {out2}")
    print(f"- {out3}")
    print("=" * 80)


if __name__ == "__main__":
    main()
