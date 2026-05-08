#!/usr/bin/env python3
"""
将 BOLT alpha/power sweep 输出整理为可视化友好数据。

输入:
- sweep_summary.json
- 各 combo 的 selection_records.json

输出:
- viz_data/selection_long.csv
- viz_data/selection_wide.csv
- viz_data/chunk_pairwise_jaccard.csv
- viz_data/combo_similarity_matrix.csv
- viz_data/summary.json
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from itertools import combinations


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_mean(values):
    return (sum(values) / len(values)) if values else None


def stringify_indices(indices):
    return ",".join(str(x) for x in indices)


def jaccard(a, b):
    sa, sb = set(a), set(b)
    union = sa | sb
    if not union:
        return 1.0
    inter = sa & sb
    return len(inter) / len(union)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def load_combo_files(sweep_root):
    summary_path = os.path.join(sweep_root, "sweep_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"sweep_summary.json not found: {summary_path}")
    summary = read_json(summary_path)

    combo_files = []
    for item in summary.get("combo_outputs", []):
        tag = item["tag"]
        path = item["selection_json"]
        if not os.path.isabs(path):
            path = os.path.join(sweep_root, path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"selection_records.json not found: {path}")
        combo_files.append((tag, path))
    if not combo_files:
        raise RuntimeError("No combo_outputs found in sweep_summary.json")
    return summary, combo_files


def main():
    parser = argparse.ArgumentParser(description="Prepare viz data from offline bolt sweep outputs")
    parser.add_argument(
        "--sweep_root",
        default="output_bolt_sweep_train_offline",
        help="Directory containing sweep_summary.json and combo folders",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory for viz files (default: <sweep_root>/viz_data)",
    )
    args = parser.parse_args()

    sweep_root = args.sweep_root
    out_dir = args.out_dir or os.path.join(sweep_root, "viz_data")
    ensure_dir(out_dir)

    summary, combo_files = load_combo_files(sweep_root)

    # 每个 key=(sample_path, current_chunk_idx) 下，各 combo 的 selected_indices
    chunk_combo_selected = defaultdict(dict)
    # key -> static info
    chunk_info = {}
    # long rows
    long_rows = []
    # 记录 combo 参数
    combo_meta = {}

    for tag, combo_json_path in combo_files:
        payload = read_json(combo_json_path)
        alpha = float(payload["alpha"])
        power = float(payload["power"])
        combo_meta[tag] = {"alpha": alpha, "power": power}

        for sample_entry in payload.get("records", []):
            sample_path = sample_entry["sample_path"]
            prompt_raw = sample_entry.get("prompt_raw", "")
            for chunk_entry in sample_entry.get("chunks", []):
                current_chunk_idx = int(chunk_entry["current_chunk_idx"])
                key = (sample_path, current_chunk_idx)
                selected_indices = chunk_entry.get("selected_indices", [])
                selected_details = chunk_entry.get("selected_details", [])
                candidate_indices = chunk_entry.get("candidate_indices", [])
                combined_scores = chunk_entry.get("combined_scores", [])

                avg_visual = safe_mean([d.get("visual", 0.0) for d in selected_details])
                avg_text = safe_mean([d.get("text", 0.0) for d in selected_details])
                avg_combined = safe_mean([d.get("combined", 0.0) for d in selected_details])
                avg_distance = safe_mean([d.get("distance_to_current", 0.0) for d in selected_details])

                long_rows.append(
                    {
                        "sample_path": sample_path,
                        "prompt_raw": prompt_raw,
                        "current_chunk_idx": current_chunk_idx,
                        "combo_tag": tag,
                        "alpha": alpha,
                        "power": power,
                        "num_candidates": int(chunk_entry.get("num_candidates", 0)),
                        "candidate_indices": stringify_indices(candidate_indices),
                        "selected_count": len(selected_indices),
                        "selected_indices": stringify_indices(selected_indices),
                        "avg_selected_visual": avg_visual,
                        "avg_selected_text": avg_text,
                        "avg_selected_combined": avg_combined,
                        "avg_selected_distance": avg_distance,
                        "combined_scores": json.dumps(combined_scores, ensure_ascii=False),
                    }
                )

                chunk_combo_selected[key][tag] = list(selected_indices)
                chunk_info[key] = {
                    "sample_path": sample_path,
                    "prompt_raw": prompt_raw,
                    "current_chunk_idx": current_chunk_idx,
                }

    # 写 long format
    long_csv = os.path.join(out_dir, "selection_long.csv")
    long_fields = [
        "sample_path",
        "prompt_raw",
        "current_chunk_idx",
        "combo_tag",
        "alpha",
        "power",
        "num_candidates",
        "candidate_indices",
        "selected_count",
        "selected_indices",
        "avg_selected_visual",
        "avg_selected_text",
        "avg_selected_combined",
        "avg_selected_distance",
        "combined_scores",
    ]
    with open(long_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=long_fields)
        writer.writeheader()
        writer.writerows(long_rows)

    # 写 wide format
    all_tags = [tag for tag, _ in combo_files]
    wide_csv = os.path.join(out_dir, "selection_wide.csv")
    wide_fields = ["sample_path", "current_chunk_idx"] + all_tags
    with open(wide_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=wide_fields)
        writer.writeheader()
        for key in sorted(chunk_combo_selected.keys(), key=lambda x: (x[0], x[1])):
            sample_path, current_chunk_idx = key
            row = {"sample_path": sample_path, "current_chunk_idx": current_chunk_idx}
            for tag in all_tags:
                row[tag] = stringify_indices(chunk_combo_selected[key].get(tag, []))
            writer.writerow(row)

    # 每个 chunk 内部，组合两两 jaccard
    pairwise_rows = []
    jaccard_pool = defaultdict(list)  # (tag_a, tag_b) -> [j1, j2, ...]
    for key, per_combo in chunk_combo_selected.items():
        sample_path, current_chunk_idx = key
        for tag_a, tag_b in combinations(all_tags, 2):
            sel_a = per_combo.get(tag_a, [])
            sel_b = per_combo.get(tag_b, [])
            jac = jaccard(sel_a, sel_b)
            inter = len(set(sel_a) & set(sel_b))
            uni = len(set(sel_a) | set(sel_b))
            pairwise_rows.append(
                {
                    "sample_path": sample_path,
                    "current_chunk_idx": current_chunk_idx,
                    "combo_a": tag_a,
                    "combo_b": tag_b,
                    "jaccard": jac,
                    "intersection_size": inter,
                    "union_size": uni,
                    "selected_a": stringify_indices(sel_a),
                    "selected_b": stringify_indices(sel_b),
                }
            )
            jaccard_pool[(tag_a, tag_b)].append(jac)

    pairwise_csv = os.path.join(out_dir, "chunk_pairwise_jaccard.csv")
    pairwise_fields = [
        "sample_path",
        "current_chunk_idx",
        "combo_a",
        "combo_b",
        "jaccard",
        "intersection_size",
        "union_size",
        "selected_a",
        "selected_b",
    ]
    with open(pairwise_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=pairwise_fields)
        writer.writeheader()
        writer.writerows(pairwise_rows)

    # 组合级别平均相似度矩阵
    # matrix[row_tag][col_tag]，对角线=1
    matrix = {r: {c: 1.0 if r == c else None for c in all_tags} for r in all_tags}
    for (tag_a, tag_b), vals in jaccard_pool.items():
        m = safe_mean(vals)
        matrix[tag_a][tag_b] = m
        matrix[tag_b][tag_a] = m

    matrix_csv = os.path.join(out_dir, "combo_similarity_matrix.csv")
    with open(matrix_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["combo_tag"] + all_tags)
        for r in all_tags:
            row = [r]
            for c in all_tags:
                v = matrix[r][c]
                row.append("" if v is None else f"{v:.6f}")
            writer.writerow(row)

    out_summary = {
        "source_sweep_root": os.path.abspath(sweep_root),
        "num_combos": len(all_tags),
        "num_chunks_total": len(chunk_combo_selected),
        "combo_meta": combo_meta,
        "files": {
            "selection_long_csv": os.path.abspath(long_csv),
            "selection_wide_csv": os.path.abspath(wide_csv),
            "chunk_pairwise_jaccard_csv": os.path.abspath(pairwise_csv),
            "combo_similarity_matrix_csv": os.path.abspath(matrix_csv),
        },
        "sweep_summary_excerpt": {
            "mode": summary.get("mode"),
            "settings_fixed": summary.get("settings_fixed"),
        },
    }
    summary_json = os.path.join(out_dir, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(out_summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("Prepared visualization data:")
    for k, v in out_summary["files"].items():
        print(f"- {k}: {v}")
    print(f"- summary_json: {os.path.abspath(summary_json)}")
    print("=" * 80)


if __name__ == "__main__":
    main()
