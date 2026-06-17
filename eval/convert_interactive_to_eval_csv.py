"""
将交互式多 prompt CSV 转换为评估用的单 prompt CSV。

输入格式: id, prompt_index, prompt
输出格式: id, prompt, duration

同一 id 的多条 prompt 按 prompt_index 排序后用空格拼接。
duration 默认按 prompt 数量 × 每段帧数计算（可通过参数指定）。
"""

import argparse
import csv
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="交互式 prompt CSV 路径")
    parser.add_argument("--output", required=True, help="输出评估 CSV 路径")
    parser.add_argument("--frames_per_chunk", type=int, default=120,
                        help="每个 prompt 段对应的帧数 (默认 120)")
    args = parser.parse_args()

    # 按 id 分组
    groups = defaultdict(list)
    with open(args.input, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid_id = int(row["id"])
            prompt_idx = int(row["prompt_index"])
            prompt = row["prompt"].strip()
            groups[vid_id].append((prompt_idx, prompt))

    # 排序并拼接
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "prompt", "duration"])
        for vid_id in sorted(groups.keys()):
            prompts = sorted(groups[vid_id], key=lambda x: x[0])
            full_prompt = " ".join(p for _, p in prompts)
            duration = len(prompts) * args.frames_per_chunk
            writer.writerow([vid_id, full_prompt, duration])

    print(f"转换完成: {len(groups)} 个视频 → {args.output}")


if __name__ == "__main__":
    main()
