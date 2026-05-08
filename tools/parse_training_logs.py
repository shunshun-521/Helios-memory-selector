#!/usr/bin/env python3
"""从两份训练 log 中提取 avg_mse 和 alpha，输出 CSV。

用法:
    python parse_training_logs.py

输出: /root/autodl-fs/output/bolt_ref_attn2/training_metrics.csv
"""

import re
import csv

LOG1 = "/root/autodl-fs/output/bolt_ref_attn2/train_0414_1136.log"
LOG2 = "/root/autodl-fs/output/bolt_ref_attn2/train_0414_1553.log"
OUT  = "/root/autodl-fs/output/bolt_ref_attn2/training_metrics.csv"

# 匹配: [Epoch X] avg_mse=0.090939, time=576.8s
# 以及 tqdm 行末尾的 α=0.0002
EPOCH_MSE_RE = re.compile(r"\[Epoch\s+(\d+)\]\s+avg_mse=([\d.]+)")
ALPHA_RE     = re.compile(r"α=([\d.]+)")


def extract_metrics(log_path, max_epochs=25):
    """从 log 文件提取每个 epoch 的 avg_mse 和最后出现的 alpha。
    
    只取每个 epoch 第一次出现的 avg_mse。
    alpha 取该 epoch 最后一条 tqdm 行中的值。
    """
    mse_dict = {}    # epoch -> avg_mse (first occurrence)
    alpha_dict = {}  # epoch -> alpha (last seen in that epoch's tqdm lines)
    
    current_epoch = None
    
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            # 检测当前 epoch（从 tqdm 行 [Epoch X]: 或 [Epoch X] 开头）
            ep_match = re.search(r"\[Epoch\s+(\d+)\]", line)
            if ep_match:
                current_epoch = int(ep_match.group(1))
            
            # 提取 alpha（从 tqdm 进度行）
            alpha_match = ALPHA_RE.search(line)
            if alpha_match and current_epoch is not None and current_epoch < max_epochs:
                alpha_dict[current_epoch] = float(alpha_match.group(1))
            
            # 提取 avg_mse（从 epoch 总结行）
            mse_match = EPOCH_MSE_RE.search(line)
            if mse_match:
                ep = int(mse_match.group(1))
                if ep < max_epochs and ep not in mse_dict:
                    mse_dict[ep] = float(mse_match.group(2))
    
    return mse_dict, alpha_dict


def main():
    mse1, alpha1 = extract_metrics(LOG1, max_epochs=25)
    mse2, alpha2 = extract_metrics(LOG2, max_epochs=25)
    
    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch_index", "avg_mse", "alpha"])
        
        # Log1: epoch 0~24 → index 0~24
        for ep in range(25):
            mse_val = mse1.get(ep, "")
            alpha_val = alpha1.get(ep, "")
            writer.writerow([ep, mse_val, alpha_val])
        
        # Log2: epoch 0~24 → index 25~49
        for ep in range(25):
            idx = ep + 25
            mse_val = mse2.get(ep, "")
            alpha_val = alpha2.get(ep, "")
            writer.writerow([idx, mse_val, alpha_val])
    
    print(f"Done! Saved to {OUT}")
    print(f"  Log1: {len(mse1)} epochs with mse, {len(alpha1)} epochs with alpha")
    print(f"  Log2: {len(mse2)} epochs with mse, {len(alpha2)} epochs with alpha")


if __name__ == "__main__":
    main()
