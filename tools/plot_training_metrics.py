#!/usr/bin/env python3
"""根据 training_metrics.csv 生成 avg_mse 和 alpha 的散点连线图。"""

import csv
import matplotlib.pyplot as plt

CSV_PATH = "/root/autodl-fs/output/bolt_ref_attn2/training_metrics.csv"
OUT_DIR  = "/root/autodl-fs/output/bolt_ref_attn2"

epochs, mse_vals, alpha_vals = [], [], []
with open(CSV_PATH) as f:
    reader = csv.DictReader(f)
    for row in reader:
        epochs.append(int(row["epoch_index"]))
        mse_vals.append(float(row["avg_mse"]))
        alpha_vals.append(float(row["alpha"]))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- avg_mse ---
ax1.plot(epochs, mse_vals, "o-", color="#2196F3", markersize=4, linewidth=1.2)
ax1.axvline(x=24.5, color="gray", linestyle="--", alpha=0.6, label="LR change (1e-4 → 5e-5)")
ax1.set_ylabel("avg_mse")
ax1.set_title("Bolt Ref-Attn Training: avg_mse over epochs")
ax1.legend()
ax1.grid(True, alpha=0.3)

# --- alpha ---
ax2.plot(epochs, alpha_vals, "s-", color="#FF5722", markersize=4, linewidth=1.2)
ax2.axvline(x=24.5, color="gray", linestyle="--", alpha=0.6, label="LR change (1e-4 → 5e-5)")
ax2.set_xlabel("epoch_index")
ax2.set_ylabel("alpha (α)")
ax2.set_title("Bolt Ref-Attn Training: alpha (gate) over epochs")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = f"{OUT_DIR}/training_metrics_plot.png"
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
