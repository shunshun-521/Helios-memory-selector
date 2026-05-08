"""从训练日志中提取 loss 并绘制 baseline vs selector 对比曲线。

用法:
    python tools/plot_loss.py
    python tools/plot_loss.py --baseline /path/to/baseline.log --selector /path/to/selector.log --out loss_curve.png
"""
import argparse
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def extract_loss(log_path):
    """提取每个 step 的最后一次 loss（tqdm 会重复打印同一 step）。"""
    step_loss = {}
    pattern = re.compile(r"(\d+)/\d+\s*\[.*?loss=([\d.]+)")
    with open(log_path) as f:
        for line in f:
            for m in pattern.finditer(line):
                step = int(m.group(1))
                loss = float(m.group(2))
                step_loss[step] = loss  # 后出现的覆盖前面的（去重）
    steps = sorted(step_loss.keys())
    losses = [step_loss[s] for s in steps]
    return steps, losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="/root/autodl-tmp/log/v3_baseline.log",
                        help="Baseline (无 selector) 训练日志")
    parser.add_argument("--selector", default="/root/autodl-tmp/Helios/train_1_init_v3.log",
                        help="Selector 版训练日志")
    parser.add_argument("--out", default="/root/autodl-tmp/Helios/tools/loss_curve.png")
    args = parser.parse_args()

    plt.figure(figsize=(10, 6))

    for log_path, label, color in [
        (args.baseline, "Baseline (Helios)", "#1f77b4"),
        (args.selector, "Selector (Helios + SFI)", "#ff7f0e"),
    ]:
        steps, losses = extract_loss(log_path)
        if steps:
            plt.plot(steps, losses, label=label, color=color, linewidth=1.8, marker="o", markersize=3)
            print(f"[{label}] {len(steps)} steps, loss range: {min(losses):.4f} ~ {max(losses):.4f}")
        else:
            print(f"[{label}] WARNING: 未从 {log_path} 中提取到 loss 数据")

    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Overfitting Experiment: Baseline vs Selector", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
