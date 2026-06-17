"""
train_selector.py — Stage 1: VLM Selector 训练
=================================================

训练 Qwen2.5-VL + LoRA 和 SelectorHead，学习从 GAP 帧候选中选出最有用的参考帧。

【与 selector_v2/v3 的区别】
- selector_v2/v3: 轻量 ~1.16M 参数的 Gumbel-Softmax 选帧器，端到端与 DiT 联合训练
- 本方案 (VLM Selector): Qwen2.5-VL-3B + LoRA (~10M 参数) 独立训练，不碰 DiT
  - 更强的语义理解 (光照变化、人物旋转等高层判断)
  - 两阶段解耦: Stage 1 只学"选什么"，Stage 2 只学"怎么注入"
  - 即插即用: 训练后冻结，不影响 DiT 主模型

【数据来源】
- selector_meta_with_psoft.json: 由 build_selector_dataset.py + compute_p_soft.py 构建
- 每条样本包含: context_pixel_path, gap_pixel_paths, p_soft, prompt_raw

【训练目标】
- Loss = KL(P_pred || P_soft)
- P_pred = SelectorHead(Q=encode_lhs(context), K=encode_lhs(gap_frames))
- P_soft = softmax(ΔMSE / τ) (由 compute_p_soft.py 预计算)

用法:
  python Helios/train_selector.py \
    --meta_json Helios/example/lighting_change/selector_meta_with_psoft.json \
    --vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
    --output_dir /root/autodl-fs/output/vlm_selector_stage1 \
    --num_epochs 50 \
    --lr 1e-4 \
    --batch_size 1
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Helios root
HELIOS_ROOT = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, HELIOS_ROOT)

from helios.modules.selector_vlm import VLMSelector


# ═══════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════

class SelectorDataset(Dataset):
    """VLM Selector 训练数据集。

    每条样本:
    - context_pixel_path: str — chunk_{t-1} 的像素帧 (jpg)
    - gap_pixel_paths: list[str] — GAP 候选帧路径列表
    - p_soft: list[float] — softmax 软标签
    - prompt_raw: str — 文本 prompt
    """

    def __init__(self, meta_json: str, max_gap_frames: int = 50):
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.samples = []
        self.max_gap_frames = max_gap_frames
        # Fix #1: 用 meta_json 所在目录作为相对路径的基准目录
        base_dir = os.path.dirname(os.path.abspath(meta_json))

        skipped_no_ctx = 0
        skipped_no_gap = 0

        for entry in meta:
            prompt = entry["prompt_raw"]
            for sample in entry["selector_samples"]:
                if sample.get("p_soft") is None:
                    continue  # 跳过未计算 P_soft 的样本

                # Fix #1: 相对路径 → 绝对路径
                ctx_path = sample["context_pixel_path"]
                if not os.path.isabs(ctx_path):
                    ctx_path = os.path.join(base_dir, ctx_path)
                gap_paths = sample["gap_pixel_paths"]
                p_soft = sample["p_soft"]

                # 校验: 路径存在 且 p_soft 长度匹配
                if not os.path.exists(ctx_path):
                    skipped_no_ctx += 1
                    continue
                valid_gaps = []
                for gp, ps in zip(gap_paths, p_soft):
                    gp_abs = gp if os.path.isabs(gp) else os.path.join(base_dir, gp)
                    if os.path.exists(gp_abs):
                        valid_gaps.append((gp_abs, ps))
                if len(valid_gaps) < 2:
                    skipped_no_gap += 1
                    continue

                gap_paths_valid = [x[0] for x in valid_gaps]
                p_soft_valid = [x[1] for x in valid_gaps]

                # Fix #3: 截断过长的 GAP 列表 — 按 P_soft 排序保留最有用的帧
                if len(gap_paths_valid) > max_gap_frames:
                    sorted_pairs = sorted(
                        zip(p_soft_valid, gap_paths_valid), reverse=True
                    )
                    sorted_pairs = sorted_pairs[:max_gap_frames]
                    p_soft_valid = [x[0] for x in sorted_pairs]
                    gap_paths_valid = [x[1] for x in sorted_pairs]
                    # 重新归一化
                    total = sum(p_soft_valid)
                    if total > 0:
                        p_soft_valid = [p / total for p in p_soft_valid]

                self.samples.append({
                    "context_path": ctx_path,
                    "gap_paths": gap_paths_valid,
                    "p_soft": p_soft_valid,
                    "prompt": prompt,
                    "choice_idx": sample["choice_idx"],
                })

        print(f"[SelectorDataset] Loaded {len(self.samples)} valid samples from {meta_json}")
        if skipped_no_ctx > 0:
            print(f"  [WARN] 跳过 {skipped_no_ctx} 个样本 (context 像素帧不存在)")
        if skipped_no_gap > 0:
            print(f"  [WARN] 跳过 {skipped_no_gap} 个样本 (有效 GAP 帧 < 2)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """返回原始路径和标签，VLM 编码在训练循环中进行（需要 processor）。"""
        return self.samples[idx]


def collate_fn(batch):
    """自定义 collate: 保持 dict 列表格式 (batch_size=1 时直接取第一个)。"""
    return batch


# ═══════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════

def train_one_epoch(
    selector: VLMSelector,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: str = "cuda",
    grad_accum_steps: int = 4,
    max_grad_norm: float = 1.0,
):
    """Stage 1 单 epoch 训练。

    流程:
    1. 加载 context_image + gap_images
    2. encode_lhs: VLM 编码得到 Q (context) 和 K (gap frames)
    3. SelectorHead(Q, K) → P_pred
    4. KL(P_pred || P_soft) → backward
    """
    selector.train()
    total_loss = 0.0
    num_samples = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}]", leave=True)
    for step, batch in enumerate(pbar):
        sample = batch[0]  # batch_size=1
        prompt = sample["prompt"]
        p_soft_list = sample["p_soft"]

        # 加载图片
        try:
            context_img = Image.open(sample["context_path"]).convert("RGB")
            gap_imgs = [Image.open(gp).convert("RGB") for gp in sample["gap_paths"]]
        except Exception as e:
            print(f"  [WARN] 跳过样本 (图片加载失败): {e}")
            continue

        # ── Step 1: VLM 编码 ──
        # Q: context chunk 的完整 LHS (保留 seq_len)
        query_lhs = selector.encode_lhs([context_img], prompt, device=device)  # (1, S_q, D)

        # Fix #2: GAP 帧分批编码避免 OOM
        gap_lhs_parts = []
        vlm_batch_size = 8
        for bi in range(0, len(gap_imgs), vlm_batch_size):
            part = selector.encode_lhs(gap_imgs[bi:bi+vlm_batch_size], prompt, device=device)
            gap_lhs_parts.append(part)
        gap_lhs = torch.cat(gap_lhs_parts, dim=0)  # (N_gap, S, D)
        gap_lhs_pooled = VLMSelector.pool_lhs(gap_lhs)  # (N_gap, D)
        key_lhs = gap_lhs_pooled.unsqueeze(0)  # (1, N_gap, D)

        # P_soft 标签
        p_soft = torch.tensor([p_soft_list], dtype=torch.float32, device=device)  # (1, N_gap)

        # ── Step 2: Forward + Loss ──
        result = selector(query_lhs, key_lhs, p_soft=p_soft)
        loss = result["loss"] / grad_accum_steps

        # ── Step 3: Backward ──
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in selector.parameters() if p.requires_grad],
                max_grad_norm,
            )
            optimizer.step()
            optimizer.zero_grad()

        total_loss += result["loss"].item()
        num_samples += 1

        # 日志
        top_k_idx = result["top_k_indices"][0].cpu().tolist()
        p_pred_max = result["p_pred"][0].max().item()
        pbar.set_postfix({
            "loss": f"{result['loss'].item():.4f}",
            "avg_loss": f"{total_loss / num_samples:.4f}",
            "top_k": str(top_k_idx[:4]),
            "p_max": f"{p_pred_max:.3f}",
        })

    # Fix #4: 处理最后一个不完整的梯度累积 batch
    if num_samples % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(
            [p for p in selector.parameters() if p.requires_grad],
            max_grad_norm,
        )
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(num_samples, 1)


def save_checkpoint(selector: VLMSelector, optimizer, epoch, output_dir, avg_loss):
    """保存 checkpoint: LoRA weights + SelectorHead weights。"""
    ckpt_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch:03d}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 保存 SelectorHead
    torch.save(
        selector.selector_head.state_dict(),
        os.path.join(ckpt_dir, "selector_head.pth"),
    )

    # 保存 LoRA weights (如果使用了 LoRA)
    if selector._vlm is not None and selector.use_lora:
        try:
            from peft.utils import get_peft_model_state_dict
            lora_state = get_peft_model_state_dict(selector._vlm)
            torch.save(lora_state, os.path.join(ckpt_dir, "vlm_lora_weights.pth"))
        except Exception as e:
            print(f"  [WARN] 保存 LoRA weights 失败: {e}")

    # 保存 optimizer
    torch.save(optimizer.state_dict(), os.path.join(ckpt_dir, "optimizer.pth"))

    # 保存元信息
    meta = {"epoch": epoch, "avg_loss": avg_loss}
    with open(os.path.join(ckpt_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    print(f"  [SAVE] Checkpoint saved to {ckpt_dir}")


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Train VLM Selector (LoRA + SelectorHead)")
    parser.add_argument("--meta_json", type=str, required=True,
                        help="selector_meta_with_psoft.json (含 P_soft 的训练数据)")
    parser.add_argument("--vlm_model_path", type=str, default="/root/autodl-fs/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-fs/output/vlm_selector_stage1")
    parser.add_argument("--num_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=1, help="目前仅支持 batch_size=1")
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--k_select", type=int, default=1, help="top-k 选帧数")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--no_lora", action="store_true", help="不使用 LoRA (freeze VLM)")
    parser.add_argument("--save_every", type=int, default=5, help="每 N 个 epoch 保存一次")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default=None, help="从 checkpoint 恢复训练")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── 初始化 Selector ──
    print(f"[INFO] Initializing VLM Selector...")
    selector = VLMSelector(
        vlm_model_path=args.vlm_model_path,
        vlm_hidden_dim=2048,
        k_select=args.k_select,
        use_lora=not args.no_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
    )
    # 显式加载 VLM (而非 lazy load)
    selector.init_vlm(device=device)
    selector.selector_head.to(device)

    # ── 恢复 checkpoint ──
    start_epoch = 0
    if args.resume_from and os.path.isdir(args.resume_from):
        head_path = os.path.join(args.resume_from, "selector_head.pth")
        if os.path.exists(head_path):
            selector.selector_head.load_state_dict(torch.load(head_path, map_location=device))
            print(f"[INFO] Resumed SelectorHead from {head_path}")
        lora_path = os.path.join(args.resume_from, "vlm_lora_weights.pth")
        if os.path.exists(lora_path) and selector.use_lora:
            from peft import set_peft_model_state_dict
            lora_state = torch.load(lora_path, map_location=device)
            set_peft_model_state_dict(selector._vlm, lora_state)
            print(f"[INFO] Resumed LoRA weights from {lora_path}")
        meta_path = os.path.join(args.resume_from, "meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                start_epoch = json.load(f).get("epoch", 0) + 1

    # ── 收集可训练参数 ──
    trainable_params = []
    # SelectorHead 参数
    trainable_params += list(selector.selector_head.parameters())
    # VLM LoRA 参数 (如果有)
    if selector._vlm is not None and selector.use_lora:
        trainable_params += [p for p in selector._vlm.parameters() if p.requires_grad]

    total_trainable = sum(p.numel() for p in trainable_params)
    print(f"[INFO] Total trainable parameters: {total_trainable:,} ({total_trainable / 1e6:.1f}M)")

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=0.01)

    if args.resume_from:
        opt_path = os.path.join(args.resume_from, "optimizer.pth")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

    # ── 数据集 ──
    dataset = SelectorDataset(args.meta_json)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0,
    )

    # ── 训练循环 ──
    print(f"\n{'='*60}")
    print(f"  Stage 1: VLM Selector Training")
    print(f"  Epochs: {start_epoch} → {args.num_epochs}")
    print(f"  Samples: {len(dataset)}")
    print(f"  LR: {args.lr}, Grad Accum: {args.grad_accum_steps}")
    print(f"  k_select: {args.k_select}")
    print(f"{'='*60}\n")

    best_loss = float("inf")
    for epoch in range(start_epoch, args.num_epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(
            selector, dataloader, optimizer, epoch,
            device=device,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
        )
        dt = time.time() - t0
        print(f"[Epoch {epoch}] avg_loss={avg_loss:.4f}, time={dt:.1f}s")

        # 保存
        if (epoch + 1) % args.save_every == 0 or avg_loss < best_loss:
            save_checkpoint(selector, optimizer, epoch, args.output_dir, avg_loss)
            if avg_loss < best_loss:
                best_loss = avg_loss

    print(f"\n[INFO] Training complete. Best loss: {best_loss:.4f}")
    print(f"[INFO] Checkpoints saved in: {args.output_dir}")


if __name__ == "__main__":
    main()
