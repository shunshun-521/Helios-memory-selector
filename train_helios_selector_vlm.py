#!/usr/bin/env python
"""
train_helios_selector_vlm.py — 用 CLIP+ITS 软标签蒸馏 VLM Selector
====================================================================

与 md/CLIP-base-selector-vlm.md §3.2 对齐。

训练目标
---------
冻结 VLM 主干与 DiT，仅训练：
    · Qwen2.5-VL 的 LoRA 适配器（默认挂在 q/k/v/o_proj）
    · (可选) 一个轻量选帧 head

VLM 打分模式（``--score_mode``）
---------------------------------
* ``yes_no`` (默认，MVP)：
    单次 forward 喂入 "context + candidate" 两张图和 "Does this help ... yes/no" 问句，
    取 ``P('yes') - P('no')`` 的 *可微* logit 作为该候选的分数。
    LoRA 直接作用在 LM 上，无需额外 head，用少量样本也能稳定梯度。
* ``hidden_head``：按 md §2.2 A 路径，
    取 candidate 图像最后一个 visual token 的 hidden state → ``nn.Linear(hidden, 1)``。

Loss 配方（Hinton soft + hard）
--------------------------------
    p_pred  = softmax(vlm_logits / τ)                    # 唯一输出
    p_teach = softmax(soft_scores / τ_teacher)           # CLIP combined score 软标签
    loss_kl = KL(p_pred || p_teach)                      # 主目标
    loss_ce = -log p_pred[hard_positions].mean()         # ITS 硬目标
    loss    = kl_lambda * loss_kl + ce_lambda * loss_ce

用法
-----
    python train_helios_selector_vlm.py \
        --data_jsonl /root/autodl-fs/selector_vlm_data/train.jsonl \
        --frames_dir /root/autodl-fs/selector_vlm_data/frames_mid \
        --vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
        --output_dir /root/autodl-fs/output/selector_vlm \
        --num_epochs 10 --batch_size 1 --grad_accum_steps 1 \
        --vlm_lr 2e-4
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader

HELIOS_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(HELIOS_ROOT))


# ─── CLI ─────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train VLM Selector (LoRA, distillation from CLIP+ITS)")
    # Data
    p.add_argument("--data_jsonl", required=True)
    p.add_argument("--frames_dir", required=True)
    p.add_argument("--val_split", type=float, default=0.0,
                   help="Reserve a fraction of samples for eval (0 = no split).")
    # VLM
    p.add_argument("--vlm_model_path", required=True)
    p.add_argument("--vlm_dtype", default="bfloat16")
    p.add_argument("--vlm_image_resize", type=int, nargs=2, default=[256, 448])
    p.add_argument("--score_mode", choices=["yes_no", "hidden_head"], default="yes_no")
    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--lora_target_modules", nargs="+",
                   default=["q_proj", "k_proj", "v_proj", "o_proj"])
    # Loss
    p.add_argument("--temperature", type=float, default=0.7, help="τ on p_pred")
    p.add_argument("--temperature_teacher", type=float, default=0.5, help="τ_teacher on p_teach")
    p.add_argument("--kl_lambda", type=float, default=1.0)
    p.add_argument("--ce_lambda", type=float, default=0.3)
    p.add_argument("--min_candidates", type=int, default=2,
                   help="Skip samples with fewer candidates (KL/CE on 1-cand is meaningless).")
    # Optim
    p.add_argument("--vlm_lr", type=float, default=2e-4)
    p.add_argument("--head_lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=1, help="微批量：每 step 处理多少 jsonl 条目")
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=0)
    # Misc
    p.add_argument("--output_dir", required=True)
    p.add_argument("--save_every", type=int, default=1)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


# ─── Dataset ─────────────────────────────────────────────────────────

class SelectorVLMDataset(Dataset):
    def __init__(self, jsonl_path: str, frames_dir: str, min_candidates: int = 2):
        self.frames_dir = Path(frames_dir)
        entries = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if len(rec.get("gap_chunk_indices", [])) < min_candidates:
                    continue
                entries.append(rec)
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def _load_png(self, uttid: str, chunk_idx: int) -> Image.Image:
        p = self.frames_dir / uttid / f"chunk_{int(chunk_idx):02d}.png"
        return Image.open(p).convert("RGB")

    def __getitem__(self, i):
        rec = self.entries[i]
        uttid = rec["uttid"]
        candidates = [self._load_png(uttid, c) for c in rec["gap_chunk_indices"]]
        context = self._load_png(uttid, rec["context_chunk_idx"])
        return {
            "uttid": uttid,
            "candidates_pil": candidates,
            "context_pil": context,
            "prompt": rec["prompt_raw"],
            "soft_scores": torch.tensor(rec["soft_scores"], dtype=torch.float32),
            "hard_positions": rec.get("hard_positions", []),
        }


def collate_passthrough(batch):
    return batch  # 每条样本内部 N_cand 不同，不做 stack


# ─── Tokenizer 工具 ─────────────────────────────────────────────────

YES_TOKENS = ("yes", "Yes", "YES")
NO_TOKENS = ("no", "No", "NO")


def collect_token_ids(tokenizer, variants):
    ids = set()
    for v in variants:
        for cand in (v, " " + v):
            try:
                tids = tokenizer.encode(cand, add_special_tokens=False)
            except Exception:
                continue
            if len(tids) == 1:
                ids.add(int(tids[0]))
    return sorted(ids)


# ─── 模型构建 ────────────────────────────────────────────────────────

def build_model(args, device):
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    dtype_map = {
        "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
        "float16": torch.float16, "fp16": torch.float16,
        "float32": torch.float32, "fp32": torch.float32,
    }
    dtype = dtype_map.get(args.vlm_dtype.lower(), torch.bfloat16)

    print(f"[train_selector_vlm] loading Qwen2.5-VL from {args.vlm_model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.vlm_model_path, torch_dtype=dtype
    )
    processor = AutoProcessor.from_pretrained(args.vlm_model_path)

    # ── LoRA ──
    from peft import LoraConfig, get_peft_model, TaskType

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.lora_target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.to(device)
    model.train()

    yes_ids = collect_token_ids(processor.tokenizer, YES_TOKENS)
    no_ids = collect_token_ids(processor.tokenizer, NO_TOKENS)
    if not yes_ids or not no_ids:
        raise RuntimeError("Cannot find yes/no token ids in tokenizer.")
    print(f"[train_selector_vlm] yes_ids={yes_ids}  no_ids={no_ids}")

    image_token_id = getattr(model.config, "image_token_id", None)
    if image_token_id is None:
        # Qwen2.5-VL 默认 image token id；优先使用 config，缺失时兜底。
        image_token_id = 151655
    hidden_dim = int(getattr(model.config, "hidden_size", 2048))

    selector_head = None
    if args.score_mode == "hidden_head":
        selector_head = nn.Linear(hidden_dim, 1, bias=True).to(device=device, dtype=dtype)
        nn.init.xavier_uniform_(selector_head.weight)
        if selector_head.bias is not None:
            nn.init.zeros_(selector_head.bias)
        selector_head.train()
        n_head = sum(p.numel() for p in selector_head.parameters())
        print(
            f"[train_selector_vlm] score_mode=hidden_head, "
            f"selector_head=Linear({hidden_dim},1), params={n_head}"
        )
    else:
        print("[train_selector_vlm] score_mode=yes_no")

    return model, processor, yes_ids, no_ids, selector_head, int(image_token_id), hidden_dim


# ─── 打分 ───────────────────────────────────────────────────────────

def build_messages(context_img, candidate_img, prompt, system_prompt):
    return [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": [
            {"type": "image", "image": context_img},
            {"type": "image", "image": candidate_img},
            {"type": "text", "text": (
                "The first image is the latest context frame. "
                "The second image is a candidate history frame. "
                "Does the candidate help continue the video towards the following prompt? "
                f"Prompt: \"{prompt}\". Answer strictly with a single word: yes or no."
            )},
        ]},
    ]


def score_candidate(model, processor, context_img, cand_img, prompt,
                    image_resize, yes_ids, no_ids, device):
    """返回 scalar tensor: yes_logit - no_logit（可微）。"""
    sys_prompt = "You are a precise video-frame scoring assistant."
    messages = build_messages(context_img, cand_img, prompt, sys_prompt)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    h, w = image_resize
    images = [context_img.resize((w, h)), cand_img.resize((w, h))]
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}
    out = model(**inputs)
    logits = out.logits[:, -1, :]  # (1, vocab)
    yes_logit = logits[0, yes_ids].max()
    no_logit = logits[0, no_ids].max()
    return (yes_logit - no_logit).float()


def score_candidate_hidden_head(
    model,
    processor,
    context_img,
    cand_img,
    prompt,
    image_resize,
    selector_head,
    image_token_id,
    device,
):
    """返回 scalar tensor: head(last_visual_hidden)。

    last_visual_hidden 定义为 candidate 图像对应的最后一个 image token hidden。
    """
    if selector_head is None:
        raise RuntimeError("score_candidate_hidden_head requires a non-None selector_head.")

    sys_prompt = "You are a precise video-frame scoring assistant."
    messages = build_messages(context_img, cand_img, prompt, sys_prompt)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    h, w = image_resize
    images = [context_img.resize((w, h)), cand_img.resize((w, h))]
    inputs = processor(text=[text], images=images, return_tensors="pt", padding=True)
    inputs = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    out = model(**inputs, output_hidden_states=True, return_dict=True)
    hidden_last = out.hidden_states[-1]  # (1, seq_len, hidden)

    input_ids = inputs.get("input_ids", None)
    if input_ids is None:
        raise RuntimeError("input_ids is missing; cannot locate visual token position.")
    visual_pos = (input_ids[0] == int(image_token_id)).nonzero(as_tuple=False)
    if visual_pos.numel() == 0:
        raise RuntimeError(
            f"Cannot find image_token_id={image_token_id} in input_ids; "
            "cannot pool candidate last visual token."
        )
    last_vis_pos = int(visual_pos[-1].item())
    pooled = hidden_last[0, last_vis_pos, :]  # (hidden,)
    return selector_head(pooled).squeeze(-1).float()


def compute_loss_for_sample(
    model,
    processor,
    sample,
    args,
    yes_ids,
    no_ids,
    selector_head,
    image_token_id,
    device,
):
    """返回 (loss, 日志 dict)。"""
    cand_imgs = sample["candidates_pil"]
    n_cand = len(cand_imgs)
    if n_cand < args.min_candidates:
        return None, {"skipped": True}

    logits = []
    for img in cand_imgs:
        if args.score_mode == "hidden_head":
            logits.append(
                score_candidate_hidden_head(
                    model=model,
                    processor=processor,
                    context_img=sample["context_pil"],
                    cand_img=img,
                    prompt=sample["prompt"],
                    image_resize=tuple(args.vlm_image_resize),
                    selector_head=selector_head,
                    image_token_id=image_token_id,
                    device=device,
                )
            )
        else:
            logits.append(
                score_candidate(
                    model=model,
                    processor=processor,
                    context_img=sample["context_pil"],
                    cand_img=img,
                    prompt=sample["prompt"],
                    image_resize=tuple(args.vlm_image_resize),
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                    device=device,
                )
            )
    logits = torch.stack(logits)  # (N_cand,)

    soft = sample["soft_scores"].to(device=logits.device, dtype=logits.dtype)

    log_p_pred = F.log_softmax(logits / max(args.temperature, 1e-4), dim=0)
    p_teach = F.softmax(soft / max(args.temperature_teacher, 1e-4), dim=0)
    loss_kl = F.kl_div(log_p_pred, p_teach, reduction="batchmean")

    hard_positions = sample["hard_positions"]
    if hard_positions:
        hard_idx = torch.tensor(
            [int(x) for x in hard_positions if int(x) < n_cand],
            dtype=torch.long, device=logits.device,
        )
        loss_ce = (-log_p_pred[hard_idx]).mean() if len(hard_idx) > 0 else torch.zeros_like(loss_kl)
    else:
        loss_ce = torch.zeros_like(loss_kl)

    loss = args.kl_lambda * loss_kl + args.ce_lambda * loss_ce

    with torch.no_grad():
        p_pred = torch.exp(log_p_pred)
        pred_top1 = int(p_pred.argmax().item())
        teach_top1 = int(p_teach.argmax().item())
    info = {
        "n_cand": n_cand,
        "loss": float(loss.detach().item()),
        "loss_kl": float(loss_kl.detach().item()),
        "loss_ce": float(loss_ce.detach().item()),
        "pred_top1": pred_top1,
        "teach_top1": teach_top1,
        "top1_match": int(pred_top1 == teach_top1),
    }
    return loss, info


# ─── 训练主循环 ─────────────────────────────────────────────────────

def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Data
    ds = SelectorVLMDataset(args.data_jsonl, args.frames_dir, min_candidates=args.min_candidates)
    print(f"[train_selector_vlm] dataset size (after min_candidates filter): {len(ds)}")
    if len(ds) == 0:
        print("  No usable samples. Abort.")
        return

    # Optional val split
    val_ds = None
    if args.val_split > 0 and len(ds) >= 4:
        n_val = max(1, int(round(len(ds) * args.val_split)))
        indices = list(range(len(ds)))
        random.Random(args.seed).shuffle(indices)
        val_idx = set(indices[:n_val])
        val_ds = [ds[i] for i in sorted(val_idx)]
        train_idx = [i for i in range(len(ds)) if i not in val_idx]
    else:
        train_idx = list(range(len(ds)))

    train_loader = DataLoader(
        [ds[i] for i in train_idx],
        batch_size=args.batch_size, shuffle=True,
        num_workers=0, collate_fn=collate_passthrough,
    )

    # Model
    (
        model,
        processor,
        yes_ids,
        no_ids,
        selector_head,
        image_token_id,
        hidden_dim,
    ) = build_model(args, device)

    # Optim
    lora_params = [p for p in model.parameters() if p.requires_grad]
    param_groups = [{"params": lora_params, "lr": args.vlm_lr}]
    if args.score_mode == "hidden_head":
        if selector_head is None:
            raise RuntimeError("score_mode=hidden_head but selector_head is None.")
        param_groups.append({"params": list(selector_head.parameters()), "lr": args.head_lr})
    optimizer = torch.optim.AdamW(param_groups, lr=args.vlm_lr, weight_decay=args.weight_decay)
    trainable = [p for g in param_groups for p in g["params"]]
    scheduler = None
    if args.warmup_steps > 0:
        from torch.optim.lr_scheduler import LambdaLR
        def lr_lambda(step):
            return min(1.0, (step + 1) / max(1, args.warmup_steps))
        scheduler = LambdaLR(optimizer, lr_lambda)

    global_step = 0
    log_path = out_dir / "train_log.jsonl"
    log_fw = open(log_path, "w", encoding="utf-8")

    for epoch in range(args.num_epochs):
        t0 = time.time()
        epoch_loss = 0.0; epoch_kl = 0.0; epoch_ce = 0.0; n_seen = 0; n_skip = 0; n_match = 0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader):
            # batch is list of samples (size=batch_size). 我们按 grad_accum 累积
            micro_loss = None
            micro_info = []
            for sample in batch:
                loss, info = compute_loss_for_sample(
                    model=model,
                    processor=processor,
                    sample=sample,
                    args=args,
                    yes_ids=yes_ids,
                    no_ids=no_ids,
                    selector_head=selector_head,
                    image_token_id=image_token_id,
                    device=device,
                )
                if loss is None:
                    n_skip += 1
                    continue
                micro_loss = loss if micro_loss is None else (micro_loss + loss)
                micro_info.append(info)
                n_seen += 1
                epoch_loss += info["loss"]; epoch_kl += info["loss_kl"]; epoch_ce += info["loss_ce"]
                n_match += info["top1_match"]

            if micro_loss is None:
                continue

            micro_loss = micro_loss / max(1, len(micro_info)) / max(1, args.grad_accum_steps)
            micro_loss.backward()

            if (step + 1) % max(1, args.grad_accum_steps) == 0:
                if args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

                if global_step % args.log_every == 0:
                    line = {
                        "step": global_step, "epoch": epoch,
                        "lr": optimizer.param_groups[0]["lr"],
                        "score_mode": args.score_mode,
                        "loss": info["loss"], "kl": info["loss_kl"], "ce": info["loss_ce"],
                        "n_cand": info["n_cand"],
                        "pred_top1": info["pred_top1"], "teach_top1": info["teach_top1"],
                    }
                    log_fw.write(json.dumps(line) + "\n"); log_fw.flush()

        # 处理余下未 step 的梯度
        if (step + 1) % max(1, args.grad_accum_steps) != 0:
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        dur = time.time() - t0
        mean_loss = epoch_loss / max(1, n_seen)
        mean_kl = epoch_kl / max(1, n_seen)
        mean_ce = epoch_ce / max(1, n_seen)
        top1_acc = n_match / max(1, n_seen)
        print(
            f"[epoch {epoch:02d}] n={n_seen} skip={n_skip} loss={mean_loss:.4f} "
            f"kl={mean_kl:.4f} ce={mean_ce:.4f} top1_match={top1_acc:.3f}  "
            f"({dur:.1f}s)"
        )
        log_fw.write(json.dumps({
            "epoch_summary": epoch, "n_seen": n_seen, "skipped": n_skip,
            "score_mode": args.score_mode,
            "mean_loss": mean_loss, "mean_kl": mean_kl, "mean_ce": mean_ce,
            "top1_match": top1_acc, "duration_sec": dur,
        }) + "\n"); log_fw.flush()

        # Save
        if (epoch + 1) % args.save_every == 0 or epoch == args.num_epochs - 1:
            ep_dir = out_dir / f"epoch_{epoch:03d}"
            ep_dir.mkdir(parents=True, exist_ok=True)
            # Save LoRA adapter only (base model is frozen, no need to save)
            try:
                model.save_pretrained(str(ep_dir))
            except Exception as exc:
                print(f"  [WARN] save_pretrained failed: {exc}")
            if args.score_mode == "hidden_head" and selector_head is not None:
                head_payload = {
                    "state_dict": selector_head.state_dict(),
                    "hidden_dim": int(hidden_dim),
                    "image_token_id": int(image_token_id),
                    "score_mode": "hidden_head",
                }
                torch.save(head_payload, ep_dir / "selector_head.pt")
                with open(ep_dir / "score_mode.json", "w", encoding="utf-8") as sm_fw:
                    json.dump(
                        {"score_mode": "hidden_head", "hidden_dim": int(hidden_dim), "image_token_id": int(image_token_id)},
                        sm_fw,
                        ensure_ascii=False,
                        indent=2,
                    )
            print(f"  saved adapter to {ep_dir}")

    log_fw.close()
    print(f"[train_selector_vlm] done. logs: {log_path}")


if __name__ == "__main__":
    main()
