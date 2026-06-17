"""
train_helios_bolt.py — BOLT Ref-Attn 训练
==========================================

冻结 Helios DiT + VAE + Text Encoder，只训练 BoltReferenceAttentionLayers。
单 pass forward: DiT frozen (requires_grad=False) + Ref-Attn hooks (有梯度)。
Loss = Flow Matching MSE。

【训练策略】
- DiT 全部参数 requires_grad=False，Ref-Attn 参数 requires_grad=True
- 正常执行 transformer(...) 一次 forward，hook 自动在 self-attn 后注入 Ref-Attn
- Ref-Attn 的输出会影响后续所有 DiT 层的计算，因此必须在同一次 forward 中完成
- loss.backward() 梯度自动只流向 Ref-Attn 参数

【显存优化】
- DiT 使用 enable_group_offload (leaf_level)，权重按需从 CPU 加载到 GPU
- Ref-Attn 参数常驻 GPU（参数量小，约几十 M）

用法:
  python train_helios_bolt.py \
    --feature_folders /path/to/precomputed_latents \
    --transformer_path /root/autodl-fs/BestWishYSH/Helios-Base \
    --output_dir /root/autodl-fs/output/bolt_ref_attn \
    --num_epochs 30 --bolt_lr 5e-5
"""

print("[BOOT] train_helios_bolt.py starting import...", flush=True)

import argparse
import inspect
import json
import math
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from tqdm import tqdm

HELIOS_ROOT = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, HELIOS_ROOT)

from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_rmsnorm_with_fp32,
    replace_all_norms_with_flash_norms,
    replace_rope_with_flash_rope,
)
from helios.modules.extract_feature import CLIP, DINOv2, decode_middle_frame, extract_chunk_feature
from helios.modules.select_frames import inverse_transform_sampling
from helios.modules.ref_attn_bolt import BoltReferenceAttentionLayers, patchify_selected_latents
from helios.modules.select_frames_vlm import VLMFrameSelector

from diffusers import AutoencoderKLWan
from diffusers.training_utils import compute_loss_weighting_for_sd3


# ═══════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════

LATENT_WINDOW_SIZE = 9
HISTORY_SIZES = [16, 2, 1]
HISTORY_WINDOW_SIZE = sum(HISTORY_SIZES)


# ═══════════════════════════════════════════
# Dataset (reuses Stage 1 dataloader format)
# ═══════════════════════════════════════════

class BoltTrainDataset(torch.utils.data.Dataset):
    """加载预计算的 VAE latent + prompt embed。

    每个 .pt 文件包含:
    - vae_latent: (num_chunks, C, T, H, W)
    - prompt_embed: (1, S, D) or (S, D)
    """

    def __init__(self, feature_folders, min_chunks=4):
        if isinstance(feature_folders, str):
            feature_folders = [feature_folders]

        self.samples = []
        for folder in feature_folders:
            for f in os.listdir(folder):
                if not f.endswith(".pt"):
                    continue
                path = os.path.join(folder, f)
                # 解析文件名获取 metadata
                parts = f.replace(".pt", "").split("_")
                if len(parts) >= 3:
                    try:
                        num_frame = int(parts[-3])
                        height = int(parts[-2])
                        width = int(parts[-1])
                    except ValueError:
                        continue
                    # 至少需要 min_chunks 个 chunk 才有 GAP 帧可选
                    num_chunks = num_frame  # 近似，实际在 __getitem__ 中检查
                    self.samples.append({
                        "path": path,
                        "num_frame": num_frame,
                        "height": height,
                        "width": width,
                    })

        print(f"[BoltTrainDataset] {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        data = torch.load(sample["path"], map_location="cpu", weights_only=False)
        return {
            "vae_latent": data["vae_latent"],       # (num_chunks, C, T, H, W)
            "prompt_embed": data["prompt_embed"],     # (1, S, D) or (S, D)
            "prompt_raw": data.get("prompt_raw", None),  # str or None
            # Optional: chunk-aligned prompt segments (Selector_VLM format).
            # - prompt_embeds_by_segment: (N_seg, S, D)
            # - segments: list[dict] with start_chunk/end_chunk/prompt_raw
            "prompt_embeds_by_segment": data.get("prompt_embeds_by_segment", None),
            "segments": data.get("segments", None),
            "path": sample["path"],
        }


def collate_fn(batch):
    return batch


# ═══════════════════════════════════════════
# History construction helpers
# ═══════════════════════════════════════════

def build_history_and_target(vae_latent, choice_idx):
    """从连续 chunk latent 中构建 target + 多尺度历史帧。

    Args:
        vae_latent: (num_chunks, C, T, H, W)
        choice_idx: int — 目标 chunk 索引

    Returns:
        target_latent: (1, C, T, H, W)
        history_short: (1, C, 2, H, W)
        history_mid: (1, C, 2, H, W)
        history_long: (1, C, 16, H, W)
        indices: dict of frame indices for RoPE
        gap_latents: list of (1, C, T, H, W) — 可用于 ITS 选帧的 GAP chunk
    """
    num_chunks, C, T, H, W = vae_latent.shape

    # 拼成连续帧序列
    continue_latent = rearrange(vae_latent, "b c t h w -> c (b t) h w")
    total_frames = continue_latent.shape[1]

    # 前面补零
    zero_pad = torch.zeros(C, HISTORY_WINDOW_SIZE, H, W,
                           device=continue_latent.device, dtype=continue_latent.dtype)
    padded = torch.cat([zero_pad, continue_latent], dim=1)

    # Target
    target_start = HISTORY_WINDOW_SIZE + choice_idx * T
    target_latent = padded[:, target_start:target_start + T].unsqueeze(0)  # (1, C, T, H, W)

    # History: target 之前的 HISTORY_WINDOW_SIZE 帧
    hist_start = target_start - HISTORY_WINDOW_SIZE
    history = padded[:, hist_start:target_start]  # (C, 19, H, W)

    # 拆分为 long(16) + mid(2) + short(1)
    history_long = history[:, :16].unsqueeze(0)    # (1, C, 16, H, W)
    history_mid = history[:, 16:18].unsqueeze(0)   # (1, C, 2, H, W)
    history_short = history[:, 18:19].unsqueeze(0) # (1, C, 1, H, W)

    # x0: 第一帧
    x0 = padded[:, HISTORY_WINDOW_SIZE:HISTORY_WINDOW_SIZE + 1].unsqueeze(0)  # (1, C, 1, H, W)

    # 如果 short 只有 1 帧，补上 x0 使其为 2 帧
    history_short = torch.cat([x0, history_short], dim=2)  # (1, C, 2, H, W)

    # Frame indices for RoPE
    target_frame_start = choice_idx * T
    idx_target = torch.arange(target_frame_start, target_frame_start + T)
    idx_short = torch.tensor([0, target_frame_start - 1]) if target_frame_start > 0 else torch.tensor([0, 0])
    idx_mid = torch.tensor([max(0, target_frame_start - 3), max(0, target_frame_start - 2)])
    idx_long = torch.arange(max(0, target_frame_start - 16), target_frame_start).tolist()
    # Pad to 16
    while len(idx_long) < 16:
        idx_long.insert(0, 0)
    idx_long = torch.tensor(idx_long)

    # GAP chunks: 所有 choice_idx 之前的 chunk (排除紧邻的 context)
    gap_latents = []
    gap_indices = []
    for i in range(max(0, choice_idx - 1)):
        gap_latents.append(vae_latent[i:i+1])  # (1, C, T, H, W)
        gap_indices.append(i)

    return (
        target_latent, history_short, history_mid, history_long,
        idx_target.unsqueeze(0), idx_short.unsqueeze(0),
        idx_mid.unsqueeze(0), idx_long.unsqueeze(0),
        gap_latents, gap_indices,
    )


# ═══════════════════════════════════════════
# Training
# ═══════════════════════════════════════════

def _decode_middle_frame_tensor(
    chunk_latent: torch.Tensor,
    vae: AutoencoderKLWan,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
) -> torch.Tensor:
    """Differentiable middle-frame decode for ID loss.

    Returns:
        Tensor (B, 3, H, W) in [0, 1]
    """
    if chunk_latent.ndim == 4:
        chunk_latent = chunk_latent.unsqueeze(0)
    if chunk_latent.ndim != 5:
        raise ValueError(f"chunk_latent must be 4D/5D, got shape={tuple(chunk_latent.shape)}")

    t_mid = int(chunk_latent.shape[2]) // 2
    mid_latent = chunk_latent[:, :, t_mid : t_mid + 1, :, :]

    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype

    mid_latent = mid_latent.to(device=vae_device, dtype=vae_dtype)
    lm = latents_mean.to(device=vae_device, dtype=vae_dtype)
    ls = latents_std.to(device=vae_device, dtype=vae_dtype)
    normalized = mid_latent / ls + lm

    pixel = vae.decode(normalized).sample  # (B, 3, 1, H, W)
    frame = pixel[:, :, 0].clamp(-1, 1).add(1).div(2)  # (B, 3, H, W), [0,1]
    return frame


def _dino_forward_on_frames(frames_01: torch.Tensor, dino_encoder: DINOv2) -> torch.Tensor:
    """Run DINOv2 on frame tensor and return pooled feature.

    Args:
        frames_01: (B, 3, H, W), range [0,1]
    Returns:
        Tensor (B, D)
    """
    if frames_01.ndim != 4:
        raise ValueError(f"frames_01 must be 4D, got shape={tuple(frames_01.shape)}")

    proc = dino_encoder.processor
    model = dino_encoder.model
    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    # DINOv2 default resolution is typically 224x224.
    target_h = 224
    target_w = 224
    if hasattr(proc, "size") and isinstance(proc.size, dict):
        target_h = int(proc.size.get("height", proc.size.get("shortest_edge", 224)))
        target_w = int(proc.size.get("width", proc.size.get("shortest_edge", 224)))
    elif hasattr(proc, "crop_size") and isinstance(proc.crop_size, dict):
        target_h = int(proc.crop_size.get("height", 224))
        target_w = int(proc.crop_size.get("width", 224))

    x = F.interpolate(frames_01, size=(target_h, target_w), mode="bilinear", align_corners=False)
    mean = torch.tensor(proc.image_mean, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    x = (x - mean) / std
    x = x.to(device=model_device, dtype=model_dtype)

    out = model(pixel_values=x)
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        feat = out.pooler_output
    else:
        feat = out.last_hidden_state[:, 0]
    return feat

def train_one_epoch(
    transformer,
    bolt_layers: BoltReferenceAttentionLayers,
    clip_model: CLIP,
    vlm_selector: VLMFrameSelector | None,
    vae: AutoencoderKLWan,
    dataloader: DataLoader,
    optimizer,
    epoch: int,
    device="cuda",
    dtype=torch.bfloat16,
    grad_accum_steps=2,
    max_grad_norm=1.0,
    bolt_k_select=4,
    bolt_alpha=0.6,
    bolt_power=2.0,
    latents_mean=None,
    latents_std=None,
    choice_idx_random_min: int = 4,
    weighting_scheme: str = "none",
    loss_w_ema_beta: float = 0.95,
    optimize_target: str = "mse",
    id_loss_lambda: float = 0.0,
    id_dino_encoder: DINOv2 | None = None,
    id_loss_max_refs: int = 1,
):
    bolt_layers.train()
    transformer.eval()

    total_mse = 0.0
    total_weighted_loss = 0.0
    total_id_loss = 0.0
    total_opt_loss = 0.0
    ema_loss_w = None
    num_samples = 0
    num_skipped = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}]")
    for step, batch in enumerate(pbar):
        sample = batch[0]

        try:
            vae_latent = sample["vae_latent"]  # (num_chunks, C, T, H, W)
            prompt_embed = sample["prompt_embed"]
            prompt_raw = sample.get("prompt_raw", None)
            prompt_embeds_by_segment = sample.get("prompt_embeds_by_segment", None)
            segments = sample.get("segments", None)
        except Exception as e:
            print(f"  [WARN] Skip: {e}")
            num_skipped += 1
            continue

        num_chunks = vae_latent.shape[0]
        if num_chunks < 3:
            num_skipped += 1
            continue

        # 随机 target chunk：默认下界 choice_idx_random_min（如 4=第三幕起），保证仍有 GAP；
        # 若 num_chunks 太小则退回 legacy 区间 [2, num_chunks)。
        _low = max(2, int(choice_idx_random_min))
        if _low < num_chunks:
            choice_idx = torch.randint(_low, num_chunks, (1,)).item()
        else:
            choice_idx = torch.randint(2, num_chunks, (1,)).item()

        # If segmented prompts exist, align conditioning (and CLIP text scoring) to the current target chunk.
        seg_idx = None
        if (
            isinstance(prompt_embeds_by_segment, torch.Tensor)
            and prompt_embeds_by_segment.ndim == 3
            and isinstance(segments, list)
            and len(segments) == int(prompt_embeds_by_segment.shape[0])
            and len(segments) > 0
        ):
            for i_s, seg in enumerate(segments):
                try:
                    sc = int(seg.get("start_chunk", 0) or 0)
                    ec = int(seg.get("end_chunk", 0) or 0)
                except Exception:
                    sc, ec = 0, 0
                if sc <= choice_idx < ec:
                    seg_idx = i_s
                    break
            if seg_idx is None:
                seg_idx = len(segments) - 1

            # Override DiT conditioning prompt embedding for this training step.
            prompt_embed = prompt_embeds_by_segment[seg_idx]

            # Override prompt_raw for CLIP text similarity (and logging).
            seg_prompt_raw = segments[seg_idx].get("prompt_raw", None)
            if isinstance(seg_prompt_raw, str) and len(seg_prompt_raw) > 0:
                prompt_raw = seg_prompt_raw

            if os.environ.get("HELIOS_DEBUG_SEGMENT_PROMPT", "0") == "1":
                bounds = (
                    int(segments[seg_idx].get("start_chunk", 0) or 0),
                    int(segments[seg_idx].get("end_chunk", 0) or 0),
                )
                snippet = (prompt_raw[:180] + "...") if isinstance(prompt_raw, str) and len(prompt_raw) > 180 else prompt_raw
                uttid = os.path.basename(sample.get("path", ""))
                print(
                    f"[DEBUG][bolt-seg-prompt] uttid={uttid} choice_idx={choice_idx} "
                    f"seg_idx={seg_idx} bounds={bounds} prompt='{snippet}'"
                )

        try:
            (target_latent, hist_short, hist_mid, hist_long,
             idx_target, idx_short, idx_mid, idx_long,
             gap_latents, gap_history_indices) = build_history_and_target(vae_latent, choice_idx)
        except Exception as e:
            print(f"  [WARN] build_history failed: {e}")
            num_skipped += 1
            continue

        if not gap_latents:
            num_skipped += 1
            continue

        target_latent = target_latent.to(device, dtype=dtype)
        prompt_embed = prompt_embed.to(device, dtype=dtype)
        if prompt_embed.ndim == 2:
            prompt_embed = prompt_embed.unsqueeze(0)

        # ── Selector: CLIP+ITS (default) or VLM (optional) ──
        # 训练 Ref-Attn 时，“选哪些参考帧”是一个可替换模块：
        # - CLIP+ITS: 与现有训练逻辑一致（默认，稳定）
        # - VLM selector: decode middle frames -> VLM scoring -> top-k/ITS -> selected_latents
        with torch.no_grad():
            # Context chunk (紧邻 target 的前一个)
            context_latent = vae_latent[choice_idx - 1:choice_idx]  # (1, C, T, H, W)
            context_feat, context_mid = extract_chunk_feature(
                context_latent, vae, clip_model, latents_mean, latents_std
            )

            has_text = prompt_raw is not None and isinstance(prompt_raw, str) and len(prompt_raw) > 0

            if vlm_selector is not None:
                # VLM 训练选帧：对 GAP 候选 decode middle_frame 作为像素输入
                candidates = []
                for gl, abs_idx in zip(gap_latents, gap_history_indices):
                    mid = decode_middle_frame(gl, vae, latents_mean, latents_std)
                    candidates.append(
                        {
                            "chunk_idx": int(abs_idx),
                            "latent": gl.detach().cpu(),
                            "decoded_frame": mid,
                            # clip_feat 仅用于 fallback / 兼容接口（VLM backbone 正常时不会走到）
                            "clip_feat": None,
                        }
                    )
                context_entry = {
                    "chunk_idx": int(choice_idx - 1),
                    "latent": context_latent.detach().cpu(),
                    "decoded_frame": context_mid,
                    "clip_feat": context_feat,
                }
                selected_latents, selected_abs_indices = vlm_selector.select_from_candidates(
                    candidates=candidates,
                    context_entry=context_entry,
                    current_prompt=(prompt_raw or ""),
                )
                sampled_positions = None  # VLM 路径不再用 positions 表示
                selected_latents = [sl.to(device=None) if isinstance(sl, torch.Tensor) else sl for sl in selected_latents]
            else:
                # CLIP+ITS：GAP chunks 的 CLIP 特征
                gap_feats = []
                for gl in gap_latents:
                    gf, _ = extract_chunk_feature(gl, vae, clip_model, latents_mean, latents_std)
                    gap_feats.append(gf)
                # 注意: CLIP 在 CPU 上，所有 similarity 计算也在 CPU
                clip_dev = clip_model.device
                gap_feats_stack = torch.stack(gap_feats, dim=0).to(clip_dev)  # (N_gap, 768)

                # Visual score
                visual_query = context_feat.unsqueeze(0).to(clip_dev)
                visual_scores = clip_model.compute_similarity(gap_feats_stack, visual_query).numpy()

                # Text score: 有 prompt_raw 时用 CLIP text encoder，否则退化为纯 visual
                if has_text:
                    text_query = clip_model.extract_text_features(prompt_raw)
                    text_scores = clip_model.compute_similarity(gap_feats_stack, text_query).numpy()
                else:
                    text_scores = np.zeros_like(visual_scores)

                # Combined score
                combined_scores = bolt_alpha * visual_scores + (1 - bolt_alpha) * text_scores

                # ITS 选帧
                actual_k = min(bolt_k_select, len(gap_latents))
                sampled_positions = inverse_transform_sampling(combined_scores, n=actual_k, power=bolt_power)
                sampled_positions = list(dict.fromkeys(sampled_positions.tolist()))

                selected_latents = [gap_latents[p] for p in sampled_positions]
                selected_abs_indices = [gap_history_indices[p] for p in sampled_positions]

        # ── 选帧监控 ──
        selected_info = []
        sel_v_mean = 0.0
        if vlm_selector is None and sampled_positions is not None:
            for pos in sampled_positions:
                info = {
                    "chunk_idx":      gap_history_indices[pos],  # 绝对位置
                    "visual_score":   float(visual_scores[pos]),
                    "text_score":     float(text_scores[pos]),
                    "combined_score": float(combined_scores[pos]),
                    "time_distance":  choice_idx - gap_history_indices[pos],
                }
                selected_info.append(info)
            # 被选中帧的平均 visual score
            sel_v_mean = float(np.mean([visual_scores[p] for p in sampled_positions])) if sampled_positions else 0.0

        if step % 20 == 0:
            # VLM 路径不用 CLIP text 分数；CLIP+ITS 路径才有 visual/text/combined
            if vlm_selector is not None:
                if isinstance(segments, list) and len(segments) > 0:
                    prompt_note = "分段prompt(segments)"
                elif has_text:
                    prompt_note = "prompt_raw(整段)"
                else:
                    prompt_note = "无文本prompt"
            else:
                prompt_note = "CLIP_text" if has_text else "off(no prompt_raw)"
            sel_mode = "VLM" if vlm_selector is not None else "CLIP+ITS"
            n_sel_log = (
                len(selected_abs_indices)
                if vlm_selector is not None
                else (len(sampled_positions) if sampled_positions is not None else 0)
            )
            print(f"\n  [选帧日志] step={step}, target=chunk_{choice_idx}, "
                  f"N_gap={len(gap_latents)}, 选了{n_sel_log}帧, "
                  f"{prompt_note}")
            print(f"  selector={sel_mode}, selected_abs={selected_abs_indices}")
            if vlm_selector is None:
                print(f"  所有 GAP combined 分数: {combined_scores.round(3).tolist()}")
                if has_text:
                    print(f"  所有 GAP text 分数:     {text_scores.round(3).tolist()}")
                for info in selected_info:
                    print(f"    → chunk_{info['chunk_idx']:02d} "
                          f"(距target {info['time_distance']} 步) "
                          f"visual={info['visual_score']:.3f} "
                          f"text={info['text_score']:.3f} "
                          f"combined={info['combined_score']:.3f}")

        # ── 加噪 ──
        sigma = torch.rand(1).item() * 0.95 + 0.025  # uniform [0.025, 0.975]
        sigma_t = torch.tensor([sigma], device=device, dtype=dtype).reshape(1, 1, 1, 1, 1)
        noise = torch.randn_like(target_latent)
        noisy_input = (1.0 - sigma_t) * target_latent + sigma_t * noise
        target_flow = noise - target_latent
        timestep_val = sigma * 1000.0
        timestep_t = torch.tensor([timestep_val], device=device, dtype=dtype)

        # ── 注册 Bolt Ref-Attn hooks ──
        # offload_bolt=False: bolt 参数常驻 GPU (参数量小)
        # detach 在 hook 内部完成，切断 DiT 反向图
        hooks = bolt_layers.register_hooks(
            transformer, selected_latents=selected_latents, offload_bolt=False,
        )

        # ── Single-pass forward ──
        # 不用 torch.no_grad(): 虽然 DiT frozen，但 hook 内的 bolt 需要梯度
        # hook 内部 detach hidden_states，切断 DiT 反向图，只保留 bolt 分支的梯度
        # DiT 自身因为 requires_grad=False，不会为其参数建立反向图
        try:
            model_pred = transformer(
                hidden_states=noisy_input,
                timestep=timestep_t,
                encoder_hidden_states=prompt_embed,
                indices_hidden_states=idx_target.to(device),
                indices_latents_history_short=idx_short.to(device),
                indices_latents_history_mid=idx_mid.to(device),
                indices_latents_history_long=idx_long.to(device),
                latents_history_short=hist_short.to(device, dtype=dtype),
                latents_history_mid=hist_mid.to(device, dtype=dtype),
                latents_history_long=hist_long.to(device, dtype=dtype),
                return_dict=False,
            )
        finally:
            BoltReferenceAttentionLayers.remove_hooks(hooks)

        if isinstance(model_pred, tuple):
            model_pred = model_pred[0]

        # ── Loss ──
        # Keep optimization target unchanged (plain MSE), and additionally log
        # a train_helios-style weighted flow loss for trend comparability.
        mse = F.mse_loss(model_pred, target_flow)
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=weighting_scheme,
            sigmas=sigma_t,
        )
        weighted_flow_loss = torch.mean(
            (weighting.float() * (model_pred.float() - target_flow.float()) ** 2).reshape(target_flow.shape[0], -1),
            dim=1,
        ).mean()
        if ema_loss_w is None:
            ema_loss_w = float(weighted_flow_loss.item())
        else:
            ema_loss_w = float(loss_w_ema_beta) * float(ema_loss_w) + (1.0 - float(loss_w_ema_beta)) * float(
                weighted_flow_loss.item()
            )
        id_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
        if id_loss_lambda > 0.0 and id_dino_encoder is not None and len(selected_latents) > 0:
            try:
                # flow model predicts v=noise-x => x_hat=noise-v
                pred_clean_latent = noise - model_pred
                pred_frame = _decode_middle_frame_tensor(
                    pred_clean_latent, vae, latents_mean, latents_std
                )  # differentiable path

                ref_frames = []
                max_refs = max(1, int(id_loss_max_refs))
                for sl in selected_latents[:max_refs]:
                    if not isinstance(sl, torch.Tensor):
                        continue
                    ref_lat = sl
                    if ref_lat.ndim == 4:
                        ref_lat = ref_lat.unsqueeze(0)
                    ref_frame = _decode_middle_frame_tensor(ref_lat, vae, latents_mean, latents_std)
                    ref_frames.append(ref_frame.detach())

                if len(ref_frames) > 0:
                    ref_frame = torch.cat(ref_frames, dim=0).mean(dim=0, keepdim=True)
                    pred_feat = _dino_forward_on_frames(pred_frame, id_dino_encoder)
                    ref_feat = _dino_forward_on_frames(ref_frame, id_dino_encoder).detach()
                    id_loss = (1.0 - F.cosine_similarity(pred_feat.float(), ref_feat.float(), dim=-1)).mean()
            except Exception as exc:  # noqa: BLE001
                if step % 20 == 0:
                    print(f"[WARN] id_loss compute failed at step={step}: {exc}")
                id_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        primary_obj = weighted_flow_loss if str(optimize_target).lower() in {"weighted", "loss_w"} else mse
        total_obj = primary_obj + float(id_loss_lambda) * id_loss
        loss = total_obj / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(bolt_layers.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            # P0: γ 硬 clamp，防 outlier 通道跑飞（|γ|_max > 0.4 已实证不安全）
            bolt_layers.clamp_gamma(-0.4, 0.4)

        total_mse += float(mse.item())
        total_weighted_loss += float(weighted_flow_loss.item())
        total_id_loss += float(id_loss.item())
        total_opt_loss += float(total_obj.item())
        num_samples += 1

        # ── 监控 LayerScale γ / 权重位移 / 注意力分布 / 有效注入比 r ──
        # S1: |γ| 进入 [1e-2, 1e-1]；S3: ‖out_proj‖ 相对初始化翻倍；
        # S4: r ∈ [0.05, 0.2] 表示 Ref-Attn 正常侵入 DiT
        # 内在收敛信号: H_attn 单调下降 + dW_rel 单调上升 + Δmse(S5) < 0
        stats = bolt_layers.collect_stats()
        avg_mse = total_mse / num_samples
        avg_weighted_loss = total_weighted_loss / num_samples
        avg_id_loss = total_id_loss / num_samples
        avg_opt_loss = total_opt_loss / num_samples

        # ── S5: 每 50 step 临时置 γ=0 跑一次 forward，对比 Δmse(γ vs γ=0) ──
        # 这是与 timestep 噪声解耦的"Ref-Attn 是否真的在帮"判据。
        # 期望：训练有效时 Δmse < 0 且绝对值随 epoch 增大；长期 ≥ 0 说明在拖后腿。
        s5_log = ""
        if (step + 1) % 50 == 0:
            saved_gammas = [m.gamma.data.clone() for m in bolt_layers.ref_attn_modules.values()]
            try:
                for m in bolt_layers.ref_attn_modules.values():
                    m.gamma.data.zero_()
                hooks_s5 = bolt_layers.register_hooks(
                    transformer, selected_latents=selected_latents, offload_bolt=False,
                )
                try:
                    with torch.no_grad():
                        pred0 = transformer(
                            hidden_states=noisy_input,
                            timestep=timestep_t,
                            encoder_hidden_states=prompt_embed,
                            indices_hidden_states=idx_target.to(device),
                            indices_latents_history_short=idx_short.to(device),
                            indices_latents_history_mid=idx_mid.to(device),
                            indices_latents_history_long=idx_long.to(device),
                            latents_history_short=hist_short.to(device, dtype=dtype),
                            latents_history_mid=hist_mid.to(device, dtype=dtype),
                            latents_history_long=hist_long.to(device, dtype=dtype),
                            return_dict=False,
                        )
                    if isinstance(pred0, tuple):
                        pred0 = pred0[0]
                    mse_gamma0 = F.mse_loss(pred0.float(), target_flow.float()).item()
                finally:
                    BoltReferenceAttentionLayers.remove_hooks(hooks_s5)
            finally:
                for m, g in zip(bolt_layers.ref_attn_modules.values(), saved_gammas):
                    m.gamma.data.copy_(g)
            mse_gamma = float(mse.item())
            delta_mse = mse_gamma - mse_gamma0
            s5_log = (
                f"  [S5 step={step+1}] mse(γ)={mse_gamma:.5f}, "
                f"mse(γ=0)={mse_gamma0:.5f}, Δmse={delta_mse:+.5f}"
            )
            del pred0
            torch.cuda.empty_cache()

        pbar.set_postfix({
            "mse": f"{mse.item():.4f}",
            "avg_mse": f"{avg_mse:.4f}",
            "loss_w": f"{weighted_flow_loss.item():.4f}",
            "avg_loss_w": f"{avg_weighted_loss:.4f}",
            "loss_w_ema": f"{ema_loss_w:.4f}",
            "id_loss": f"{id_loss.item():.4f}",
            "opt_loss": f"{total_obj.item():.4f}",
            "|γ|": f"{stats['gamma_abs_mean']:.4f}",
            "|γ|max": f"{stats['gamma_abs_max']:.4f}",
            "r": f"{stats['r_mean']:.3f}",
            "dW_rel": f"{stats['out_proj_rel_change']:.4f}",
            "H_attn": f"{stats['attn_entropy']:.3f}",
            "p_max": f"{stats['attn_max_prob']:.3f}",
            "op_fro": f"{stats['out_proj_fro']:.2f}",
            "sel": str(selected_abs_indices),
            "v_score": f"{sel_v_mean:.3f}",
        })

        if (step + 1) % 50 == 0:
            print(
                f"\n  [Step {step+1}] mse={mse.item():.6f}, "
                f"avg_mse={avg_mse:.6f}, "
                f"loss_w={weighted_flow_loss.item():.6f}, "
                f"avg_loss_w={avg_weighted_loss:.6f}, "
                f"loss_w_ema={ema_loss_w:.6f}, "
                f"id_loss={id_loss.item():.6f}, "
                f"avg_id_loss={avg_id_loss:.6f}, "
                f"avg_opt_loss={avg_opt_loss:.6f}, "
                f"|γ|_mean={stats['gamma_abs_mean']:.5f}, "
                f"|γ|_max={stats['gamma_abs_max']:.5f}, "
                f"r_mean={stats['r_mean']:.4f}, "
                f"r_max={stats['r_max']:.4f}, "
                f"H_attn={stats['attn_entropy']:.4f}, "
                f"p_max={stats['attn_max_prob']:.4f}, "
                f"kv_norm={stats['kv_norm']:.3f}, "
                f"op_fro={stats['out_proj_fro']:.3f}, "
                f"op_dW={stats['out_proj_delta_fro']:.3f} "
                f"(rel={stats['out_proj_rel_change']:.4f}), "
                f"qkv_dW={stats['qkv_delta_fro']:.3f} "
                f"(rel={stats['qkv_rel_change']:.4f}), "
                f"samples={num_samples}, skipped={num_skipped}"
            )
            if s5_log:
                print(s5_log)

        # 清理显存
        del model_pred, mse, weighted_flow_loss, id_loss, total_obj, loss, noisy_input, target_flow, noise
        torch.cuda.empty_cache()

    # 处理最后一批未对齐 grad_accum_steps 的残余梯度
    if num_samples % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(bolt_layers.parameters(), max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
        bolt_layers.clamp_gamma(-0.4, 0.4)

    avg_mse = total_mse / max(num_samples, 1)
    avg_weighted_loss = total_weighted_loss / max(num_samples, 1)
    avg_id_loss = total_id_loss / max(num_samples, 1)
    avg_opt_loss = total_opt_loss / max(num_samples, 1)
    return {
        "avg_mse": float(avg_mse),
        "avg_weighted_loss": float(avg_weighted_loss),
        "avg_id_loss": float(avg_id_loss),
        "avg_opt_loss": float(avg_opt_loss),
        "loss_w_ema_last": float(0.0 if ema_loss_w is None else ema_loss_w),
    }


@torch.no_grad()
def evaluate_flow_matching_avg_mse(
    transformer,
    bolt_layers: BoltReferenceAttentionLayers,
    clip_model: CLIP,
    vlm_selector: VLMFrameSelector | None,
    vae: AutoencoderKLWan,
    dataloader: DataLoader,
    *,
    device="cuda",
    dtype=torch.bfloat16,
    bolt_k_select=4,
    bolt_alpha=0.6,
    bolt_power=2.0,
    latents_mean=None,
    latents_std=None,
    choice_idx_random_min: int = 4,
    seed: int = 42,
    max_batches: int = 0,
    weighting_scheme: str = "none",
):
    """与训练目标一致的验证：仅计算 Flow Matching MSE（不反传）。"""
    bolt_layers.eval()
    transformer.eval()
    np.random.seed(int(seed))

    total_mse = 0.0
    total_weighted_loss = 0.0
    num_samples = 0
    num_skipped = 0
    gen = torch.Generator(device=device).manual_seed(int(seed))

    for step, batch in enumerate(dataloader):
        if max_batches and step >= int(max_batches):
            break
        sample = batch[0]
        try:
            vae_latent = sample["vae_latent"]
            prompt_embed = sample["prompt_embed"]
            prompt_raw = sample.get("prompt_raw", None)
            prompt_embeds_by_segment = sample.get("prompt_embeds_by_segment", None)
            segments = sample.get("segments", None)
        except Exception:
            num_skipped += 1
            continue

        num_chunks = vae_latent.shape[0]
        if num_chunks < 3:
            num_skipped += 1
            continue

        # 为了让 val avg 可比较，使用确定性的 target chunk 选择（不走随机）。
        _low = max(2, int(choice_idx_random_min))
        choice_idx = _low if _low < num_chunks else 2

        seg_idx = None
        if (
            isinstance(prompt_embeds_by_segment, torch.Tensor)
            and prompt_embeds_by_segment.ndim == 3
            and isinstance(segments, list)
            and len(segments) == int(prompt_embeds_by_segment.shape[0])
            and len(segments) > 0
        ):
            for i_s, seg in enumerate(segments):
                try:
                    sc = int(seg.get("start_chunk", 0) or 0)
                    ec = int(seg.get("end_chunk", 0) or 0)
                except Exception:
                    sc, ec = 0, 0
                if sc <= choice_idx < ec:
                    seg_idx = i_s
                    break
            if seg_idx is None:
                seg_idx = len(segments) - 1
            prompt_embed = prompt_embeds_by_segment[seg_idx]
            seg_prompt_raw = segments[seg_idx].get("prompt_raw", None)
            if isinstance(seg_prompt_raw, str) and len(seg_prompt_raw) > 0:
                prompt_raw = seg_prompt_raw

        try:
            (
                target_latent,
                hist_short,
                hist_mid,
                hist_long,
                idx_target,
                idx_short,
                idx_mid,
                idx_long,
                gap_latents,
                gap_history_indices,
            ) = build_history_and_target(vae_latent, choice_idx)
        except Exception:
            num_skipped += 1
            continue

        if not gap_latents:
            num_skipped += 1
            continue

        target_latent = target_latent.to(device, dtype=dtype)
        prompt_embed = prompt_embed.to(device, dtype=dtype)
        if prompt_embed.ndim == 2:
            prompt_embed = prompt_embed.unsqueeze(0)

        context_latent = vae_latent[choice_idx - 1 : choice_idx]
        context_feat, context_mid = extract_chunk_feature(
            context_latent, vae, clip_model, latents_mean, latents_std
        )
        has_text = prompt_raw is not None and isinstance(prompt_raw, str) and len(prompt_raw) > 0

        if vlm_selector is not None:
            candidates = []
            for gl, abs_idx in zip(gap_latents, gap_history_indices):
                mid = decode_middle_frame(gl, vae, latents_mean, latents_std)
                candidates.append(
                    {
                        "chunk_idx": int(abs_idx),
                        "latent": gl.detach().cpu(),
                        "decoded_frame": mid,
                        "clip_feat": None,
                    }
                )
            context_entry = {
                "chunk_idx": int(choice_idx - 1),
                "latent": context_latent.detach().cpu(),
                "decoded_frame": context_mid,
                "clip_feat": context_feat,
            }
            selected_latents, _ = vlm_selector.select_from_candidates(
                candidates=candidates,
                context_entry=context_entry,
                current_prompt=(prompt_raw or ""),
            )
            selected_latents = [sl.to(device=None) if isinstance(sl, torch.Tensor) else sl for sl in selected_latents]
        else:
            gap_feats = []
            for gl in gap_latents:
                gf, _ = extract_chunk_feature(gl, vae, clip_model, latents_mean, latents_std)
                gap_feats.append(gf)
            clip_dev = clip_model.device
            gap_feats_stack = torch.stack(gap_feats, dim=0).to(clip_dev)
            visual_query = context_feat.unsqueeze(0).to(clip_dev)
            visual_scores = clip_model.compute_similarity(gap_feats_stack, visual_query).numpy()
            if has_text:
                text_query = clip_model.extract_text_features(prompt_raw)
                text_scores = clip_model.compute_similarity(gap_feats_stack, text_query).numpy()
            else:
                text_scores = np.zeros_like(visual_scores)
            combined_scores = bolt_alpha * visual_scores + (1 - bolt_alpha) * text_scores
            actual_k = min(bolt_k_select, len(gap_latents))
            sampled_positions = inverse_transform_sampling(combined_scores, n=actual_k, power=bolt_power)
            sampled_positions = list(dict.fromkeys(sampled_positions.tolist()))
            selected_latents = [gap_latents[p] for p in sampled_positions]

        sigma = torch.rand((1,), generator=gen, device=device).item() * 0.95 + 0.025
        sigma_t = torch.tensor([sigma], device=device, dtype=dtype).reshape(1, 1, 1, 1, 1)
        noise = torch.randn(target_latent.shape, generator=gen, device=device, dtype=dtype)
        noisy_input = (1.0 - sigma_t) * target_latent + sigma_t * noise
        target_flow = noise - target_latent
        timestep_val = sigma * 1000.0
        timestep_t = torch.tensor([timestep_val], device=device, dtype=dtype)

        hooks = bolt_layers.register_hooks(transformer, selected_latents=selected_latents, offload_bolt=False)
        try:
            model_pred = transformer(
                hidden_states=noisy_input,
                timestep=timestep_t,
                encoder_hidden_states=prompt_embed,
                indices_hidden_states=idx_target.to(device),
                indices_latents_history_short=idx_short.to(device),
                indices_latents_history_mid=idx_mid.to(device),
                indices_latents_history_long=idx_long.to(device),
                latents_history_short=hist_short.to(device, dtype=dtype),
                latents_history_mid=hist_mid.to(device, dtype=dtype),
                latents_history_long=hist_long.to(device, dtype=dtype),
                return_dict=False,
            )
        finally:
            BoltReferenceAttentionLayers.remove_hooks(hooks)

        if isinstance(model_pred, tuple):
            model_pred = model_pred[0]
        mse = F.mse_loss(model_pred.float(), target_flow.float()).item()
        weighting = compute_loss_weighting_for_sd3(
            weighting_scheme=weighting_scheme,
            sigmas=sigma_t,
        )
        weighted_loss = torch.mean(
            (weighting.float() * (model_pred.float() - target_flow.float()) ** 2).reshape(target_flow.shape[0], -1),
            dim=1,
        ).mean().item()
        total_mse += float(mse)
        total_weighted_loss += float(weighted_loss)
        num_samples += 1

    avg = total_mse / max(num_samples, 1)
    avg_weighted = total_weighted_loss / max(num_samples, 1)
    return {
        "avg_mse": float(avg),
        "avg_weighted_loss": float(avg_weighted),
        "num_samples": int(num_samples),
        "num_skipped": int(num_skipped),
    }


# ═══════════════════════════════════════════
# Validation helpers (align with infer_helios_bolt.py)
# ═══════════════════════════════════════════

def _weight_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return "fp32"


def build_vlm_selector_for_validation(clip_model: CLIP, args, device: str, dtype: torch.dtype) -> VLMFrameSelector:
    """与 infer_helios_bolt 一致：CLIP pre-filter + VLM backbone + rank mode。"""
    from helios.modules.vlm_backbones import load_default_backbone
    import json as _json

    if not getattr(args, "vlm_model_path", None):
        raise ValueError("validate (VLM): set --vlm_model_path or validation 配置 vlm_model_path")

    resolved_score_mode = "yes_no"
    st = getattr(args, "vlm_score_mode", None)
    if st in {"yes_no", "hidden_head"}:
        resolved_score_mode = st
    elif getattr(args, "vlm_lora_path", None):
        meta_path = os.path.join(args.vlm_lora_path, "score_mode.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = _json.load(f)
                m = str(meta.get("score_mode", "")).strip()
                if m in {"yes_no", "hidden_head"}:
                    resolved_score_mode = m
            except Exception:
                pass

    head_path = None
    if getattr(args, "vlm_lora_path", None):
        head_path = os.path.join(args.vlm_lora_path, "selector_head.pt")
        if not os.path.exists(head_path):
            head_path = None

    vlm_backbone = load_default_backbone(
        model_path=args.vlm_model_path,
        lora_path=getattr(args, "vlm_lora_path", None),
        dtype=_weight_dtype_to_str(dtype),
        device=device,
        score_mode=resolved_score_mode,
        head_path=head_path,
    )
    vk = getattr(args, "vlm_k_select", None)
    if vk is None:
        vk = getattr(args, "bolt_k_select", 4)
    vpow = getattr(args, "vlm_power", None)
    if vpow is None:
        vpow = getattr(args, "bolt_power", 2.0)
    vmin = getattr(args, "vlm_min_chunk_distance", None)
    if vmin is None:
        vmin = getattr(args, "bolt_min_chunk_distance", 3)
    return VLMFrameSelector(
        clip_model=clip_model,
        k=int(vk),
        power=float(vpow),
        min_chunk_distance=int(vmin),
        max_candidates=int(getattr(args, "vlm_max_candidates", 16)),
        prefilter_alpha=float(getattr(args, "vlm_prefilter_alpha", getattr(args, "bolt_alpha", 0.6))),
        temperature=float(getattr(args, "vlm_temperature", 0.7)),
        vlm_backbone=vlm_backbone,
        device=device,
        fallback_to_random=False,
        vlm_rank_mode=str(getattr(args, "vlm_rank_mode", "its")).lower(),
    )


# ═══════════════════════════════════════════
# Validation (Interactive Inference)
# ═══════════════════════════════════════════

def validate_epoch(
    transformer,
    bolt_layers: BoltReferenceAttentionLayers,
    clip_model: CLIP,
    vae: AutoencoderKLWan,
    val_cfg: dict,
    epoch: int,
    output_dir: str,
    bolt_k_select: int = 4,
    bolt_alpha: float = 0.6,
    bolt_power: float = 2.0,
    bolt_min_chunk_distance: int = 3,
    device="cuda",
    dtype=torch.bfloat16,
    args=None,
):
    """执行一次 interactive 推理验证，输出视频并打印选帧详情。

    - selector_type=clip_its: CLIP+ITS 选帧（旧路径）
    - selector_type=vlm: 与 infer_helios_bolt 对齐：VLM +（默认开启）LongMemory codebook + fast/slow cache
    """
    from helios.pipelines.pipeline_helios import HeliosPipeline
    from helios.scheduler.scheduling_helios import HeliosScheduler
    from helios.modules.select_frames import select_gap_frames
    from helios.modules.extract_feature import DINOv2, decode_all_frames, extract_tail_embedding
    from helios.modules.select_frames_vlm import (
        compute_slow_step_chunks,
        parse_slow_step_chunks,
    )
    from helios.modules.memory_bank import VLMSelectorCache, evict_history
    from helios.modules.long_memory import (
        LongMemoryStore,
        build_codebook_candidates_for_vlm,
        cosine_similarity,
    )
    from diffusers.utils import export_to_video

    if args is None:
        args = argparse.Namespace(
            selector_type="clip_its",
            bolt_k_select=4,
            bolt_alpha=0.6,
            bolt_power=2.0,
            bolt_min_chunk_distance=3,
            enable_long_memory=False,
            vlm_model_path=None,
            vlm_lora_path=None,
            vlm_score_mode="yes_no",
            vlm_rank_mode="its",
            vlm_max_candidates=16,
            vlm_prefilter_alpha=0.6,
            vlm_temperature=0.7,
            vlm_power=None,
            vlm_min_chunk_distance=None,
            vlm_k_select=None,
            vlm_slow_step_chunks=None,
            vlm_extra_slow_at_segment_mid=False,
            vlm_fallback_to_clip=False,
            vlm_use_chunk_video=False,
            vlm_video_max_frames=None,
            mb_max_history_chunks=32,
            mb_keep_recent_k=8,
            mb_evict_strategy="farthest_lowclip",
            mb_evict_alpha=0.5,
            no_vlm_cache_invalidate_on_evict=False,
            dino_model_path=None,
            lm_tau_merge=0.85,
            lm_tau_cut=0.35,
            lm_ema_alpha_new=0.2,
            lm_codebook_max_size=512,
            lm_codebook_topm=16,
            lm_codebook_evict="lru",
            lm_debug=False,
            bolt_log_every_chunk=False,
            weight_dtype="bfloat16",
        )

    bolt_layers.eval()
    transformer.eval()

    val_dir = os.path.join(output_dir, "validation", f"epoch_{epoch:03d}")
    os.makedirs(val_dir, exist_ok=True)

    # VAE normalization 参数
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(vae.device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        vae.device, vae.dtype
    )

    # ── 构建 Pipeline (复用已加载的 transformer 和 vae) ──
    print(f"\n{'='*60}")
    print(f"  [VALIDATION] Epoch {epoch} — Interactive Inference")
    sel_t = getattr(args, "selector_type", "clip_its")
    print(f"  selector_type={sel_t} (VLM 验证默认启用 LongMemory codebook，可用 validation_config 覆盖)")
    print(f"{'='*60}")

    base_model_path = val_cfg.get("base_model_path", None)
    scheduler = HeliosScheduler.from_pretrained(base_model_path, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(
        base_model_path, transformer=transformer, vae=vae, scheduler=scheduler,
        torch_dtype=dtype,
    )

    # 训练阶段 transformer 已经启用了 group offload，进入验证前必须先清理旧 hooks，
    # 否则再次 enable_group_offload 会因同名 hook 重复注册而报错。
    _hook_names_to_clean = ["layer_execution_tracker", "lazy_prefetch_group_offloading", "group_offloading"]
    for _component in [transformer, vae]:
        for _name, _mod in _component.named_modules():
            if hasattr(_mod, "_diffusers_hook"):
                for _hname in _hook_names_to_clean:
                    if _mod._diffusers_hook.get_hook(_hname) is not None:
                        _mod._diffusers_hook.remove_hook(_hname, recurse=False)

    if val_cfg.get("enable_low_vram_mode", True):
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type=val_cfg.get("group_offloading_type", "leaf_level"),
            use_stream=True, record_stream=True,
        )
    else:
        pipe = pipe.to(device)

    # ── 选帧日志收集 ──
    all_selection_logs = []
    print("  [VAL] dino_ref_vs_gt 指标已停用（deprecated）。")

    def _val_cfg_bool(key, default):
        if key in val_cfg:
            return bool(val_cfg[key])
        return default

    def _val_cfg_get(key, default):
        return val_cfg.get(key, default)

    use_vlm_path = (getattr(args, "selector_type", "clip_its") == "vlm")
    # 与训练/推理对齐：VLM 验证默认走 LongMemory；可被 validation_config.enable_long_memory=false 关闭
    if "enable_long_memory" in val_cfg:
        use_long_mem = _val_cfg_bool("enable_long_memory", True)
    else:
        use_long_mem = True if use_vlm_path else False

    dino_for_gt = None
    if use_long_mem or use_vlm_path:
        dino_path = _val_cfg_get("dino_model_path", getattr(args, "dino_model_path", None))
        wd = getattr(args, "weight_dtype", "bfloat16")
        dino_for_gt = DINOv2(device=device, model_id_or_path=dino_path, dtype="fp16" if wd == "fp16" else "bf16")
    if _val_cfg_get("validation_gt_pt", None) is not None:
        print("  [VAL] validation_gt_pt 已忽略（dino_ref_vs_gt 指标已停用）。")

    selector_vlm = None
    if use_vlm_path:
        selector_vlm = build_vlm_selector_for_validation(clip_model, args, device, dtype)
        print("  [VAL] VLM selector built (CLIP pre-filter + VLM, same as infer_helios_bolt).")

    def make_chunk_callback(
        prompt_text,
        history_ref,
        active_hook_ref,
        video_id,
        interpolate_time_list=None,
    ):
        if isinstance(prompt_text, list) and interpolate_time_list is not None:
            from itertools import accumulate
            seg_boundaries = list(accumulate(interpolate_time_list))
        else:
            seg_boundaries = None

        def _get_current_prompt(chunk_idx):
            if seg_boundaries is None:
                return prompt_text if isinstance(prompt_text, str) else prompt_text[0]
            for seg_i, boundary in enumerate(seg_boundaries):
                if chunk_idx < boundary:
                    return prompt_text[seg_i]
            return prompt_text[-1]

        if not use_vlm_path:
            # ── CLIP+ITS（旧路径）──
            def chunk_callback(chunk_idx: int, chunk_latent: torch.Tensor):
                with torch.no_grad():
                    clip_feat, _ = extract_chunk_feature(
                        chunk_latent, vae, clip_model, latents_mean, latents_std
                    )
                    entry = {
                        "chunk_idx": chunk_idx,
                        "latent": chunk_latent.detach().cpu(),
                        "clip_feat": clip_feat,
                    }
                    history_ref.append(entry)

                    if active_hook_ref[0] is not None:
                        BoltReferenceAttentionLayers.remove_hooks(active_hook_ref[0])
                        active_hook_ref[0] = None

                    current_prompt = _get_current_prompt(chunk_idx + 1)
                    selected_latents, selected_indices = select_gap_frames(
                        history=history_ref,
                        current_chunk_idx=chunk_idx + 1,
                        current_prompt=current_prompt,
                        clip_model=clip_model,
                        k=bolt_k_select,
                        alpha=bolt_alpha,
                        power=bolt_power,
                        min_chunk_distance=bolt_min_chunk_distance,
                        device=device,
                    )

                    log_entry = {
                        "video_id": video_id,
                        "chunk_idx": chunk_idx,
                        "next_chunk": chunk_idx + 1,
                        "prompt_snippet": current_prompt[:60],
                        "history_size": len(history_ref),
                        "selected_indices": [int(x) for x in selected_indices],
                        "num_selected": len(selected_latents),
                        "schedule_mode": "clip_its",
                    }

                    if selected_latents:
                        gap_history = history_ref[:chunk_idx]
                        if len(gap_history) > 0 and len(selected_indices) > 0:
                            gap_feats = torch.stack([e["clip_feat"] for e in gap_history], dim=0).to(clip_model.device)
                            context_feat = history_ref[chunk_idx]["clip_feat"].unsqueeze(0).to(clip_model.device)
                            v_scores = clip_model.compute_similarity(gap_feats, context_feat).numpy()
                            t_query = clip_model.extract_text_features(current_prompt).to(clip_model.device)
                            t_scores = clip_model.compute_similarity(gap_feats, t_query).numpy()
                            detail_list = []
                            for sel_idx in selected_indices:
                                if sel_idx < len(v_scores):
                                    detail_list.append({
                                        "chunk": int(sel_idx),
                                        "visual": float(v_scores[sel_idx]),
                                        "text": float(t_scores[sel_idx]),
                                        "combined": float(bolt_alpha * v_scores[sel_idx] + (1 - bolt_alpha) * t_scores[sel_idx]),
                                    })
                            log_entry["selection_detail"] = detail_list

                        print(f"  [VAL chunk {chunk_idx}] clip_its → next={chunk_idx+1} "
                              f"selected={selected_indices}")
                        hooks = bolt_layers.register_hooks(transformer, selected_latents=selected_latents)
                        active_hook_ref[0] = hooks
                    else:
                        if chunk_idx > 0:
                            print(f"  [VAL chunk {chunk_idx}] 无 GAP 帧可选 (history={len(history_ref)})")

                    all_selection_logs.append(log_entry)

            return chunk_callback

        # ── VLM + LongMemory（与 infer_helios_bolt 对齐）──
        long_memory = None
        prev_tail_emb = None
        if use_long_mem and dino_for_gt is not None:
            long_memory = LongMemoryStore(
                tau_merge=float(_val_cfg_get("lm_tau_merge", getattr(args, "lm_tau_merge", 0.85))),
                tau_cut=float(_val_cfg_get("lm_tau_cut", getattr(args, "lm_tau_cut", 0.35))),
                ema_alpha_new=float(_val_cfg_get("lm_ema_alpha_new", getattr(args, "lm_ema_alpha_new", 0.2))),
                max_size=int(_val_cfg_get("lm_codebook_max_size", getattr(args, "lm_codebook_max_size", 512))),
                evict_strategy=str(_val_cfg_get("lm_codebook_evict", getattr(args, "lm_codebook_evict", "lru"))),
            )
            print(f"  [VAL] LongMemory enabled: tau_cut={long_memory.tau_cut}, codebook_max={long_memory.max_size}")
        elif use_vlm_path and not use_long_mem:
            print("  [VAL] LongMemory disabled (enable_long_memory=false); VLM 仅 history+prefilter 路径。")

        selector_cache = VLMSelectorCache()
        explicit_slow = parse_slow_step_chunks(getattr(args, "vlm_slow_step_chunks", None))
        auto_slow = compute_slow_step_chunks(
            interpolate_time_list,
            extra_mid=bool(getattr(args, "vlm_extra_slow_at_segment_mid", False)),
        )
        slow_triggers = explicit_slow if explicit_slow is not None else auto_slow

        def _next_slow_trigger(idx: int) -> int:
            future = sorted([x for x in slow_triggers if x > idx])
            return future[0] if future else (idx + 1)

        def chunk_callback(chunk_idx: int, chunk_latent: torch.Tensor):
            nonlocal prev_tail_emb
            with torch.no_grad():
                clip_feat, middle_frame = extract_chunk_feature(
                    chunk_latent, vae, clip_model, latents_mean, latents_std
                )
                tail_emb = None
                if dino_for_gt is not None:
                    tail_emb = extract_tail_embedding(
                        chunk_latent, vae, dino_for_gt, latents_mean, latents_std, return_frame=False,
                    )
                decoded_frames = None
                if getattr(args, "vlm_use_chunk_video", False):
                    decoded_frames = decode_all_frames(
                        chunk_latent, vae, latents_mean, latents_std,
                        max_frames=getattr(args, "vlm_video_max_frames", None),
                    )

                # 与训练侧一致：为 VLM 提供 chunk middle 像素，避免 __call__ 路径缺图退回 CLIP 占位
                history_ref.append({
                    "chunk_idx": int(chunk_idx),
                    "latent": chunk_latent.detach().cpu(),
                    "clip_feat": clip_feat,
                    "decoded_frame": middle_frame,
                    "decoded_frames": decoded_frames,
                    "tail_emb": tail_emb,
                })

                evicted = evict_history(
                    history_ref,
                    max_n=int(getattr(args, "mb_max_history_chunks", 32)),
                    keep_recent=int(getattr(args, "mb_keep_recent_k", 8)),
                    strategy=str(getattr(args, "mb_evict_strategy", "farthest_lowclip")),
                    current_chunk_idx=int(chunk_idx),
                    protected_chunk_idx=selector_cache.selected_chunk_idx,
                    alpha=float(getattr(args, "mb_evict_alpha", 0.5)),
                )
                if evicted and not getattr(args, "no_vlm_cache_invalidate_on_evict", False):
                    if any(int(x) in set(evicted) for x in selector_cache.selected_chunk_idx):
                        selector_cache.update([], int(chunk_idx), "", [])

                if active_hook_ref[0] is not None:
                    BoltReferenceAttentionLayers.remove_hooks(active_hook_ref[0])
                    active_hook_ref[0] = None

                next_chunk_idx = int(chunk_idx) + 1
                clip_prompt = _get_current_prompt(next_chunk_idx)

                def _run_clip_its():
                    return select_gap_frames(
                        history=history_ref,
                        current_chunk_idx=next_chunk_idx,
                        current_prompt=clip_prompt,
                        clip_model=clip_model,
                        k=bolt_k_select,
                        alpha=bolt_alpha,
                        power=bolt_power,
                        min_chunk_distance=bolt_min_chunk_distance,
                        device=device,
                    )

                if long_memory is not None and tail_emb is not None:
                    try:
                        action, slot_i, best_sim = long_memory.update(
                            e_tail=tail_emb,
                            chunk_idx=int(chunk_idx),
                            latent=chunk_latent.detach().cpu(),
                            decoded_frame=middle_frame,
                        )
                        _tm = float(_val_cfg_get("lm_tau_merge", getattr(args, "lm_tau_merge", 0.85)))
                        _ea = float(_val_cfg_get("lm_ema_alpha_new", getattr(args, "lm_ema_alpha_new", 0.2)))
                        print(
                            f"  [Codebook][update] chunk={int(chunk_idx)} action={action} slot={slot_i} "
                            f"best_sim={best_sim:.4f} τ_merge={_tm:.3f} ema_α={_ea:.3f} |N={len(long_memory)}"
                        )
                        if getattr(args, "lm_debug", False) or getattr(args, "bolt_log_every_chunk", False):
                            print(
                                f"  [LongMemory][update] chunk={int(chunk_idx)} "
                                f"action={action} slot={slot_i} best_sim={best_sim:.3f} "
                                f"codebook_size={len(long_memory)}"
                            )
                    except Exception as exc:
                        print(f"  [VAL][LongMemory] update failed: {exc}")

                # cut / codebook 诊断（写入 log_entry）
                schedule_mode = "fast"
                is_cut = False
                dist_prev = None
                sim_prev = None
                _tau_c = float(_val_cfg_get("lm_tau_cut", getattr(args, "lm_tau_cut", 0.35)))
                codebook_candidates_meta = None

                if long_memory is not None and tail_emb is not None and prev_tail_emb is not None:
                    sim_prev = float(cosine_similarity(prev_tail_emb, tail_emb).item())
                    dist_prev = 1.0 - sim_prev
                    is_cut = dist_prev >= _tau_c

                if is_cut and long_memory is not None and tail_emb is not None:
                    try:
                        candidates = build_codebook_candidates_for_vlm(
                            store=long_memory,
                            query_embedding=tail_emb,
                            topm=int(_val_cfg_get("lm_codebook_topm", getattr(args, "lm_codebook_topm", 16))),
                            exclude_chunk_idx=[int(chunk_idx)],
                        )
                        for c in candidates:
                            if c.get("decoded_frame") is None:
                                c["decoded_frame"] = decode_middle_frame(
                                    c["latent"], vae, latents_mean, latents_std
                                )
                        context_entry = {"decoded_frame": middle_frame, "clip_feat": clip_feat}
                        cand_summary = [
                            f"c{int(c['chunk_idx'])}:{float(c.get('_codebook_score', 0.0)):.3f}"
                            for c in candidates[:8]
                        ]
                        more = "" if len(candidates) <= 8 else f" …(+{len(candidates) - 8})"
                        print(
                            f"  [VAL chunk {chunk_idx}] slow/longmem "
                            f"(d_prev={dist_prev:.4f}≥τ_cut={_tau_c:.3f} sim_prev={sim_prev:.4f}) "
                            f"codebook→{len(candidates)} cand [{', '.join(cand_summary)}{more}]"
                        )
                        selected_latents, selected_indices = selector_vlm.select_from_candidates(
                            candidates=candidates,
                            context_entry=context_entry,
                            current_prompt=clip_prompt,
                        )
                        schedule_mode = "slow/longmem"
                        codebook_candidates_meta = [
                            {"chunk": int(c["chunk_idx"]), "score": float(c.get("_codebook_score", 0.0))}
                            for c in candidates
                        ]
                    except Exception as exc:
                        if getattr(args, "vlm_fallback_to_clip", False):
                            print(f"  [VAL] VLM slow failed ({exc}), fallback CLIP+ITS.")
                            selected_latents, selected_indices = _run_clip_its()
                            schedule_mode = "slow/clip_fallback"
                        else:
                            raise
                else:
                    # fast：由 cut 门控决定，不注入 ref-attn（仅短程 DiT history）。
                    schedule_mode = "fast"
                    selected_latents, selected_indices = [], []

                if tail_emb is not None:
                    prev_tail_emb = tail_emb

                log_entry = {
                    "video_id": video_id,
                    "chunk_idx": chunk_idx,
                    "next_chunk": next_chunk_idx,
                    "prompt_snippet": clip_prompt[:60],
                    "history_size": len(history_ref),
                    "selected_indices": [int(x) for x in selected_indices],
                    "num_selected": len(selected_latents),
                    "schedule_mode": schedule_mode,
                    "cut_tau": float(_tau_c),
                    "cut_is_cut": bool(is_cut),
                }
                if dist_prev is not None:
                    log_entry["cut_dist_prev"] = float(dist_prev)
                if sim_prev is not None:
                    log_entry["cut_sim_prev"] = float(sim_prev)
                if codebook_candidates_meta is not None:
                    log_entry["codebook_candidates"] = codebook_candidates_meta

                if selected_latents:
                    _extra = ""
                    if codebook_candidates_meta:
                        _cb = ", ".join(
                            f"c{x['chunk']}:{x['score']:.3f}" for x in codebook_candidates_meta[:6]
                        )
                        _suf = "" if len(codebook_candidates_meta) <= 6 else f" …(+{len(codebook_candidates_meta) - 6})"
                        _extra = f" | codebook_top=[{_cb}{_suf}]"
                    elif schedule_mode == "fast":
                        _extra = (
                            f" | cut d={dist_prev:.4f} τ={_tau_c:.3f} is_cut={is_cut}"
                            if dist_prev is not None
                            else f" | cut τ={_tau_c:.3f} is_cut={is_cut} (no_prev_tail)"
                        )
                    print(
                        f"  [VAL chunk {chunk_idx}] {schedule_mode} → next={next_chunk_idx} "
                        f"sel={selected_indices}{_extra}"
                    )
                    hooks = bolt_layers.register_hooks(transformer, selected_latents=selected_latents)
                    active_hook_ref[0] = hooks
                elif schedule_mode == "fast":
                    print(
                        f"  [VAL chunk {chunk_idx}] fast → next={next_chunk_idx} "
                        f"(no ref-attn; cache={list(selector_cache.selected_chunk_idx)} 仅调度)"
                    )
                elif chunk_idx > 0:
                    print(f"  [VAL chunk {chunk_idx}] no ref frames (history={len(history_ref)})")

                all_selection_logs.append(log_entry)

        return chunk_callback

    val_csv = val_cfg.get("validation_interactive_csv", None)
    val_prompts = val_cfg.get("validation_prompts", [])
    seed = val_cfg.get("validation_seed", 42)
    generator = torch.Generator(device=device).manual_seed(seed)

    baseline_negative_prompt = (
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, "
        "worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, "
        "deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
    )

    common_kwargs = dict(
        negative_prompt=val_cfg.get("negative_prompt", baseline_negative_prompt),
        height=val_cfg.get("validation_height", 384),
        width=val_cfg.get("validation_width", 640),
        num_frames=val_cfg.get("validation_num_frames", 97),
        num_inference_steps=val_cfg.get("validation_num_inference_steps", 50),
        guidance_scale=val_cfg.get("validation_guidance_scale", 5.0),
        generator=generator,
        history_sizes=[16, 2, 1],
        latent_window_size=val_cfg.get("validation_latent_window_size", 9),
        is_keep_x0=True,
        use_zero_init=val_cfg.get("use_zero_init", True),
        use_cfg_zero_star=val_cfg.get("use_cfg_zero_star", False),
        zero_steps=val_cfg.get("zero_steps", 1),
    )

    videos_to_generate = []

    if val_csv and os.path.exists(val_csv):
        import pandas as pd
        df = pd.read_csv(val_csv)
        df = df.sort_values(by=["id", "prompt_index"])
        max_videos = int(_val_cfg_get("validation_max_videos", 2))
        all_ids = df["id"].unique()[:max_videos]
        for vid_id in all_ids:
            group = df[df["id"] == vid_id]
            if "refined_prompt" in df.columns:
                prompts = group["refined_prompt"].fillna(group["prompt"]).tolist()
            else:
                prompts = group["prompt"].tolist()
            videos_to_generate.append({
                "video_id": str(vid_id),
                "prompts": prompts,
                "is_interactive": len(prompts) > 1,
            })
    elif val_prompts:
        for i, p in enumerate(val_prompts):
            videos_to_generate.append({
                "video_id": str(i),
                "prompts": [p],
                "is_interactive": False,
            })

    if not videos_to_generate:
        print("  [VAL] No validation prompts configured, skipping.")
        bolt_layers.train()
        # Keep training path stable: decode_middle_frame in training expects VAE on CPU.
        vae.to("cpu", dtype=torch.float32)
        # 清理 pipeline 引用 (不 del transformer/vae，它们是外部传入的)
        del pipe, scheduler
        torch.cuda.empty_cache()
        return

    print(f"  [VAL] Generating {len(videos_to_generate)} validation video(s)...")

    for vi, vinfo in enumerate(videos_to_generate):
        vid_id = vinfo["video_id"]
        prompts = vinfo["prompts"]
        is_interactive = vinfo["is_interactive"]

        print(f"\n  ── Video [{vid_id}] ({len(prompts)} prompt(s)) ──")
        for pi, p in enumerate(prompts):
            print(f"    [{pi}] {p[:80]}...")

        history = []
        active_hooks = [None]
        interpolate_time = val_cfg.get("interpolate_time", 3)
        interpolate_time_list = [interpolate_time] * len(prompts)

        callback = make_chunk_callback(
            prompts if is_interactive else prompts[0],
            history,
            active_hooks,
            vid_id,
            interpolate_time_list if is_interactive else None,
        )

        pipe_kwargs = dict(common_kwargs)
        pipe_kwargs["prompt"] = prompts if is_interactive else prompts[0]

        if is_interactive and val_cfg.get("use_interpolate_prompt", True):
            pipe_kwargs["use_interpolate_prompt"] = True
            pipe_kwargs["interpolation_steps"] = val_cfg.get("interpolation_steps", 1)
            pipe_kwargs["interpolate_time_list"] = interpolate_time_list

        if "chunk_callback" in inspect.signature(pipe.__call__).parameters:
            pipe_kwargs["chunk_callback"] = callback

        try:
            with torch.no_grad():
                output = pipe(**pipe_kwargs)
        finally:
            if active_hooks[0] is not None:
                BoltReferenceAttentionLayers.remove_hooks(active_hooks[0])

        frames = output.frames[0]
        fps = val_cfg.get("validation_fps", 24)
        out_path = os.path.join(val_dir, f"val_{vid_id}_epoch{epoch:03d}.mp4")
        export_to_video(frames, out_path, fps=fps)
        print(f"  [VAL] Saved: {out_path} ({len(frames)} frames)")

    # ── 保存选帧日志 ──
    log_path = os.path.join(val_dir, "selection_log.json")
    with open(log_path, "w") as f:
        json.dump(all_selection_logs, f, indent=2, ensure_ascii=False)
    print(f"  [VAL] Selection log: {log_path}")

    # ── 打印选帧汇总 ──
    print(f"\n  ── 选帧汇总 (Epoch {epoch}) ──")
    for log in all_selection_logs:
        vid = log["video_id"]
        cidx = log["chunk_idx"]
        sel = log.get("selected_indices", [])
        n_sel = log.get("num_selected", 0)
        if n_sel > 0:
            detail = log.get("selection_detail") or []
            if detail:
                scores_str = ", ".join(
                    f"chunk_{d['chunk']}(c={d['combined']:.3f})" for d in detail
                )
            else:
                sm = log.get("schedule_mode", "")
                scores_str = f"{sm} {sel}" if sm else str(sel)
            print(f"    video={vid} chunk_{cidx}→chunk_{cidx+1}: "
                  f"选了{n_sel}帧 [{scores_str}]")

    print(f"  [VAL] Epoch {epoch} validation complete.\n")

    # 恢复训练模式
    bolt_layers.train()

    # 清理 validation pipeline 在 transformer/vae 上注册的所有 offload hooks
    _hook_names_to_clean = ["layer_execution_tracker", "lazy_prefetch_group_offloading", "group_offloading"]
    for _component in [transformer, vae]:
        for _name, _mod in _component.named_modules():
            if hasattr(_mod, "_diffusers_hook"):
                for _hname in _hook_names_to_clean:
                    if _mod._diffusers_hook.get_hook(_hname) is not None:
                        _mod._diffusers_hook.remove_hook(_hname, recurse=False)

    del pipe, scheduler
    torch.cuda.empty_cache()

    # 重新注册训练用的 group offload (恢复 DiT 的 CPU↔GPU 按需加载)
    transformer.enable_group_offload(
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=True,
        record_stream=True,
    )
    print("  [VAL] Restored training group offload on transformer.")
    # Validation may move VAE to CUDA via pipeline; move it back for training decode path.
    vae.to("cpu", dtype=torch.float32)
    torch.cuda.empty_cache()


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def load_config_from_yaml(yaml_path):
    """从 yaml 文件加载配置，返回 flat dict。"""
    import yaml
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    flat = {}
    flat["output_dir"] = cfg.get("output_dir", "/root/autodl-fs/output/bolt_ref_attn")
    flat["seed"] = cfg.get("seed", 42)

    data = cfg.get("data_config", {})
    flat["feature_folders"] = data.get("feature_folders", [])

    model = cfg.get("model_config", {})
    flat["transformer_path"] = model.get("transformer_path", "/root/autodl-fs/BestWishYSH/Helios-Base")
    flat["base_model_path"] = model.get("base_model_path", None)
    flat["clip_model_path"] = model.get("clip_model_path", None)

    bolt = cfg.get("bolt_config", {})
    flat["bolt_active_layers"] = bolt.get("active_layers", "19-26")
    flat["bolt_attn_dim"] = bolt.get("attn_dim", 2560)
    flat["bolt_num_heads"] = bolt.get("num_heads", 20)
    flat["bolt_k_select"] = bolt.get("k_select", 4)
    flat["bolt_alpha"] = bolt.get("alpha", 0.6)
    flat["bolt_power"] = bolt.get("power", 2.0)
    flat["bolt_min_chunk_distance"] = bolt.get("min_chunk_distance", 3)

    train = cfg.get("training_config", {})
    flat["num_epochs"] = train.get("num_epochs", 30)
    flat["bolt_lr"] = train.get("bolt_lr", 5e-5)
    # LayerScale γ 专用学习率 (首选方案 A1)：远大于主权重 lr
    flat["lr_gamma"] = train.get("lr_gamma", 1e-2)
    flat["grad_accum_steps"] = train.get("grad_accum_steps", 2)
    flat["max_grad_norm"] = train.get("max_grad_norm", 1.0)
    flat["weighting_scheme"] = train.get("weighting_scheme", "none")
    flat["loss_w_ema_beta"] = float(train.get("loss_w_ema_beta", 0.95))
    flat["optimize_target"] = str(train.get("optimize_target", "mse"))
    flat["id_loss_lambda"] = float(train.get("id_loss_lambda", 0.0))
    flat["id_loss_max_refs"] = int(train.get("id_loss_max_refs", 1))
    flat["id_dino_model_path"] = train.get("id_dino_model_path", None)
    flat["save_every"] = train.get("save_every", 5)
    flat["resume_from"] = train.get("resume_from", None)
    # 约 7 chunk、三幕结构时第三幕常从 chunk>=4；训练随机 target 仅从此下界起抽（不足则退回 2..）
    _cmin = train.get("choice_idx_random_min", 4)
    flat["choice_idx_random_min"] = int(4 if _cmin is None else _cmin)

    # Validation config (保持为 dict，不展平)
    val = cfg.get("validation_config", {})
    if val:
        # validation 的 base_model_path 默认使用 model_config.base_model_path（若有），否则退回 transformer_path
        val.setdefault("base_model_path", flat["base_model_path"] or flat["transformer_path"])
        flat["validation_config"] = val
    else:
        flat["validation_config"] = None

    # 与 infer 对齐的 VLM / LongMemory 参数（可写在 yaml 顶层 selector_inference）
    si = cfg.get("selector_inference", {})
    if isinstance(si, dict):
        for k, v in si.items():
            if v is not None:
                flat[k] = v

    # 训练选帧：selector_type / vlm_* 等（与 argparse 同名，便于一份 yaml 管全）
    st = cfg.get("selector_training", {})
    if isinstance(st, dict):
        for k, v in st.items():
            if v is not None:
                flat[k] = v

    return flat


def main():
    parser = argparse.ArgumentParser(description="Train BOLT Reference Attention")
    parser.add_argument("--config", default=None, help="Path to yaml config file")
    parser.add_argument("--feature_folders", nargs="+", default=None,
                        help="Directories with precomputed .pt files")
    parser.add_argument("--transformer_path", default=None)
    parser.add_argument(
        "--base_model_path",
        default=None,
        help="Base model directory that contains 'vae' (and other components). Defaults to transformer_path.",
    )
    parser.add_argument("--clip_model_path", default=None, help="CLIP model local path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--bolt_lr", type=float, default=None)
    parser.add_argument("--lr_gamma", type=float, default=None,
                        help="LayerScale γ 专用学习率（默认 1e-2，远大于主权重 lr）")
    parser.add_argument("--bolt_active_layers", default=None)
    parser.add_argument("--bolt_k_select", type=int, default=None)
    parser.add_argument("--bolt_alpha", type=float, default=None)
    parser.add_argument("--bolt_power", type=float, default=None)
    parser.add_argument("--bolt_min_chunk_distance", type=int, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=None)
    parser.add_argument("--save_every", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume_from", default=None, help="Path to .pth to resume training")
    parser.add_argument(
        "--choice_idx_random_min",
        type=int,
        default=None,
        help="Random target chunk idx lower bound (inclusive). Default 4 ≈ third act for ~7-chunk videos; "
        "falls back to [2,num_chunks) if too few chunks. Set 2 to restore fully random early targets.",
    )
    # ── Optional: train with VLM selector (instead of CLIP+ITS) ──
    parser.add_argument(
        "--selector_type",
        choices=["clip_its", "vlm"],
        default="vlm",
        help="训练选帧：clip_its（CLIP+ITS）或 vlm（默认 vlm，见 yaml selector_training）。",
    )
    parser.add_argument("--vlm_model_path", default="/root/autodl-fs/Qwen2.5-VL-3B-Instruct", help="VLM backbone path/id (for selector_type=vlm).")
    parser.add_argument("--vlm_lora_path", default=None, help="VLM LoRA path (optional).")
    parser.add_argument(
        "--vlm_score_mode",
        choices=["yes_no", "hidden_head"],
        default="yes_no",
        help="VLM scoring mode used by load_default_backbone.",
    )
    parser.add_argument(
        "--vlm_rank_mode",
        choices=["its", "topk"],
        default="topk",
        help="VLM selection: its sampling or greedy top-k by score.",
    )
    parser.add_argument(
        "--enable_long_memory",
        action="store_true",
        help="Validation/inference-style: use LongMemory codebook when selector_type=vlm (default on for VLM in validate_epoch).",
    )
    parser.add_argument("--dino_model_path", default="/root/autodl-fs/dinov2-base", help="DINOv2 for LongMemory + ref-vs-GT metric.")
    parser.add_argument("--vlm_k_select", type=int, default=1, help="Override k for VLM selector (default: bolt_k_select).")
    parser.add_argument("--vlm_max_candidates", type=int, default=16, help="CLIP pre-filter top-M for VLM validation.")
    parser.add_argument("--vlm_prefilter_alpha", type=float, default=0.6, help="VLM prefilter alpha.")
    parser.add_argument("--vlm_temperature", type=float, default=0.7, help="VLM temperature.")
    parser.add_argument("--vlm_power", type=float, default=None, help="VLM ITS power (default: bolt_power).")
    parser.add_argument("--vlm_min_chunk_distance", type=int, default=None, help="VLM min chunk distance (default: bolt_min_chunk_distance).")
    # ── 验证/交互推理：何时强制「slow」重算 VLM；以及多帧输入（与 validate_epoch / infer 对齐）──
    parser.add_argument(
        "--vlm_slow_step_chunks",
        default=None,
        help='显式指定哪些 next_chunk_idx 必须走 slow（逗号分隔 chunk 索引），如 "0,7,14"。'
        "不设则从 interactive 的 interpolate_time_list 推断（每段起点等）。",
    )
    parser.add_argument(
        "--vlm_extra_slow_at_segment_mid",
        action="store_true",
        help="在自动推断的 slow 集合上，再给每段 prompt 的中点多加一个 slow（更密重算选帧）。",
    )
    parser.add_argument(
        "--vlm_fallback_to_clip",
        action="store_true",
        help="验证时 VLM slow 抛错则退回 CLIP+ITS 选帧（默认关闭，便于暴露错误）。",
    )
    parser.add_argument(
        "--vlm_use_chunk_video",
        action="store_true",
        help="VLM 打分用整 chunk 多帧 decode（decoded_frames）而非仅 middle；需 backbone 支持 score_video。",
    )
    parser.add_argument(
        "--vlm_video_max_frames",
        type=int,
        default=None,
        help="与 --vlm_use_chunk_video 配合：每个 chunk 最多解码多少帧（控显存/时间）。",
    )
    # ── 验证 history 条数上限与驱逐（memory_bank.evict_history）──
    parser.add_argument(
        "--mb_max_history_chunks",
        type=int,
        default=32,
        help="验证时 history_ref 最多保留多少个已生成 chunk（超出则驱逐旧项）。",
    )
    parser.add_argument(
        "--mb_keep_recent_k",
        type=int,
        default=8,
        help="驱逐时尽量保留的最近 chunk 数（与 mb_evict_strategy 一起用）。",
    )
    parser.add_argument(
        "--mb_evict_strategy",
        choices=["farthest_lowclip", "oldest"],
        default="farthest_lowclip",
        help="history 满时驱逐策略：farthest_lowclip=远且 CLIP 与上下文不相似优先；oldest=最旧优先。",
    )
    parser.add_argument(
        "--mb_evict_alpha",
        type=float,
        default=0.5,
        help="farthest_lowclip 中「时间距离」与「1-CLIP 相似度」的混合权重 α（越大越看重距离）。",
    )
    parser.add_argument(
        "--no_vlm_cache_invalidate_on_evict",
        action="store_true",
        help="驱逐 history 后不要因「选中 chunk 被踢」而清空 VLMSelectorCache（默认会清）。",
    )
    # ── LongMemory 码本（DINO tail + merge/cut；主要影响验证 longmem 分支）──
    parser.add_argument(
        "--lm_tau_merge",
        type=float,
        default=0.85,
        help="码本更新：与最近槽余弦相似度≥此值则 EMA 合并该槽，否则新开槽（越高越难合并、槽更细）。",
    )
    parser.add_argument(
        "--lm_tau_cut",
        type=float,
        default=0.35,
        help="切换门控：相邻 chunk 的 DINO tail 距离 d=1-cos≥此值视为 cut，触发从码本取候选再走 VLM。",
    )
    parser.add_argument(
        "--lm_ema_alpha_new",
        type=float,
        default=0.2,
        help="码本槽 EMA 更新时新观测权重（key ← α*new + (1-α)*old）。",
    )
    parser.add_argument(
        "--lm_codebook_max_size",
        type=int,
        default=512,
        help="LongMemory 码本最大槽位数，满则按 lm_codebook_evict 驱逐。",
    )
    parser.add_argument(
        "--lm_codebook_topm",
        type=int,
        default=16,
        help="cut 触发后从码本按相似度取 Top-M 个候选 chunk 交给 VLM 精排。",
    )
    parser.add_argument(
        "--lm_codebook_evict",
        choices=["lru", "lfu", "oldest"],
        default="lru",
        help="码本满时驱逐策略：lru / lfu / oldest。",
    )
    parser.add_argument(
        "--lm_debug",
        action="store_true",
        help="LongMemory 内部调试输出（若实现中有开关则生效）。",
    )
    parser.add_argument(
        "--bolt_log_every_chunk",
        action="store_true",
        help="验证/训练循环中更频繁打印与 chunk 相关的选帧日志（若下游支持）。",
    )
    parser.add_argument(
        "--weight_dtype",
        default="bfloat16",
        choices=["bfloat16", "fp16", "fp32"],
        help="部分模块权重/计算精度（如 DINO 侧 fp16/bf16）；与 DiT dtype 可独立。",
    )
    parser.add_argument(
        "--weighting_scheme",
        default=None,
        help="Flow loss weighting scheme (aligned with train_helios.py).",
    )
    parser.add_argument(
        "--optimize_target",
        choices=["mse", "weighted", "loss_w"],
        default=None,
        help="Primary objective for backprop: mse or weighted/loss_w.",
    )
    parser.add_argument(
        "--id_loss_lambda",
        type=float,
        default=None,
        help="Lambda for ID consistency loss. Set >0 to enable L_id.",
    )
    parser.add_argument(
        "--id_loss_max_refs",
        type=int,
        default=None,
        help="Max number of selected reference chunks used to build ID target frame.",
    )
    parser.add_argument(
        "--id_dino_model_path",
        default=None,
        help="DINOv2 model path for ID loss; defaults to --dino_model_path when unset.",
    )
    parser.add_argument(
        "--loss_w_ema_beta",
        type=float,
        default=None,
        help="EMA smoothing beta for loss_w logging (0~1, closer to 1 = smoother).",
    )
    cli_args = parser.parse_args()

    # ── 合并配置: yaml 为底，CLI 覆盖 ──
    defaults = {
        "feature_folders": [], "transformer_path": "/root/autodl-fs/BestWishYSH/Helios-Base",
        "base_model_path": None,
        "clip_model_path": None, "output_dir": "/root/autodl-fs/output/bolt_ref_attn",
        "num_epochs": 30, "bolt_lr": 5e-5, "lr_gamma": 1e-2, "bolt_active_layers": "19-26",
        "bolt_attn_dim": 2560, "bolt_num_heads": 20,
        "bolt_k_select": 4, "bolt_alpha": 0.6, "bolt_power": 2.0, "bolt_min_chunk_distance": 3,
        "grad_accum_steps": 2, "max_grad_norm": 1.0,         "save_every": 5,
        "seed": 42, "resume_from": None,
        "choice_idx_random_min": 4,
        "selector_type": "vlm",
        "vlm_model_path": None,
        "vlm_lora_path": None,
        "vlm_score_mode": "yes_no",
        "vlm_rank_mode": "topk",
        "enable_long_memory": False,
        "dino_model_path": None,
        "vlm_k_select": None,
        "vlm_max_candidates": 16,
        "vlm_prefilter_alpha": 0.6,
        "vlm_temperature": 0.7,
        "vlm_power": None,
        "vlm_min_chunk_distance": None,
        "vlm_slow_step_chunks": None,
        "vlm_extra_slow_at_segment_mid": False,
        "vlm_fallback_to_clip": False,
        "vlm_use_chunk_video": False,
        "vlm_video_max_frames": None,
        "mb_max_history_chunks": 32,
        "mb_keep_recent_k": 8,
        "mb_evict_strategy": "farthest_lowclip",
        "mb_evict_alpha": 0.5,
        "no_vlm_cache_invalidate_on_evict": False,
        "lm_tau_merge": 0.85,
        "lm_tau_cut": 0.35,
        "lm_ema_alpha_new": 0.2,
        "lm_codebook_max_size": 512,
        "lm_codebook_topm": 16,
        "lm_codebook_evict": "lru",
        "lm_debug": False,
        "bolt_log_every_chunk": False,
        "weight_dtype": "bfloat16",
        "weighting_scheme": "none",
        "optimize_target": "mse",
        "id_loss_lambda": 0.0,
        "id_loss_max_refs": 1,
        "id_dino_model_path": None,
        "loss_w_ema_beta": 0.95,
    }

    if cli_args.config:
        yaml_cfg = load_config_from_yaml(cli_args.config)
        defaults.update({k: v for k, v in yaml_cfg.items() if v is not None})

    # CLI 参数覆盖 yaml（仅覆盖用户显式传入的项，避免 argparse 默认值反向覆盖 yaml）
    raw_argv = sys.argv[1:]
    provided_cli_flags = set()
    for tok in raw_argv:
        if not tok.startswith("--"):
            continue
        flag = tok[2:].split("=", 1)[0].strip()
        if flag:
            provided_cli_flags.add(flag)

    for key in defaults:
        flag_name = key.replace("_", "-")
        if flag_name not in provided_cli_flags:
            continue
        cli_val = getattr(cli_args, key, None)
        if cli_val is not None:
            defaults[key] = cli_val

    args = argparse.Namespace(**defaults)

    if not args.feature_folders:
        parser.error("必须通过 --config 或 --feature_folders 指定训练数据目录")

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    base_model_path = args.base_model_path or args.transformer_path

    # ── Parse active layers ──
    start_l, end_l = map(int, args.bolt_active_layers.split("-"))
    active_layers = list(range(start_l, end_l + 1))

    # ── Load frozen DiT ──
    print("[INFO] Loading frozen Helios DiT...")
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer" if os.path.isdir(os.path.join(args.transformer_path, "transformer")) else None,
        torch_dtype=dtype,
        transformer_additional_kwargs={
            "has_multi_term_memory_patch": True,
            "zero_history_timestep": True,
            "guidance_cross_attn": True,
        },
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()
    transformer.eval()
    transformer.requires_grad_(False)
    # Group offload: DiT 权重按需从 CPU 加载到 GPU
    transformer.enable_group_offload(
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=True,
        record_stream=True,
    )
    print("[INFO] DiT loaded with group offloading (leaf_level)")

    # ── Load VAE (for CLIP feature extraction) ──
    print("[INFO] Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(
        base_model_path, subfolder="vae", torch_dtype=torch.float32,
    )
    vae.eval()
    vae.requires_grad_(False)
    vae.to("cpu")  # 只在需要时移到 GPU

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(vae.device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        vae.device, vae.dtype
    )

    # ── Load CLIP ──
    print("[INFO] Loading CLIP...")
    clip_kwargs = {"device": device}
    if args.clip_model_path:
        clip_kwargs["model_path"] = args.clip_model_path
    clip_model = CLIP(**clip_kwargs)

    # ── Offload CLIP to CPU (方案 B: 释放 ~900MB GPU 显存) ──
    print("[INFO] Offloading CLIP to CPU (float32) to save GPU memory...")
    clip_model.model.to(dtype=torch.float32, device="cpu")
    clip_model.device = "cpu"
    torch.cuda.empty_cache()

    # ── Optional: DINO encoder for ID consistency loss ──
    id_dino_encoder = None
    if float(getattr(args, "id_loss_lambda", 0.0) or 0.0) > 0.0:
        id_dino_path = args.id_dino_model_path or args.dino_model_path
        if not id_dino_path:
            raise ValueError("id_loss_lambda > 0 requires --id_dino_model_path or --dino_model_path")
        print(f"[INFO] Loading DINOv2 for id loss from: {id_dino_path}")
        id_dino_encoder = DINOv2(device=device, model_id_or_path=id_dino_path, dtype=args.weight_dtype)
        id_dino_encoder.model.requires_grad_(False)
        id_dino_encoder.model.eval()

    # ── Optional: VLM selector backbone ──
    vlm_selector = None
    if args.selector_type == "vlm":
        if not args.vlm_model_path:
            raise ValueError("--selector_type vlm requires --vlm_model_path")
        print("[INFO] Loading VLM backbone for selector...")
        try:
            from helios.modules.vlm_backbones import load_default_backbone
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                "Failed to import helios.modules.vlm_backbones.load_default_backbone; "
                "please ensure VLM backbone module exists."
            ) from exc
        vlm_backbone = load_default_backbone(
            model_path=args.vlm_model_path,
            lora_path=args.vlm_lora_path,
            dtype="bf16" if dtype == torch.bfloat16 else "fp16",
            device=device,
            score_mode=args.vlm_score_mode,
            head_path=None,
        )
        _vk = args.vlm_k_select if args.vlm_k_select is not None else args.bolt_k_select
        _vpower = args.vlm_power if args.vlm_power is not None else args.bolt_power
        _vmin_dist = args.vlm_min_chunk_distance if args.vlm_min_chunk_distance is not None else args.bolt_min_chunk_distance
        vlm_selector = VLMFrameSelector(
            clip_model=None,  # training uses VLM pixels; no CLIP prefilter needed
            k=int(_vk),
            power=float(_vpower),
            min_chunk_distance=int(_vmin_dist),
            max_candidates=int(getattr(args, "vlm_max_candidates", 16) or 16),
            prefilter_alpha=float(args.vlm_prefilter_alpha),
            temperature=float(args.vlm_temperature),
            vlm_backbone=vlm_backbone,
            device=device,
            fallback_to_random=False,
            vlm_rank_mode=args.vlm_rank_mode,
        )
        print(
            f"[INFO] selector_type=vlm | k={_vk} (vlm_k_select or bolt_k_select), "
            f"rank={args.vlm_rank_mode}, score={args.vlm_score_mode}, temp={args.vlm_temperature}"
        )

    # ── Create Bolt Ref-Attn Layers (trainable) ──
    bolt_attn_dim = getattr(args, "bolt_attn_dim", 2560)
    bolt_num_heads = getattr(args, "bolt_num_heads", 20)
    bolt_layers = BoltReferenceAttentionLayers(
        dit_dim=5120,
        latent_patch_dim=512,
        num_heads=bolt_num_heads,
        attn_dim=bolt_attn_dim,
        active_layers=active_layers,
    ).to(device, dtype=dtype)

    if args.resume_from and os.path.exists(args.resume_from):
        bolt_layers.load_state_dict(torch.load(args.resume_from, map_location=device))
        print(f"[INFO] Resumed from {args.resume_from}")

    # P0: 把 _*_init buffer 重新对齐到"本次训练起点"。
    # 新 run 时 __init__ 已自动 snap 到 xavier 初值；resume 时这里覆盖为 ckpt 权重，
    # 这样 ‖W - W_init‖_F 度量的是"本次会话内"的位移，而不是"距最初随机 xavier"。
    bolt_layers.snapshot_initial_weights()
    print("[INFO] Snapshotted initial weights for ‖W - W_init‖_F monitoring.")

    # ── Optimizer ──
    # LayerScale γ 单独 param group：lr_gamma（默认 1e-2），weight_decay=0
    # 其余（Wq/Wk/Wv/out_proj）维持 bolt_lr（默认 5e-5），weight_decay=0.01
    # 详见 md/bolt_integration_plan_alpha.md §四（A1 + B2 首选方案）
    gamma_params = list(bolt_layers.gamma_parameters())
    other_params = list(bolt_layers.non_gamma_parameters())
    trainable = sum(p.numel() for p in gamma_params + other_params if p.requires_grad)
    print(
        f"[INFO] Bolt Ref-Attn trainable params: {trainable:,} ({trainable/1e6:.1f}M)"
        f" | γ: {sum(p.numel() for p in gamma_params):,}, "
        f"W(q/k/v/out): {sum(p.numel() for p in other_params):,}"
    )
    lr_gamma = getattr(args, "lr_gamma", 1e-2)
    print(
        f"[INFO] Optimizer param groups: "
        f"gamma lr={lr_gamma:g} (wd=0), "
        f"others lr={args.bolt_lr:g} (wd=0.01)"
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": gamma_params, "lr": lr_gamma, "weight_decay": 0.0},
            {"params": other_params, "lr": args.bolt_lr, "weight_decay": 0.01},
        ]
    )

    # ── Dataset ──
    dataset = BoltTrainDataset(args.feature_folders)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    # ── Save config ──
    config = vars(args)
    config["active_layers"] = active_layers
    config["trainable_params"] = trainable
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Validation config ──
    val_cfg = getattr(args, "validation_config", None)
    if val_cfg:
        val_epochs = val_cfg.get("validation_epochs", 5)
        val_fm_seed = int(val_cfg.get("flow_matching_val_seed", args.seed))
        val_fm_max_batches = int(val_cfg.get("flow_matching_val_max_batches", 0) or 0)
        print(f"  Validation: every {val_epochs} epochs (interactive inference)")
        print(
            f"  Validation(FM avg): enabled, seed={val_fm_seed}, "
            f"max_batches={'all' if val_fm_max_batches <= 0 else val_fm_max_batches}"
        )
        val_csv = val_cfg.get("validation_interactive_csv", None)
        if val_csv:
            print(f"  Validation CSV: {val_csv}")
    else:
        val_epochs = 0
        val_fm_seed = int(args.seed)
        val_fm_max_batches = 0

    # ── Train ──
    print(f"\n{'='*60}")
    print(f"  BOLT Reference Attention Training")
    print(f"  Active layers: {active_layers[0]}-{active_layers[-1]}")
    print(f"  Epochs: {args.num_epochs}, LR(W): {args.bolt_lr}, LR(γ): {lr_gamma}")
    print(f"  weighting_scheme: {args.weighting_scheme}")
    print(f"  optimize_target: {args.optimize_target}")
    print(f"  loss_w_ema_beta: {args.loss_w_ema_beta}")
    print(
        f"  id_loss_lambda: {args.id_loss_lambda} "
        f"(max_refs={args.id_loss_max_refs}, dino={args.id_dino_model_path or args.dino_model_path})"
    )
    print(f"  selector_type: {args.selector_type}")
    if args.selector_type == "vlm":
        _vk = args.vlm_k_select if args.vlm_k_select is not None else args.bolt_k_select
        print(
            f"  VLM (train): model={args.vlm_model_path}, rank={args.vlm_rank_mode}, "
            f"score_mode={args.vlm_score_mode}, k={_vk}, temp={args.vlm_temperature}, "
            f"max_cand={args.vlm_max_candidates}"
        )
        if args.vlm_lora_path:
            print(f"  VLM LoRA: {args.vlm_lora_path}")
    else:
        print(
            f"  CLIP+ITS (train): K_select={args.bolt_k_select}, Alpha={args.bolt_alpha}, "
            f"Power={args.bolt_power}, MinDist={args.bolt_min_chunk_distance}"
        )
    if args.selector_type == "vlm":
        print(
            f"  BOLT (shared / ref-attn & VLM fallback knobs): bolt_k_select={args.bolt_k_select}, "
            f"alpha={args.bolt_alpha}, power={args.bolt_power}, min_chunk_distance={args.bolt_min_chunk_distance}"
        )
    print(f"  choice_idx_random_min: {args.choice_idx_random_min} (random target chunk idx ≥ this when possible)")
    if val_epochs > 0:
        print(f"  Validation: every {val_epochs} epochs")
    print(f"{'='*60}\n")

    # ── Epoch -1: 训练前先 validate 一次 (baseline) ──
    if val_cfg and val_epochs > 0:
        print("[INFO] Running pre-training validation (epoch=-1, baseline)...")
        try:
            fm_stats = evaluate_flow_matching_avg_mse(
                transformer=transformer,
                bolt_layers=bolt_layers,
                clip_model=clip_model,
                vlm_selector=vlm_selector,
                vae=vae,
                dataloader=val_dataloader,
                device=device,
                dtype=dtype,
                bolt_k_select=args.bolt_k_select,
                bolt_alpha=args.bolt_alpha,
                bolt_power=args.bolt_power,
                latents_mean=latents_mean,
                latents_std=latents_std,
                choice_idx_random_min=args.choice_idx_random_min,
                seed=val_fm_seed + 100000,
                max_batches=val_fm_max_batches,
                weighting_scheme=args.weighting_scheme,
            )
            print(
                f"[VAL][FM] epoch=-1 val_avg_mse={fm_stats['avg_mse']:.6f}, "
                f"val_avg_loss_w={fm_stats['avg_weighted_loss']:.6f} "
                f"(samples={fm_stats['num_samples']}, skipped={fm_stats['num_skipped']})"
            )
            validate_epoch(
                transformer=transformer,
                bolt_layers=bolt_layers,
                clip_model=clip_model,
                vae=vae,
                val_cfg=val_cfg,
                epoch=-1,
                output_dir=args.output_dir,
                bolt_k_select=args.bolt_k_select,
                bolt_alpha=args.bolt_alpha,
                bolt_power=args.bolt_power,
                bolt_min_chunk_distance=args.bolt_min_chunk_distance,
                device=device,
                dtype=dtype,
                args=args,
            )
        except Exception as e:
            print(f"  [VAL ERROR] Pre-training validation failed: {e}")
            import traceback
            traceback.print_exc()
            bolt_layers.train()
            torch.cuda.empty_cache()

    for epoch in range(args.num_epochs):
        t0 = time.time()
        train_stats = train_one_epoch(
            transformer, bolt_layers, clip_model, vlm_selector, vae, dataloader, optimizer, epoch,
            device=device, dtype=dtype,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            bolt_k_select=args.bolt_k_select,
            bolt_alpha=args.bolt_alpha,
            bolt_power=args.bolt_power,
            latents_mean=latents_mean,
            latents_std=latents_std,
            choice_idx_random_min=args.choice_idx_random_min,
            weighting_scheme=args.weighting_scheme,
            loss_w_ema_beta=args.loss_w_ema_beta,
            optimize_target=args.optimize_target,
            id_loss_lambda=args.id_loss_lambda,
            id_dino_encoder=id_dino_encoder,
            id_loss_max_refs=args.id_loss_max_refs,
        )
        dt = time.time() - t0
        print(
            f"[Epoch {epoch}] avg_mse={train_stats['avg_mse']:.6f}, "
            f"avg_loss_w={train_stats['avg_weighted_loss']:.6f}, "
            f"avg_id_loss={train_stats['avg_id_loss']:.6f}, "
            f"avg_opt_loss={train_stats['avg_opt_loss']:.6f}, "
            f"loss_w_ema_last={train_stats['loss_w_ema_last']:.6f}, time={dt:.1f}s"
        )

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"bolt_ref_attn_epoch{epoch:03d}.pth")
            torch.save(bolt_layers.state_dict(), ckpt_path)
            print(f"  [SAVE] {ckpt_path}")

        # ── Validation ──
        if val_cfg and val_epochs > 0 and (epoch + 1) % val_epochs == 0:
            try:
                fm_stats = evaluate_flow_matching_avg_mse(
                    transformer=transformer,
                    bolt_layers=bolt_layers,
                    clip_model=clip_model,
                    vlm_selector=vlm_selector,
                    vae=vae,
                    dataloader=val_dataloader,
                    device=device,
                    dtype=dtype,
                    bolt_k_select=args.bolt_k_select,
                    bolt_alpha=args.bolt_alpha,
                    bolt_power=args.bolt_power,
                    latents_mean=latents_mean,
                    latents_std=latents_std,
                    choice_idx_random_min=args.choice_idx_random_min,
                    seed=val_fm_seed + epoch,
                    max_batches=val_fm_max_batches,
                    weighting_scheme=args.weighting_scheme,
                )
                print(
                    f"[VAL][FM] epoch={epoch} val_avg_mse={fm_stats['avg_mse']:.6f}, "
                    f"val_avg_loss_w={fm_stats['avg_weighted_loss']:.6f} "
                    f"(samples={fm_stats['num_samples']}, skipped={fm_stats['num_skipped']})"
                )
                validate_epoch(
                    transformer=transformer,
                    bolt_layers=bolt_layers,
                    clip_model=clip_model,
                    vae=vae,
                    val_cfg=val_cfg,
                    epoch=epoch,
                    output_dir=args.output_dir,
                    bolt_k_select=args.bolt_k_select,
                    bolt_alpha=args.bolt_alpha,
                    bolt_power=args.bolt_power,
                    bolt_min_chunk_distance=args.bolt_min_chunk_distance,
                    device=device,
                    dtype=dtype,
                    args=args,
                )
            except Exception as e:
                print(f"  [VAL ERROR] Epoch {epoch} validation failed: {e}")
                import traceback
                traceback.print_exc()
                # 恢复训练模式
                bolt_layers.train()
                torch.cuda.empty_cache()

    # Final save
    final_path = os.path.join(args.output_dir, "bolt_ref_attn_final.pth")
    torch.save(bolt_layers.state_dict(), final_path)
    print(f"\n[INFO] Training complete. Final: {final_path}")


if __name__ == "__main__":
    main()
