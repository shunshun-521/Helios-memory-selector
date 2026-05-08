#!/usr/bin/env python
"""
build_vlm_selector_dataset.py — 把 latents_short/*.pt 离线转为 VLM Selector 训练集
===================================================================================

与 md/CLIP-base-selector-vlm.md §3.1 对齐。

产出
-----
<output_dir>/
  frames_mid/{uttid}/chunk_XX.png    — 每个 chunk 的中间帧（PNG，~30-80 KB）
  train.jsonl                        — 每条样本:
      {"uttid": str,
       "choice_idx": int,
       "gap_chunk_indices": [int, ...],   # 做完 min_chunk_distance 过滤后的候选位置
       "context_chunk_idx": int,
       "prompt_raw": str,
       "soft_scores": [float, ...],        # 对 gap_chunk_indices 的 CLIP combined score
       "hard_positions": [int, ...]}       # ITS 实际选中的 position（索引到 gap_chunk_indices）

用法
-----
    python tools/build_vlm_selector_dataset.py \
        --input_pt /root/autodl-tmp/output_4_13/nice_baseline_only_one/latents_short/12_0-297_281_384_640.pt \
        --output_dir /root/autodl-fs/selector_vlm_data \
        --vae_path /root/autodl-fs/BestWishYSH/Helios-Base

说明
-----
* 支持单个 .pt（--input_pt），也支持整目录（--input_dir）扫描 *.pt；
* 复用 helios.modules.select_frames.select_gap_frames 计算 combined_scores + ITS；
* 默认 min_chunk_distance / alpha / power 与 bolt_ref_attn.yaml 保持一致，
  这样 VLM 学到的分布正好可以蒸馏回 CLIP+ITS 教师。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

HELIOS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HELIOS_ROOT))

from helios.modules.extract_feature import CLIP, decode_middle_frame  # noqa: E402
from helios.modules.select_frames import (  # noqa: E402
    inverse_transform_sampling,
    select_gap_frames,
)


# ─── utils ───────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Build VLM Selector dataset from latents_short .pt")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input_pt", help="Single .pt path")
    src.add_argument("--input_dir", help="Directory to scan for *.pt")

    p.add_argument("--output_dir", required=True)
    p.add_argument("--vae_path", default="/root/autodl-fs/BestWishYSH/Helios-Base",
                   help="HF path with subfolder 'vae' (AutoencoderKLWan)")
    p.add_argument("--clip_path", default=None, help="CLIP ViT-L/14 path; None=use default")
    p.add_argument("--device", default="cuda")

    # selector teacher 超参（对齐推理 / bolt_ref_attn.yaml）
    p.add_argument("--k_select", type=int, default=4)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--power", type=float, default=1.5)
    p.add_argument("--min_chunk_distance", type=int, default=3)

    p.add_argument("--overwrite_png", action="store_true", help="Re-render middle frames even if PNG exists")
    p.add_argument("--limit", type=int, default=None, help="Only process first N .pt files")
    return p.parse_args()


def _list_pts(args) -> list[Path]:
    if args.input_pt:
        return [Path(args.input_pt)]
    pts = sorted(Path(args.input_dir).rglob("*.pt"))
    return pts[: args.limit] if args.limit else pts


def _uttid_of(pt_path: Path) -> str:
    return pt_path.stem


# ─── main ────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.output_dir)
    frames_root = out_dir / "frames_mid"
    jsonl_path = out_dir / "train.jsonl"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_root.mkdir(parents=True, exist_ok=True)

    # VAE
    from diffusers.models import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(
        args.vae_path, subfolder="vae", torch_dtype=torch.float32
    ).to(device).eval()

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(device, dtype=torch.float32)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, dtype=torch.float32
    )

    # CLIP
    clip_model = CLIP(device=device, model_path=args.clip_path)

    pts = _list_pts(args)
    print(f"[build_vlm_selector_dataset] processing {len(pts)} .pt -> {out_dir}")
    if not pts:
        print("[build_vlm_selector_dataset] no .pt found, abort.")
        return

    n_samples = 0
    n_skipped = 0
    with open(jsonl_path, "w", encoding="utf-8") as jfw:
        for pt_path in tqdm(pts, desc="pt files"):
            uttid = _uttid_of(pt_path)
            d = torch.load(pt_path, map_location="cpu", weights_only=False)
            if "vae_latent" not in d:
                print(f"  [skip] {uttid}: no vae_latent key")
                n_skipped += 1
                continue
            vae_latents = d["vae_latent"]  # (num_chunks, 16, 9, H, W)
            if vae_latents.ndim != 5:
                print(f"  [skip] {uttid}: vae_latent ndim != 5 ({tuple(vae_latents.shape)})")
                n_skipped += 1
                continue
            num_chunks = int(vae_latents.shape[0])

            # Prefer chunk-aligned segmented prompts if present (Selector_VLM format).
            segments = d.get("segments", None)
            prompts_per_chunk = None
            if isinstance(segments, list) and len(segments) > 0:
                prompts_per_chunk = _prompts_per_chunk_from_segments(segments, num_chunks)

            if prompts_per_chunk is None:
                prompt_raw = d.get("prompt_raw", "")
                if isinstance(prompt_raw, list):
                    # 若多 prompt，按段均分到 chunks（与 Helios interpolate 约定一致）
                    prompts_per_chunk = _broadcast_prompt_to_chunks(prompt_raw, num_chunks)
                else:
                    prompts_per_chunk = [str(prompt_raw)] * num_chunks

            # 预计算每个 chunk 的 mid PNG + CLIP feat
            chunk_entries = []
            uttid_dir = frames_root / uttid
            uttid_dir.mkdir(parents=True, exist_ok=True)
            for ci in range(num_chunks):
                png_path = uttid_dir / f"chunk_{ci:02d}.png"
                if args.overwrite_png or not png_path.exists():
                    with torch.no_grad():
                        pil = decode_middle_frame(
                            vae_latents[ci : ci + 1].to(device=device, dtype=torch.float32),
                            vae, latents_mean, latents_std,
                        )
                    pil.save(png_path)
                else:
                    from PIL import Image
                    pil = Image.open(png_path).convert("RGB")

                clip_feat = clip_model.extract_visual_features([pil]).squeeze(0).cpu()
                chunk_entries.append({
                    "chunk_idx": ci,
                    "latent": vae_latents[ci : ci + 1].clone(),  # (1,C,T,H,W)
                    "clip_feat": clip_feat,
                    "decoded_frame": None,  # 训练时按 uttid/chunk_XX.png 读
                })

            # 对每个 choice_idx 生成一条样本
            for choice_idx in range(2, num_chunks):
                history_slice = chunk_entries[:choice_idx]  # 含 context = chunk_{choice_idx-1}
                # 手动复刻 select_gap_frames 的候选筛选（否则拿不到 combined_scores）
                gap_raw = [
                    e for e in history_slice[: choice_idx - 1]
                    if (choice_idx - int(e["chunk_idx"])) >= int(args.min_chunk_distance)
                ]
                if len(gap_raw) == 0:
                    continue

                context_entry = history_slice[choice_idx - 1]
                prompt = prompts_per_chunk[min(choice_idx, num_chunks - 1)]

                gap_feats = torch.stack([e["clip_feat"] for e in gap_raw], dim=0).to(device)
                vq = context_entry["clip_feat"].unsqueeze(0).to(device)
                v_score = clip_model.compute_similarity(gap_feats, vq).numpy()
                tq = clip_model.extract_text_features(prompt).to(device)
                t_score = clip_model.compute_similarity(gap_feats, tq).numpy()
                combined = args.alpha * v_score + (1.0 - args.alpha) * t_score

                actual_k = min(args.k_select, len(gap_raw))
                hard_positions = inverse_transform_sampling(
                    combined, n=actual_k, power=args.power
                )
                hard_positions = list(dict.fromkeys([int(x) for x in hard_positions.tolist()]))

                # sanity check: compare with select_gap_frames API
                _, ref_indices = select_gap_frames(
                    history=history_slice, current_chunk_idx=choice_idx,
                    current_prompt=prompt, clip_model=clip_model,
                    k=args.k_select, alpha=args.alpha, power=args.power,
                    min_chunk_distance=args.min_chunk_distance, device=device,
                )

                record = {
                    "uttid": uttid,
                    "choice_idx": int(choice_idx),
                    "gap_chunk_indices": [int(e["chunk_idx"]) for e in gap_raw],
                    "context_chunk_idx": int(context_entry["chunk_idx"]),
                    "prompt_raw": prompt,
                    "soft_scores": [float(x) for x in combined.tolist()],
                    "hard_positions": hard_positions,
                    "ref_hard_indices": [int(x) for x in ref_indices],
                    "teacher_config": {
                        "alpha": float(args.alpha),
                        "power": float(args.power),
                        "k_select": int(args.k_select),
                        "min_chunk_distance": int(args.min_chunk_distance),
                    },
                }
                jfw.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_samples += 1

    print(f"[build_vlm_selector_dataset] done. samples={n_samples}, skipped={n_skipped}")
    print(f"  frames_mid: {frames_root}")
    print(f"  train.jsonl: {jsonl_path}")


def _broadcast_prompt_to_chunks(prompt_list, num_chunks):
    n = len(prompt_list)
    if n == 0:
        return [""] * num_chunks
    if n == 1:
        return [prompt_list[0]] * num_chunks
    per_seg = max(1, num_chunks // n)
    out = []
    for ci in range(num_chunks):
        seg_i = min(n - 1, ci // per_seg)
        out.append(prompt_list[seg_i])
    return out


def _prompts_per_chunk_from_segments(segments, num_chunks: int):
    """
    Build per-chunk prompt list from segments:
      segments: list[{start_chunk,end_chunk,prompt_raw|prompt}]
    """
    out = ["" for _ in range(num_chunks)]
    ok = False
    for seg in segments:
        if not isinstance(seg, dict):
            continue
        try:
            sc = int(seg.get("start_chunk", 0) or 0)
            ec = int(seg.get("end_chunk", 0) or 0)
        except Exception:
            continue
        if ec <= sc:
            continue
        prompt = seg.get("prompt_raw", None)
        if not isinstance(prompt, str) or len(prompt) == 0:
            prompt = seg.get("prompt", "")
        prompt = str(prompt or "")
        sc = max(0, min(num_chunks, sc))
        ec = max(0, min(num_chunks, ec))
        for i in range(sc, ec):
            out[i] = prompt
        ok = True
    return out if ok else None


if __name__ == "__main__":
    main()
