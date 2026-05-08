#!/usr/bin/env python
"""
compare_selector_clipits_vs_vlm_zeroshot.py
==========================================

目的
----
在不跑 DiT（避免 OOM）的前提下，对同一条样本、同一 choice_idx（目标 chunk）
对比两种选帧器的行为：

1) CLIP+ITS（visual + text combined score + inverse_transform_sampling）
2) VLM zero-shot（yes/no logits），再按 ``--vlm_rank_mode`` 选 chunk：
   - ``its``：inverse_transform_sampling（与 CLIP 侧一致）
   - ``topk``：分数最高的 K 个候选（贪心，不经 ITS）

输入
----
特征目录：包含 get_short-latents.py 生成的 *.pt（需含 vae_latent）。
可选支持 Selector_VLM 的分段 prompt：
  - segments: list[{start_chunk,end_chunk,prompt_raw|prompt}]

输出
----
对每个 sample 的每次 trial 打印：
  - uttid, num_chunks, choice_idx, seg_idx/bounds, prompt_snippet
  - GAP candidates (chunk indices)
  - CLIP scores（visual/text/combined）+ ITS 选中
  - VLM scores + （ ``--vlm_rank_mode`` 为 its 则用 ITS；为 topk 则取分数最高的 K 个）
  - overlap/intersection

示例命令（VLM 多帧 chunk；VLM 侧用贪心 top-k，不经 ITS）
-----------------------------
python tools/compare_selector_clipits_vs_vlm_zeroshot.py \
  --feature_folder /root/autodl-tmp/Selector_VLM/example_long/latents_short \
  --vae_path /root/autodl-fs/BestWishYSH/Helios-Base \
  --vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
  --device cuda \
  --vlm_device cuda \
  --choice_mode all \
  --k_select 1 \
  --alpha 0.6 \
  --power 2.0 \
  --min_chunk_distance 2 \
  --vlm_use_chunk_video \
  --vlm_video_max_frames 9 \
  --vlm_rank_mode topk \
  --print_topn 8 
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch

# Ensure `import helios` works regardless of current working directory.
HELIOS_ROOT = Path(__file__).resolve().parents[1]
if str(HELIOS_ROOT) not in sys.path:
    sys.path.insert(0, str(HELIOS_ROOT))


def _list_pt_files(feature_folder: str) -> List[Path]:
    p = Path(feature_folder)
    pts = sorted([x for x in p.rglob("*.pt") if x.is_file()])
    return pts


def _pick_prompt_for_choice_idx(
    choice_idx: int,
    prompt_raw_fallback: str,
    segments: Optional[list],
) -> Tuple[str, Optional[int], Optional[Tuple[int, int]]]:
    if isinstance(segments, list) and len(segments) > 0:
        seg_idx = None
        for i_s, seg in enumerate(segments):
            if not isinstance(seg, dict):
                continue
            try:
                sc = int(seg.get("start_chunk", 0) or 0)
                ec = int(seg.get("end_chunk", 0) or 0)
            except Exception:
                continue
            if sc <= choice_idx < ec:
                seg_idx = i_s
                bounds = (sc, ec)
                prompt = seg.get("prompt_raw", None)
                if not isinstance(prompt, str) or len(prompt) == 0:
                    prompt = seg.get("prompt", "")
                return str(prompt or ""), seg_idx, bounds
        # fallback to last seg
        last = segments[-1] if isinstance(segments[-1], dict) else {}
        prompt = last.get("prompt_raw", None)
        if not isinstance(prompt, str) or len(prompt) == 0:
            prompt = last.get("prompt", "")
        bounds = (
            int(last.get("start_chunk", 0) or 0),
            int(last.get("end_chunk", 0) or 0),
        )
        return str(prompt or ""), len(segments) - 1, bounds
    return str(prompt_raw_fallback or ""), None, None


def parse_args():
    p = argparse.ArgumentParser(description="Compare CLIP+ITS vs VLM zero-shot selection on latents_short .pt")
    p.add_argument("--feature_folder", required=True, help="Folder containing *.pt (vae_latent + prompt_raw/segments)")
    p.add_argument("--vae_path", default="/root/autodl-fs/BestWishYSH/Helios-Base", help="Path containing subfolder vae/")
    p.add_argument("--clip_path", default="/root/autodl-fs/clip-vit-large-patch14/AI-ModelScope/clip-vit-large-patch14", help="Optional CLIP model path; None uses default")
    p.add_argument("--vlm_model_path", default="/root/autodl-fs/Qwen2.5-VL-3B-Instruct", help="e.g. /root/autodl-fs/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--device", default="cpu", help="cuda/cpu for VAE decode + CLIP feature")
    p.add_argument("--vlm_device", default="cpu", help="cuda/cpu for VLM scoring (zero-shot)")
    p.add_argument("--vlm_offload_to_cpu", action="store_true", help="After each scoring, offload VLM to CPU")
    p.add_argument("--vlm_image_resize", type=int, nargs=2, default=[256, 448])
    p.add_argument(
        "--vlm_use_chunk_video",
        action="store_true",
        help="Use full chunk frames (multi-image) for VLM scoring instead of middle-frame only.",
    )
    p.add_argument(
        "--vlm_video_max_frames",
        type=int,
        default=None,
        help="Optional cap for decoded frames per chunk when vlm_use_chunk_video is enabled.",
    )
    p.add_argument(
        "--vlm_rank_mode",
        choices=["its", "topk"],
        default="its",
        help="After VLM scores: its=inverse_transform_sampling; topk=greedy top-k by score.",
    )
    # Teacher params (align with CLIP+ITS)
    p.add_argument("--k_select", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.3)
    p.add_argument("--power", type=float, default=1.0)
    p.add_argument("--min_chunk_distance", type=int, default=2)
    # Trials
    p.add_argument(
        "--choice_mode",
        choices=["random", "all"],
        default="all",
        help="random: sample choice_idx; all: iterate choice_idx from 0..last chunk",
    )
    p.add_argument("--trials_per_video", type=int, default=9, help="Only used when choice_mode=random")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None, help="Only process first N .pt files")
    p.add_argument("--print_topn", type=int, default=8, help="Print top-N ranked candidates for each method")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"

    # Lazy imports to keep script startup light.
    from diffusers.models import AutoencoderKLWan
    from helios.modules.extract_feature import CLIP, decode_all_frames, decode_middle_frame
    from helios.modules.select_frames import inverse_transform_sampling
    from helios.modules.vlm_backbones import QwenVLZeroShotBackbone

    # VAE
    vae = AutoencoderKLWan.from_pretrained(args.vae_path, subfolder="vae", torch_dtype=torch.float32).to(device).eval()
    latents_mean = (
        torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, dtype=torch.float32)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        device, dtype=torch.float32
    )

    # CLIP
    clip_model = CLIP(device=device, model_path=args.clip_path)

    # VLM zero-shot backbone
    vlm = QwenVLZeroShotBackbone(
        model_path=args.vlm_model_path,
        lora_path=None,
        dtype="bfloat16",
        device=args.vlm_device,
        image_resize=tuple(args.vlm_image_resize),
        offload_to_cpu=bool(args.vlm_offload_to_cpu),
    )

    pts = _list_pt_files(args.feature_folder)
    if args.limit:
        pts = pts[: args.limit]
    print(f"[compare] pts={len(pts)} folder={args.feature_folder}")

    for pt_path in pts:
        d = torch.load(pt_path, map_location="cpu", weights_only=False)
        if "vae_latent" not in d:
            continue
        vae_latents = d["vae_latent"]  # (num_chunks, C, T, H, W)
        num_chunks = int(vae_latents.shape[0])
        if num_chunks < 4:
            continue

        segments = d.get("segments", None)
        prompt_raw_fallback = d.get("prompt_raw", "")
        if isinstance(prompt_raw_fallback, list):
            prompt_raw_fallback = str(prompt_raw_fallback[0]) if prompt_raw_fallback else ""

        uttid = pt_path.name
        print("\n" + "=" * 90)
        print(f"[video] {uttid}  num_chunks={num_chunks}")

        # Precompute per-chunk:
        # - decoded_middle_frames: middle frame for CLIP+ITS / legacy VLM scoring
        # - decoded_chunk_videos: full chunk frames for video-level VLM scoring (optional)
        decoded_middle_frames = []
        decoded_chunk_videos = []
        clip_feats = []
        for ci in range(num_chunks):
            chunk_latent = vae_latents[ci : ci + 1].to(device=device, dtype=torch.float32)
            pil = decode_middle_frame(
                chunk_latent,
                vae,
                latents_mean,
                latents_std,
            )
            decoded_middle_frames.append(pil)
            vf = clip_model.extract_visual_features([pil]).squeeze(0)
            clip_feats.append(vf.detach().cpu())
            if args.vlm_use_chunk_video:
                chunk_frames = decode_all_frames(
                    chunk_latent,
                    vae,
                    latents_mean,
                    latents_std,
                    max_frames=args.vlm_video_max_frames,
                )
                decoded_chunk_videos.append(chunk_frames)
            else:
                decoded_chunk_videos.append(None)

        if args.vlm_use_chunk_video and len(decoded_chunk_videos) > 0:
            first_len = len(decoded_chunk_videos[0]) if decoded_chunk_videos[0] is not None else 0
            print(
                f"[video-mode] enabled  max_frames={args.vlm_video_max_frames}  "
                f"frames_per_chunk(example)={first_len}"
            )

        if args.choice_mode == "all":
            choice_indices = list(range(0, num_chunks))
        else:
            choice_indices = [random.randint(2, num_chunks - 1) for _ in range(args.trials_per_video)]

        if len(choice_indices) > 0:
            print(
                f"[choice] mode={args.choice_mode}  n={len(choice_indices)}  "
                f"first={choice_indices[0]}  last={choice_indices[-1]}"
            )

        for t, choice_idx in enumerate(choice_indices):
            try:
                prompt, seg_idx, bounds = _pick_prompt_for_choice_idx(choice_idx, str(prompt_raw_fallback), segments)
                snippet = (prompt[:140] + "...") if isinstance(prompt, str) and len(prompt) > 140 else prompt
                print("-" * 90)
                if seg_idx is None:
                    print(f"[trial {t}] choice_idx={choice_idx}  seg=None  prompt='{snippet}'")
                else:
                    print(
                        f"[trial {t}] choice_idx={choice_idx}  seg_idx={seg_idx}  bounds={bounds}  prompt='{snippet}'"
                    )

                # GAP candidates: < choice_idx-1 and time_distance >= min_chunk_distance
                gap_abs = []
                for gi in range(0, choice_idx - 1):
                    if (choice_idx - gi) >= int(args.min_chunk_distance):
                        gap_abs.append(gi)
                if len(gap_abs) == 0:
                    print("  [skip] no GAP candidates (need choice_idx>=2 and enough history)")
                    continue

                context_idx = choice_idx - 1
                context_feat = clip_feats[context_idx].unsqueeze(0).to(device)
                gap_feats = torch.stack([clip_feats[i] for i in gap_abs], dim=0).to(device)

                # CLIP visual/text score
                v_scores = clip_model.compute_similarity(gap_feats, context_feat).numpy()
                t_query = clip_model.extract_text_features(prompt).to(device)
                t_scores = clip_model.compute_similarity(gap_feats, t_query).numpy()
                combined = args.alpha * v_scores + (1.0 - args.alpha) * t_scores
                k = min(int(args.k_select), len(gap_abs))
                clip_pos = inverse_transform_sampling(combined, n=k, power=float(args.power)).tolist()
                clip_pos = list(dict.fromkeys(int(x) for x in clip_pos))
                clip_sel_abs = [gap_abs[p] for p in clip_pos]

                # VLM scores (zero-shot yes/no)
                if args.vlm_use_chunk_video:
                    if not hasattr(vlm, "score_video"):
                        raise RuntimeError("Current VLM backbone does not implement score_video().")
                    candidate_videos = [decoded_chunk_videos[i] for i in gap_abs]
                    context_video = decoded_chunk_videos[context_idx]
                    vlm_scores = vlm.score_video(
                        context_frames=context_video,
                        candidate_videos=candidate_videos,
                        prompt=prompt,
                    )
                else:
                    candidate_imgs = [decoded_middle_frames[i] for i in gap_abs]
                    context_img = decoded_middle_frames[context_idx]
                    vlm_scores = vlm.score(context_image=context_img, candidate_images=candidate_imgs, prompt=prompt)
                if args.vlm_rank_mode == "topk":
                    order = np.argsort(-vlm_scores)[:k]
                    vlm_pos = [int(i) for i in order]
                else:
                    vlm_pos = inverse_transform_sampling(vlm_scores, n=k, power=float(args.power)).tolist()
                    vlm_pos = list(dict.fromkeys(int(x) for x in vlm_pos))
                vlm_sel_abs = [gap_abs[p] for p in vlm_pos]

                # Print summary
                inter = sorted(set(clip_sel_abs).intersection(set(vlm_sel_abs)))
                print(f"  GAP candidates(abs): {gap_abs}")
                print(f"  CLIP sel(abs):       {clip_sel_abs}")
                print(f"  VLM  sel(abs):       {vlm_sel_abs}")
                print(f"  overlap(abs):        {inter}")

                # Print per-candidate scores (small lists only; keep readable)
                def _round_list(x):
                    return [float(np.round(v, 3)) for v in x.tolist()]

                print(f"  scores visual:   {_round_list(v_scores)}")
                print(f"  scores text:     {_round_list(t_scores)}")
                print(f"  scores combined: {_round_list(combined)}")
                print(f"  scores vlm:      {[float(np.round(v, 3)) for v in vlm_scores.tolist()]}")

                # Ranked candidates for clearer comparison
                topn = max(1, int(args.print_topn))
                clip_rank = sorted(
                    [(gap_abs[i], float(combined[i])) for i in range(len(gap_abs))],
                    key=lambda x: -x[1],
                )[:topn]
                vlm_rank = sorted(
                    [(gap_abs[i], float(vlm_scores[i])) for i in range(len(gap_abs))],
                    key=lambda x: -x[1],
                )[:topn]
                print(f"  top{topn} CLIP(combined): {[(i, round(s,3)) for i,s in clip_rank]}")
                print(f"  top{topn} VLM (yes-no):   {[(i, round(s,3)) for i,s in vlm_rank]}")
            except Exception as exc:  # noqa: BLE001
                print(f"  [error] trial={t} choice_idx={choice_idx} failed: {type(exc).__name__}: {exc}")
                continue


if __name__ == "__main__":
    main()

