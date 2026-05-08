#!/usr/bin/env python3
"""
基于训练样本做 BOLT alpha/power 离线 sweep（不生成视频）。

核心目标:
1) 使用固定训练样本（.pt）中的 latent + prompt_raw；
2) 先一次性提取每个 chunk 的 CLIP 特征；
3) 在同一组固定 visual/text 分数上扫描不同 alpha/power。

这样可严格保证：除 alpha/power 外，其它因素完全一致。

示例:
bash -c 
'python /root/autodl-tmp/Helios/tools/run_bolt_alpha_power_sweep.py \
  --feature_folders /root/autodl-tmp/output_4_13/nice_baseline_only_one \
  --transformer_path /root/autodl-fs/BestWishYSH/Helios-Base \
  --output_root output_bolt_sweep_train_offline \
  --combos "0.2:1.0,0.2:3.0,0.5:1.0,0.5:3.0,0.8:1.0,0.8:3.0,0.6:2.0" \
  --bolt_k_select 4 \
  --bolt_min_chunk_distance 3 \
  --max_samples 10  2>&1 | tee output_bolt_sweep_train_offline/run_$(date +%F_%H-%M-%S).log'

保存完整运行日志（stdout + stderr）:
bash -c 'python tools/run_bolt_alpha_power_sweep.py ... 2>&1 | tee output_bolt_sweep_train_offline/run_$(date +%F_%H-%M-%S).log'
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from tqdm import tqdm

HELIOS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys
sys.path.insert(0, HELIOS_ROOT)

from helios.modules.extract_feature import CLIP, extract_chunk_feature
from helios.modules.select_frames import inverse_transform_sampling


DEFAULT_COMBOS = [
    (0.2, 1.0),
    (0.2, 3.0),
    (0.3, 1.0),
    (0.3, 3.0),
    (0.5, 1.0),
    (0.5, 3.0),
    (0.8, 1.0),
    (0.8, 3.0),
    (0.6, 2.0),
]

DEFAULT_FEATURE_FOLDER = "/root/autodl-tmp/output_4_13/nice_baseline_only_one"


def parse_combo_string(combo_str: str):
    combo_str = combo_str.strip()
    if not combo_str:
        return DEFAULT_COMBOS
    pairs = []
    for raw_item in combo_str.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Invalid combo '{item}', expected alpha:power")
        alpha_s, power_s = item.split(":", 1)
        alpha = float(alpha_s.strip())
        power = float(power_s.strip())
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0,1], got {alpha}")
        if power <= 0:
            raise ValueError(f"power must be > 0, got {power}")
        pairs.append((alpha, power))
    return pairs or DEFAULT_COMBOS


def combo_tag(alpha: float, power: float):
    a = f"{alpha:.2f}".replace(".", "p")
    p = f"{power:.2f}".replace(".", "p")
    return f"alpha_{a}__power_{p}"


def collect_pt_files(feature_folders):
    files = []
    for folder in feature_folders:
        if not os.path.isdir(folder):
            raise FileNotFoundError(f"feature folder not found: {folder}")
        for fname in sorted(os.listdir(folder)):
            if fname.endswith(".pt"):
                files.append(os.path.join(folder, fname))
    if not files:
        raise RuntimeError("No .pt files found in feature_folders")
    return files


def to_float_list(arr):
    return [float(x) for x in np.asarray(arr).tolist()]


def build_base_chunk_records(chunk_feats, prompt_raw, clip_model, min_chunk_distance):
    """
    对每个 current_chunk_idx 预计算固定分数（与 alpha/power 无关）。
    current_chunk_idx 含义与 select_gap_frames 保持一致: 正在为该 chunk 选参考帧。
    """
    num_chunks = len(chunk_feats)
    has_text = isinstance(prompt_raw, str) and len(prompt_raw.strip()) > 0
    text_query = clip_model.extract_text_features(prompt_raw) if has_text else None

    base_records = []
    for current_chunk_idx in range(2, num_chunks):
        context_idx = current_chunk_idx - 1
        candidate_indices = [
            idx for idx in range(0, context_idx)
            if (current_chunk_idx - idx) >= int(min_chunk_distance)
        ]
        if not candidate_indices:
            base_records.append(
                {
                    "current_chunk_idx": int(current_chunk_idx),
                    "context_chunk_idx": int(context_idx),
                    "candidate_indices": [],
                    "visual_scores": [],
                    "text_scores": [],
                    "has_text": bool(has_text),
                }
            )
            continue

        gap_feats = torch.stack([chunk_feats[idx] for idx in candidate_indices], dim=0).to(clip_model.device)
        context_feat = chunk_feats[context_idx].unsqueeze(0).to(clip_model.device)
        visual_scores = clip_model.compute_similarity(gap_feats, context_feat).numpy()

        if has_text:
            text_scores = clip_model.compute_similarity(gap_feats, text_query.to(clip_model.device)).numpy()
        else:
            text_scores = np.zeros_like(visual_scores)

        base_records.append(
            {
                "current_chunk_idx": int(current_chunk_idx),
                "context_chunk_idx": int(context_idx),
                "candidate_indices": [int(x) for x in candidate_indices],
                "visual_scores": to_float_list(visual_scores),
                "text_scores": to_float_list(text_scores),
                "has_text": bool(has_text),
            }
        )
    return base_records


def apply_combo_to_base_record(base_record, alpha, power, k_select):
    candidate_indices = base_record["candidate_indices"]
    visual_scores = np.asarray(base_record["visual_scores"], dtype=np.float32)
    text_scores = np.asarray(base_record["text_scores"], dtype=np.float32)

    if len(candidate_indices) == 0:
        return {
            "current_chunk_idx": base_record["current_chunk_idx"],
            "context_chunk_idx": base_record["context_chunk_idx"],
            "num_candidates": 0,
            "selected_indices": [],
            "selected_details": [],
        }

    combined_scores = alpha * visual_scores + (1.0 - alpha) * text_scores
    actual_k = min(int(k_select), len(candidate_indices))
    sampled_positions = inverse_transform_sampling(combined_scores, n=actual_k, power=power)
    sampled_positions = list(dict.fromkeys(sampled_positions.tolist()))

    selected_indices = [int(candidate_indices[pos]) for pos in sampled_positions]
    selected_details = []
    for pos in sampled_positions:
        selected_details.append(
            {
                "chunk_idx": int(candidate_indices[pos]),
                "visual": float(visual_scores[pos]),
                "text": float(text_scores[pos]),
                "combined": float(combined_scores[pos]),
                "distance_to_current": int(base_record["current_chunk_idx"] - candidate_indices[pos]),
            }
        )

    return {
        "current_chunk_idx": base_record["current_chunk_idx"],
        "context_chunk_idx": base_record["context_chunk_idx"],
        "num_candidates": int(len(candidate_indices)),
        "candidate_indices": candidate_indices,
        "combined_scores": to_float_list(combined_scores),
        "selected_indices": selected_indices,
        "selected_details": selected_details,
    }


def main():
    parser = argparse.ArgumentParser(description="Offline alpha/power sweep on training features")
    parser.add_argument(
        "--feature_folders",
        nargs="+",
        default=[DEFAULT_FEATURE_FOLDER],
        help="Folders containing training .pt features",
    )
    parser.add_argument("--transformer_path", required=True,default="/root/autodl-fs/BestWishYSH/Helios-Base", help="Model path containing VAE subfolder")
    parser.add_argument("--clip_model_path", default="/root/autodl-fs/clip-vit-large-patch14/AI-ModelScope/clip-vit-large-patch14", help="Optional local CLIP model path")
    parser.add_argument("--output_root", default="output_bolt_sweep_train_offline")
    parser.add_argument("--combos", default="", help="Comma-separated alpha:power list")
    parser.add_argument("--bolt_k_select", type=int, default=4)
    parser.add_argument("--bolt_min_chunk_distance", type=int, default=3)
    parser.add_argument("--max_samples", type=int, default=10, help="Limit number of .pt samples to process")
    parser.add_argument("--start_sample_idx", type=int, default=0, help="Start offset in sorted .pt file list")
    parser.add_argument("--device", default=None, help="cuda/cpu, default auto")
    args = parser.parse_args()

    if args.bolt_k_select <= 0:
        raise ValueError("--bolt_k_select must be > 0")
    if args.bolt_min_chunk_distance <= 0:
        raise ValueError("--bolt_min_chunk_distance must be > 0")

    combos = parse_combo_string(args.combos)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_root, exist_ok=True)

    print("=" * 80)
    print("BOLT alpha/power offline sweep (training features)")
    print(f"device                : {device}")
    print(f"feature_folders       : {args.feature_folders}")
    print(f"transformer_path      : {args.transformer_path}")
    print(f"output_root           : {args.output_root}")
    print(f"combos                : {combos}")
    print(f"bolt_k_select         : {args.bolt_k_select}")
    print(f"bolt_min_chunk_dist   : {args.bolt_min_chunk_distance}")
    print("=" * 80)

    pt_files = collect_pt_files(args.feature_folders)
    s = max(0, int(args.start_sample_idx))
    e = min(len(pt_files), s + max(1, int(args.max_samples)))
    selected_files = pt_files[s:e]
    print(f"[INFO] Using samples: {len(selected_files)} / {len(pt_files)} (slice: {s}:{e})")

    print("[INFO] Loading VAE...")
    vae = AutoencoderKLWan.from_pretrained(args.transformer_path, subfolder="vae", torch_dtype=torch.float32)
    vae.eval()
    vae.requires_grad_(False)
    # 在部分环境下，wan VAE 的 3D conv decode 在 CUDA 后端不可用，固定走 CPU 更稳。
    vae.to("cpu")

    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(vae.device, vae.dtype)

    print("[INFO] Loading CLIP...")
    clip_kwargs = {"device": device}
    if args.clip_model_path:
        clip_kwargs["model_path"] = args.clip_model_path
    clip_model = CLIP(**clip_kwargs)

    # 先缓存每个样本每个 chunk 的固定分数（不依赖 alpha/power）
    per_sample_base = []
    for sample_path in tqdm(selected_files, desc="Precompute fixed scores"):
        data = torch.load(sample_path, map_location="cpu", weights_only=False)
        vae_latent = data["vae_latent"]  # (num_chunks, C, T, H, W)
        prompt_raw = data.get("prompt_raw", None)
        if vae_latent.ndim != 5:
            print(f"[WARN] Skip invalid latent shape: {sample_path}")
            continue
        num_chunks = int(vae_latent.shape[0])
        if num_chunks < 3:
            print(f"[WARN] Skip short sample (<3 chunks): {sample_path}")
            continue

        chunk_feats = []
        with torch.no_grad():
            for chunk_idx in range(num_chunks):
                chunk_latent = vae_latent[chunk_idx:chunk_idx + 1]
                feat, _ = extract_chunk_feature(chunk_latent, vae, clip_model, latents_mean, latents_std)
                chunk_feats.append(feat)

        base_records = build_base_chunk_records(
            chunk_feats=chunk_feats,
            prompt_raw=prompt_raw,
            clip_model=clip_model,
            min_chunk_distance=args.bolt_min_chunk_distance,
        )
        per_sample_base.append(
            {
                "sample_path": sample_path,
                "prompt_raw": prompt_raw if isinstance(prompt_raw, str) else "",
                "num_chunks": num_chunks,
                "base_records": base_records,
            }
        )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "mode": "training_offline_no_video",
        "output_root": os.path.abspath(args.output_root),
        "settings_fixed": {
            "feature_folders": args.feature_folders,
            "transformer_path": args.transformer_path,
            "clip_model_path": args.clip_model_path,
            "bolt_k_select": int(args.bolt_k_select),
            "bolt_min_chunk_distance": int(args.bolt_min_chunk_distance),
            "num_samples_processed": len(per_sample_base),
        },
        "combos": [{"alpha": float(a), "power": float(p)} for a, p in combos],
        "combo_outputs": [],
    }

    # 在完全相同的 base_records 上套每个 alpha/power
    for alpha, power in combos:
        tag = combo_tag(alpha, power)
        run_dir = os.path.join(args.output_root, tag)
        os.makedirs(run_dir, exist_ok=True)
        out_json = os.path.join(run_dir, "selection_records.json")

        combo_records = []
        print(f"\n[COMBO] {tag}")
        for sample in per_sample_base:
            sample_out = {
                "sample_path": sample["sample_path"],
                "prompt_raw": sample["prompt_raw"],
                "num_chunks": int(sample["num_chunks"]),
                "chunks": [],
            }
            for base_record in sample["base_records"]:
                chunk_result = apply_combo_to_base_record(
                    base_record=base_record,
                    alpha=float(alpha),
                    power=float(power),
                    k_select=int(args.bolt_k_select),
                )
                sample_out["chunks"].append(chunk_result)

                cidx = chunk_result["current_chunk_idx"]
                print(
                    f"  sample={os.path.basename(sample['sample_path'])} "
                    f"chunk={cidx} selected={chunk_result['selected_indices']}"
                )
            combo_records.append(sample_out)

        payload = {
            "alpha": float(alpha),
            "power": float(power),
            "k_select": int(args.bolt_k_select),
            "min_chunk_distance": int(args.bolt_min_chunk_distance),
            "records": combo_records,
        }
        with open(out_json, "w", encoding="utf-8") as fw:
            json.dump(payload, fw, ensure_ascii=False, indent=2)

        summary["combo_outputs"].append({"tag": tag, "selection_json": os.path.abspath(out_json)})
        print(f"  [OK] saved: {out_json}")

    summary_path = os.path.join(args.output_root, "sweep_summary.json")
    with open(summary_path, "w", encoding="utf-8") as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)
    print("\n" + "=" * 80)
    print(f"Done. Summary: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
