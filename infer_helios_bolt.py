"""
infer_helios_bolt.py — 推理: BOLT ITS 选帧 + Bolt Ref-Attn 注入
================================================================

在 Helios interactive 推理模式下，引入 BOLT 的 ITS 选帧思想。
用 CLIP 计算 visual score + text score 的加权融合作为 ITS 的输入分布，
从历史 GAP chunk 中选出最有参考价值的帧，以 Bolt Ref-Attn 形式注入 DiT。

【与 inference_with_gap_injection.py 的区别】
- gap_injection: VLM Selector 选帧 → 替换 history_long (零额外参数)
- 本方案: CLIP + ITS 选帧 → Bolt Ref-Attn 注入 (需要训练好的 Ref-Attn 权重)

用法:
  python infer_helios_bolt.py \
    --bolt_ckpt /path/to/bolt_ref_attn.pth \
    --interactive_prompt_csv_path example/prompt_interactive_helios.csv
"""

import importlib
import os

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import json
import sys
import time

import numpy as np
import torch
from tqdm import tqdm

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
else:
    torch_npu = None

HELIOS_ROOT = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, HELIOS_ROOT)

from helios.diffusers_version.pipeline_helios_diffusers import HeliosPipeline
from helios.diffusers_version.scheduling_helios_diffusers import HeliosScheduler
from helios.diffusers_version.transformer_helios_diffusers import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_rmsnorm_with_fp32,
    replace_all_norms_with_flash_norms,
    replace_rope_with_flash_rope,
)
from helios.utils.utils_base import load_extra_components
from helios.modules.ref_attn_bolt import BoltReferenceAttentionLayers

from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video


def parse_args():
    p = argparse.ArgumentParser(description="BOLT ITS + Ref-Attn Inference")
    # Model paths
    p.add_argument("--base_model_path", default="/root/autodl-fs/BestWishYSH/Helios-Base")
    p.add_argument("--transformer_path", default="/root/autodl-fs/BestWishYSH/Helios-Base")
    p.add_argument("--lora_path", default=None)
    p.add_argument("--partial_path", default=None)
    # Bolt Ref-Attn
    p.add_argument("--bolt_ckpt", default=None, help="Path to trained BoltRefAttnLayers .pth")
    p.add_argument("--bolt_active_layers", default="36-39", help="DiT layer range for Ref-Attn")
    p.add_argument("--bolt_attn_dim", type=int, default=1280, help="Ref-Attn bottleneck dim (must match training)")
    p.add_argument("--bolt_num_heads", type=int, default=10, help="Ref-Attn num heads (must match training)")
    p.add_argument(
        "--enable_bolt_injection",
        action="store_true",
        help="Enable CLIP+ITS selection and Bolt Ref-Attn hook injection. Default: off (baseline parity).",
    )
    p.add_argument(
        "--selector_type",
        choices=["clip_its", "vlm", "vlm_zeroshot", "random"],
        default="clip_its",
        help="Frame selector route: keep CLIP+ITS, or add VLM/Random selector. "
             "vlm_zeroshot uses VLM yes/no scoring without training.",
    )
    # BOLT ITS params
    p.add_argument("--bolt_k_select", type=int, default=4, help="ITS 选帧数量")
    p.add_argument("--bolt_alpha", type=float, default=0.6, help="visual/text score 融合权重")
    p.add_argument("--bolt_power", type=float, default=2.0, help="ITS 锐度系数")
    p.add_argument("--bolt_min_chunk_distance", type=int, default=3, help="ITS 候选的最小 chunk 距离阈值")
    # VLM selector params (新增，不替代 CLIP+ITS)
    p.add_argument("--vlm_k_select", type=int, default=4, help="VLM selector 最终选帧数")
    p.add_argument("--vlm_power", type=float, default=2.0, help="VLM selector ITS 锐度")
    p.add_argument("--vlm_min_chunk_distance", type=int, default=3, help="VLM selector 最小 chunk 距离")
    p.add_argument("--vlm_model_path", default=None, help="预留：VLM backbone 路径（后续接入真实 VLM）")
    p.add_argument("--vlm_lora_path", default=None, help="预留：VLM LoRA 路径（后续接入真实 VLM）")
    p.add_argument(
        "--vlm_score_mode",
        default=None,
        help="VLM 打分模式: yes_no | hidden_head。留空时尝试从 vlm_lora_path/score_mode.json 自动读取。",
    )
    p.add_argument("--vlm_max_candidates", type=int, default=16, help="VLM 前的 CLIP pre-filter top-M")
    p.add_argument("--vlm_prefilter_alpha", type=float, default=0.6, help="VLM pre-filter visual/text 融合权重")
    p.add_argument("--vlm_temperature", type=float, default=0.7, help="VLM selector 温度参数")
    p.add_argument(
        "--vlm_use_chunk_video",
        action="store_true",
        help="Decode full chunk frames and feed as multi-image video input for VLM scoring.",
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
        help="After VLM scores on candidates: its=inverse_transform_sampling; topk=greedy top-k by score.",
    )
    p.add_argument("--vlm_slow_step_chunks", default=None, help='显式 slow step 列表，如 "0,7,14"')
    p.add_argument("--vlm_extra_slow_at_segment_mid", action="store_true", help="在每段中点额外触发 slow")
    p.add_argument("--vlm_fallback_to_clip", action="store_true", help="VLM 路径异常时回退 CLIP+ITS")
    # Memory bank params (VLM 路径)
    p.add_argument("--mb_max_history_chunks", type=int, default=32, help="history_ref 长度上限")
    p.add_argument("--mb_keep_recent_k", type=int, default=8, help="最近 K 个 chunk 不驱逐")
    p.add_argument(
        "--mb_evict_strategy",
        choices=["farthest_lowclip", "oldest"],
        default="farthest_lowclip",
        help="history_ref 驱逐策略",
    )
    p.add_argument("--mb_evict_alpha", type=float, default=0.5, help="farthest_lowclip 策略权重")
    p.add_argument(
        "--no_vlm_cache_invalidate_on_evict",
        action="store_true",
        help="关闭: cache 命中的 chunk 被驱逐时自动重算 slow",
    )
    p.add_argument(
        "--bolt_log_every_chunk",
        action="store_true",
        help="Print selection details for every chunk instead of sparse logging.",
    )
    p.add_argument(
        "--bolt_selection_dump_path",
        default=None,
        help="Optional JSON path to dump per-chunk selected indices during generation.",
    )
    # LongMemory (codebook + cut-gated retrieval)
    p.add_argument("--enable_long_memory", action="store_true", help="Enable cut-gated codebook retrieval (LongMemory).")
    p.add_argument("--dino_model_path", default=None, help="DINOv2 model id/path for LongMemory embeddings.")
    p.add_argument("--lm_tau_merge", type=float, default=0.85, help="LongMemory: merge threshold for codebook update.")
    p.add_argument("--lm_tau_cut", type=float, default=0.35, help="LongMemory: cut threshold (1-cos) for retrieval gate.")
    p.add_argument("--lm_ema_alpha_new", type=float, default=0.2, help="LongMemory: EMA new weight for refresh.")
    p.add_argument("--lm_codebook_max_size", type=int, default=512, help="LongMemory: codebook size cap.")
    p.add_argument("--lm_codebook_topm", type=int, default=16, help="LongMemory: retrieve top-M candidates on cut.")
    p.add_argument(
        "--lm_codebook_evict",
        choices=["lru", "lfu", "oldest"],
        default="lru",
        help="LongMemory: eviction strategy when codebook is full.",
    )
    p.add_argument(
        "--lm_debug",
        action="store_true",
        help="Verbose LongMemory debug prints (update/retrieve/selected chunks).",
    )
    # Generation
    p.add_argument("--prompt", default="A person slowly turns around")
    p.add_argument(
        "--negative_prompt",
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )
    p.add_argument("--prompt_txt_path", default=None)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--num_frames", type=int, default=99)
    p.add_argument("--num_inference_steps", type=int, default=50)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_folder", default="output_bolt")
    p.add_argument("--weight_dtype", default="bfloat16")
    # Pipeline options
    p.add_argument("--enable_low_vram_mode", action="store_true")
    p.add_argument("--group_offloading_type", default="leaf_level")
    p.add_argument("--is_enable_stage2", action="store_true")
    p.add_argument("--pyramid_num_inference_steps_list", nargs="+", type=int, default=[20, 20, 20])
    p.add_argument("--is_skip_first_chunk", action="store_true")
    p.add_argument("--is_amplify_first_chunk", action="store_true")
    p.add_argument("--use_zero_init", action="store_true")
    p.add_argument("--zero_steps", type=int, default=1)
    # Interactive
    p.add_argument("--interactive_prompt_csv_path", default=None)
    p.add_argument("--use_interpolate_prompt", action="store_true")
    p.add_argument("--interpolation_steps", type=int, default=3)
    p.add_argument("--interpolate_time", type=int, default=7)
    return p.parse_args()


def main():
    args = parse_args()
    dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(args.weight_dtype, torch.bfloat16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_folder, exist_ok=True)
    print(f"[INFO] enable_bolt_injection={bool(args.enable_bolt_injection)}")
    print(f"[INFO] selector_type={args.selector_type}")
    if args.enable_bolt_injection and args.selector_type in {"vlm", "vlm_zeroshot"}:
        print(f"[INFO] vlm_rank_mode={args.vlm_rank_mode}")

    # ── 1. Load Pipeline ──
    print("[1/4] Loading Helios Pipeline...")
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
        torch_dtype=dtype,
    )
    transformer = replace_rmsnorm_with_fp32(transformer)
    transformer = replace_all_norms_with_flash_norms(transformer)
    replace_rope_with_flash_rope()

    try:
        transformer.set_attention_backend("_flash_3_hub")
    except Exception:
        try:
            transformer.set_attention_backend("flash_hub")
        except Exception:
            pass

    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path, transformer=transformer, vae=vae, scheduler=scheduler,
        torch_dtype=dtype,
    )

    if args.lora_path:
        pipe.load_lora_weights(args.lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[1.0])
        if args.partial_path:
            from argparse import Namespace
            infer_args = Namespace(training_config=Namespace(
                is_enable_stage1=True, restrict_self_attn=False,
                is_amplify_history=False, is_use_gan=False, use_selector=False,
            ))
            load_extra_components(infer_args, transformer, args.partial_path)

    if args.enable_low_vram_mode:
        pipe.enable_group_offload(
            onload_device=torch.device("cuda"),
            offload_device=torch.device("cpu"),
            offload_type=args.group_offloading_type,
            use_stream=True, record_stream=True,
        )
    else:
        pipe = pipe.to(device)

    # ── Optional: BOLT injection (CLIP+ITS + hook injection) ──
    if args.enable_bolt_injection:
        from helios.modules.extract_feature import (
            CLIP,
            DINOv2,
            decode_all_frames,
            decode_middle_frame,
            extract_chunk_feature,
            extract_tail_embedding,
        )
        from helios.modules.select_frames import select_gap_frames
        from helios.modules.select_frames_vlm import (
            RandomSelector,
            VLMFrameSelector,
            compute_slow_step_chunks,
            parse_slow_step_chunks,
        )
        from helios.modules.memory_bank import VLMSelectorCache, evict_history
        from helios.modules.long_memory import (
            LongMemoryStore,
            build_codebook_candidates_for_vlm,
            cosine_similarity,
        )

        print("[2/4] Loading CLIP model (for ITS selection)...")
        clip_model = CLIP(device=device)
        dino_encoder = None
        long_memory = None
        prev_tail_emb = None
        if args.enable_long_memory and args.selector_type in {"vlm", "vlm_zeroshot"}:
            print("[2/4] Loading DINOv2 encoder (for LongMemory)...")
            dino_encoder = DINOv2(
                device=device,
                model_id_or_path=args.dino_model_path,
                dtype=args.weight_dtype,
            )
            long_memory = LongMemoryStore(
                tau_merge=args.lm_tau_merge,
                tau_cut=args.lm_tau_cut,
                ema_alpha_new=args.lm_ema_alpha_new,
                max_size=args.lm_codebook_max_size,
                evict_strategy=args.lm_codebook_evict,
            )

        # VAE normalization 参数 (for feature extraction)
        latents_mean = (
            torch.tensor(vae.config.latents_mean)
            .view(1, vae.config.z_dim, 1, 1, 1)
            .to(vae.device, vae.dtype)
        )
        latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
            vae.device, vae.dtype
        )

        print("[3/4] Loading Bolt Reference Attention...")
        start_l, end_l = map(int, args.bolt_active_layers.split("-"))
        active_layers = list(range(start_l, end_l + 1))

        bolt_layers = BoltReferenceAttentionLayers(
            dit_dim=5120,
            latent_patch_dim=512,
            num_heads=args.bolt_num_heads,
            attn_dim=args.bolt_attn_dim,
            active_layers=active_layers,
        ).to(device, dtype=dtype)

        if args.bolt_ckpt and os.path.exists(args.bolt_ckpt):
            state = torch.load(args.bolt_ckpt, map_location=device)
            bolt_layers.load_state_dict(state)
            print(f"  Loaded Bolt Ref-Attn from {args.bolt_ckpt}")
        else:
            print(
                "  [WARN] No bolt_ckpt provided, using default init "
                "(LayerScale γ=1e-4, out_proj xavier) — Ref-Attn 的注入强度约 r~1e-4，"
                "对 DiT 影响极小。"
            )

        bolt_layers.eval()

        # history_ref: list of {"chunk_idx": int, "latent": Tensor(B,C,T,H,W) on CPU, "clip_feat": Tensor(768) on CPU, "decoded_frame": PIL|None, "decoded_frames": list[PIL]|None}
        history_ref = []
        chunk_selection_records = []
        active_hooks = [None]
        selector_cache = VLMSelectorCache()

        def _build_selector():
            if args.selector_type == "random":
                return RandomSelector(
                    k=args.vlm_k_select,
                    power=args.vlm_power,
                    min_chunk_distance=args.vlm_min_chunk_distance,
                    seed=args.seed,
                )
            if args.selector_type in {"vlm", "vlm_zeroshot"}:
                vlm_backbone = None
                if args.vlm_model_path:
                    try:
                        from helios.modules.vlm_backbones import load_default_backbone
                        resolved_score_mode = "yes_no"
                        if args.selector_type == "vlm_zeroshot":
                            # Force zero-shot yes/no scoring, ignore any LoRA/head settings.
                            resolved_score_mode = "yes_no"
                        elif args.vlm_score_mode in {"yes_no", "hidden_head"}:
                            resolved_score_mode = args.vlm_score_mode
                        elif args.vlm_lora_path:
                            score_mode_meta = os.path.join(args.vlm_lora_path, "score_mode.json")
                            if os.path.exists(score_mode_meta):
                                try:
                                    with open(score_mode_meta, "r", encoding="utf-8") as sm_f:
                                        meta = json.load(sm_f)
                                    meta_mode = str(meta.get("score_mode", "")).strip()
                                    if meta_mode in {"yes_no", "hidden_head"}:
                                        resolved_score_mode = meta_mode
                                        print(
                                            f"  [VLM] auto-detected score_mode={resolved_score_mode} "
                                            f"from {score_mode_meta}"
                                        )
                                except Exception as exc:  # noqa: BLE001
                                    print(f"  [WARN] Failed to read score_mode metadata ({exc}); use yes_no.")
                        elif args.vlm_score_mode is not None:
                            print(
                                f"  [WARN] Unsupported vlm_score_mode={args.vlm_score_mode}; use yes_no."
                            )

                        vlm_backbone = load_default_backbone(
                            model_path=args.vlm_model_path,
                            lora_path=(None if args.selector_type == "vlm_zeroshot" else args.vlm_lora_path),
                            dtype=args.weight_dtype,
                            device=device,
                            score_mode=resolved_score_mode,
                            head_path=(
                                None
                                if (args.selector_type == "vlm_zeroshot" or not args.vlm_lora_path)
                                else os.path.join(args.vlm_lora_path, "selector_head.pt")
                            ),
                        )
                        print(f"  [VLM] score_mode={resolved_score_mode}")
                        if vlm_backbone is None:
                            print(
                                "  [WARN] VLM backbone load failed; VLMFrameSelector will "
                                "use CLIP-placeholder scoring (functional pipeline, no VLM)."
                            )
                    except Exception as exc:  # noqa: BLE001
                        print(f"  [WARN] VLM backbone import/load error: {exc}")
                        vlm_backbone = None
                return VLMFrameSelector(
                    clip_model=clip_model,
                    k=args.vlm_k_select,
                    power=args.vlm_power,
                    min_chunk_distance=args.vlm_min_chunk_distance,
                    max_candidates=args.vlm_max_candidates,
                    prefilter_alpha=args.vlm_prefilter_alpha,
                    temperature=args.vlm_temperature,
                    vlm_backbone=vlm_backbone,
                    device=device,
                    fallback_to_random=False,
                    vlm_rank_mode=args.vlm_rank_mode,
                )
            return None

        selector = _build_selector()

        def _get_clip_prompt_for_next_chunk(prompt_input, next_chunk_idx: int, interpolate_time_list=None):
            # For CLIP selection we use the prompt of the next chunk.
            if not isinstance(prompt_input, list):
                return prompt_input
            if interpolate_time_list is None:
                return prompt_input[0]
            from itertools import accumulate

            seg_boundaries = list(accumulate(interpolate_time_list))
            for seg_i, boundary in enumerate(seg_boundaries):
                if next_chunk_idx < boundary:
                    return prompt_input[seg_i]
            return prompt_input[-1]

        def make_chunk_callback(prompt_input, interpolate_time_list=None):
            explicit_slow = parse_slow_step_chunks(args.vlm_slow_step_chunks)
            auto_slow = compute_slow_step_chunks(
                interpolate_time_list,
                extra_mid=args.vlm_extra_slow_at_segment_mid,
            )
            slow_triggers = explicit_slow if explicit_slow is not None else auto_slow

            def _next_slow_trigger(idx: int) -> int:
                future = sorted([x for x in slow_triggers if x > idx])
                return future[0] if future else (idx + 1)

            def chunk_callback(chunk_idx: int, chunk_latents: torch.Tensor):
                with torch.no_grad():
                    clip_feat, middle_frame = extract_chunk_feature(
                        chunk_latents, vae, clip_model, latents_mean, latents_std
                    )
                    tail_emb = None
                    if long_memory is not None and dino_encoder is not None:
                        # LongMemory 用 chunk 尾帧做 embedding（按需检索/更新）
                        tail_emb = extract_tail_embedding(
                            chunk_latents,
                            vae,
                            dino_encoder,
                            latents_mean,
                            latents_std,
                            return_frame=False,
                        )
                    decoded_frames = None
                    if args.selector_type in {"vlm", "vlm_zeroshot"} and args.vlm_use_chunk_video:
                        decoded_frames = decode_all_frames(
                            chunk_latents,
                            vae,
                            latents_mean,
                            latents_std,
                            max_frames=args.vlm_video_max_frames,
                        )
                    history_ref.append(
                        {
                            "chunk_idx": int(chunk_idx),
                            "latent": chunk_latents.detach().cpu(),
                            "clip_feat": clip_feat,
                            # B 方案：不长期缓存像素帧（VLM/LongMemory 需要时再 decode）
                            "decoded_frame": None,
                            "decoded_frames": decoded_frames,
                            "tail_emb": tail_emb,
                        }
                    )
                    if decoded_frames is not None and (args.bolt_log_every_chunk or (chunk_idx % 2 == 0)):
                        print(
                            f"  [VLM-VIDEO] chunk={chunk_idx} decoded_frames={len(decoded_frames)} "
                            f"(cap={args.vlm_video_max_frames})"
                        )

                    if args.selector_type in {"vlm", "vlm_zeroshot", "random"}:
                        evicted = evict_history(
                            history_ref,
                            max_n=args.mb_max_history_chunks,
                            keep_recent=args.mb_keep_recent_k,
                            strategy=args.mb_evict_strategy,
                            current_chunk_idx=int(chunk_idx),
                            protected_chunk_idx=selector_cache.selected_chunk_idx,
                            alpha=args.mb_evict_alpha,
                        )
                        if evicted and not args.no_vlm_cache_invalidate_on_evict:
                            if any(int(x) in set(evicted) for x in selector_cache.selected_chunk_idx):
                                selector_cache.update([], int(chunk_idx), "", [])

                    if active_hooks[0] is not None:
                        BoltReferenceAttentionLayers.remove_hooks(active_hooks[0])
                        active_hooks[0] = None

                    next_chunk_idx = int(chunk_idx) + 1
                    clip_prompt = _get_clip_prompt_for_next_chunk(prompt_input, next_chunk_idx, interpolate_time_list)

                    def _run_clip_its():
                        return select_gap_frames(
                            history=history_ref,
                            current_chunk_idx=next_chunk_idx,
                            current_prompt=clip_prompt,
                            clip_model=clip_model,
                            k=args.bolt_k_select,
                            alpha=args.bolt_alpha,
                            power=args.bolt_power,
                            min_chunk_distance=args.bolt_min_chunk_distance,
                            device=device,
                        )

                    if args.selector_type == "clip_its":
                        selected_latents, selected_indices = _run_clip_its()
                        schedule_mode = "clip_its"
                    else:
                        # ── LongMemory: cut gate + codebook retrieval ──
                        if long_memory is not None and tail_emb is not None:
                            nonlocal prev_tail_emb
                            # update codebook with current tail_emb + latent (no pixel cached)
                            try:
                                action, slot_i, best_sim = long_memory.update(
                                    e_tail=tail_emb,
                                    chunk_idx=int(chunk_idx),
                                    latent=chunk_latents.detach().cpu(),
                                    decoded_frame=None,
                                )
                                # 流程图「更新 Codebook」：best_sim=与码本最近槽的余弦；action=refresh 表示 EMA 合并该槽，insert 表示新开槽
                                print(
                                    f"  [Codebook][update] chunk={int(chunk_idx)} action={action} slot={slot_i} "
                                    f"best_sim={best_sim:.4f} τ_merge={float(args.lm_tau_merge):.3f} "
                                    f"ema_α={float(args.lm_ema_alpha_new):.3f} |N={len(long_memory)}"
                                )
                                if args.lm_debug or args.bolt_log_every_chunk:
                                    print(
                                        f"  [LongMemory][update] chunk={int(chunk_idx)} "
                                        f"action={action} slot={slot_i} best_sim={best_sim:.3f} "
                                        f"codebook_size={len(long_memory)}"
                                    )
                            except Exception as exc:
                                print(f"  [LongMemory] update failed: {exc}")

                        if long_memory is not None and tail_emb is not None and prev_tail_emb is not None:
                            sim_prev = float(cosine_similarity(prev_tail_emb, tail_emb).item())
                            dist_prev = 1.0 - sim_prev
                            is_cut = dist_prev >= float(args.lm_tau_cut)
                            if args.lm_debug or args.bolt_log_every_chunk:
                                print(
                                    f"  [LongMemory][gate] chunk={int(chunk_idx)} "
                                    f"cos(prev,curr)={sim_prev:.3f} dist={dist_prev:.3f} "
                                    f"tau_cut={float(args.lm_tau_cut):.3f} -> cut={bool(is_cut)}"
                                )
                        else:
                            is_cut = False

                        if is_cut and long_memory is not None and tail_emb is not None:
                            try:
                                # ① codebook embedding top-M
                                candidates = build_codebook_candidates_for_vlm(
                                    store=long_memory,
                                    query_embedding=tail_emb,
                                    topm=args.lm_codebook_topm,
                                    exclude_chunk_idx=[int(chunk_idx)],
                                )
                                if args.lm_debug or args.bolt_log_every_chunk:
                                    cand_chunks = [int(c["chunk_idx"]) for c in candidates]
                                    cand_scores = [float(c.get("_codebook_score", 0.0)) for c in candidates]
                                    print(
                                        f"  [LongMemory][retrieve] chunk={int(chunk_idx)} "
                                        f"topM={len(candidates)} chunks={cand_chunks} "
                                        f"scores={[round(x,3) for x in cand_scores]}"
                                    )
                                # ② decode 像素帧供 VLM 精排
                                for c in candidates:
                                    if c.get("decoded_frame") is None:
                                        c["decoded_frame"] = decode_middle_frame(
                                            c["latent"], vae, latents_mean, latents_std
                                        )
                                # context frame：用当前 chunk 的 middle_frame 近似（推理时可见）
                                context_entry = {"decoded_frame": middle_frame, "clip_feat": clip_feat}
                                selected_latents, selected_indices = selector.select_from_candidates(
                                    candidates=candidates,
                                    context_entry=context_entry,
                                    current_prompt=clip_prompt,
                                )
                                schedule_mode = "slow/longmem"
                                if args.lm_debug or args.bolt_log_every_chunk:
                                    print(
                                        f"  [LongMemory][select] chunk={int(chunk_idx)} "
                                        f"selected={len(selected_indices)} idx={selected_indices}"
                                    )
                            except Exception as exc:
                                if args.vlm_fallback_to_clip:
                                    print(f"  [Selector:{args.selector_type}] slow failed ({exc}), fallback to CLIP+ITS.")
                                    selected_latents, selected_indices = _run_clip_its()
                                else:
                                    raise
                        else:
                            # fast：由 cut 门控决定，不注入 ref-attn。
                            schedule_mode = "fast"
                            selected_latents, selected_indices = [], []

                        if long_memory is not None and tail_emb is not None:
                            prev_tail_emb = tail_emb

                    chunk_selection_records.append(
                        {
                            "chunk_idx": int(chunk_idx),
                            "next_chunk_idx": int(next_chunk_idx),
                            "selector_type": args.selector_type,
                            "schedule_mode": schedule_mode,
                            "selected_count": int(len(selected_indices)),
                            "selected_indices": [int(x) for x in selected_indices],
                            "alpha": float(args.bolt_alpha),
                            "power": float(args.bolt_power),
                            "min_chunk_distance": int(args.bolt_min_chunk_distance),
                            "k_select": int(args.bolt_k_select),
                        }
                    )

                    if selected_latents:
                        active_hooks[0] = bolt_layers.register_hooks(transformer, selected_latents=selected_latents)
                    if args.bolt_log_every_chunk or (chunk_idx % 2 == 0):
                        print(
                            f"  [BOLT] chunk={chunk_idx} → next={next_chunk_idx} "
                            f"selector={args.selector_type}/{schedule_mode} "
                            f"selected={len(selected_indices)} GAP (idx={selected_indices})"
                        )

            return chunk_callback
    else:
        clip_model = None
        latents_mean = None
        latents_std = None
        bolt_layers = None
        history_ref = None
        active_hooks = None
        make_chunk_callback = None

    def get_common_pipe_kwargs(generator):
        return dict(
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            history_sizes=[16, 2, 1],
            num_latent_frames_per_chunk=args.num_latent_frames_per_chunk,
            keep_first_frame=True,
            is_enable_stage2=args.is_enable_stage2,
            pyramid_num_inference_steps_list=args.pyramid_num_inference_steps_list,
            is_skip_first_chunk=args.is_skip_first_chunk,
            is_amplify_first_chunk=args.is_amplify_first_chunk,
            use_zero_init=args.use_zero_init,
            zero_steps=args.zero_steps,
        )

    def generate_one_video(prompt_input, output_path, seed_offset=0,
                           interpolate_time_list=None):
        if args.enable_bolt_injection:
            chunk_selection_records.clear()
        generator = torch.Generator(device=device).manual_seed(args.seed + seed_offset)
        pipe_kwargs = get_common_pipe_kwargs(generator)
        pipe_kwargs["prompt"] = prompt_input

        if isinstance(prompt_input, list) and args.use_interpolate_prompt:
            pipe_kwargs["use_interpolate_prompt"] = True
            pipe_kwargs["interpolation_steps"] = args.interpolation_steps
            pipe_kwargs["interpolate_time_list"] = interpolate_time_list

        try:
            if args.enable_bolt_injection:
                pipe_kwargs["chunk_callback"] = make_chunk_callback(prompt_input, interpolate_time_list)
            with torch.no_grad():
                output = pipe(**pipe_kwargs)
        finally:
            if args.enable_bolt_injection and active_hooks is not None and active_hooks[0] is not None:
                BoltReferenceAttentionLayers.remove_hooks(active_hooks[0])
                active_hooks[0] = None

        frames = output.frames[0]
        true_duration = len(frames)
        # 替换 output_path 中的 {TRUE_DUR} 占位符为实际帧数
        output_path = output_path.replace("{TRUE_DUR}", str(true_duration))
        export_to_video(frames, output_path, fps=args.fps)
        print(f"  Saved: {output_path}")
        if args.enable_bolt_injection and args.bolt_selection_dump_path:
            dump_dir = os.path.dirname(args.bolt_selection_dump_path)
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
            with open(args.bolt_selection_dump_path, "w", encoding="utf-8") as fw:
                json.dump(
                    {
                        "alpha": float(args.bolt_alpha),
                        "power": float(args.bolt_power),
                        "k_select": int(args.bolt_k_select),
                        "min_chunk_distance": int(args.bolt_min_chunk_distance),
                        "records": chunk_selection_records,
                    },
                    fw,
                    ensure_ascii=False,
                    indent=2,
                )
            print(f"  [BOLT] selection records dumped to: {args.bolt_selection_dump_path}")

    # ── 4. Generate ──
    print("[4/4] Generating (diffusers_version + BOLT injection)...")

    if args.interactive_prompt_csv_path is not None:
        import pandas as pd
        df = pd.read_csv(args.interactive_prompt_csv_path)
        df = df.sort_values(by=["id", "prompt_index"])
        all_video_ids = df["id"].unique()

        print(f"  [Interactive] {len(all_video_ids)} videos from CSV")

        for video_id in tqdm(all_video_ids, desc="Processing videos"):
            group_df = df[df["id"] == video_id]
            if "refined_prompt" in df.columns:
                prompt_list = group_df["refined_prompt"].fillna(group_df["prompt"]).tolist()
            else:
                prompt_list = group_df["prompt"].tolist()

            # 目标帧数: pipeline 会根据 interpolate_time 自动扩展 chunk 数
            # 计算实际的 num_latent_sections (与 pipeline 逻辑一致)
            from itertools import accumulate
            itl = [args.interpolate_time] * len(prompt_list)
            cumul = list(accumulate(itl))
            window_num_frames = (args.num_latent_frames_per_chunk - 1) * 4 + 1  # 33
            base_sections = max(1, (args.num_frames + window_num_frames - 1) // window_num_frames)
            if args.use_interpolate_prompt and base_sections < max(cumul):
                actual_sections = max(cumul)
            else:
                actual_sections = base_sections
            target_duration = actual_sections * window_num_frames

            # eval 命名格式: {id}_{target-duration}_ori{true-duration}.mp4
            # true-duration 在生成后才知道，用占位符 {TRUE_DUR}
            output_path = os.path.join(
                args.output_folder,
                f"{video_id}_{target_duration}_ori{{TRUE_DUR}}.mp4"
            )

            # 检查是否已有该 id 的视频（前缀匹配）
            existing = [f for f in os.listdir(args.output_folder)
                        if f.startswith(f"{video_id}_") and f.endswith(".mp4")]
            if existing:
                print(f"  skipping video_id={video_id} (found {existing[0]})")
                continue

            interpolate_time_list = [args.interpolate_time] * len(prompt_list)

            print(f"\n{'─'*50}")
            print(f"Video [{video_id}]: {len(prompt_list)} prompts, target_duration={target_duration}")
            for i, p in enumerate(prompt_list):
                print(f"  [{i+1}] {p[:80]}...")

            generate_one_video(
                prompt_input=prompt_list,
                output_path=output_path,
                seed_offset=int(video_id),
                interpolate_time_list=interpolate_time_list,
            )
    else:
        if args.prompt_txt_path and os.path.exists(args.prompt_txt_path):
            with open(args.prompt_txt_path) as f:
                prompts = [l.strip() for l in f if l.strip()]
        else:
            prompts = [args.prompt]

        for pi, prompt in enumerate(prompts):
            print(f"\n{'─'*50}")
            print(f"Prompt [{pi}]: {prompt[:80]}...")
            vid_id = pi + 1  # eval CSV 的 id 从 1 开始
            target_duration = args.num_frames
            out_path = os.path.join(
                args.output_folder,
                f"{vid_id}_{target_duration}_{{TRUE_DUR}}.mp4"
            )
            generate_one_video(prompt, out_path, seed_offset=pi)

    print(f"\n[INFO] All done. Max VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"[INFO] Output: {args.output_folder}")


if __name__ == "__main__":
    main()
