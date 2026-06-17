"""
inference_with_gap_injection.py — 推理: VLM Selector + GAP History Injection
=============================================================================

将 VLM Selector 选出的 GAP 帧直接注入 Helios 的 latents_history_long，
完全复用 Helios 已有的 history self-attention 机制。

【与 ref_attn 方案的区别】
- ref_attn: 需要额外的 attention 层 + Stage 2 训练 + 显存开销
- 本方案: 零额外参数，零训练，直接替换 history_long 中的帧

【原理】
Helios 的 history_long (16帧) 只包含最近的历史帧。
随着视频变长，早期重要帧（人物正脸、特定光照）被挤出窗口。
本方案用 VLM Selector 选出的 GAP 帧替换 history_long 中最远的帧，
让 DiT 的 self-attention 能看到更有信息量的历史参考帧。

用法:
  python inference_with_gap_injection.py \
    --selector_ckpt /path/to/selector_ckpt \
    --prompt "A camera orbiting around a person..." \
    --k_inject 4
"""

import importlib
import os

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import sys
import time

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
else:
    torch_npu = None

HELIOS_ROOT = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, HELIOS_ROOT)

from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.scheduler.scheduling_helios import HeliosScheduler
from helios.modules.helios_kernels import (
    replace_rmsnorm_with_fp32,
    replace_all_norms_with_flash_norms,
    replace_rope_with_flash_rope,
)
from helios.utils.utils_base import load_extra_components
from diffusers.models import AutoencoderKLWan

from helios.modules.selector_vlm import VLMSelector
from helios.modules.gap_history_injector import GAPHistoryInjector


def parse_args():
    p = argparse.ArgumentParser()
    # Model paths
    p.add_argument("--base_model_path", default="/root/autodl-fs/BestWishYSH/Helios-Base")
    p.add_argument("--transformer_path", default="/root/autodl-fs/BestWishYSH/Helios-Base")
    p.add_argument("--lora_path", default=None)
    p.add_argument("--partial_path", default=None)
    # Selector
    p.add_argument("--selector_ckpt", required=True)
    p.add_argument("--vlm_model_path", default="/root/autodl-fs/Qwen2.5-VL-3B-Instruct")
    p.add_argument("--k_select", type=int, default=4, help="Selector top-k")
    p.add_argument("--k_inject", type=int, default=4, help="注入 history_long 的帧数")
    p.add_argument("--max_bank_size", type=int, default=64)
    # Generation
    p.add_argument("--prompt", default="A person slowly turns around")
    p.add_argument("--negative_prompt", default="")
    p.add_argument("--prompt_txt_path", default=None)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--width", type=int, default=832)
    p.add_argument("--num_frames", type=int, default=97)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--latent_window_size", type=int, default=9)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_folder", default="output_gap_inject")
    p.add_argument("--weight_dtype", default="bfloat16")
    # Pipeline options
    p.add_argument("--enable_low_vram_mode", action="store_true")
    p.add_argument("--group_offloading_type", default="leaf_level")
    p.add_argument("--is_enable_stage2", action="store_true")
    p.add_argument("--pyramid_num_inference_steps_list", nargs="+", type=int, default=[10, 10, 10])
    p.add_argument("--is_skip_first_chunk", action="store_true")
    p.add_argument("--is_amplify_first_chunk", action="store_true")
    p.add_argument("--use_zero_init", action="store_true")
    p.add_argument("--use_cfg_zero_star", action="store_true")
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

    # ── 1. Load Pipeline ──
    print("[1/3] Loading Helios Pipeline...")
    transformer_kwargs = {
        "has_multi_term_memory_patch": True,
        "zero_history_timestep": True,
        "guidance_cross_attn": True,
    }
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path, subfolder="transformer",
        torch_dtype=dtype, transformer_additional_kwargs=transformer_kwargs,
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

    # ── 2. Load Selector ──
    print("[2/3] Loading VLM Selector...")
    selector = VLMSelector(
        vlm_model_path=args.vlm_model_path,
        vlm_hidden_dim=2048, k_select=args.k_select, use_lora=True,
    )
    selector.init_vlm(device=device)
    head_path = os.path.join(args.selector_ckpt, "selector_head.pth")
    if os.path.exists(head_path):
        selector.selector_head.load_state_dict(torch.load(head_path, map_location=device))
    lora_path = os.path.join(args.selector_ckpt, "vlm_lora_weights.pth")
    if os.path.exists(lora_path):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(selector._vlm, torch.load(lora_path, map_location=device))
    selector.selector_head = selector.selector_head.to(device)
    selector.eval()
    for p in selector.parameters():
        p.requires_grad = False

    # ── 3. Create Injector ──
    injector = GAPHistoryInjector(k_inject=args.k_inject)

    # VAE normalization 参数（与 pipeline 内部一致）
    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(vae.device, vae.dtype)
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(
        vae.device, vae.dtype
    )

    # ── Helpers ──
    from helios.modules.memory_bank import MemoryBank
    from diffusers.utils import export_to_video
    import inspect

    def make_chunk_callback(prompt_text, bank_ref, injector_ref,
                            active_hook_ref, interpolate_time_list=None):
        """为每个视频创建 chunk_callback 闭包。支持 interactive 多 prompt。"""
        if isinstance(prompt_text, list) and interpolate_time_list is not None:
            from itertools import accumulate
            seg_boundaries = list(accumulate(interpolate_time_list))
        else:
            seg_boundaries = None

        if isinstance(prompt_text, list):
            all_prompts_joined = " ".join(prompt_text)
        else:
            all_prompts_joined = prompt_text

        def _get_current_prompt(chunk_idx):
            if seg_boundaries is None:
                return prompt_text if isinstance(prompt_text, str) else prompt_text[0]
            for seg_i, boundary in enumerate(seg_boundaries):
                if chunk_idx < boundary:
                    return prompt_text[seg_i]
            return prompt_text[-1]

        def chunk_callback(chunk_idx: int, chunk_latent: torch.Tensor):
            with torch.no_grad():
                mid_frame_idx = chunk_latent.shape[2] // 2
                single_latent = chunk_latent[:, :, mid_frame_idx:mid_frame_idx+1]
                try:
                    # 问题5修复: VAE decode 前做 normalization（与 pipeline 一致）
                    normalized_latent = single_latent.to(vae.dtype) / latents_std + latents_mean
                    pixel = vae.decode(normalized_latent).sample
                    pixel = pixel[0, :, 0].clamp(-1, 1).add(1).div(2)
                    pixel = pixel.permute(1, 2, 0).cpu().float().numpy()
                    pixel = (pixel * 255).astype(np.uint8)
                    context_img = Image.fromarray(pixel)
                except Exception as e:
                    print(f"  [chunk {chunk_idx}] VAE decode failed: {e}")
                    return

                q_prompt = _get_current_prompt(chunk_idx)
                q_lhs = selector.encode_lhs([context_img], q_prompt, device=device)
                k_lhs = selector.encode_lhs([context_img], all_prompts_joined, device=device) #problem!?
                k_lhs_pooled = VLMSelector.pool_lhs(k_lhs)

                # Step 1: 获取所有 LHS 给 Selector
                all_lhs = bank_ref.get_all_lhs()
                selected_indices = None
                if all_lhs is not None and all_lhs.shape[0] > 0:
                    result = selector(q_lhs, all_lhs.unsqueeze(0))
                    top_k_idx = result["top_k_indices"][0]
                    selected_indices = top_k_idx
                    print(f"  [chunk {chunk_idx}] seg='{q_prompt[:40]}...' → "
                          f"{len(top_k_idx)} GAP frames (bank={len(bank_ref)})")

                    # 问题4修复: 更新被选中条目的 score（用于淘汰排序）
                    scores = result["p_pred"][0, top_k_idx]
                    bank_ref.update_scores(top_k_idx, scores)

                # 问题3修复: 移除旧 hook，注册新 hook（传入 bank 作为数据源）
                if active_hook_ref[0] is not None:
                    active_hook_ref[0].remove()
                    active_hook_ref[0] = None
                if selected_indices is not None:
                    active_hook_ref[0] = injector_ref.register_hook(
                        transformer, bank_ref, selected_indices)

                # Step 5: chunk 生成完毕后加入新条目
                bank_ref.add(
                    lhs=k_lhs_pooled.squeeze(0),
                    latent=chunk_latent.squeeze(0).cpu(),
                    chunk_idx=chunk_idx,
                )
        return chunk_callback

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
            latent_window_size=args.latent_window_size,
            is_keep_x0=True,
            is_enable_stage2=args.is_enable_stage2,
            stage2_num_inference_steps_list=args.pyramid_num_inference_steps_list,
            is_skip_first_section=args.is_skip_first_chunk,
            is_amplify_first_chunk=args.is_amplify_first_chunk,
            use_zero_init=args.use_zero_init,
            use_cfg_zero_star=args.use_cfg_zero_star,
            zero_steps=args.zero_steps,
        )

    def generate_one_video(prompt_input, output_path, seed_offset=0,
                           interpolate_time_list=None):
        generator = torch.Generator(device=device).manual_seed(args.seed + seed_offset)
        bank = MemoryBank(max_size=args.max_bank_size, device=device)
        active_hook = [None]

        callback = make_chunk_callback(
            prompt_input, bank, injector, active_hook, interpolate_time_list)

        pipe_kwargs = get_common_pipe_kwargs(generator)
        pipe_kwargs["prompt"] = prompt_input

        if isinstance(prompt_input, list) and args.use_interpolate_prompt:
            pipe_kwargs["use_interpolate_prompt"] = True
            pipe_kwargs["interpolation_steps"] = args.interpolation_steps
            pipe_kwargs["interpolate_time_list"] = interpolate_time_list
        
        if "chunk_callback" in inspect.signature(pipe.__call__).parameters:
            pipe_kwargs["chunk_callback"] = callback
            print(f"  [INFO] chunk_callback active (GAP injection)")
        else:
            print(f"  [WARN] Pipeline does NOT support chunk_callback")

        try:
            with torch.no_grad():
                output = pipe(**pipe_kwargs)
        finally:
            if active_hook[0] is not None:
                active_hook[0].remove()

        frames = output.frames[0]
        if hasattr(frames, 'shape'):
            print(f"  [DEBUG] frames type={type(frames)}, shape={frames.shape}, dtype={frames.dtype}")
        else:
            print(f"  [DEBUG] frames type={type(frames)}, len={len(frames)}")
            if len(frames) > 0:
                f0 = frames[0]
                print(f"  [DEBUG] frames[0] type={type(f0)}, shape={getattr(f0, 'shape', 'N/A')}")
        export_to_video(frames, output_path, fps=args.fps)
        print(f"  Saved: {output_path}")

    # ── 4. Generate ──
    print("[3/3] Generating...")

    # ═══ Mode A: Interactive CSV (multi-prompt per video) ═══
    if args.interactive_prompt_csv_path is not None:
        import pandas as pd
        df = pd.read_csv(args.interactive_prompt_csv_path)
        df = df.sort_values(by=["id", "prompt_index"])
        all_video_ids = df["id"].unique()

        print(f"  [Interactive] {len(all_video_ids)} videos from CSV")

        for video_id in tqdm(all_video_ids, desc="Processing videos"):
            output_path = os.path.join(args.output_folder, f"{video_id}.mp4")
            if os.path.exists(output_path):
                print(f"  skipping {output_path}")
                continue

            group_df = df[df["id"] == video_id]
            if "refined_prompt" in df.columns:
                prompt_list = group_df["refined_prompt"].fillna(group_df["prompt"]).tolist()
            else:
                prompt_list = group_df["prompt"].tolist()

            interpolate_time_list = [args.interpolate_time] * len(prompt_list)

            print(f"\n{'─'*50}")
            print(f"Video [{video_id}]: {len(prompt_list)} prompts, "
                  f"interpolate_time={args.interpolate_time}")
            for i, p in enumerate(prompt_list):
                print(f"  [{i+1}] {p[:80]}...")

            generate_one_video(
                prompt_input=prompt_list,
                output_path=output_path,
                seed_offset=int(video_id),
                interpolate_time_list=interpolate_time_list,
            )

    # ═══ Mode B: Single prompt / prompt txt ═══
    else:
        if args.prompt_txt_path and os.path.exists(args.prompt_txt_path):
            with open(args.prompt_txt_path) as f:
                prompts = [l.strip() for l in f if l.strip()]
        else:
            prompts = [args.prompt]

        for pi, prompt in enumerate(prompts):
            print(f"\n{'─'*50}")
            print(f"Prompt [{pi}]: {prompt[:80]}...")
            out_path = os.path.join(
                args.output_folder, f"{pi:04d}_gap_inject_{args.seed}.mp4")
            generate_one_video(prompt, out_path, seed_offset=pi)

    print(f"\n[INFO] All done. Max VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"[INFO] Output: {args.output_folder}")


if __name__ == "__main__":
    main()
