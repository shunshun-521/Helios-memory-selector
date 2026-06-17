"""
inference_with_selector.py — 推理: VLM Selector + Ref-Attn 集成
==================================================================

改造 infer_helios_v3.py，在自回归循环中接入:
  MemoryBank + VLMSelector + ReferenceAttentionLayers

【与 infer_helios_v3.py (selector_v2) 的区别】
- v2/v3: selector 内嵌在 DiT 的 forward 中，GAP 帧 concat 进 self-attn
- 本方案: selector 在 pipeline 外部调用 (每 chunk 一次)，Ref-Attn 通过 hook 注入
  - 更强的语义选帧 (VLM)
  - 更干净的注入方式 (独立 Ref-Attn + zero-init α)
  - Memory Bank 缓存历史帧 LHS/latent，避免重复 VLM 编码

【每 chunk 推理流程】
1. bank.get_all_lhs()           → 所有候选帧 LHS
2. selector(Q=context, K=bank)  → top-k 索引
3. bank.update_scores(top_k)    → 更新选中帧分数
4. bank.get_selected(top_k)     → 取 LHS+latent 给 Ref-Attn
5. DiT forward with Ref-Attn    → 生成 chunk_t
6. bank.add(new_chunk)          → 入库 + 可能淘汰

用法:
  # 单 prompt 推理
  python inference_with_selector.py \
    --selector_ckpt /path/to/selector_ckpt \
    --ref_attn_ckpt /path/to/ref_attn.pth \
    --prompt "A camera orbiting around a person..."

  # Interactive 多 prompt 推理 (CSV 格式: id, prompt_index, prompt)
  python inference_with_selector.py \
    --selector_ckpt /path/to/selector_ckpt \
    --ref_attn_ckpt /path/to/ref_attn.pth \
    --interactive_prompt_csv_path example/prompt_interactive_helios_3.csv \
    --use_interpolate_prompt \
    --interpolation_steps 3 \
    --interpolate_time 7 \
    --num_frames 726 \
    --enable_low_vram_mode
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
import torch.distributed as dist
from PIL import Image
from tqdm import tqdm

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
else:
    torch_npu = None

HELIOS_ROOT = os.path.dirname(__file__)
sys.path.insert(0, HELIOS_ROOT)

from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.scheduler.scheduling_helios import HeliosScheduler
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.utils.utils_base import load_extra_components

from helios.modules.selector_vlm import VLMSelector
from helios.modules.ref_attn import ReferenceAttentionLayers
from helios.modules.memory_bank import MemoryBank

from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video


# ═══════════════════════════════════════════
# Ref-Attn Hook Manager (Fix #9: 使用 ref_attn.py 统一接口)
# ═══════════════════════════════════════════

class RefAttnHookManager:
    """管理 DiT forward 过程中的 Ref-Attn hook 注入。

    使用 ReferenceAttentionLayers 的统一 hook 接口。
    支持在推理过程中动态更新 ref_key/ref_value。

    用法:
        manager = RefAttnHookManager(ref_attn_layers)

        # 每 chunk 更新参考特征并重新注册 hook
        manager.update_and_register(transformer, ref_key, ref_value)

        # pipeline forward (hook 自动生效)
        output = pipe(...)

        # 生成完毕后移除
        manager.remove()
    """

    def __init__(self, ref_attn_layers: ReferenceAttentionLayers):
        self.ref_attn_layers = ref_attn_layers
        self.hooks = []

    def update_and_register(self, transformer, ref_key, ref_value=None):
        """更新参考特征并重新注册 hook。每 chunk 调用一次。"""
        self.remove()
        self.hooks = self.ref_attn_layers.register_hooks(transformer, ref_key, ref_value)

    def register_empty(self, transformer):
        """注册空 hook (ref_key=None，Ref-Attn 输出为零)。"""
        self.remove()
        self.hooks = self.ref_attn_layers.register_hooks(transformer, ref_key=None)

    def remove(self):
        """移除所有 hooks。"""
        ReferenceAttentionLayers.remove_hooks(self.hooks)


# ═══════════════════════════════════════════
# Args
# ═══════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="Inference with VLM Selector + Ref-Attn")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-fs/BestWishYSH/Helios-Base")
    parser.add_argument("--transformer_path", type=str, default="/root/autodl-fs/BestWishYSH/Helios-Base")
    parser.add_argument("--lora_path", type=str, default=None, help="Helios DiT LoRA (optional)")
    parser.add_argument("--partial_path", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default="/root/autodl-tmp/output_4_6")

    # === VLM Selector ===
    parser.add_argument("--selector_ckpt", type=str, required=True,default="/root/autodl-fs/output/vlm_selector_stage1/checkpoint-epoch003",
                        help="Stage 1 checkpoint dir")
    parser.add_argument("--vlm_model_path", type=str, default="/root/autodl-fs/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--k_select", type=int, default=2)

    # === Ref-Attn ===
    parser.add_argument("--ref_attn_ckpt", type=str, required=True,
                        help="Stage 2 Ref-Attn checkpoint (.pth)")
    parser.add_argument("--dit_dim", type=int, default=5120)
    parser.add_argument("--active_layers", type=str, default="10-30")

    # === Memory Bank ===
    parser.add_argument("--memory_bank_max", type=int, default=50)

    # === Generation ===
    parser.add_argument("--weight_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=99)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--latent_window_size", type=int, default=9)

    # === Prompt ===
    parser.add_argument("--prompt", type=str, default="A dynamic time-lapse video...")
    parser.add_argument("--negative_prompt", type=str, default="worst quality, low quality")
    parser.add_argument("--prompt_txt_path", type=str, default=None)

    # === Interactive (multi-prompt CSV) ===
    parser.add_argument("--interactive_prompt_csv_path", type=str, default=None,
                        help="CSV with columns: id, prompt_index, prompt")
    parser.add_argument("--use_interpolate_prompt", action="store_true")
    parser.add_argument("--interpolation_steps", type=int, default=3)
    parser.add_argument("--interpolate_time", type=int, default=7)

    # === Pipeline extras ===
    parser.add_argument("--is_enable_stage2", action="store_true")
    parser.add_argument("--pyramid_num_inference_steps_list", type=int, nargs="+", default=None)
    parser.add_argument("--is_skip_first_chunk", action="store_true")
    parser.add_argument("--is_amplify_first_chunk", action="store_true")
    parser.add_argument("--use_zero_init", action="store_true")
    parser.add_argument("--zero_steps", type=int, default=1)
    parser.add_argument("--enable_low_vram_mode", action="store_true")
    parser.add_argument("--group_offloading_type", type=str, default="leaf_level",
                        choices=["leaf_level", "block_level"])
    parser.add_argument("--num_blocks_per_group", type=int, default=4)

    return parser.parse_args()


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    args = parse_args()

    if args.weight_dtype == "fp32":
        dtype = torch.float32
    elif args.weight_dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_folder, exist_ok=True)

    # ──────────────────────────────
    # 1. Load Helios Pipeline
    # ──────────────────────────────
    print("[1/4] Loading Helios Pipeline...")
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
            num_blocks_per_group=args.num_blocks_per_group if args.group_offloading_type == "block_level" else None,
            use_stream=True,
            record_stream=True,
        )
    else:
        pipe = pipe.to(device)

    # ──────────────────────────────
    # 2. Load VLM Selector (frozen)
    # ──────────────────────────────
    print("[2/4] Loading VLM Selector...")
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
    selector.eval()
    for p in selector.parameters():
        p.requires_grad = False

    # ──────────────────────────────
    # 3. Load Ref-Attn Layers
    # ──────────────────────────────
    print("[3/4] Loading Reference Attention Layers...")
    start_l, end_l = map(int, args.active_layers.split("-"))
    active_layers = list(range(start_l, end_l + 1))
    ref_attn_layers = ReferenceAttentionLayers(
        dit_dim=args.dit_dim, vlm_hidden_dim=2048, active_layers=active_layers,
    ).to(device, dtype=dtype)
    ref_attn_layers.load_state_dict(torch.load(args.ref_attn_ckpt, map_location=device))
    ref_attn_layers.eval()
    for p in ref_attn_layers.parameters():
        p.requires_grad = False

    # Hook manager
    hook_manager = RefAttnHookManager(ref_attn_layers)

    # ──────────────────────────────
    # 4. Generate
    # ──────────────────────────────
    print("[4/4] Generating...")

    # ── Helper: chunk_callback factory ──
    def make_chunk_callback(prompt_text, bank_ref, interpolate_time_list=None):
        """为每个视频创建 chunk_callback 闭包。

        interactive 模式下根据 chunk_idx 所在 segment 动态切换 prompt。
        """
        # 预计算 segment 边界: cumulative chunk counts
        if isinstance(prompt_text, list) and interpolate_time_list is not None:
            from itertools import accumulate
            seg_boundaries = list(accumulate(interpolate_time_list))  # e.g. [7, 14, 21]
        else:
            seg_boundaries = None

        # Q prompt: 当前 segment prompt (动态切换)
        # K prompt: 所有 segment prompts 拼接 (全局语义)
        if isinstance(prompt_text, list):
            all_prompts_joined = " ".join(prompt_text)
        else:
            all_prompts_joined = prompt_text

        def _get_current_prompt(chunk_idx):
            """根据 chunk_idx 返回当前 segment 对应的 prompt (用于 Q)。"""
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
                    pixel = vae.decode(single_latent.to(vae.dtype)).sample
                    pixel = pixel[0, :, 0].clamp(-1, 1).add(1).div(2)
                    pixel = pixel.permute(1, 2, 0).cpu().float().numpy()
                    pixel = (pixel * 255).astype(np.uint8)
                    context_img = Image.fromarray(pixel)
                except Exception as e:
                    print(f"  [chunk_callback] VAE decode failed for chunk {chunk_idx}: {e}")
                    return

                # Q: 当前 segment prompt + chunk_{t-1} 像素帧
                q_prompt = _get_current_prompt(chunk_idx)
                q_lhs = selector.encode_lhs([context_img], q_prompt, device=device)

                # K/V 入库: 用所有 segment prompts 拼接 + 当前帧编码
                # 这样 bank 中的 LHS 包含全局语义，selector 检索时 K 有完整上下文
                k_lhs = selector.encode_lhs([context_img], all_prompts_joined, device=device)
                k_lhs_pooled = VLMSelector.pool_lhs(k_lhs)

                # 用 Q 去检索 bank 中的 K
                all_lhs = bank_ref.get_all_lhs()
                if all_lhs is not None and all_lhs.shape[0] > 0:
                    result = selector(q_lhs, all_lhs.unsqueeze(0))
                    top_k_idx = result["top_k_indices"][0]
                    scores = result["p_pred"][0, top_k_idx]
                    bank_ref.update_scores(top_k_idx, scores)

                    sel_lhs, sel_latent = bank_ref.get_selected(top_k_idx)
                    if sel_lhs is not None:
                        ref_key = sel_lhs.unsqueeze(0).to(device, dtype=dtype)
                        hook_manager.update_and_register(transformer, ref_key)
                        print(f"  [chunk {chunk_idx}] seg='{q_prompt[:30]}...' → {len(top_k_idx)} ref frames (bank={bank_ref.size})")
                else:
                    hook_manager.register_empty(transformer)

                # 入库时用全局 prompt 编码的 LHS (K 语义)
                bank_ref.add(
                    lhs=k_lhs_pooled.squeeze(0),
                    latent=chunk_latent.squeeze(0).cpu(),
                    chunk_idx=chunk_idx,
                )
        return chunk_callback

    # ── Helper: common pipeline kwargs ──
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
            zero_steps=args.zero_steps,
        )

    # ── Helper: run one video ──
    def generate_one_video(prompt_input, output_path, seed_offset=0,
                           interpolate_time_list=None):
        """生成一个视频。prompt_input 可以是 str 或 list[str] (interactive)。"""
        generator = torch.Generator(device=device).manual_seed(args.seed + seed_offset)
        bank = MemoryBank(max_size=args.memory_bank_max, device=device)

        callback = make_chunk_callback(prompt_input, bank, interpolate_time_list)
        hook_manager.register_empty(transformer)

        pipe_kwargs = get_common_pipe_kwargs(generator)
        pipe_kwargs["prompt"] = prompt_input

        # Interactive mode
        if isinstance(prompt_input, list) and args.use_interpolate_prompt:
            pipe_kwargs["use_interpolate_prompt"] = True
            pipe_kwargs["interpolation_steps"] = args.interpolation_steps
            pipe_kwargs["interpolate_time_list"] = interpolate_time_list

        # Inject chunk_callback if pipeline supports it
        import inspect
        if "chunk_callback" in inspect.signature(pipe.__call__).parameters:
            pipe_kwargs["chunk_callback"] = callback
            print(f"  [INFO] chunk_callback active")
        else:
            print(f"  [WARN] Pipeline does NOT support chunk_callback — Ref-Attn inactive")

        try:
            with torch.no_grad():
                output = pipe(**pipe_kwargs)
        finally:
            hook_manager.remove()

        frames = output.frames[0] if hasattr(output, "frames") and isinstance(output.frames, list) else output.frames if hasattr(output, "frames") else output[0]
        export_to_video(frames, output_path, fps=args.fps)
        print(f"  Saved: {output_path}")

    # ═══════════════════════════════════════
    # Mode A: Interactive CSV (multi-prompt per video)
    # ═══════════════════════════════════════
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

    # ═══════════════════════════════════════
    # Mode B: Single prompt / prompt txt
    # ═══════════════════════════════════════
    else:
        if args.prompt_txt_path and os.path.exists(args.prompt_txt_path):
            with open(args.prompt_txt_path) as f:
                prompts = [l.strip() for l in f if l.strip()]
        else:
            prompts = [args.prompt]

        for pi, prompt in enumerate(prompts):
            print(f"\n{'─'*50}")
            print(f"Prompt [{pi}]: {prompt[:80]}...")
            out_path = os.path.join(args.output_folder, f"{pi:04d}_vlm_selector_{args.seed}.mp4")
            generate_one_video(prompt, out_path, seed_offset=pi)

    print(f"\n[INFO] All done. Max VRAM: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")
    print(f"[INFO] Output: {args.output_folder}")


if __name__ == "__main__":
    main()
