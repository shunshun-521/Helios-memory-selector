"""
Inference script for Helios v3 (supports both baseline and selector_v2).
Based on infer_helios.py but uses transformer_helios_v3 + pipeline_helios (non-diffusers)
which has built-in temporal token selection (selector) support.

Usage examples:
  # Baseline (no selector, merged weights):
  python infer_helios_v3.py \
      --transformer_path /root/autodl-fs/output/v3_baseline_2/merged \
      --output_folder ./output_helios/baseline_2

  # Selector_v2 (merged weights):
  python infer_helios_v3.py \
      --transformer_path /root/autodl-fs/output/v3_selector_v2_try_2026_03_30_1814/merged \
      --output_folder ./output_helios/selector_v2

  # With LoRA (not merged):
  python infer_helios_v3.py \
      --lora_path /root/autodl-fs/output/v3_baseline_2/checkpoint-1000/pytorch_lora_weights.safetensors \
      --partial_path /root/autodl-fs/output/v3_baseline_2/checkpoint-1000/transformer_partial.pth \
      --output_folder ./output_helios/baseline_2_lora

  # Selector_v2 with LoRA (not merged):
  python infer_helios_v3.py \
      --use_selector \
      --lora_path /root/autodl-fs/output/v3_selector_v2_try_2026_03_30_1814/checkpoint-1000/pytorch_lora_weights.safetensors \
      --partial_path /root/autodl-fs/output/v3_selector_v2_try_2026_03_30_1814/checkpoint-1000/transformer_partial.pth \
      --output_folder ./output_helios/selector_v2_lora
"""
import importlib
import os

os.environ["HF_ENABLE_PARALLEL_LOADING"] = "yes"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "8"

import argparse
import time

import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

if importlib.util.find_spec("torch_npu") is not None:
    import torch_npu
else:
    torch_npu = None

from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.scheduler.scheduling_helios import HeliosScheduler
from helios.modules.transformer_helios_v3 import HeliosTransformer3DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.utils.utils_base import load_extra_components

from diffusers import ContextParallelConfig
from diffusers.models import AutoencoderKLWan
from diffusers.utils import export_to_video, load_image, load_video


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video with Helios v3 model")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="/root/autodl-fs/BestWishYSH/HeliosDistillede")
    parser.add_argument("--transformer_path", type=str, default="/root/autodl-fs/Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--partial_path", type=str, default=None)
    parser.add_argument("--output_folder", type=str, default="./output_helios")
    parser.add_argument("--enable_compile", action="store_true")

    # === Selector ===
    parser.add_argument("--use_selector", action="store_true",
                        help="Enable temporal token selection (selector_v2). "
                             "Required when loading selector_v2 checkpoints with --partial_path.")

    # === Generation parameters ===
    parser.add_argument("--sample_type", type=str, default="t2v", choices=["t2v", "i2v", "v2v"])
    parser.add_argument("--weight_dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--num_frames", type=int, default=99)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    # cfg zero
    parser.add_argument("--use_zero_init", action="store_true")
    parser.add_argument("--zero_steps", type=int, default=1)
    # stage 1
    parser.add_argument("--latent_window_size", type=int, default=9)
    # stage 2
    parser.add_argument("--is_enable_stage2", action="store_true")
    parser.add_argument("--pyramid_num_inference_steps_list", type=int, nargs="+", default=[20, 20, 20])
    # stage 3
    parser.add_argument("--is_skip_first_chunk", action="store_true")
    parser.add_argument("--is_amplify_first_chunk", action="store_true")

    # === Prompts ===
    parser.add_argument("--use_interpolate_prompt", action="store_true")
    parser.add_argument("--interpolation_steps", type=int, default=3)
    parser.add_argument("--interpolate_time", type=int, default=7)
    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--image_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--image_noise_sigma_max", type=float, default=0.135)
    parser.add_argument("--video_path", type=str, default=None)
    parser.add_argument("--video_noise_sigma_min", type=float, default=0.111)
    parser.add_argument("--video_noise_sigma_max", type=float, default=0.135)
    parser.add_argument("--prompt", type=str, default=(
        "A dynamic time-lapse video showing the rapidly moving scenery from the window of a speeding train. "
        "The camera captures various elements such as lush green fields, towering trees, quaint countryside houses, "
        "and distant mountain ranges passing by quickly. The train window frames the view, adding a sense of speed "
        "and motion as the landscape rushes past. The camera remains static but emphasizes the fast-paced movement "
        "outside. The overall atmosphere is serene yet exhilarating, capturing the essence of travel and exploration. "
        "Medium shot focusing on the train window and the rushing scenery beyond."
    ))
    parser.add_argument("--negative_prompt", type=str, default=(
        "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, "
        "overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, "
        "poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, "
        "messy background, three legs, many people in the background, walking backwards"
    ))
    parser.add_argument("--prompt_txt_path", type=str, default=None)

    # === Context parallelism ===
    parser.add_argument("--enable_parallelism", action="store_true")
    parser.add_argument("--cp_backend", type=str, choices=["ring", "ulysses", "unified", "ulysses_anything"],
                        default="ulysses")

    # === Group-Offloading ===
    parser.add_argument("--enable_low_vram_mode", action="store_true")
    parser.add_argument("--group_offloading_type", type=str, choices=["leaf_level", "block_level"],
                        default="leaf_level")
    parser.add_argument("--num_blocks_per_group", type=str, default="4")

    return parser.parse_args()


def main():
    args = parse_args()

    assert not (args.enable_low_vram_mode and args.enable_compile), \
        "enable_low_vram_mode and enable_compile cannot be used together."

    if args.weight_dtype == "fp32":
        args.weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        args.weight_dtype = torch.float16
    else:
        args.weight_dtype = torch.bfloat16

    os.makedirs(args.output_folder, exist_ok=True)

    if dist.is_available() and "RANK" in os.environ:
        if args.cp_backend == "ulysses_anything":
            dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        else:
            dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        device = torch.device("cuda", rank % torch.cuda.device_count())
        world_size = dist.get_world_size()
        torch.cuda.set_device(device)
        assert world_size == 1 or not args.enable_low_vram_mode
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1

    # Resolve prompt / image / video
    prompt = args.prompt
    image_path = args.image_path
    video_path = args.video_path
    interpolate_time_list = None

    if args.sample_type == "i2v" and image_path is None and prompt == parse_args.__defaults__:
        image_path = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/astronaut.jpg"
        prompt = "An astronaut hatching from an egg, on the surface of the moon."

    # Build transformer with v3 (has selector modules built-in when has_multi_term_memory_patch=True)
    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": True,
        "zero_history_timestep": True,
        "guidance_cross_attn": True,
        "restrict_self_attn": False,
        "is_train_restrict_lora": False,
        "restrict_lora": False,
        "restrict_lora_rank": 128,
    }

    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
        torch_dtype=args.weight_dtype,
        transformer_additional_kwargs=transformer_additional_kwargs,
    )
    if not args.enable_compile:
        transformer = replace_rmsnorm_with_fp32(transformer)
        transformer = replace_all_norms_with_flash_norms(transformer)
        replace_rope_with_flash_rope()
    try:
        transformer.set_attention_backend("_flash_3_hub")
    except Exception:
        try:
            transformer.set_attention_backend("flash_hub")
        except Exception:
            print("Warning: Could not set flash attention backend, using default attention.")

    vae = AutoencoderKLWan.from_pretrained(args.base_model_path, subfolder="vae", torch_dtype=torch.float32)
    scheduler = HeliosScheduler.from_pretrained(args.base_model_path, subfolder="scheduler")
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=args.weight_dtype,
    )

    # Load LoRA + partial weights if provided
    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[1.0])

        if args.partial_path is not None:
            from argparse import Namespace
            infer_args = Namespace()
            infer_args.training_config = Namespace()
            infer_args.training_config.is_enable_stage1 = True
            infer_args.training_config.restrict_self_attn = False
            infer_args.training_config.is_amplify_history = False
            infer_args.training_config.is_use_gan = False
            infer_args.training_config.use_selector = args.use_selector
            load_extra_components(infer_args, transformer, args.partial_path)

    if args.enable_compile:
        torch.backends.cudnn.benchmark = True
        pipe.text_encoder.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.vae.compile(mode="max-autotune-no-cudagraphs", dynamic=False)
        pipe.transformer.compile(mode="max-autotune-no-cudagraphs", dynamic=False)

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

    if world_size > 1 and args.enable_parallelism:
        cp_config_map = {
            "ring": ContextParallelConfig(ring_degree=world_size),
            "unified": ContextParallelConfig(ring_degree=world_size // 2, ulysses_degree=world_size // 2),
            "ulysses": ContextParallelConfig(ulysses_degree=world_size),
            "ulysses_anything": ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True),
        }
        pipe.transformer.enable_parallelism(config=cp_config_map[args.cp_backend])

    # Common generation kwargs
    gen_kwargs = dict(
        use_selector=args.use_selector,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
        history_sizes=[16, 2, 1],
        latent_window_size=args.latent_window_size,
        is_keep_x0=True,
        is_enable_stage2=args.is_enable_stage2,
        stage2_num_inference_steps_list=args.pyramid_num_inference_steps_list,
        is_skip_first_section=args.is_skip_first_chunk,
        is_amplify_first_chunk=args.is_amplify_first_chunk,
        use_zero_init=args.use_zero_init,
        zero_steps=args.zero_steps,
        image=load_image(image_path).resize((args.width, args.height)) if image_path else None,
        image_noise_sigma_min=args.image_noise_sigma_min,
        image_noise_sigma_max=args.image_noise_sigma_max,
        video=load_video(video_path) if video_path else None,
        video_noise_sigma_min=args.video_noise_sigma_min,
        video_noise_sigma_max=args.video_noise_sigma_max,
        use_interpolate_prompt=args.use_interpolate_prompt,
        interpolation_steps=args.interpolation_steps,
        interpolate_time_list=interpolate_time_list,
    )

    if args.prompt_txt_path is not None:
        with open(args.prompt_txt_path, "r") as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]
        if not args.enable_parallelism:
            prompt_list_with_idx = list(enumerate(prompt_list))[rank::world_size]
        else:
            prompt_list_with_idx = list(enumerate(prompt_list))

        for idx, p in tqdm(prompt_list_with_idx, desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{idx}.mp4")
            if os.path.exists(output_path):
                print("skipping!")
                continue
            with torch.no_grad():
                try:
                    output = pipe(prompt=p, **gen_kwargs).frames[0]
                except Exception as e:
                    print(f"Error: {e}")
                    continue
            if not args.enable_parallelism or rank == 0:
                export_to_video(output, output_path, fps=args.fps)
    else:
        with torch.no_grad():
            output = pipe(prompt=prompt, **gen_kwargs).frames[0]

        if not args.enable_parallelism or rank == 0:
            file_count = len([f for f in os.listdir(args.output_folder)
                              if os.path.isfile(os.path.join(args.output_folder, f))])
            output_path = os.path.join(args.output_folder,
                                       f"{file_count:04d}_{args.sample_type}_{int(time.time())}.mp4")
            export_to_video(output, output_path, fps=args.fps)

    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")


if __name__ == "__main__":
    main()
