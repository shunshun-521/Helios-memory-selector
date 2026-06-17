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

from helios.pipelines.pipeline_helios_v2 import HeliosPipeline
from helios.scheduler.scheduling_helios import HeliosScheduler
from helios.modules.transformer_helios_v2 import HeliosTransformer1DModel
from helios.modules.helios_kernels import (
    replace_all_norms_with_flash_norms,
    replace_rmsnorm_with_fp32,
    replace_rope_with_flash_rope,
)
from helios.utils.utils_base_v2 import load_extra_components

from diffusers import ContextParallelConfig
from diffusers.utils import export_to_video, load_image, load_video
from helios.dataset.dac_vae import DAC
import soundfile as sf


def parse_args():
    parser = argparse.ArgumentParser(description="Generate audio with model")

    # === Model paths ===
    parser.add_argument("--base_model_path", type=str, default="BestWishYsh/Helios-Base")
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="BestWishYsh/Helios-Base",
    )
    parser.add_argument("--audio_vae_path", type=str, default="/root/autodl-fs/dac_vae/audio_vae")
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--partial_path",
        type=str,
        default=None,
    )
    parser.add_argument("--output_folder", type=str, default="./output_helios_audio")
    parser.add_argument("--enable_compile", action="store_true")

    # === Generation parameters ===
    # environment
    parser.add_argument(
        "--sample_type",
        type=str,
        default="t2a",
        choices=["t2a"],
    )
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model weights.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed for random number generator.")
    # base
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--sample_rate", type=int, default=48000)
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
    parser.add_argument(
        "--prompt",
        type=str,
        default="A high quality audio recording of a person speaking in a clear and calm voice. The speaker is articulating their words carefully and thoughtfully. The tone is informative and engaging. There is no background noise or music, just the natural sound of the person's voice. The recording is suitable for a podcast, audiobook, or voiceover application.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="low quality, bad quality, noisy, distorted, artifact, poor articulation, muzzled, muffled",
    )
    parser.add_argument(
        "--prompt_txt_path",
        type=str,
        default=None,
    )

    # === Context parallelism ===
    parser.add_argument("--enable_parallelism", action="store_true")
    parser.add_argument(
        "--cp_backend",
        type=str,
        choices=["ring", "ulysses", "unified", "ulysses_anything"],
        default="ulysses",
        help="Context parallel backend to use.",
    )

    # === Group-Offloading ===
    parser.add_argument("--enable_low_vram_mode", action="store_true")
    parser.add_argument(
        "--group_offloading_type",
        type=str,
        choices=["leaf_level", "block_level"],
        default="leaf_level",
        help="Specifies the granularity for group CPU offloading.",
    )
    parser.add_argument(
        "--num_blocks_per_group",
        type=str,
        default="4",
        help="The number of blocks to bundle together in each offloading group.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    assert not (args.enable_low_vram_mode and args.enable_compile), (
        "enable_low_vram_mode and enable_compile cannot be used together."
    )

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
        assert world_size == 1 or not args.enable_low_vram_mode, "enable_low_vram_mode is only for single GPU."
    else:
        rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1

    prompt = args.prompt
    interpolate_time_list = None

    transformer = HeliosTransformer1DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer",
        torch_dtype=args.weight_dtype,
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

    vae = DAC.from_pretrained(args.audio_vae_path)
    scheduler = HeliosScheduler.from_pretrained(
        args.base_model_path,
        subfolder="scheduler",
    )
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=args.weight_dtype,
    )

    if args.lora_path is not None:
        pipe.load_lora_weights(args.lora_path, adapter_name="default")
        pipe.set_adapters(["default"], adapter_weights=[1.0])

        if args.partial_path is not None:
            if not hasattr(args, "training_config"):
                from argparse import Namespace

                args.training_config = Namespace()
            args.training_config.is_enable_stage1 = True
            args.training_config.restrict_self_attn = True
            args.training_config.is_amplify_history = True
            args.training_config.is_use_gan = True
            load_extra_components(args, transformer, args.partial_path)

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
        if args.cp_backend == "ring":
            cp_config = ContextParallelConfig(ring_degree=world_size)
        elif args.cp_backend == "unified":
            cp_config = ContextParallelConfig(ring_degree=world_size // 2, ulysses_degree=world_size // 2)
        elif args.cp_backend == "ulysses":
            cp_config = ContextParallelConfig(ulysses_degree=world_size)
        elif args.cp_backend == "ulysses_anything":
            cp_config = ContextParallelConfig(ulysses_degree=world_size, ulysses_anything=True)
        else:
            raise ValueError(f"Unsupported cp_backend: {args.cp_backend}")

        pipe.transformer.enable_parallelism(config=cp_config)

    if args.prompt_txt_path is not None:
        with open(args.prompt_txt_path, "r") as f:
            prompt_list = [line.strip() for line in f.readlines() if line.strip()]
        if not args.enable_parallelism:
            prompt_list_with_idx = [(i, prompt) for i, prompt in enumerate(prompt_list)]
            prompt_list_with_idx = prompt_list_with_idx[rank::world_size]
        else:
            prompt_list_with_idx = [(i, prompt) for i, prompt in enumerate(prompt_list)]

        for idx, prompt in tqdm(prompt_list_with_idx, desc="Processing prompts"):
            output_path = os.path.join(args.output_folder, f"{idx}.wav")
            if os.path.exists(output_path):
                print("skipping!")
                continue

            with torch.no_grad():
                try:
                    output = pipe(
                        prompt=prompt,
                        negative_prompt=args.negative_prompt,
                        duration=args.duration,
                        sample_rate=args.sample_rate,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=args.guidance_scale,
                        generator=torch.Generator(device="cuda").manual_seed(args.seed),
                        # stage 1
                        history_sizes=[16, 2, 1],
                        latent_window_size=args.latent_window_size,
                        is_keep_x0=True,
                        # stage 2
                        is_enable_stage2=args.is_enable_stage2,
                        stage2_num_inference_steps_list=args.pyramid_num_inference_steps_list,
                        # stage 3
                        is_skip_first_section=args.is_skip_first_chunk,
                        is_amplify_first_chunk=args.is_amplify_first_chunk,
                        # cfg zero
                        use_zero_init=args.use_zero_init,
                        zero_steps=args.zero_steps,
                        # interpolate_prompt
                        use_interpolate_prompt=args.use_interpolate_prompt,
                        interpolation_steps=args.interpolation_steps,
                        interpolate_time_list=interpolate_time_list,
                        output_type="np"
                    ).frames[0]
                except Exception as e:
                    print(f"Error processing prompt: {e}")
                    continue
            if not args.enable_parallelism or rank == 0:
                sf.write(output_path, output.T, args.sample_rate)
    else:
        with torch.no_grad():
            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                duration=args.duration,
                sample_rate=args.sample_rate,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                # stage 1
                history_sizes=[16, 2, 1],
                latent_window_size=args.latent_window_size,
                is_keep_x0=True,
                # stage 2
                is_enable_stage2=args.is_enable_stage2,
                stage2_num_inference_steps_list=args.pyramid_num_inference_steps_list,
                # stage 3
                is_skip_first_section=args.is_skip_first_chunk,
                is_amplify_first_chunk=args.is_amplify_first_chunk,
                # cfg zero
                use_zero_init=args.use_zero_init,
                zero_steps=args.zero_steps,
                # interpolate_prompt
                use_interpolate_prompt=args.use_interpolate_prompt,
                interpolation_steps=args.interpolation_steps,
                interpolate_time_list=interpolate_time_list,
                output_type="np"
            ).frames[0]

        if not args.enable_parallelism or rank == 0:
            file_count = len(
                [f for f in os.listdir(args.output_folder) if os.path.isfile(os.path.join(args.output_folder, f))]
            )
            output_path = os.path.join(
                args.output_folder, f"{file_count:04d}_{args.sample_type}_{int(time.time())}.wav"
            )
            sf.write(output_path, output.T, args.sample_rate)

    print(f"Max memory: {torch.cuda.max_memory_allocated() / 1024**3:.3f} GB")


if __name__ == "__main__":
    main()