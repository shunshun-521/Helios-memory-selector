#!/usr/bin/env python3
"""Ref-Short inference via HeliosPipeline + RefShortValidationCtx."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from omegaconf import OmegaConf

HELIOS_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HELIOS_ROOT)

from helios.modules.extract_feature import CLIP
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.scheduler.scheduling_helios import HeliosScheduler
from helios.utils.ref_short_validation import build_inference_ctx
from helios.utils.train_config import Args, SelectorInferenceConfig, SelectorTrainingConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--pretrained_model_name_or_path", type=str, required=True)
    p.add_argument("--transformer_model_name_or_path", type=str, required=True)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--partial_path", type=str, default=None, help="transformer_partial.pth (patch_ref + clean_patch)")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
    )
    p.add_argument("--output_path", type=str, default="ref_short_out.mp4")
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--num_inference_steps", type=int, default=30)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--scheduler_type", type=str, default="unipc", choices=["unipc", "euler"])
    p.add_argument("--use_cfg_zero_star", action="store_true")
    p.add_argument("--use_zero_init", action="store_true")
    p.add_argument("--zero_steps", type=int, default=1)
    p.add_argument("--vlm_model_path", type=str, required=True)
    p.add_argument("--vlm_k_select", type=int, default=None)
    p.add_argument("--ref_frames_per_chunk", type=int, default=None)
    p.add_argument(
        "--selector_force_slow",
        action="store_true",
        default=None,
        help="Force every chunk slow (debug); default uses yaml selector_inference",
    )
    p.add_argument(
        "--enable_long_memory",
        action="store_true",
        default=None,
        help="DINO cut + codebook (default from yaml)",
    )
    p.add_argument("--dino_model_path", type=str, default=None)
    p.add_argument("--lm_tau_cut", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    conf = OmegaConf.structured(Args)
    if args.config:
        conf = OmegaConf.merge(conf, OmegaConf.load(args.config))
    conf.training_config.use_ref_short = True
    if args.ref_frames_per_chunk is not None:
        conf.training_config.ref_frames_per_chunk = args.ref_frames_per_chunk

    selector_training_overrides = {
        "vlm_model_path": args.vlm_model_path,
    }
    if args.vlm_k_select is not None:
        selector_training_overrides["vlm_k_select"] = args.vlm_k_select
    if args.selector_force_slow is not None:
        selector_training_overrides["selector_force_slow"] = args.selector_force_slow

    conf.selector_training = OmegaConf.merge(
        OmegaConf.structured(SelectorTrainingConfig),
        selector_training_overrides,
    )

    selector_inference_overrides = {}
    if args.selector_force_slow is not None:
        selector_inference_overrides["selector_force_slow"] = args.selector_force_slow
    if args.enable_long_memory is not None:
        selector_inference_overrides["enable_long_memory"] = args.enable_long_memory
    if args.vlm_k_select is not None:
        selector_inference_overrides["vlm_k_select"] = args.vlm_k_select
    if args.dino_model_path is not None:
        selector_inference_overrides["dino_model_path"] = args.dino_model_path

    conf.selector_inference = OmegaConf.merge(
        OmegaConf.structured(SelectorInferenceConfig),
        selector_inference_overrides,
    )
    if args.lm_tau_cut is not None:
        conf.selector_inference.lm_tau_cut = args.lm_tau_cut

    # Align with infer_helios.py: VAE in fp32, and pass the SAME VAE into pipeline for decode.
    vae = AutoencoderKLWan.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    ).to(device)
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)
    # pipeline_helios expects latents_std = 1 / std (see pipeline_helios.py)
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, vae.config.z_dim, 1, 1, 1).to(device, vae.dtype)

    clip_model = CLIP(device=str(device))
    ref_ctx = build_inference_ctx(
        conf,
        clip_model=clip_model,
        vae=vae,
        latents_mean=latents_mean,
        latents_std=latents_std,
        device=device,
        dtype=dtype,
    )

    # Align scheduler behavior with infer_helios.py
    scheduler = HeliosScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # IMPORTANT: explicitly load Helios' own transformer implementation.
    # Passing a path into DiffusionPipeline.from_pretrained can accidentally load the diffusers-side transformer class,
    # whose forward() doesn't accept ref-short kwargs like `latents_history_ref`.
    tr_path = Path(args.transformer_model_name_or_path)
    if not tr_path.exists():
        raise FileNotFoundError(f"--transformer_model_name_or_path not found: {str(tr_path)}")
    transformer = HeliosTransformer3DModel.from_pretrained(
        str(tr_path),
        torch_dtype=dtype,
    )

    pipe = HeliosPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=dtype,
    )
    if args.lora_path:
        pipe.load_lora_weights(args.lora_path)
    if args.partial_path:
        from helios.utils.utils_base import load_extra_components

        load_extra_components(conf, pipe.transformer, args.partial_path)
    pipe = pipe.to(device)

    generator = torch.Generator(device=device).manual_seed(args.seed)
    ref_ctx.reset()
    video = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        history_sizes=conf.training_config.history_sizes,
        latent_window_size=conf.training_config.latent_window_size[0],
        ref_short_validation_ctx=ref_ctx,
        scheduler_type=args.scheduler_type,
        use_cfg_zero_star=args.use_cfg_zero_star,
        use_zero_init=args.use_zero_init,
        zero_steps=args.zero_steps,
        output_type="np",
    ).frames[0]

    export_to_video(video, args.output_path, fps=24)
    print(f"Saved {args.output_path}")


if __name__ == "__main__":
    main()
