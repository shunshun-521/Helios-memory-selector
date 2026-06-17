#!/usr/bin/env python3
"""
Merge a ref-short LoRA checkpoint into Helios-Base transformer weights.

This follows the same pattern as tools/merge_lora_base.py, but is parameterized:
- load a clean transformer init (Wan2.1 Diffusers transformer)
- load Helios-Base pipeline with that transformer
- load LoRA safetensors + transformer_partial.pth from a checkpoint folder
- fuse LoRA and save merged transformer to <out_dir>/transformer
"""

from __future__ import annotations

import os
import sys
from argparse import ArgumentParser, Namespace
from pathlib import Path

import torch


def _resolve_ckpt_files(ckpt_dir: Path, use_ema: bool) -> tuple[Path, Path]:
    base = ckpt_dir / "model_ema" if use_ema else ckpt_dir
    lora = base / "pytorch_lora_weights.safetensors"
    partial = base / "transformer_partial.pth"
    if not lora.exists():
        raise FileNotFoundError(f"LoRA weights not found: {lora}")
    if not partial.exists():
        raise FileNotFoundError(f"transformer_partial.pth not found: {partial}")
    return lora, partial


def _print_patch_ref_merge_check(transformer, partial_path: Path) -> None:
    """Print whether patch_ref in transformer matches partial checkpoint."""
    partial_sd = torch.load(str(partial_path), map_location="cpu")
    partial_w = partial_sd.get("patch_ref.weight")
    partial_b = partial_sd.get("patch_ref.bias")
    has_patch_ref_module = hasattr(transformer, "patch_ref")
    print(f"[ref-short-check] transformer has patch_ref module: {has_patch_ref_module}")
    print(f"[ref-short-check] partial has patch_ref.weight: {partial_w is not None}")
    print(f"[ref-short-check] partial has patch_ref.bias: {partial_b is not None}")

    if not has_patch_ref_module or partial_w is None or partial_b is None:
        print("[ref-short-check] SKIP compare due to missing module/keys.")
        return

    merged_w = transformer.patch_ref.weight.detach().cpu()
    merged_b = transformer.patch_ref.bias.detach().cpu()
    weight_match = torch.allclose(merged_w, partial_w)
    bias_match = torch.allclose(merged_b, partial_b)
    print(f"[ref-short-check] patch_ref.weight matches partial: {weight_match}")
    print(f"[ref-short-check] patch_ref.bias matches partial: {bias_match}")
    print(f"[ref-short-check] patch_ref merged_ok: {bool(weight_match and bias_match)}")


def main() -> None:
    p = ArgumentParser()
    p.add_argument("--base_model_path", type=str, required=True)
    p.add_argument("--ckpt_dir", type=str, required=True, help="e.g. /root/autodl-fs/output/.../checkpoint-2000")
    p.add_argument("--out_dir", type=str, default=None, help="default: <ckpt_dir>/merged")
    p.add_argument(
        "--wan_transformer_init_path",
        type=str,
        default="/root/autodl-fs/Wan-AI/Wan2.1-T2V-14B-Diffusers",
        help="Transformer init source (diffusers format)",
    )
    p.add_argument("--use_ema", action="store_true", help="use ckpt_dir/model_ema/* if present")
    p.add_argument("--adapter_weight", type=float, default=1.0)
    args = p.parse_args()

    helios_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(helios_root))

    from helios.modules.transformer_helios import HeliosTransformer3DModel
    from helios.pipelines.pipeline_helios import HeliosPipeline
    from helios.utils.utils_base import load_extra_components

    ckpt_dir = Path(args.ckpt_dir)
    out_dir = Path(args.out_dir) if args.out_dir else (ckpt_dir / "merged")
    out_transformer_dir = out_dir / "transformer"
    out_transformer_dir.mkdir(parents=True, exist_ok=True)

    lora_path, partial_path = _resolve_ckpt_files(ckpt_dir, bool(args.use_ema))

    transformer_additional_kwargs = {
        "has_multi_term_memory_patch": True,
        "zero_history_timestep": True,
        "guidance_cross_attn": True,
        "restrict_self_attn": False,
        "is_train_restrict_lora": False,
        "restrict_lora": False,
        "restrict_lora_rank": 128,
    }

    # Make merge deterministic & CPU-friendly by default (same as merge_lora_base.py).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    torch.set_default_device("cpu")

    transformer = HeliosTransformer3DModel.from_pretrained(
        args.wan_transformer_init_path,
        subfolder="transformer",
        transformer_additional_kwargs=transformer_additional_kwargs,
    )
    pipe = HeliosPipeline.from_pretrained(
        args.base_model_path,
        transformer=transformer,
    )

    pipe.load_lora_weights(str(lora_path), adapter_name="default")
    pipe.set_adapters(["default"], adapter_weights=[float(args.adapter_weight)])

    cfg = Namespace()
    if not hasattr(cfg, "training_config"):
        cfg.training_config = Namespace()
    cfg.training_config.is_enable_stage1 = True
    # Important: enable ref-short component loading from transformer_partial.pth.
    # Without this flag, load_extra_components() will skip patch_ref and the merged
    # transformer keeps patch_ref as an untrained init copy.
    cfg.training_config.use_ref_short = True
    cfg.training_config.restrict_self_attn = False
    cfg.training_config.is_amplify_history = False
    cfg.training_config.is_use_gan = False

    load_extra_components(cfg, transformer, str(partial_path))
    _print_patch_ref_merge_check(transformer, partial_path)

    pipe.fuse_lora()
    pipe.unload_lora_weights()
    pipe.transformer.save_pretrained(str(out_transformer_dir))

    print(str(out_transformer_dir))


if __name__ == "__main__":
    main()

