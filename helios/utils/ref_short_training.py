"""Setup SelectorRuntime for ref-short Stage1 training."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Optional

import torch

from helios.modules.extract_feature import decode_middle_frame
from helios.modules.select_frames_vlm import VLMFrameSelector
from helios.modules.selector_runtime import SelectorRuntime
from helios.utils.utils_helios_ref_short import set_selector_runtime


logger = logging.getLogger(__name__)


def _weight_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return "fp32"


def build_vlm_selector_from_config(args, clip_model, device: str, dtype: torch.dtype) -> VLMFrameSelector:
    from helios.modules.vlm_backbones import load_default_backbone

    st = getattr(args.selector_training, "selector_type", "vlm")
    if st not in {"vlm", "vlm_zeroshot"}:
        raise ValueError(f"ref-short training requires selector_type vlm, got {st}")

    model_path = getattr(args.selector_training, "vlm_model_path", None)
    if not model_path:
        raise ValueError("selector_training.vlm_model_path is required for ref-short training")

    score_mode = getattr(args.selector_training, "vlm_score_mode", "yes_no")
    head_path = getattr(args.selector_training, "vlm_head_path", None)
    lora_path = getattr(args.selector_training, "vlm_lora_path", None)

    vlm_backbone = load_default_backbone(
        model_path=model_path,
        lora_path=lora_path,
        dtype=_weight_dtype_to_str(dtype),
        device=device,
        score_mode=score_mode,
        head_path=head_path,
    )

    return VLMFrameSelector(
        clip_model=clip_model,
        k=int(args.selector_training.vlm_k_select),
        power=float(getattr(args.selector_training, "vlm_power", 2.0)),
        min_chunk_distance=int(getattr(args.selector_training, "vlm_min_chunk_distance", 3)),
        max_candidates=int(getattr(args.selector_training, "vlm_max_candidates", 16)),
        prefilter_alpha=float(getattr(args.selector_training, "vlm_prefilter_alpha", 0.6)),
        temperature=float(getattr(args.selector_training, "vlm_temperature", 0.7)),
        vlm_backbone=vlm_backbone,
        device=device,
        fallback_to_random=True,
        vlm_rank_mode=str(getattr(args.selector_training, "vlm_rank_mode", "topk")).lower(),
    )


def setup_ref_short_training(
    args,
    *,
    clip_model=None,
    vae=None,
    latents_mean=None,
    latents_std=None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> SelectorRuntime:
    vlm_selector = build_vlm_selector_from_config(args, clip_model, device, dtype)

    decode_fn = None
    if vae is not None and latents_mean is not None and latents_std is not None:

        def decode_fn(latent):
            return decode_middle_frame(latent, vae, latents_mean, latents_std)

    runtime = SelectorRuntime(
        selector=vlm_selector,
        ref_frames_per_chunk=int(args.training_config.ref_frames_per_chunk),
        vlm_k_select=int(args.selector_training.vlm_k_select),
        selector_force_slow=bool(args.selector_training.selector_force_slow),
        enable_long_memory=bool(args.selector_training.enable_long_memory),
        min_chunk_distance=int(getattr(args.selector_training, "vlm_min_chunk_distance", 3)),
        decode_middle_frame_fn=decode_fn,
        debug_log=bool(getattr(args.selector_training, "debug_log", False)),
        debug_every=int(getattr(args.selector_training, "debug_every", 1)),
    )
    set_selector_runtime(runtime)
    logger.info(
        "Ref-short SelectorRuntime ready: k=%s ref_frames_per_chunk=%s force_slow=%s min_chunk_distance=%s",
        args.selector_training.vlm_k_select,
        args.training_config.ref_frames_per_chunk,
        args.selector_training.selector_force_slow,
        runtime.min_chunk_distance,
    )
    return runtime


def attach_ref_short_to_stage1_batch(
    args,
    *,
    batch,
    model_input,
    indices_hidden_states,
    indices_latents_history_short,
    indices_latents_history_mid,
    indices_latents_history_long,
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    weight_dtype,
    device,
    vae=None,
    latents_mean=None,
    latents_std=None,
):
    from helios.utils.utils_helios_ref_short import attach_ref_to_stage1_tensors, get_selector_runtime

    runtime = get_selector_runtime()
    if runtime is None:
        raise RuntimeError("SelectorRuntime not initialized; call setup_ref_short_training first")

    decode_fn = None
    if vae is not None and latents_mean is not None and latents_std is not None:

        def decode_fn(latent):
            return decode_middle_frame(latent, vae, latents_mean, latents_std)

    bsz = model_input.shape[0]
    if bsz != 1:
        raise NotImplementedError("ref-short training currently supports batch_size=1 per GPU")

    vae_latent = batch["vae_latent"][0] if isinstance(batch["vae_latent"], torch.Tensor) else batch["vae_latent"][0]
    choice_idx = int(batch["choice_idx"][0] if isinstance(batch["choice_idx"], torch.Tensor) else batch["choice_idx"][0])
    prompt_raw = ""
    if "prompt_raws" in batch:
        pr = batch["prompt_raws"]
        prompt_raw = pr[0] if isinstance(pr, list) else str(pr)
    elif "prompt_raw" in batch:
        prompt_raw = batch["prompt_raw"]

    return attach_ref_to_stage1_tensors(
        model_input=model_input,
        indices_hidden_states=indices_hidden_states,
        indices_latents_history_short=indices_latents_history_short,
        indices_latents_history_mid=indices_latents_history_mid,
        indices_latents_history_long=indices_latents_history_long,
        latents_history_short=latents_history_short,
        latents_history_mid=latents_history_mid,
        latents_history_long=latents_history_long,
        vae_latent=vae_latent,
        choice_idx=choice_idx,
        prompt_raw=prompt_raw,
        selector_runtime=runtime,
        batch_size=bsz,
        device=device,
        dtype=weight_dtype,
        decode_middle_frame_fn=decode_fn,
    )
