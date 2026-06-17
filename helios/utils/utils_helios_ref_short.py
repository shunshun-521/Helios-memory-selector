"""Ref-short helpers for Stage1 training and inference."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import torch

from helios.modules.ref_short_builder import (
    build_history_and_target_from_vae_latent,
    build_ref_short_tensors,
    merge_stage1_history_with_dataloader_tensors,
)
from helios.modules.selector_runtime import SelectorRuntime


logger = logging.getLogger(__name__)

_SELECTOR_RUNTIME: Optional[SelectorRuntime] = None


def set_selector_runtime(runtime: Optional[SelectorRuntime]) -> None:
    global _SELECTOR_RUNTIME
    _SELECTOR_RUNTIME = runtime


def get_selector_runtime() -> Optional[SelectorRuntime]:
    return _SELECTOR_RUNTIME


def init_patch_ref_trainable(transformer, args) -> None:
    if not hasattr(transformer, "patch_ref"):
        return
    if getattr(args.training_config, "is_train_full_patch_ref", False):
        for name, param in transformer.named_parameters():
            if "patch_ref" in name:
                param.requires_grad = True
    if getattr(args.training_config, "is_train_lora_patch_ref", False):
        for name, param in transformer.named_parameters():
            if "patch_ref" in name and "lora" in name:
                param.requires_grad = True


@torch.no_grad()
def build_selector_output_for_batch(
    *,
    vae_latent: torch.Tensor,
    choice_idx: int,
    prompt_raw: str,
    context_entry: Optional[dict],
    selector_runtime: SelectorRuntime,
) -> Tuple[Any, int]:
    (
        _target,
        _short,
        _mid,
        _long,
        indices,
        gap_latents,
        gap_indices,
    ) = build_history_and_target_from_vae_latent(
        vae_latent,
        int(choice_idx),
        min_chunk_distance=int(getattr(selector_runtime, "min_chunk_distance", 3)),
    )

    selector_output = selector_runtime.select_for_next_chunk(
        mode="train_offline",
        current_prompt=prompt_raw or "",
        current_chunk_idx=int(choice_idx),
        choice_idx=int(choice_idx),
        gap_latents=gap_latents,
        gap_chunk_indices=gap_indices,
        context_entry=context_entry,
    )
    t0 = int(indices["t0"])
    return selector_output, t0


def attach_ref_to_stage1_tensors(
    *,
    model_input,
    indices_hidden_states,
    indices_latents_history_short,
    indices_latents_history_mid,
    indices_latents_history_long,
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    vae_latent: torch.Tensor,
    choice_idx: int,
    prompt_raw: str,
    selector_runtime: SelectorRuntime,
    batch_size: int,
    device,
    dtype,
    decode_middle_frame_fn=None,
):
    """After prepare_stage1_clean_input_from_latents, build ref branch tensors."""
    # Expect per-sample latent chunks as (num_chunks, C, T, H, W).
    # Dataloader may still carry a leading batch dim in some paths.
    if vae_latent.ndim == 6:
        if vae_latent.shape[0] != 1:
            raise ValueError(
                f"ref-short expects per-rank batch_size=1 for vae_latent, got shape={tuple(vae_latent.shape)}"
            )
        vae_latent = vae_latent[0]
    elif vae_latent.ndim == 4:
        # Single chunk fallback: (C, T, H, W) -> (1, C, T, H, W)
        vae_latent = vae_latent.unsqueeze(0)
    elif vae_latent.ndim != 5:
        raise ValueError(f"Unexpected vae_latent shape for ref-short: {tuple(vae_latent.shape)}")

    num_chunks = int(vae_latent.shape[0])
    if not (0 <= int(choice_idx) < num_chunks):
        raise ValueError(f"choice_idx out of range: choice_idx={choice_idx}, num_chunks={num_chunks}")

    context_latent = None
    if int(choice_idx) > 0:
        context_latent = vae_latent[int(choice_idx) - 1 : int(choice_idx)]
    context_entry = None
    if context_latent is not None:
        context_entry = {
            "chunk_idx": int(choice_idx) - 1,
            "latent": context_latent.detach().cpu(),
            "decoded_frame": decode_middle_frame_fn(context_latent) if decode_middle_frame_fn else None,
            "clip_feat": None,
        }

    selector_output, t0 = build_selector_output_for_batch(
        vae_latent=vae_latent,
        choice_idx=int(choice_idx),
        prompt_raw=prompt_raw,
        context_entry=context_entry,
        selector_runtime=selector_runtime,
    )
    ref_tensors = build_ref_short_tensors(
        selector_output,
        t0=t0,
        target_slot_start=int(indices_hidden_states[0, 0].item()) if indices_hidden_states is not None else 0,
        batch_size=batch_size,
        device=device,
        dtype=dtype,
    )
    ref_slot_count = len(selector_output.ref_frame_indices)
    if ref_slot_count > 0:
        from helios.modules.ref_short_rope import shift_stage1_indices_by_ref_count

        (
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
        ) = shift_stage1_indices_by_ref_count(
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            ref_slot_count=ref_slot_count,
        )
    return (
        model_input,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        ref_tensors.latents_history_ref,
        ref_tensors.indices_latents_history_ref,
        selector_output,
    )
