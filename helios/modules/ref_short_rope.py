"""Ref-short RoPE slot remapping: x0=0, ref=1..N, history+target shifted +N."""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

# Maximum reserved ref slots immediately after x0.
REF_SLOT_BASE = 1
REF_SLOT_COUNT = 6
REF_SLOT_MAX = REF_SLOT_BASE + REF_SLOT_COUNT - 1  # 6

# Legacy (non-ref-short) target start: 1 + 16 + 2 + 1 = 20
LEGACY_TARGET_SLOT_START = 20
# Ref-short maximum target start after +6 shift: 26.
REF_SHORT_TARGET_SLOT_START = LEGACY_TARGET_SLOT_START + REF_SLOT_COUNT


def apply_ref_short_history_target_shift(indices: torch.Tensor, ref_slot_count: int = REF_SLOT_COUNT) -> torch.Tensor:
    """Shift history+target RoPE slots by the actual ref count; x0 (slot 0) unchanged."""
    ref_slot_count = int(ref_slot_count)
    if ref_slot_count <= 0:
        return indices.clone()
    if ref_slot_count > REF_SLOT_COUNT:
        raise ValueError(
            f"ref_slot_count {ref_slot_count} exceeds reserved maximum {REF_SLOT_COUNT}; "
            "reduce vlm_k_select or ref_frames_per_chunk"
        )
    out = indices.clone()
    if out.ndim == 1:
        out[1:] = out[1:] + ref_slot_count
    else:
        out[:, 1:] = out[:, 1:] + ref_slot_count
    return out


def shift_stage1_indices_by_ref_count(
    *,
    indices_hidden_states: torch.Tensor,
    indices_latents_history_short: torch.Tensor,
    indices_latents_history_mid: torch.Tensor,
    indices_latents_history_long: torch.Tensor,
    ref_slot_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shift split Stage1 indices after ref selection.

    `indices_latents_history_short` includes x0 at position 0, so only its
    non-x0 entries are shifted.
    """
    ref_slot_count = int(ref_slot_count)
    if ref_slot_count <= 0:
        return (
            indices_hidden_states,
            indices_latents_history_short,
            indices_latents_history_mid,
            indices_latents_history_long,
        )
    if ref_slot_count > REF_SLOT_COUNT:
        raise ValueError(
            f"ref_slot_count {ref_slot_count} exceeds reserved maximum {REF_SLOT_COUNT}; "
            "reduce vlm_k_select or ref_frames_per_chunk"
        )

    indices_hidden_states = indices_hidden_states + ref_slot_count
    indices_latents_history_mid = indices_latents_history_mid + ref_slot_count
    indices_latents_history_long = indices_latents_history_long + ref_slot_count
    indices_latents_history_short = indices_latents_history_short.clone()
    if indices_latents_history_short.ndim == 1:
        indices_latents_history_short[1:] = indices_latents_history_short[1:] + ref_slot_count
    else:
        indices_latents_history_short[:, 1:] = indices_latents_history_short[:, 1:] + ref_slot_count
    return (
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
    )


def build_ref_rope_indices_remapped(
    ref_frame_indices_global: Sequence[int],
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Assign ref frames fixed slots REF_SLOT_BASE.. in global-time order."""
    if not ref_frame_indices_global:
        return None
    n = len(ref_frame_indices_global)
    if n > REF_SLOT_COUNT:
        raise ValueError(
            f"ref frame count {n} exceeds reserved slots {REF_SLOT_COUNT}; "
            "reduce vlm_k_select or ref_frames_per_chunk"
        )
    order = sorted(range(n), key=lambda i: int(ref_frame_indices_global[i]))
    slots: List[int] = [0] * n
    for rank, src_i in enumerate(order):
        slots[src_i] = REF_SLOT_BASE + rank
    idx = torch.tensor(slots, dtype=torch.long, device=device)
    return idx.unsqueeze(0).expand(batch_size, -1)
