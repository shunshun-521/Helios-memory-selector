"""Build ref-short tensors and RoPE indices for Helios Stage1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
from einops import rearrange

from helios.modules.ref_short_gap import (
    DEFAULT_MIN_CHUNK_DISTANCE,
    filter_gap_chunk_pairs,
    is_gap_chunk_eligible,
)
from helios.modules.ref_short_rope import build_ref_rope_indices_remapped
from helios.modules.selector_runtime import SelectorOutput, evenly_spaced_latent_frame_indices


HISTORY_WINDOW_SIZE = 19
LATENT_FRAMES_PER_CHUNK = 9


@dataclass
class RefShortBatchTensors:
    latents_history_ref: Optional[torch.Tensor]
    indices_latents_history_ref: Optional[torch.Tensor]
    ref_frame_indices_global: List[int]
    ref_chunk_indices: List[int]


def build_history_and_target_from_vae_latent(
    vae_latent: torch.Tensor,
    choice_idx: int,
    *,
    history_window_size: int = HISTORY_WINDOW_SIZE,
    latent_frames_per_chunk: int = LATENT_FRAMES_PER_CHUNK,
    min_chunk_distance: int = DEFAULT_MIN_CHUNK_DISTANCE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict, List[torch.Tensor], List[int]]:
    """Mirror train_helios_bolt.build_history_and_target for offline selector input."""
    num_chunks, c, t, h, w = vae_latent.shape
    continue_latent = rearrange(vae_latent, "b c t h w -> c (b t) h w")
    zero_pad = torch.zeros(c, history_window_size, h, w, device=continue_latent.device, dtype=continue_latent.dtype)
    padded = torch.cat([zero_pad, continue_latent], dim=1)

    target_start = history_window_size + choice_idx * t
    target_latent = padded[:, target_start : target_start + t].unsqueeze(0)

    hist_start = target_start - history_window_size
    history = padded[:, hist_start:target_start]
    history_long = history[:, :16].unsqueeze(0)
    history_mid = history[:, 16:18].unsqueeze(0)
    history_short = history[:, 18:19].unsqueeze(0)

    x0 = padded[:, history_window_size : history_window_size + 1].unsqueeze(0)
    history_short = torch.cat([x0, history_short], dim=2)

    target_frame_start = choice_idx * t
    indices = {
        "target": torch.arange(target_frame_start, target_frame_start + t),
        "short": torch.tensor([0, max(0, target_frame_start - 1)]),
        "mid": torch.tensor([max(0, target_frame_start - 3), max(0, target_frame_start - 2)]),
        "long": _pad_long_indices(target_frame_start),
        "t0": target_frame_start,
    }

    gap_latents: List[torch.Tensor] = []
    gap_indices: List[int] = []
    for i in range(max(0, choice_idx - 1)):
        if is_gap_chunk_eligible(i, choice_idx, min_chunk_distance=min_chunk_distance):
            gap_latents.append(vae_latent[i : i + 1])
            gap_indices.append(i)

    return (
        target_latent,
        history_short,
        history_mid,
        history_long,
        indices,
        gap_latents,
        gap_indices,
    )


def _pad_long_indices(target_frame_start: int) -> torch.Tensor:
    idx_long = list(range(max(0, target_frame_start - 16), target_frame_start))
    while len(idx_long) < 16:
        idx_long.insert(0, 0)
    return torch.tensor(idx_long)


def extract_ref_latents_from_chunks(
    selected_chunk_latents: Sequence[torch.Tensor],
    selected_chunk_indices: Sequence[int],
    *,
    ref_frames_per_chunk: int = 3,
    latent_frames_per_chunk: int = LATENT_FRAMES_PER_CHUNK,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Per selected chunk, take evenly-spaced latent frames (design §3.3)."""
    local_idx = evenly_spaced_latent_frame_indices(latent_frames_per_chunk, ref_frames_per_chunk)
    ref_latents: List[torch.Tensor] = []
    ref_frame_indices: List[int] = []
    for chunk_latent, chunk_idx in zip(selected_chunk_latents, selected_chunk_indices):
        if chunk_latent.ndim == 4:
            chunk_latent = chunk_latent.unsqueeze(0)
        for li in local_idx:
            ref_latents.append(chunk_latent[:, :, li : li + 1, :, :].contiguous())
            ref_frame_indices.append(int(chunk_idx) * latent_frames_per_chunk + int(li))
    return ref_latents, ref_frame_indices


def pack_ref_latents(ref_latents: List[torch.Tensor], device=None, dtype=None) -> Optional[torch.Tensor]:
    if not ref_latents:
        return None
    stacked = torch.cat(ref_latents, dim=2)
    if device is not None:
        stacked = stacked.to(device=device)
    if dtype is not None:
        stacked = stacked.to(dtype=dtype)
    return stacked


def build_ref_rope_indices_for_pipeline(
    ref_frame_indices_global: Sequence[int],
    chunk_idx: int = 0,
    *,
    latent_window_size: int = LATENT_FRAMES_PER_CHUNK,
    history_sizes: Sequence[int] = (16, 2, 1),
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Assign ref frames to reserved RoPE slots 1..6 (global-time order)."""
    del chunk_idx, latent_window_size, history_sizes  # legacy args kept for call-site compat
    return build_ref_rope_indices_remapped(
        ref_frame_indices_global,
        batch_size=batch_size,
        device=device,
    )


def build_ref_rope_indices(
    ref_frame_indices_global: Sequence[int],
    t0: int = 0,
    target_slot_start: int = 0,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
) -> Optional[torch.Tensor]:
    """Assign ref frames to reserved RoPE slots 1..6 (global-time order)."""
    del t0, target_slot_start  # legacy args kept for call-site compat
    return build_ref_rope_indices_remapped(
        ref_frame_indices_global,
        batch_size=batch_size,
        device=device,
    )


def build_ref_short_tensors(
    selector_output: SelectorOutput,
    *,
    t0: int,
    target_slot_start: int = 0,
    batch_size: int = 1,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> RefShortBatchTensors:
    latents_history_ref = pack_ref_latents(selector_output.ref_latents, device=device, dtype=dtype)
    indices_latents_history_ref = build_ref_rope_indices(
        selector_output.ref_frame_indices,
        t0=t0,
        target_slot_start=target_slot_start,
        batch_size=batch_size,
        device=device,
    )
    return RefShortBatchTensors(
        latents_history_ref=latents_history_ref,
        indices_latents_history_ref=indices_latents_history_ref,
        ref_frame_indices_global=list(selector_output.ref_frame_indices),
        ref_chunk_indices=list(selector_output.ref_chunk_indices),
    )


def merge_stage1_history_with_dataloader_tensors(
  history_latents: torch.Tensor,
  target_latents: torch.Tensor,
  x0_latents: torch.Tensor,
  choice_idx: int,
  latent_window_size: int = LATENT_FRAMES_PER_CHUNK,
) -> int:
    """Return global frame index t0 for current target window."""
    return int(choice_idx) * int(latent_window_size)
