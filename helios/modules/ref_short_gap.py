"""Shared GAP chunk eligibility rules for ref-short train/infer parity."""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch

DEFAULT_MIN_CHUNK_DISTANCE = 3


def is_gap_chunk_eligible(
    chunk_idx: int,
    current_chunk_idx: int,
    *,
    min_chunk_distance: int = DEFAULT_MIN_CHUNK_DISTANCE,
) -> bool:
    """Match select_frames_vlm._get_gap_candidates / select_gap_frames rules.

    - Exclude context chunk (current_chunk_idx - 1).
    - Require (current_chunk_idx - chunk_idx) >= min_chunk_distance.
    """
    cur = int(current_chunk_idx)
    ci = int(chunk_idx)
    if cur < 2:
        return False
    if ci >= cur - 1:
        return False
    return (cur - ci) >= int(min_chunk_distance)


def filter_gap_chunk_pairs(
    gap_latents: Sequence[torch.Tensor],
    gap_chunk_indices: Sequence[int],
    current_chunk_idx: int,
    *,
    min_chunk_distance: int = DEFAULT_MIN_CHUNK_DISTANCE,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Filter offline GAP latents to the same candidate set as streaming inference."""
    filtered_latents: List[torch.Tensor] = []
    filtered_indices: List[int] = []
    for gl, ci in zip(gap_latents, gap_chunk_indices):
        if is_gap_chunk_eligible(ci, current_chunk_idx, min_chunk_distance=min_chunk_distance):
            filtered_latents.append(gl)
            filtered_indices.append(int(ci))
    return filtered_latents, filtered_indices
