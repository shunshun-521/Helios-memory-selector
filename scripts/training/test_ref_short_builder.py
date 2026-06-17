#!/usr/bin/env python3
"""Unit tests for ref_short_builder (no GPU / VLM required)."""

import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/Helios")

from helios.modules.ref_short_builder import (
    build_history_and_target_from_vae_latent,
    build_ref_rope_indices,
    build_ref_short_tensors,
    extract_ref_latents_from_chunks,
    pack_ref_latents,
)
from helios.modules.ref_short_gap import is_gap_chunk_eligible
from helios.modules.ref_short_rope import (
    REF_SHORT_TARGET_SLOT_START,
    REF_SLOT_BASE,
    REF_SLOT_MAX,
    apply_ref_short_history_target_shift,
    build_ref_rope_indices_remapped,
)
from helios.modules.selector_runtime import SelectorOutput, evenly_spaced_latent_frame_indices


def test_evenly_spaced_indices():
    idx = evenly_spaced_latent_frame_indices(9, 3)
    assert idx == [0, 4, 8], f"got {idx}"


def test_extract_ref_latents():
    chunk = torch.randn(1, 16, 9, 4, 4)
    refs, gidx = extract_ref_latents_from_chunks([chunk], [2], ref_frames_per_chunk=3)
    assert len(refs) == 3
    assert gidx == [18, 22, 26]


def test_ref_rope_remapped_slots():
    idx = build_ref_rope_indices_remapped([18, 22, 26])
    assert idx.tolist() == [[1, 2, 3]]
    idx2 = build_ref_rope_indices_remapped([26, 18, 22])
    assert idx2.tolist() == [[3, 1, 2]]
    assert (idx >= REF_SLOT_BASE).all() and (idx <= REF_SLOT_MAX).all()
    legacy = build_ref_rope_indices([18, 22, 26], t0=27)
    assert legacy.tolist() == [[1, 2, 3]]


def test_history_target_shift():
    legacy = torch.arange(0, 29)
    shifted = apply_ref_short_history_target_shift(legacy)
    assert shifted[0].item() == 0
    assert shifted[1:17].tolist() == list(range(7, 23))
    assert shifted[20].item() == REF_SHORT_TARGET_SLOT_START


def test_gap_chunk_eligibility():
    assert not is_gap_chunk_eligible(0, 2, min_chunk_distance=3)
    assert is_gap_chunk_eligible(0, 3, min_chunk_distance=3)
    assert not is_gap_chunk_eligible(1, 3, min_chunk_distance=3)
    assert is_gap_chunk_eligible(0, 4, min_chunk_distance=3)
    assert is_gap_chunk_eligible(1, 4, min_chunk_distance=3)


def test_builder_end_to_end():
    num_chunks = 5
    vae_latent = torch.randn(num_chunks, 16, 9, 8, 8)
    choice_idx = 3
    _, _, _, _, indices, gap_latents, gap_idx = build_history_and_target_from_vae_latent(vae_latent, choice_idx)
    assert len(gap_latents) == 1
    assert gap_idx == [0]
    sel = SelectorOutput(
        ref_latents=[],
        ref_chunk_indices=[],
        ref_frame_indices=[],
        schedule_mode="test",
    )
    refs, fidx = extract_ref_latents_from_chunks(gap_latents[:1], gap_idx[:1], ref_frames_per_chunk=3)
    sel.ref_latents = refs
    sel.ref_chunk_indices = gap_idx[:1]
    sel.ref_frame_indices = fidx
    out = build_ref_short_tensors(sel, t0=indices["t0"], batch_size=1)
    assert out.latents_history_ref is not None
    assert out.latents_history_ref.shape[2] == 3
    assert out.indices_latents_history_ref.shape == (1, 3)
    assert out.indices_latents_history_ref.tolist() == [[1, 2, 3]]
    assert (out.indices_latents_history_ref <= REF_SLOT_MAX).all()
    assert (out.indices_latents_history_ref >= REF_SLOT_BASE).all()


def main():
    test_evenly_spaced_indices()
    test_extract_ref_latents()
    test_ref_rope_remapped_slots()
    test_history_target_shift()
    test_gap_chunk_eligibility()
    test_builder_end_to_end()
    print("test_ref_short_builder: ALL PASSED")


if __name__ == "__main__":
    main()
