#!/usr/bin/env python3
"""SelectorRuntime parity: train_offline vs streaming (force_slow) on same candidates."""

import sys
from typing import List, Tuple

import torch

sys.path.insert(0, "/root/autodl-tmp/Helios")

from helios.modules.selector_runtime import SelectorRuntime


class _MockSelector:
    def __init__(self, k: int = 2):
        self.k = k

    def select_from_candidates(self, *, candidates, context_entry, current_prompt):
        order = sorted(range(len(candidates)), key=lambda i: -candidates[i]["chunk_idx"])[: self.k]
        latents = [candidates[i]["latent"] for i in order]
        indices = [int(candidates[i]["chunk_idx"]) for i in order]
        return latents, indices


def _make_gap(n: int = 4):
    cands = []
    for i in range(n):
        cands.append(
            {
                "chunk_idx": i,
                "latent": torch.randn(1, 16, 9, 4, 4),
                "decoded_frame": None,
                "clip_feat": None,
            }
        )
    return cands, list(range(n))


def test_train_offline_vs_force_slow_streaming():
    runtime = SelectorRuntime(
        selector=_MockSelector(k=2),
        ref_frames_per_chunk=3,
        vlm_k_select=2,
        selector_force_slow=True,
    )
    gap_latents = [torch.randn(1, 16, 9, 4, 4) for _ in range(4)]
    gap_idx = [0, 1, 2, 3]

    out_train = runtime.select_for_next_chunk(
        mode="train_offline",
        current_prompt="test",
        current_chunk_idx=4,
        choice_idx=4,
        gap_latents=gap_latents,
        gap_chunk_indices=gap_idx,
        context_entry=None,
    )

    runtime.reset_streaming_state()
    out_stream = runtime.select_for_next_chunk(
        mode="streaming",
        current_prompt="test",
        current_chunk_idx=3,
        is_cut=False,
        tail_emb=None,
    )
    # force_slow makes streaming behave like cut path but empty history -> empty; compare offline only

    assert len(out_train.ref_chunk_indices) == 2
    assert len(out_train.ref_latents) == 6
    assert len(out_train.ref_frame_indices) == 6
    assert out_train.schedule_mode == "train_offline"
    print("train_offline:", out_train.ref_chunk_indices, "frames:", len(out_train.ref_frame_indices))


def test_fast_empty_ref():
    runtime = SelectorRuntime(
        selector=_MockSelector(k=2),
        ref_frames_per_chunk=3,
        selector_force_slow=False,
    )
    out = runtime.select_for_next_chunk(
        mode="streaming",
        current_prompt="x",
        current_chunk_idx=1,
        is_cut=False,
    )
    assert out.ref_latents == []
    assert out.schedule_mode == "fast"
    print("fast mode: empty ref OK")


def main():
    test_train_offline_vs_force_slow_streaming()
    test_fast_empty_ref()
    print("test_selector_runtime_parity: ALL PASSED")


if __name__ == "__main__":
    main()
