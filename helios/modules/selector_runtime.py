"""Unified selector runtime for ref-short train/infer parity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import logging

from helios.modules.long_memory import build_codebook_candidates_for_vlm
from helios.modules.ref_short_gap import DEFAULT_MIN_CHUNK_DISTANCE, filter_gap_chunk_pairs


LATENT_FRAMES_PER_CHUNK = 9
logger = logging.getLogger(__name__)


@dataclass
class SelectorOutput:
    ref_latents: List[torch.Tensor]
    ref_chunk_indices: List[int]
    ref_frame_indices: List[int]
    schedule_mode: str
    meta: Dict[str, Any] = field(default_factory=dict)


def evenly_spaced_latent_frame_indices(
    latent_frames_per_chunk: int = LATENT_FRAMES_PER_CHUNK,
    num_frames: int = 3,
) -> List[int]:
    if num_frames <= 1:
        return [latent_frames_per_chunk // 2]
    idx = np.linspace(0, latent_frames_per_chunk - 1, num_frames)
    return [int(round(x)) for x in idx]


class SelectorRuntime:
    """Single entry for ref chunk selection (train offline + infer streaming)."""

    def __init__(
        self,
        *,
        selector: Any,
        ref_frames_per_chunk: int = 3,
        vlm_k_select: int = 2,
        selector_force_slow: bool = False,
        enable_long_memory: bool = False,
        long_memory: Any = None,
        lm_tau_cut: float = 0.35,
        lm_codebook_topm: int = 16,
        latent_frames_per_chunk: int = LATENT_FRAMES_PER_CHUNK,
        min_chunk_distance: int = DEFAULT_MIN_CHUNK_DISTANCE,
        decode_middle_frame_fn: Optional[Callable] = None,
        cosine_similarity_fn: Optional[Callable] = None,
        debug_log: bool = False,
        debug_every: int = 1,
    ) -> None:
        self.selector = selector
        self.ref_frames_per_chunk = ref_frames_per_chunk
        self.vlm_k_select = vlm_k_select
        self.selector_force_slow = selector_force_slow
        self.enable_long_memory = enable_long_memory
        self.long_memory = long_memory
        self.lm_tau_cut = lm_tau_cut
        self.lm_codebook_topm = lm_codebook_topm
        self.latent_frames_per_chunk = latent_frames_per_chunk
        self.min_chunk_distance = int(min_chunk_distance)
        self.decode_middle_frame_fn = decode_middle_frame_fn
        self.cosine_similarity_fn = cosine_similarity_fn
        self._prev_tail_emb: Optional[torch.Tensor] = None
        self.debug_log = bool(debug_log)
        self.debug_every = max(1, int(debug_every))
        self._select_calls = 0

    def select_for_next_chunk(
        self,
        *,
        mode: str,
        current_prompt: str,
        current_chunk_idx: int,
        choice_idx: Optional[int] = None,
        gap_latents: Optional[Sequence[torch.Tensor]] = None,
        gap_chunk_indices: Optional[Sequence[int]] = None,
        context_entry: Optional[Dict[str, Any]] = None,
        history_ref: Optional[List[Dict[str, Any]]] = None,
        tail_emb: Optional[torch.Tensor] = None,
        is_cut: Optional[bool] = None,
    ) -> SelectorOutput:
        if mode == "train_offline":
            return self._select_train_offline(
                gap_latents=gap_latents or [],
                gap_chunk_indices=gap_chunk_indices or [],
                context_entry=context_entry,
                current_prompt=current_prompt,
                choice_idx=choice_idx,
            )
        return self._select_streaming(
            history_ref=history_ref or [],
            current_chunk_idx=current_chunk_idx,
            current_prompt=current_prompt,
            context_entry=context_entry,
            tail_emb=tail_emb,
            is_cut=is_cut,
        )

    def _select_train_offline(
        self,
        *,
        gap_latents: Sequence[torch.Tensor],
        gap_chunk_indices: Sequence[int],
        context_entry: Optional[Dict[str, Any]],
        current_prompt: str,
        choice_idx: Optional[int],
    ) -> SelectorOutput:
        current_chunk_idx = int(choice_idx if choice_idx is not None else -1)
        gap_latents, gap_chunk_indices = filter_gap_chunk_pairs(
            gap_latents,
            gap_chunk_indices,
            current_chunk_idx,
            min_chunk_distance=self.min_chunk_distance,
        )

        if not gap_latents or self.selector is None:
            out = SelectorOutput([], [], [], "train_offline_empty", {"choice_idx": choice_idx})
            self._log_selection(
                mode=out.schedule_mode,
                current_chunk_idx=choice_idx if choice_idx is not None else -1,
                vlm_called=False,
                selected_chunk_indices=[],
                ref_frame_indices=[],
                extra={"num_candidates": 0, "reason": "empty_gap_or_no_selector"},
            )
            return out

        candidates = []
        for gl, abs_idx in zip(gap_latents, gap_chunk_indices):
            decoded = None
            if self.decode_middle_frame_fn is not None:
                decoded = self.decode_middle_frame_fn(gl)
            candidates.append(
                {
                    "chunk_idx": int(abs_idx),
                    "latent": gl.detach().cpu() if isinstance(gl, torch.Tensor) else gl,
                    "decoded_frame": decoded,
                    "clip_feat": None,
                }
            )

        # Training default: every step has ref (selector_force_slow=True per design §11).
        selected_latents, selected_indices = self.selector.select_from_candidates(
            candidates=candidates,
            context_entry=context_entry,
            current_prompt=current_prompt,
        )
        schedule_mode = "train_offline"

        ref_latents, ref_frame_indices = self._expand_chunks_to_ref_frames(selected_latents, selected_indices)
        out = SelectorOutput(
            ref_latents=ref_latents,
            ref_chunk_indices=[int(x) for x in selected_indices],
            ref_frame_indices=ref_frame_indices,
            schedule_mode=schedule_mode,
            meta={"choice_idx": choice_idx, "num_candidates": len(candidates)},
        )
        self._log_selection(
            mode=out.schedule_mode,
            current_chunk_idx=choice_idx if choice_idx is not None else -1,
            vlm_called=True,
            selected_chunk_indices=out.ref_chunk_indices,
            ref_frame_indices=out.ref_frame_indices,
            extra={"num_candidates": len(candidates)},
        )
        return out

    def _select_streaming(
        self,
        *,
        history_ref: List[Dict[str, Any]],
        current_chunk_idx: int,
        current_prompt: str,
        context_entry: Optional[Dict[str, Any]],
        tail_emb: Optional[torch.Tensor],
        is_cut: Optional[bool],
    ) -> SelectorOutput:
        if self.selector_force_slow:
            is_cut = True

        if is_cut is None and tail_emb is not None and self._prev_tail_emb is not None and self.cosine_similarity_fn:
            sim = float(self.cosine_similarity_fn(self._prev_tail_emb, tail_emb).item())
            is_cut = (1.0 - sim) >= float(self.lm_tau_cut)
        elif is_cut is None:
            is_cut = False

        if tail_emb is not None:
            self._prev_tail_emb = tail_emb.detach()

        selected_latents: List[torch.Tensor] = []
        selected_indices: List[int] = []
        schedule_mode = "fast"
        vlm_called = False
        num_candidates = 0

        if is_cut and self.enable_long_memory and self.long_memory is not None and tail_emb is not None:
            candidates = build_codebook_candidates_for_vlm(
                store=self.long_memory,
                query_embedding=tail_emb,
                topm=self.lm_codebook_topm,
                exclude_chunk_idx=[int(current_chunk_idx)],
            )
            num_candidates = len(candidates)
            if self.decode_middle_frame_fn is not None:
                for c in candidates:
                    if c.get("decoded_frame") is None:
                        c["decoded_frame"] = self.decode_middle_frame_fn(c["latent"])
            selected_latents, selected_indices = self.selector.select_from_candidates(
                candidates=candidates,
                context_entry=context_entry,
                current_prompt=current_prompt,
            )
            vlm_called = True
            schedule_mode = "slow/longmem"
        elif is_cut and history_ref:
            selected_latents, selected_indices = self.selector(
                history_ref,
                current_chunk_idx + 1,
                current_prompt,
            )
            vlm_called = True
            num_candidates = len(history_ref)
            schedule_mode = "slow/history"

        ref_latents, ref_frame_indices = self._expand_chunks_to_ref_frames(selected_latents, selected_indices)
        out = SelectorOutput(
            ref_latents=ref_latents,
            ref_chunk_indices=[int(x) for x in selected_indices],
            ref_frame_indices=ref_frame_indices,
            schedule_mode=schedule_mode,
            meta={"current_chunk_idx": current_chunk_idx, "is_cut": is_cut},
        )
        longmem_size = len(self.long_memory) if self.long_memory is not None else 0
        self._log_selection(
            mode=out.schedule_mode,
            current_chunk_idx=current_chunk_idx,
            vlm_called=vlm_called,
            selected_chunk_indices=out.ref_chunk_indices,
            ref_frame_indices=out.ref_frame_indices,
            extra={
                "is_cut": bool(is_cut),
                "force_slow": self.selector_force_slow,
                "enable_long_memory": self.enable_long_memory,
                "longmem_size": longmem_size,
                "num_candidates": num_candidates,
            },
        )
        return out

    def _expand_chunks_to_ref_frames(
        self,
        selected_latents: Sequence[torch.Tensor],
        selected_indices: Sequence[int],
    ) -> Tuple[List[torch.Tensor], List[int]]:
        local_idx = evenly_spaced_latent_frame_indices(self.latent_frames_per_chunk, self.ref_frames_per_chunk)
        ref_latents: List[torch.Tensor] = []
        ref_frame_indices: List[int] = []
        for chunk_latent, chunk_idx in zip(selected_latents, selected_indices):
            if chunk_latent.ndim == 4:
                chunk_latent = chunk_latent.unsqueeze(0)
            for li in local_idx:
                ref_latents.append(chunk_latent[:, :, li : li + 1, :, :].contiguous())
                ref_frame_indices.append(int(chunk_idx) * self.latent_frames_per_chunk + int(li))
        return ref_latents, ref_frame_indices

    def reset_streaming_state(self) -> None:
        self._prev_tail_emb = None

    def _should_log(self) -> bool:
        self._select_calls += 1
        return self.debug_log and (self._select_calls % self.debug_every == 0)

    def _log_selection(
        self,
        *,
        mode: str,
        current_chunk_idx: int,
        vlm_called: bool,
        selected_chunk_indices: Sequence[int],
        ref_frame_indices: Sequence[int],
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._should_log():
            return
        extra = extra or {}
        if mode == "fast":
            slow_fast = "fast"
        elif mode.startswith("slow/") or mode.startswith("train_offline"):
            slow_fast = "slow"
        else:
            slow_fast = "unknown"
        logger.info(
            (
                "[RefShort][Selector] slow_fast=%s mode=%s chunk=%s vlm_called=%s "
                "selected_chunks=%s selected_ref_frames=%s extra=%s"
            ),
            slow_fast,
            mode,
            int(current_chunk_idx),
            bool(vlm_called),
            [int(x) for x in selected_chunk_indices],
            [int(x) for x in ref_frame_indices],
            extra,
        )
