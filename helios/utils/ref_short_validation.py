"""Ref-short pipeline context: training validation (offline) vs inference (streaming + cut)."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional

import torch

from helios.modules.extract_feature import decode_middle_frame
from helios.modules.long_memory import LongMemoryStore, cosine_similarity
from helios.modules.ref_short_gap import filter_gap_chunk_pairs
from helios.modules.ref_short_builder import (
    build_ref_rope_indices_for_pipeline,
    pack_ref_latents,
)
from helios.modules.selector_runtime import SelectorOutput, SelectorRuntime


logger = logging.getLogger(__name__)


class RefShortPipelineCtx:
    """HeliosPipeline ref-short state (train-offline OR infer streaming)."""

    def __init__(
        self,
        *,
        runtime: SelectorRuntime,
        vae,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
        history_sizes: List[int],
        latent_window_size: int = 9,
        use_train_offline: bool = True,
        decode_middle_frame_fn: Optional[Callable] = None,
        dino_encoder=None,
        long_memory: Optional[LongMemoryStore] = None,
        lm_tau_cut: float = 0.35,
        slow_history_max_chunks: int = 0,
    ) -> None:
        self.runtime = runtime
        self.vae = vae
        self.latents_mean = latents_mean
        self.latents_std = latents_std
        self.device = device
        self.dtype = dtype
        self.history_sizes = list(history_sizes)
        self.latent_window_size = int(latent_window_size)
        self.use_train_offline = bool(use_train_offline)
        self.decode_middle_frame_fn = decode_middle_frame_fn
        self.dino_encoder = dino_encoder
        self.long_memory = long_memory
        self.lm_tau_cut = float(lm_tau_cut)
        self.slow_history_max_chunks = int(slow_history_max_chunks)

        self.history_ref: List[dict] = []
        self.latents_history_ref: Optional[torch.Tensor] = None
        self.indices_latents_history_ref: Optional[torch.Tensor] = None
        self._pending_latents: Optional[torch.Tensor] = None
        self._pending_indices: Optional[torch.Tensor] = None
        self._prev_tail_emb: Optional[torch.Tensor] = None
        self.last_schedule_mode: str = "init"
        self.debug_log: bool = bool(getattr(runtime, "debug_log", False))
        self.debug_every: int = max(1, int(getattr(runtime, "debug_every", 1)))
        self._slow_history_clock: int = 0
        self._slow_history_stats: dict[int, dict] = {}

    def reset(self) -> None:
        self.history_ref.clear()
        self.latents_history_ref = None
        self.indices_latents_history_ref = None
        self._pending_latents = None
        self._pending_indices = None
        self._prev_tail_emb = None
        self.last_schedule_mode = "init"
        self._slow_history_clock = 0
        self._slow_history_stats.clear()
        self.runtime.reset_streaming_state()
        if self.long_memory is not None and hasattr(self.long_memory, "_entries"):
            self.long_memory._entries.clear()  # noqa: SLF001 — reset per video

    def _touch_slow_history_use(self, selected_chunk_indices: List[int]) -> None:
        """Update per-chunk usage stats after selector picks historical chunks."""
        if not selected_chunk_indices:
            return
        for ci in selected_chunk_indices:
            entry = self._slow_history_stats.setdefault(int(ci), {"hits": 0, "last_hit_at": -1})
            entry["hits"] = int(entry.get("hits", 0)) + 1
            entry["last_hit_at"] = int(self._slow_history_clock)
            self._slow_history_clock += 1

    def _evict_slow_history_if_needed(self) -> None:
        """Evict history_ref entries for full-slow long-video inference.

        Policy: keep at most `slow_history_max_chunks` items by removing entries with:
        1) smallest usage count (`hits`) first
        2) if tied, least recently selected (`last_hit_at` smaller first)
        """
        max_chunks = int(self.slow_history_max_chunks)
        if max_chunks <= 0:
            return
        if len(self.history_ref) <= max_chunks:
            return

        while len(self.history_ref) > max_chunks:
            best_pos = None
            best_key = None
            for pos, ent in enumerate(self.history_ref):
                ci = int(ent["chunk_idx"])
                stat = self._slow_history_stats.get(ci, {"hits": 0, "last_hit_at": -1})
                key = (int(stat.get("hits", 0)), int(stat.get("last_hit_at", -1)))
                if best_key is None or key < best_key:
                    best_key = key
                    best_pos = pos
            if best_pos is None:
                break
            evicted_ci = int(self.history_ref[best_pos]["chunk_idx"])
            del self.history_ref[best_pos]
            self._slow_history_stats.pop(evicted_ci, None)
            if self.debug_log:
                logger.info(
                    "[RefShort][SlowHistoryEvict] evicted_chunk=%s current_size=%s max_chunks=%s",
                    evicted_ci,
                    len(self.history_ref),
                    max_chunks,
                )

    def _apply_selector_output(self, out: SelectorOutput, target_chunk_idx: int) -> None:
        self.last_schedule_mode = out.schedule_mode
        self._touch_slow_history_use(out.ref_chunk_indices)
        if out.ref_latents:
            self._pending_latents = pack_ref_latents(out.ref_latents, device=self.device, dtype=self.dtype)
            self._pending_indices = build_ref_rope_indices_for_pipeline(
                out.ref_frame_indices,
                chunk_idx=int(target_chunk_idx),
                latent_window_size=self.latent_window_size,
                history_sizes=self.history_sizes,
                batch_size=1,
                device=self.device,
            )
        else:
            self._pending_latents = None
            self._pending_indices = None

    def _log_ref_selection(self, out: SelectorOutput, target_chunk_idx: int, *, source: str) -> None:
        if not self.debug_log or (int(target_chunk_idx) % self.debug_every != 0):
            return
        logger.info(
            "[RefShort][ValidationChunk] source=%s generating_chunk=%s mode=%s "
            "selected_ref_chunks=%s selected_ref_frames=%s num_ref_latents=%s",
            source,
            int(target_chunk_idx),
            out.schedule_mode,
            [int(x) for x in out.ref_chunk_indices],
            [int(x) for x in out.ref_frame_indices],
            len(out.ref_latents),
        )

    def prepare_for_chunk(self, chunk_idx: int, prompt: str) -> None:
        if not self.use_train_offline:
            self.latents_history_ref = self._pending_latents
            self.indices_latents_history_ref = self._pending_indices
            return

        gap_latents = []
        gap_chunk_indices = []
        for ent in self.history_ref:
            ci = int(ent["chunk_idx"])
            if ci < max(0, int(chunk_idx) - 1):
                gap_latents.append(ent["latent"])
                gap_chunk_indices.append(ci)

        gap_latents, gap_chunk_indices = filter_gap_chunk_pairs(
            gap_latents,
            gap_chunk_indices,
            int(chunk_idx),
            min_chunk_distance=int(getattr(self.runtime, "min_chunk_distance", 3)),
        )

        context_entry = self._context_entry_for_chunk(int(chunk_idx) - 1)
        out = self.runtime.select_for_next_chunk(
            mode="train_offline",
            current_prompt=prompt or "",
            current_chunk_idx=int(chunk_idx),
            choice_idx=int(chunk_idx),
            gap_latents=gap_latents,
            gap_chunk_indices=gap_chunk_indices,
            context_entry=context_entry,
        )
        self._apply_selector_output(out, int(chunk_idx))
        self._log_ref_selection(out, int(chunk_idx), source="train_offline")
        self.latents_history_ref = self._pending_latents
        self.indices_latents_history_ref = self._pending_indices

    def on_chunk_done(
        self,
        chunk_idx: int,
        chunk_latents: torch.Tensor,
        *,
        prompt_for_next: str = "",
    ) -> None:
        decoded = None
        if self.decode_middle_frame_fn is not None:
            decoded = self.decode_middle_frame_fn(chunk_latents)
        self.history_ref.append(
            {
                "chunk_idx": int(chunk_idx),
                "latent": chunk_latents.detach().cpu(),
                "decoded_frame": decoded,
                "clip_feat": None,
            }
        )
        self._slow_history_stats.setdefault(int(chunk_idx), {"hits": 0, "last_hit_at": -1})
        self._evict_slow_history_if_needed()

        if self.use_train_offline:
            return

        next_chunk_idx = int(chunk_idx) + 1
        tail_emb = None
        if self.dino_encoder is not None:
            from helios.modules.extract_feature import extract_tail_embedding

            tail_emb = extract_tail_embedding(
                chunk_latents,
                self.vae,
                self.dino_encoder,
                self.latents_mean,
                self.latents_std,
                return_frame=False,
            )

        is_cut = False
        if tail_emb is not None and self._prev_tail_emb is not None:
            sim = float(cosine_similarity(self._prev_tail_emb, tail_emb).item())
            is_cut = (1.0 - sim) >= self.lm_tau_cut
            if self.debug_log and (int(chunk_idx) % self.debug_every == 0):
                logger.info(
                    "[RefShort][DINO-cut] chunk=%s sim=%.4f dist=%.4f tau_cut=%.4f is_cut=%s",
                    int(chunk_idx),
                    sim,
                    1.0 - sim,
                    float(self.lm_tau_cut),
                    bool(is_cut),
                )

        if tail_emb is not None:
            self._prev_tail_emb = tail_emb.detach()

        if self.long_memory is not None and tail_emb is not None:
            try:
                action, slot_idx, best_sim = self.long_memory.update(
                    e_tail=tail_emb,
                    chunk_idx=int(chunk_idx),
                    latent=chunk_latents.detach().cpu(),
                    decoded_frame=None,
                )
                if self.debug_log and (int(chunk_idx) % self.debug_every == 0):
                    logger.info(
                        "[RefShort][LongMemory] chunk=%s action=%s slot=%s best_sim=%.4f size=%s",
                        int(chunk_idx),
                        str(action),
                        int(slot_idx),
                        float(best_sim),
                        len(self.long_memory),
                    )
            except Exception as exc:
                logger.warning("LongMemory.update failed chunk=%s: %s", chunk_idx, exc)

        context_entry = self._context_entry_for_chunk(int(chunk_idx))
        out = self.runtime.select_for_next_chunk(
            mode="streaming",
            current_prompt=prompt_for_next or "",
            current_chunk_idx=int(chunk_idx),
            tail_emb=tail_emb,
            is_cut=is_cut,
            history_ref=self.history_ref,
            context_entry=context_entry,
        )
        self._apply_selector_output(out, next_chunk_idx)
        self._log_ref_selection(out, next_chunk_idx, source="streaming")

    def _context_entry_for_chunk(self, chunk_idx: int) -> Optional[dict]:
        if chunk_idx < 0:
            return None
        for ent in reversed(self.history_ref):
            if int(ent["chunk_idx"]) == int(chunk_idx):
                return {
                    "chunk_idx": int(chunk_idx),
                    "latent": ent["latent"],
                    "decoded_frame": ent.get("decoded_frame"),
                    "clip_feat": ent.get("clip_feat"),
                }
        return None


# Backward-compatible alias
RefShortValidationCtx = RefShortPipelineCtx


def build_validation_ctx(
    args,
    *,
    clip_model,
    vae,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> RefShortPipelineCtx:
    """Training-period validation: align with training (train_offline + force_slow)."""
    from helios.utils.ref_short_training import build_vlm_selector_from_config

    vlm_selector = build_vlm_selector_from_config(args, clip_model, str(device), dtype)
    runtime = SelectorRuntime(
        selector=vlm_selector,
        ref_frames_per_chunk=int(args.training_config.ref_frames_per_chunk),
        vlm_k_select=int(args.selector_training.vlm_k_select),
        selector_force_slow=bool(args.selector_training.selector_force_slow),
        enable_long_memory=False,
        min_chunk_distance=int(getattr(args.selector_training, "vlm_min_chunk_distance", 3)),
        decode_middle_frame_fn=lambda lat: decode_middle_frame(lat, vae, latents_mean, latents_std),
        debug_log=bool(getattr(args.selector_training, "debug_log", False)),
        debug_every=int(getattr(args.selector_training, "debug_every", 1)),
    )
    return RefShortPipelineCtx(
        runtime=runtime,
        vae=vae,
        latents_mean=latents_mean,
        latents_std=latents_std,
        device=device,
        dtype=dtype,
        history_sizes=args.training_config.history_sizes,
        latent_window_size=args.training_config.latent_window_size[0],
        use_train_offline=True,
    )


def build_inference_ctx(
    args,
    *,
    clip_model,
    vae,
    latents_mean: torch.Tensor,
    latents_std: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> RefShortPipelineCtx:
    """Deployment inference: streaming + DINO cut gate + LongMemory (infer_helios_bolt 对齐)."""
    from helios.modules.extract_feature import DINOv2
    from helios.utils.ref_short_training import build_vlm_selector_from_config

    inf = args.selector_inference
    vlm_selector = build_vlm_selector_from_config(args, clip_model, str(device), dtype)

    dino_encoder = None
    long_memory = None
    if bool(inf.enable_long_memory):
        dino_path = getattr(inf, "dino_model_path", None)
        dino_encoder = DINOv2(device=str(device), model_id_or_path=dino_path, dtype=dtype)
        long_memory = LongMemoryStore(
            tau_merge=float(getattr(inf, "lm_tau_merge", 0.85)),
            tau_cut=float(getattr(inf, "lm_tau_cut", 0.35)),
            ema_alpha_new=float(getattr(inf, "lm_ema_alpha_new", 0.2)),
            max_size=int(getattr(inf, "lm_codebook_max_size", 512)),
            evict_strategy=str(getattr(inf, "lm_codebook_evict", "lru")),
        )

    runtime = SelectorRuntime(
        selector=vlm_selector,
        ref_frames_per_chunk=int(args.training_config.ref_frames_per_chunk),
        vlm_k_select=int(getattr(inf, "vlm_k_select", args.selector_training.vlm_k_select)),
        selector_force_slow=bool(inf.selector_force_slow),
        enable_long_memory=bool(inf.enable_long_memory),
        long_memory=long_memory,
        lm_tau_cut=float(getattr(inf, "lm_tau_cut", 0.35)),
        lm_codebook_topm=int(getattr(inf, "lm_codebook_topm", 16)),
        min_chunk_distance=int(getattr(args.selector_training, "vlm_min_chunk_distance", 3)),
        decode_middle_frame_fn=lambda lat: decode_middle_frame(lat, vae, latents_mean, latents_std),
        cosine_similarity_fn=cosine_similarity,
        debug_log=bool(getattr(inf, "debug_log", False)),
        debug_every=int(getattr(inf, "debug_every", 1)),
    )
    return RefShortPipelineCtx(
        runtime=runtime,
        vae=vae,
        latents_mean=latents_mean,
        latents_std=latents_std,
        device=device,
        dtype=dtype,
        history_sizes=args.training_config.history_sizes,
        latent_window_size=args.training_config.latent_window_size[0],
        use_train_offline=False,
        dino_encoder=dino_encoder,
        long_memory=long_memory,
        lm_tau_cut=float(getattr(inf, "lm_tau_cut", 0.35)),
        slow_history_max_chunks=int(getattr(inf, "slow_history_max_chunks", 0)),
    )


def resolve_prompt_for_chunk(
    prompt,
    chunk_idx: int,
    *,
    interpolate_time_list: Optional[List[int]] = None,
) -> str:
    if not isinstance(prompt, list):
        return str(prompt)
    if not interpolate_time_list:
        return str(prompt[0])
    from itertools import accumulate

    boundaries = list(accumulate(interpolate_time_list))
    for seg_i, boundary in enumerate(boundaries):
        if int(chunk_idx) < int(boundary):
            return str(prompt[seg_i])
    return str(prompt[-1])
