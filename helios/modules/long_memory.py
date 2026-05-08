"""
long_memory.py — Cut-gated Long Memory Store (codebook + retrieval)
===================================================================

本模块实现一个最小、可扩展的 `LongMemoryStore` 抽象（codebook + 检索门控），
对应 `md/LongMemory.md` 的思路。
This module implements a minimal, extensible `LongMemoryStore` abstraction
(codebook + cut-gated retrieval) as described in `md/LongMemory.md`.

- 不额外维护短程记忆（short-term buffer）：
  Helios 本身已有短上下文（例如最近 2 个 chunk），这里不重复存。
  No extra short-term buffer is required (Helios already carries short context).

- 用长期 codebook 压缩历史（long-term compression via prototypes）：
  每个槽位存一个 embedding 原型用于快速相似度检索，同时存对应 latent 以便直接注入 DiT；
  像素帧（PIL）可选，仅用于 VLM 精排输入。
  Each slot stores an embedding prototype (fast similarity search) AND the corresponding
  latent (direct injection into DiT). Decoded frames are optional for VLM re-ranking.

- 在线更新（online update / merge-or-insert）：
  若新 embedding 与最近邻相似度 >= tau_merge → EMA 刷新该槽位；否则新增槽位。
  If the nearest similarity >= tau_merge, refresh the slot via EMA; otherwise insert.

- 切换门控检索（cut-gated retrieval）：
  仅当检测到显著切换（boundary/cut）时，才从 codebook 检索候选。
  Only retrieve from the codebook when a boundary/cut is detected.

This file is intentionally independent from VLM selector implementation; it only
manages memory entries and candidate retrieval.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


def _l2_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.detach().float()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """余弦相似度 / cosine similarity.

    输入为两个 1D 向量，输出标量张量。
    Takes two 1D vectors and returns a scalar tensor.
    """
    a = _l2_normalize(a.flatten())
    b = _l2_normalize(b.flatten())
    return (a * b).sum()


def batched_cosine_similarity(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """批量余弦相似度 / batched cosine similarity.

    query: [D], keys: [N, D] -> scores: [N]
    """
    q = _l2_normalize(query.flatten()).unsqueeze(0)  # [1, D]
    k = _l2_normalize(keys)  # [N, D]
    return (k * q).sum(dim=-1)


@dataclass
class CodebookEntry:
    """codebook 的一个槽位 / one slot in the long-term codebook."""

    key: torch.Tensor  # embedding 原型 / embedding prototype, shape [D] (CPU recommended)
    chunk_idx: int  # 槽位当前代表的 chunk 绝对位置 / absolute chunk idx this entry represents
    latent: torch.Tensor  # 可直接注入的 VAE latent / VAE latent on CPU: (B,C,T,H,W)
    decoded_frame: Any = None  # 可选像素帧 / optional decoded frame (e.g. PIL.Image) for VLM

    # 驱逐与调试元信息 / meta for eviction & debugging
    hits: int = 0
    last_used_at: int = 0
    created_at: int = 0


@dataclass
class RetrieveResult:
    """检索返回值 / return type for retrieval methods."""

    entries: List[CodebookEntry] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)  # 相似度标量 / similarity scores (same length as entries)


class LongMemoryStore:
    """切换门控的长期记忆存储 / cut-gated long-term memory store.

    设计要点 / design choices:
    - key 用 embedding 原型，支持廉价的向量相似度检索。
      Store embedding prototypes as keys for cheap similarity search.
    - latent 与 key 同槽位保存，便于最终模块（例如 Bolt Ref-Attn）直接注入，不需要从 embedding 反解像素。
      Store latents alongside keys so consumers can inject latents directly (no decode from embeddings).
    - decoded_frame 可选，只在 VLM 需要像素输入时按需填充（推荐按需 decode，避免长期缓存占 RAM）。
      decoded_frame is optional and only needed when a VLM selector requires pixel inputs.
    """

    def __init__(
        self,
        *,
        tau_merge: float = 0.85,
        tau_cut: float = 0.35,
        ema_alpha_new: float = 0.2,
        max_size: int = 512,
        evict_strategy: str = "lru",
        normalize_keys: bool = True,
    ):
        """
        Args:
            tau_merge: 合并阈值 / merge threshold.
                max cosine >= tau_merge → refresh nearest slot (EMA); else insert new.
            tau_cut: 切换阈值 / cut threshold.
                distance d = 1 - cosine. If d >= tau_cut, trigger retrieval.
            ema_alpha_new: EMA 中“新观测权重” / EMA new-observation weight.
                key <- a * e_new + (1-a) * key_old
            max_size: codebook 容量上限 / hard cap for codebook slots.
            evict_strategy: 满了以后怎么驱逐 / eviction strategy when full: "lru" | "lfu" | "oldest".
            normalize_keys: 是否保持 key 为 L2 norm=1 / keep keys L2-normalized for stable cosine search.
        """
        self.tau_merge = float(tau_merge)
        self.tau_cut = float(tau_cut)
        self.ema_alpha_new = float(ema_alpha_new)
        self.max_size = int(max_size)
        self.evict_strategy = str(evict_strategy).lower().strip()
        if self.evict_strategy not in {"lru", "lfu", "oldest"}:
            raise ValueError(f"Unsupported evict_strategy={evict_strategy!r}")
        self.normalize_keys = bool(normalize_keys)

        self._entries: List[CodebookEntry] = []
        self._step: int = 0  # 单调计数器 / monotonic counter for last_used_at & created_at

    # ─────────────────────────────────────────────────────────────
    # Properties
    # ─────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> List[CodebookEntry]:
        # 约定只读 / expose as read-only by convention; callers should not mutate in-place.
        return list(self._entries)

    # ─────────────────────────────────────────────────────────────
    # Cut gate
    # ─────────────────────────────────────────────────────────────

    def should_retrieve(self, e_prev: torch.Tensor, e_curr: torch.Tensor) -> bool:
        """判断是否触发长程检索 / decide whether to retrieve from long memory.

        距离定义 / distance: d = 1 - cosine(e_prev, e_curr).
        """
        sim = float(cosine_similarity(e_prev, e_curr).item())
        dist = 1.0 - sim
        return dist >= self.tau_cut

    # ─────────────────────────────────────────────────────────────
    # Update / Insert
    # ─────────────────────────────────────────────────────────────

    def update(
        self,
        *,
        e_tail: torch.Tensor,
        chunk_idx: int,
        latent: torch.Tensor,
        decoded_frame: Any = None,
    ) -> Tuple[str, int, float]:
        """用新观测更新 codebook / update codebook with a new observation.

        Returns:
            (action, slot_index, best_sim)
            action ∈ {"insert", "refresh"}.
        """
        self._step += 1
        key = e_tail.detach().float().flatten().cpu()
        if self.normalize_keys:
            key = _l2_normalize(key)

        if not self._entries:
            self._entries.append(
                CodebookEntry(
                    key=key,
                    chunk_idx=int(chunk_idx),
                    latent=latent.detach().cpu(),
                    decoded_frame=decoded_frame,
                    hits=0,
                    last_used_at=self._step,
                    created_at=self._step,
                )
            )
            return "insert", 0, 1.0

        keys = torch.stack([e.key for e in self._entries], dim=0)  # [N,D] CPU
        sims = batched_cosine_similarity(key, keys)  # [N]
        best_sim, best_i = torch.max(sims, dim=0)
        best_sim_f = float(best_sim.item())
        best_i_int = int(best_i.item())

        if best_sim_f >= self.tau_merge:
            # EMA 刷新 / refresh via EMA: key <- a*new + (1-a)*old
            old = self._entries[best_i_int].key
            new_key = self.ema_alpha_new * key + (1.0 - self.ema_alpha_new) * old
            if self.normalize_keys:
                new_key = _l2_normalize(new_key)

            ent = self._entries[best_i_int]
            ent.key = new_key
            ent.chunk_idx = int(chunk_idx)
            ent.latent = latent.detach().cpu()
            ent.decoded_frame = decoded_frame
            ent.last_used_at = self._step
            # refresh 不算命中 / hits not incremented on refresh; hits is for retrieval usage
            return "refresh", best_i_int, best_sim_f

        # insert new slot (may evict)
        if len(self._entries) >= self.max_size:
            self._evict_one()

        self._entries.append(
            CodebookEntry(
                key=key,
                chunk_idx=int(chunk_idx),
                latent=latent.detach().cpu(),
                decoded_frame=decoded_frame,
                hits=0,
                last_used_at=self._step,
                created_at=self._step,
            )
        )
        return "insert", len(self._entries) - 1, best_sim_f

    def update_from_history_entry(
        self,
        entry: Dict[str, Any],
        *,
        embedding_field: str = "clip_feat",
        latent_field: str = "latent",
        frame_field: str = "decoded_frame",
    ) -> Tuple[str, int, float]:
        """便捷封装：从 `history_ref` 条目更新 / convenience wrapper for Helios `history_ref` items."""
        if embedding_field not in entry:
            raise KeyError(f"history entry missing {embedding_field!r}")
        if latent_field not in entry:
            raise KeyError(f"history entry missing {latent_field!r}")
        if "chunk_idx" not in entry:
            raise KeyError("history entry missing 'chunk_idx'")

        return self.update(
            e_tail=entry[embedding_field],
            chunk_idx=int(entry["chunk_idx"]),
            latent=entry[latent_field],
            decoded_frame=entry.get(frame_field),
        )

    # ─────────────────────────────────────────────────────────────
    # Retrieval
    # ─────────────────────────────────────────────────────────────

    def retrieve_topk(
        self,
        *,
        query: torch.Tensor,
        topk: int = 4,
        exclude_chunk_idx: Optional[Sequence[int]] = None,
    ) -> RetrieveResult:
        """按余弦相似度取 top-k / retrieve top-k most similar entries by cosine similarity."""
        if not self._entries or topk <= 0:
            return RetrieveResult(entries=[], scores=[])

        self._step += 1
        q = query.detach().float().flatten().cpu()
        if self.normalize_keys:
            q = _l2_normalize(q)

        exclude_set = set(int(x) for x in (exclude_chunk_idx or []))
        kept: List[Tuple[int, CodebookEntry]] = [
            (i, e) for i, e in enumerate(self._entries) if int(e.chunk_idx) not in exclude_set
        ]
        if not kept:
            return RetrieveResult(entries=[], scores=[])

        idxs, ents = zip(*kept)
        keys = torch.stack([e.key for e in ents], dim=0)
        sims = batched_cosine_similarity(q, keys)  # [N_kept]

        k = min(int(topk), sims.numel())
        vals, pos = torch.topk(sims, k=k, largest=True)

        out_entries: List[CodebookEntry] = []
        out_scores: List[float] = []
        for j in range(k):
            kept_pos = int(pos[j].item())
            ent = ents[kept_pos]
            ent.hits += 1
            ent.last_used_at = self._step
            out_entries.append(ent)
            out_scores.append(float(vals[j].item()))
        return RetrieveResult(entries=out_entries, scores=out_scores)

    def get_latents(self, entries: Sequence[CodebookEntry]) -> List[torch.Tensor]:
        """从检索结果里取 latent（可直接注入）/ extract CPU latents (ready for injection)."""
        return [e.latent for e in entries]

    def get_frames(self, entries: Sequence[CodebookEntry]) -> List[Any]:
        """从检索结果里取像素帧（给 VLM 用）/ extract decoded frames (for VLM pixel input)."""
        return [e.decoded_frame for e in entries]

    # ─────────────────────────────────────────────────────────────
    # Eviction
    # ─────────────────────────────────────────────────────────────

    def _evict_one(self) -> None:
        """驱逐一个槽位 / evict a single entry according to strategy."""
        if not self._entries:
            return

        if self.evict_strategy == "oldest":
            victim_i = min(range(len(self._entries)), key=lambda i: self._entries[i].created_at)
        elif self.evict_strategy == "lfu":
            victim_i = min(range(len(self._entries)), key=lambda i: (self._entries[i].hits, self._entries[i].created_at))
        else:  # lru
            victim_i = min(range(len(self._entries)), key=lambda i: (self._entries[i].last_used_at, self._entries[i].created_at))

        del self._entries[victim_i]


def build_codebook_candidates_for_vlm(
    *,
    store: LongMemoryStore,
    query_embedding: torch.Tensor,
    topm: int = 16,
    exclude_chunk_idx: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    """桥接函数：把 codebook 检索结果转成 selector 候选格式 / utility bridge.

    这样可以复用现有 selector（例如 `VLMFrameSelector.select_from_candidates`），其输入是
    list[dict]，字段类似 {"chunk_idx","latent","decoded_frame","clip_feat"}。
    This makes it easy to reuse selector code that expects a list of dict candidates.
    """
    res = store.retrieve_topk(query=query_embedding, topk=topm, exclude_chunk_idx=exclude_chunk_idx)
    candidates: List[Dict[str, Any]] = []
    for ent, score in zip(res.entries, res.scores):
        candidates.append(
            {
                "chunk_idx": int(ent.chunk_idx),
                "latent": ent.latent,
                "decoded_frame": ent.decoded_frame,
                "clip_feat": ent.key,  # 这里存的是 codebook key（已归一化）/ normalized embedding prototype
                "_codebook_score": float(score),
            }
        )
    return candidates

