"""
memory_bank.py — Helios 长视频推理的 history_ref 容量管理
=============================================================

推理时 `history_ref` 在每生成一 chunk 就 append 一项；长视频 (100+ chunks)
若不加节流会把 VLM 的 visual tokens 撑爆（Memory Bank Cap），并持续占用 CPU RAM。

本模块只提供"在原列表上做 in-place 驱逐"的工具函数，不改变 history_ref 的
数据结构，保证现有 CLIP+ITS 路径与新 VLM 路径可以共享同一份 history_ref。

【数据结构约定】
history_ref: list[dict]，每项至少包含：
    "chunk_idx":      int                  — chunk 的绝对位置
    "latent":         torch.Tensor (CPU)   — 该 chunk 的 VAE latent
    "clip_feat":      torch.Tensor (CPU)   — CLIP 视觉特征 (768,)，已 L2 norm 前的原值
    "decoded_frame":  Optional[PIL.Image]  — 中间帧（VLM 路径用；CLIP 路径可为 None）

【第一版实现的驱逐策略】
- farthest_lowclip (默认): score = α·norm_time_dist + (1-α)·(1 - cos(clip_feat, recent_clip))
                            分最高的"既远又不像"被优先驱逐。不需要额外状态。
- oldest (最简 baseline / fallback): 丢 chunk_idx 最小的。

注：lru / uniform / recent_window 等策略按 md/CLIP-base-selector-vlm.md §10 Q5
    的约定放到后续版本再实现。
"""

from __future__ import annotations

from typing import List, Dict, Optional, Sequence

import torch


# ═══════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════

def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1D tensors (CPU, float32)."""
    a = a.detach().float().flatten()
    b = b.detach().float().flatten()
    a = a / (a.norm().clamp_min(1e-8))
    b = b / (b.norm().clamp_min(1e-8))
    return float(torch.dot(a, b).item())


def _score_farthest_lowclip(
    entry: Dict,
    current_chunk_idx: int,
    recent_clip_feat: Optional[torch.Tensor],
    alpha: float = 0.5,
    max_time_distance: int = 1,
) -> float:
    """驱逐优先级分：越大越应该被驱逐。

    score = alpha * norm_time_dist + (1 - alpha) * (1 - cos(entry.clip_feat, recent_clip))
    """
    time_dist = max(0, int(current_chunk_idx) - int(entry["chunk_idx"]))
    norm_t = min(1.0, time_dist / max(1, max_time_distance))

    if recent_clip_feat is None or entry.get("clip_feat") is None:
        sim = 0.0
    else:
        sim = _cosine(entry["clip_feat"], recent_clip_feat)
    dissim = 1.0 - sim

    return alpha * norm_t + (1.0 - alpha) * dissim


# ═══════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════

def evict_history(
    history_ref: List[Dict],
    *,
    max_n: int = 32,
    keep_recent: int = 8,
    strategy: str = "farthest_lowclip",
    current_chunk_idx: Optional[int] = None,
    protected_chunk_idx: Optional[Sequence[int]] = None,
    alpha: float = 0.5,
) -> List[int]:
    """按策略把 history_ref in-place 裁剪到 max_n。

    Args:
        history_ref: list of chunk 记录，见模块顶部约定
        max_n: 硬上限；len(history_ref) > max_n 时才会驱逐
        keep_recent: 最近 K 个 chunk（按 chunk_idx 降序）永不被驱逐
        strategy: "farthest_lowclip" | "oldest"
        current_chunk_idx: 用于 farthest_lowclip 的时间距离计算；缺省时取 history 中最大的 chunk_idx
        protected_chunk_idx: 额外强制保留的 chunk_idx 集合（例如当前 vlm_cache 正在引用的）
        alpha: farthest_lowclip 中时间距离 vs 语义不相似度的权重

    Returns:
        evicted_chunk_idx: 被驱逐的 chunk_idx 列表（按驱逐顺序）
    """
    if len(history_ref) <= max_n:
        return []

    if current_chunk_idx is None:
        current_chunk_idx = max(int(e["chunk_idx"]) for e in history_ref)

    # 最近 keep_recent 个强制保留
    sorted_by_idx_desc = sorted(
        range(len(history_ref)),
        key=lambda i: int(history_ref[i]["chunk_idx"]),
        reverse=True,
    )
    protected_positions = set(sorted_by_idx_desc[:max(0, keep_recent)])

    if protected_chunk_idx:
        protected_set = set(int(x) for x in protected_chunk_idx)
        for pos, e in enumerate(history_ref):
            if int(e["chunk_idx"]) in protected_set:
                protected_positions.add(pos)

    # 候选池：未被保护的 positions
    candidates = [i for i in range(len(history_ref)) if i not in protected_positions]

    # 计算驱逐打分
    if strategy == "oldest":
        scored = [(-int(history_ref[i]["chunk_idx"]), i) for i in candidates]
        # 取负的 chunk_idx：值越大代表 chunk_idx 越小（越老）
        # 但我们要"越大越应该被驱逐" → 直接用 score = -chunk_idx
        # 上面 tuple 第一个元素本身就是 score
    elif strategy == "farthest_lowclip":
        # 用 keep_recent 里最近一个的 clip_feat 作为 "recent 语义锚点"
        recent_pos = sorted_by_idx_desc[0] if sorted_by_idx_desc else None
        recent_feat = history_ref[recent_pos]["clip_feat"] if recent_pos is not None else None
        # normalize 因子
        max_td = max(
            1,
            max(
                current_chunk_idx - int(history_ref[i]["chunk_idx"])
                for i in candidates
            ),
        )
        scored = [
            (
                _score_farthest_lowclip(
                    history_ref[i], current_chunk_idx, recent_feat,
                    alpha=alpha, max_time_distance=max_td,
                ),
                i,
            )
            for i in candidates
        ]
    else:
        raise ValueError(
            f"Unknown mb_evict_strategy='{strategy}'. Supported: farthest_lowclip, oldest."
        )

    # 按分数降序排，分最高的最先被驱逐
    scored.sort(key=lambda x: x[0], reverse=True)

    n_to_evict = len(history_ref) - max_n
    evict_positions = sorted([i for _, i in scored[:n_to_evict]], reverse=True)

    evicted_chunk_idx = []
    for pos in evict_positions:
        evicted_chunk_idx.append(int(history_ref[pos]["chunk_idx"]))
        del history_ref[pos]

    return evicted_chunk_idx


def refresh_latents_from(
    history_ref: List[Dict],
    selected_chunk_idx: Sequence[int],
) -> tuple:
    """fast step 专用：按 chunk_idx 从最新 history_ref 取 latent。

    如果某个 chunk_idx 已经被驱逐，会被静默跳过（调用方可据此判断是否需要 mini-slow）。

    Returns:
        (selected_latents, recovered_chunk_idx): 仍存在的 latent 列表 和 对应 chunk_idx 列表
    """
    idx_to_entry = {int(e["chunk_idx"]): e for e in history_ref}
    selected_latents = []
    recovered = []
    for cidx in selected_chunk_idx:
        entry = idx_to_entry.get(int(cidx))
        if entry is None:
            continue
        selected_latents.append(entry["latent"])
        recovered.append(int(cidx))
    return selected_latents, recovered


# ═══════════════════════════════════════════
# VLM Selector 选帧缓存（slow/fast 调度用）
# ═══════════════════════════════════════════

class VLMSelectorCache:
    """只缓存 VLM slow step 的选帧结果，fast step 按 chunk_idx 去 history_ref 重新取 latent。

    这样做的好处：history_ref 被 memory_bank_evict 裁剪时，fast step 自然会感知到
    (refresh_latents_from 取不到就降级/重算)，不会出现 stale latent 引用。
    """

    def __init__(self):
        self.selected_chunk_idx: List[int] = []
        self.valid_for_chunks: set = set()
        self.computed_at_chunk: Optional[int] = None
        self.for_prompt: Optional[str] = None

    def is_empty(self) -> bool:
        return not self.selected_chunk_idx

    def update(
        self,
        selected_chunk_idx: Sequence[int],
        computed_at_chunk: int,
        for_prompt: str,
        valid_for: Sequence[int],
    ):
        self.selected_chunk_idx = list(int(x) for x in selected_chunk_idx)
        self.computed_at_chunk = int(computed_at_chunk)
        self.for_prompt = for_prompt
        self.valid_for_chunks = set(int(x) for x in valid_for)

    def get_latents(self, history_ref: List[Dict]) -> tuple:
        """从最新 history_ref 按 chunk_idx 取 latent。"""
        return refresh_latents_from(history_ref, self.selected_chunk_idx)

    def covers(self, chunk_idx: int) -> bool:
        return int(chunk_idx) in self.valid_for_chunks
