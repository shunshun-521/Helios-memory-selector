"""
select_frames_vlm.py — VLM 选帧器（Plug-and-play，不替代 CLIP+ITS）
====================================================================

与 ``helios/modules/select_frames.py`` 的 ``select_gap_frames`` **同输入同输出**，
仅作为 ``infer_helios_bolt.py`` 在 ``--selector_type vlm`` 时的可选分支。

现状（Step 1 / 第一轮实现，参考 md/CLIP-base-selector-vlm.md §9）：
    - 抽象基类 ``BaseSelector`` 定义统一接口
    - ``RandomSelector``：随机选帧，仅用于跑通端到端流水线（接口 / hooks / eviction）
    - ``VLMFrameSelector``：骨架 stub，真正的 VLM forward 留到 Step 3/4
    - 默认 pre-filter = ``clip_topm``（复用 CLIP compute_similarity），默认驱逐策略由
      ``helios/modules/memory_bank.py`` 提供

所有与 CLIP+ITS 相同的超参（k_select / power / min_chunk_distance）语义一致，
便于互换和 ablation。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

# 复用 ITS 抽样（与 CLIP+ITS 路径完全一致）
from helios.modules.select_frames import inverse_transform_sampling


# ═══════════════════════════════════════════
# Base Selector
# ═══════════════════════════════════════════

class BaseSelector(ABC):
    """所有 selector 的统一调用接口。

    返回签名必须与 ``select_gap_frames`` 完全一致：
        selected_latents: list of (B, C, T, H, W)
        selected_indices: list of int   # 绝对 chunk_idx
    """

    def __init__(
        self,
        k: int = 4,
        power: float = 2.0,
        min_chunk_distance: int = 3,
    ):
        self.k = int(k)
        self.power = float(power)
        self.min_chunk_distance = int(min_chunk_distance)

    @abstractmethod
    def __call__(
        self,
        history: List[Dict],
        current_chunk_idx: int,
        current_prompt: str,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        ...

    # ─── 共用：筛选合法的 GAP 候选 ───
    def _get_gap_candidates(
        self,
        history: List[Dict],
        current_chunk_idx: int,
    ) -> List[Dict]:
        """从 history 中挑出可参与选帧的 GAP 条目。

        规则与 select_gap_frames / ref_short_builder 一致：
          - 排除当前 chunk 的 context (history[current_chunk_idx - 1])
          - 与 target 的时间距离 >= min_chunk_distance
        """
        from helios.modules.ref_short_gap import is_gap_chunk_eligible

        if current_chunk_idx < 2 or len(history) < 2:
            return []

        return [
            e
            for e in history
            if is_gap_chunk_eligible(
                int(e["chunk_idx"]),
                int(current_chunk_idx),
                min_chunk_distance=self.min_chunk_distance,
            )
        ]


# ═══════════════════════════════════════════
# Random Selector (Step 1 跑通端到端用)
# ═══════════════════════════════════════════

class RandomSelector(BaseSelector):
    """随机选帧。仅用于验证 ``--selector_type`` 分流、memory_bank 驱逐、
    Bolt Ref-Attn hooks 重挂等端到端链路是否正确。**不具备任何选帧质量**，
    不应作为生产路径使用。
    """

    def __init__(
        self,
        k: int = 4,
        power: float = 2.0,
        min_chunk_distance: int = 3,
        seed: Optional[int] = None,
    ):
        super().__init__(k=k, power=power, min_chunk_distance=min_chunk_distance)
        self._rng = np.random.default_rng(seed)

    def __call__(
        self,
        history: List[Dict],
        current_chunk_idx: int,
        current_prompt: str,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        gap = self._get_gap_candidates(history, current_chunk_idx)
        if not gap:
            return [], []

        actual_k = min(self.k, len(gap))
        positions = self._rng.choice(len(gap), size=actual_k, replace=False)
        positions = sorted(int(p) for p in positions)

        selected_latents = [gap[p]["latent"] for p in positions]
        selected_indices = [int(gap[p]["chunk_idx"]) for p in positions]
        return selected_latents, selected_indices


# ═══════════════════════════════════════════
# CLIP Pre-filter（VLM 前的粗筛层）
# ═══════════════════════════════════════════

def clip_topm_prefilter(
    gap_candidates: List[Dict],
    clip_model,
    context_entry: Optional[Dict],
    current_prompt: str,
    max_candidates: int,
    alpha: float = 0.6,
    device: str = "cuda",
) -> List[Dict]:
    """用 CLIP visual + text combined score 从 gap_candidates 里取 top-M。

    与 CLIP+ITS 路径的分数计算对齐（helios/modules/select_frames.py），
    这样蒸馏出来的 VLM 正好在同一候选分布上训练。
    """
    if len(gap_candidates) <= max_candidates:
        return list(gap_candidates)
    if clip_model is None:
        return list(gap_candidates[:max_candidates])

    gap_feats = torch.stack([e["clip_feat"] for e in gap_candidates], dim=0).to(device)

    # Visual query: 上一段 chunk 的 CLIP 特征（若无 context 则只用 text）
    if context_entry is not None and context_entry.get("clip_feat") is not None:
        visual_query = context_entry["clip_feat"].unsqueeze(0).to(device)
        visual_scores = clip_model.compute_similarity(gap_feats, visual_query).numpy()
    else:
        visual_scores = np.zeros(len(gap_candidates), dtype=np.float32)

    if current_prompt:
        text_query = clip_model.extract_text_features(current_prompt).to(device)
        text_scores = clip_model.compute_similarity(gap_feats, text_query).numpy()
    else:
        text_scores = np.zeros_like(visual_scores)

    combined = alpha * visual_scores + (1.0 - alpha) * text_scores
    top_positions = np.argsort(-combined)[:max_candidates]
    return [gap_candidates[int(p)] for p in sorted(top_positions.tolist())]


# ═══════════════════════════════════════════
# VLM Frame Selector (Step 1: stub；Step 3/4 接 Qwen2.5-VL)
# ═══════════════════════════════════════════

class VLMFrameSelector(BaseSelector):
    """基于 VLM 的高粒度选帧器（stub）/ VLM-based frame selector (stub).

    当前版本（Step 1）/ current stage (Step 1):
        - 接口、CLIP pre-filter、ITS 采样已就绪
        - VLM 推理部分暂以"CLIP combined score"作为 logits 的占位实现，
          这样在没有训练好的 VLM ckpt 时仍可跑通流水线
        - Step 3 训练出 LoRA ckpt 后，只需替换 ``_score_candidates`` 的实现

    Args（参数语义中英对照）/ Args:
        clip_model: 粗筛用的 CLIP 封装（helios.modules.extract_feature.CLIP）
        k: 最终注入 Ref-Attn 的帧数（= vlm_k_select）
        power: ITS 锐度
        min_chunk_distance: GAP 候选与 target 的最小时间距离
        max_candidates: CLIP pre-filter 的 top-M（= vlm_max_candidates）
        prefilter_alpha: visual vs text 的粗筛权重
        temperature: softmax 温度（= vlm_temperature）
        vlm_backbone: 真正的 VLM 模型（Step 1 传 None 即可）
        device: 推理设备
        fallback_to_random: VLM 推理失败时是否退化到 RandomSelector
        vlm_rank_mode: ``its``（inverse_transform_sampling）或 ``topk``（分数最高的 K 个候选，不经 ITS）
    """

    def __init__(
        self,
        clip_model=None,
        k: int = 4,
        power: float = 2.0,
        min_chunk_distance: int = 3,
        max_candidates: int = 16,
        prefilter_alpha: float = 0.6,
        temperature: float = 0.7,
        vlm_backbone=None,
        device: str = "cuda",
        fallback_to_random: bool = True,
        vlm_rank_mode: str = "its",
    ):
        super().__init__(k=k, power=power, min_chunk_distance=min_chunk_distance)
        self.clip_model = clip_model
        self.max_candidates = int(max_candidates)
        self.prefilter_alpha = float(prefilter_alpha)
        self.temperature = float(temperature)
        self.vlm_rank_mode = str(vlm_rank_mode).lower().strip()
        if self.vlm_rank_mode not in {"its", "topk"}:
            raise ValueError(f"vlm_rank_mode must be 'its' or 'topk', got {vlm_rank_mode!r}")
        self.vlm = vlm_backbone
        self.device = device
        self.fallback_to_random = fallback_to_random
        self._fallback = RandomSelector(
            k=k, power=power, min_chunk_distance=min_chunk_distance,
        ) if fallback_to_random else None

    # ─── 外部可覆盖：真正的 VLM 打分 ───
    def _score_candidates(
        self,
        candidates: List[Dict],
        context_entry: Optional[Dict],
        current_prompt: str,
    ) -> np.ndarray:
        """给每个候选打分 / score each candidate.

        优先级 / priority:
        1. 注入了 ``vlm_backbone``（含 ``.score(context_image, candidate_images, prompt)``）
           → 使用真实 VLM 零样本 / LoRA 打分。
        2. 否则退化为 CLIP combined score（Step 1 的 stub，无需权重即可跑通流水线）。
        3. 再否则（没有 clip_model）：均匀分布。

        Returns:
            scores: np.array shape (N_cand,)
        """
        if self.vlm is not None:
            # 优先走视频分支：context/candidate 都提供 decoded_frames 且 backbone 支持 score_video
            context_frames = None if context_entry is None else context_entry.get("decoded_frames")
            candidate_videos = [e.get("decoded_frames") for e in candidates]
            if (
                hasattr(self.vlm, "score_video")
                and context_frames is not None
                and all(isinstance(v, list) and len(v) > 0 for v in candidate_videos)
            ):
                print(
                    "[VLMFrameSelector] using score_video "
                    f"(context_frames={len(context_frames)}, "
                    f"candidates={len(candidate_videos)}, "
                    f"candidate_frames[0]={len(candidate_videos[0])})"
                )
                return self.vlm.score_video(
                    context_frames=context_frames,
                    candidate_videos=candidate_videos,
                    prompt=current_prompt or "",
                )

            context_image = None
            if context_entry is not None:
                context_image = context_entry.get("decoded_frame")
            candidate_images = [e.get("decoded_frame") for e in candidates]
            if any(img is None for img in candidate_images):
                # decoded_frame 缺失（CLIP 路径复用 history_ref 时可能 None）→ 退回 CLIP 占位
                print(
                    "[VLMFrameSelector] decoded_frame missing in candidates; "
                    "fallback to CLIP-placeholder scoring."
                )
            else:
                return self.vlm.score(
                    context_image=context_image,
                    candidate_images=candidate_images,
                    prompt=current_prompt or "",
                )

        if self.clip_model is None:
            return np.ones(len(candidates), dtype=np.float32)

        gap_feats = torch.stack([e["clip_feat"] for e in candidates], dim=0).to(self.device)

        if context_entry is not None and context_entry.get("clip_feat") is not None:
            vq = context_entry["clip_feat"].unsqueeze(0).to(self.device)
            v_score = self.clip_model.compute_similarity(gap_feats, vq).numpy()
        else:
            v_score = np.zeros(len(candidates), dtype=np.float32)

        if current_prompt:
            tq = self.clip_model.extract_text_features(current_prompt).to(self.device)
            t_score = self.clip_model.compute_similarity(gap_feats, tq).numpy()
        else:
            t_score = np.zeros_like(v_score)

        return self.prefilter_alpha * v_score + (1.0 - self.prefilter_alpha) * t_score

    def __call__(
        self,
        history: List[Dict],
        current_chunk_idx: int,
        current_prompt: str,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        gap = self._get_gap_candidates(history, current_chunk_idx)
        if not gap:
            return [], []

        # ① CLIP pre-filter: N_gap → M=max_candidates
        context = history[current_chunk_idx - 1] if current_chunk_idx - 1 < len(history) else None
        candidates = clip_topm_prefilter(
            gap_candidates=gap,
            clip_model=self.clip_model,
            context_entry=context,
            current_prompt=current_prompt,
            max_candidates=self.max_candidates,
            alpha=self.prefilter_alpha,
            device=self.device,
        )

        # ② VLM 精排（Step 1 用 CLIP 占位；Step 3+ 替换）
        try:
            scores = self._score_candidates(candidates, context, current_prompt)
        except Exception as exc:
            if self._fallback is None:
                raise
            print(f"[VLMFrameSelector] score failed ({exc}); falling back to RandomSelector.")
            return self._fallback(history, current_chunk_idx, current_prompt)

        # ③ 按分数选 K 个候选：ITS（随机）或 top-k（贪心最高）
        actual_k = min(self.k, len(candidates))
        scores_arr = np.asarray(scores, dtype=np.float32)
        if self.vlm_rank_mode == "topk":
            order = np.argsort(-scores_arr)[:actual_k]
            sampled_positions = [int(i) for i in order]
        else:
            sampled = inverse_transform_sampling(
                scores_arr, n=actual_k, power=self.power,
            )
            sampled_positions = list(dict.fromkeys(sampled.tolist()))

        selected_latents = [candidates[p]["latent"] for p in sampled_positions]
        selected_indices = [int(candidates[p]["chunk_idx"]) for p in sampled_positions]
        return selected_latents, selected_indices

    def select_from_candidates(
        self,
        *,
        candidates: List[Dict],
        context_entry: Optional[Dict],
        current_prompt: str,
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """直接对给定候选列表做精排并选 top-k / rank given candidates and select top-k.

        用于 LongMemory codebook 检索后的二阶段流程 / for 2-stage LongMemory pipeline:
            codebook embedding top-M → VLM re-rank → inject top-k latents
        """
        if not candidates:
            return [], []

        try:
            scores = self._score_candidates(candidates, context_entry, current_prompt)
        except Exception as exc:
            if self._fallback is None:
                raise
            print(f"[VLMFrameSelector] select_from_candidates score failed ({exc}); falling back to RandomSelector.")
            # 退化：把 candidates 当做 history-like 列表随机挑 k
            actual_k = min(self.k, len(candidates))
            picked = np.random.choice(len(candidates), size=actual_k, replace=False)
            picked = sorted(int(x) for x in picked.tolist())
            return (
                [candidates[i]["latent"] for i in picked],
                [int(candidates[i]["chunk_idx"]) for i in picked],
            )

        actual_k = min(self.k, len(candidates))
        scores_arr = np.asarray(scores, dtype=np.float32)
        if self.vlm_rank_mode == "topk":
            order = np.argsort(-scores_arr)[:actual_k]
            sampled_positions = [int(i) for i in order]
        else:
            sampled = inverse_transform_sampling(scores_arr, n=actual_k, power=self.power)
            sampled_positions = list(dict.fromkeys(sampled.tolist()))

        selected_latents = [candidates[p]["latent"] for p in sampled_positions]
        selected_indices = [int(candidates[p]["chunk_idx"]) for p in sampled_positions]
        return selected_latents, selected_indices


# ═══════════════════════════════════════════
# Slow / Fast 调度工具
# ═══════════════════════════════════════════

def compute_slow_step_chunks(
    interpolate_time_list: Optional[Sequence[int]],
    extra_mid: bool = False,
) -> set:
    """根据 interactive 推理的 interpolate_time_list 推断哪些 chunk_idx 触发 slow step。

    默认规则：每个 prompt 段的起点（段切换处）触发 slow。
    可选 ``extra_mid=True`` 在每段中点再加一个 slow。

    例子：interpolate_time_list=[7,7,7] → slow = {0, 7, 14}
          若 extra_mid=True         → slow = {0, 3, 7, 10, 14, 17}
    """
    if not interpolate_time_list:
        return {0}

    boundaries, acc = [], 0
    for t in interpolate_time_list:
        acc += int(t)
        boundaries.append(acc)
    seg_starts = [0] + boundaries[:-1]

    triggers = set(seg_starts)
    if extra_mid:
        for start, end in zip(seg_starts, boundaries):
            triggers.add(start + (end - start) // 2)
    return triggers


def parse_slow_step_chunks(spec: Optional[str]) -> Optional[set]:
    """解析 CLI 的 ``--vlm_slow_step_chunks "0,7,14"``。"""
    if spec is None:
        return None
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if part:
            out.add(int(part))
    return out
