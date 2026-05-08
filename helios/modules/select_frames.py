"""
select_frames.py — 推理时在线选帧 (BOLT ITS)
================================================

每生成一个 chunk 之前，用 ITS 从历史 GAP chunk 中选参考帧。
完全在线，无任何离线预处理。

选帧结果用于 Reference Attention 注入。
"""

import torch
import numpy as np


def inverse_transform_sampling(scores, n, power=2.0):
    """BOLT 逆变换采样 (CVPR 2025)。

    Args:
        scores: np.array (N,) — 每个 GAP chunk 的综合相似度分数
        n: int — 要选几帧
        power: float — 锐度系数 (>1 集中高分帧, =1 按概率采样, <1 趋向均匀)

    Returns:
        np.array (n,) — 在 scores 中的位置索引
    """
    scores = scores - scores.min()
    if scores.max() > 1e-8:
        scores = scores / scores.max()
    else:
        return np.linspace(0, len(scores) - 1, n, dtype=int)

    scores = scores ** power
    probs = scores / scores.sum()
    cdf = np.cumsum(probs)

    uniform_pts = np.linspace(1 / n, 1 - 1 / n, n)
    sampled_indices = np.searchsorted(cdf, uniform_pts)
    sampled_indices = np.clip(sampled_indices, 0, len(scores) - 1)
    return sampled_indices


def select_gap_frames(
    history,          # list of dict: {"chunk_idx": int, "latent": (B,C,T,H,W), "clip_feat": (768,)}
    current_chunk_idx,
    current_prompt,   # str: 当前 chunk 对应的 CLIP prompt (不做插值，过渡时直接用下一个)
    clip_model,       # CLIP 实例
    k=8,              # 最多选几帧
    alpha=0.6,        # 视觉权重 (文本权重 = 1-alpha)
    power=2.0,        # ITS 锐度
    min_chunk_distance=3,  # 与 target 最小 chunk 距离，默认跳过最近两段
    device="cuda",
):
    """从历史 GAP chunk 中用 ITS 选出 k 个最有参考价值的 chunk。

    GAP 候选 = history[0 : current_chunk_idx - 1] 中满足
              (current_chunk_idx - chunk_idx) >= min_chunk_distance 的历史
    context  = history[current_chunk_idx - 1]      (上一段，提供 visual query)

    Returns:
        selected_latents: list of Tensor (B, C, T, H, W)
        selected_indices: list of int (被选中的 chunk_idx)
    """
    if current_chunk_idx < 2 or len(history) < 2:
        return [], []

    # 排除 context（上一段）后，再应用最小时间距离阈值
    raw_gap_history = history[:current_chunk_idx - 1]
    gap_history = [
        e for e in raw_gap_history
        if (current_chunk_idx - int(e["chunk_idx"])) >= int(min_chunk_distance)
    ]
    context_entry = history[current_chunk_idx - 1]
    N_gap = len(gap_history)
    if N_gap == 0:
        return [], []

    # visual query: context chunk 的 CLIP 特征
    visual_query = context_entry["clip_feat"].unsqueeze(0).to(device)

    # text query: 当前 prompt 的 CLIP 特征
    text_query = clip_model.extract_text_features(current_prompt).to(device)

    # 堆叠所有 GAP 帧的 CLIP 特征
    gap_feats = torch.stack([e["clip_feat"] for e in gap_history], dim=0).to(device)

    # 计算相似度
    visual_scores = clip_model.compute_similarity(gap_feats, visual_query).numpy()
    text_scores = clip_model.compute_similarity(gap_feats, text_query).numpy()
    combined_scores = alpha * visual_scores + (1 - alpha) * text_scores

    # ITS 选帧
    actual_k = min(k, N_gap)
    sampled_positions = inverse_transform_sampling(combined_scores, n=actual_k, power=power)
    sampled_positions = list(dict.fromkeys(sampled_positions.tolist()))  # 去重

    selected_latents = [gap_history[p]["latent"] for p in sampled_positions]
    selected_indices = [gap_history[p]["chunk_idx"] for p in sampled_positions]

    return selected_latents, selected_indices
