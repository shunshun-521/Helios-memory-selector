"""
Frame Selector: 从 GAP 帧中选择关键帧，借鉴 SFI 的 Reverse-KL Fusion + Log-score Refinement。

用于 Helios Stage 1 训练，从金字塔历史帧与首帧之间的 GAP 区域中挑选关键帧作为额外历史输入。
"""

import torch
import torch.nn.functional as F


def select_key_frames_from_gap(
    gap_latents,            # (B, C, T_gap, H, W) — GAP 区域所有帧的 latent
    gap_frame_indices,      # (B, T_gap) — 每帧的绝对位置索引
    history_latents,        # (B, C, T_hist, H, W) — 已有的金字塔历史帧
    target_latents,         # (B, C, T_tgt, H, W) — 要预测的目标帧
    k_select,               # int — 要选择的关键帧数量
    alpha=1.0,              # log-score 温度
    cross_frame_alpha=0.85, # 去冗余系数
    pos_power=1.8,          # 位置先验幂次
    pos_eta=0.4,            # 位置先验权重
):
    """
    从 GAP 帧中选择 K 个关键帧。

    Returns:
        selected_latents: (B, C, K, H, W)
        selected_frame_indices: (B, K)
    """
    B, C, T_gap, H, W = gap_latents.shape

    if T_gap <= k_select:
        return gap_latents, gap_frame_indices

    with torch.no_grad():
        # 全部用 float32 计算（避免 bfloat16 + F.normalize 的 dtype 不一致）
        gap_flat = gap_latents.float().flatten(3).permute(0, 2, 1, 3).flatten(2)       # (B, T_gap, C*H*W)
        target_flat = target_latents.float().flatten(3).permute(0, 2, 1, 3).flatten(2) # (B, T_tgt, C*H*W)

        gap_norm = F.normalize(gap_flat, dim=-1)
        target_norm = F.normalize(target_flat, dim=-1)

        # 1. Attention-like scores: (B, T_gap, T_tgt)
        scores = torch.bmm(gap_norm, target_norm.transpose(1, 2))

        # 2. Log-score Refinement
        log_scores = torch.logsumexp(scores / alpha, dim=2)  # (B, T_gap)

        # 3. Reverse-KL Fusion: 去冗余
        if history_latents is not None and history_latents.shape[2] > 0:
            hist_flat = history_latents.float().flatten(3).permute(0, 2, 1, 3).flatten(2)
            hist_norm = F.normalize(hist_flat, dim=-1)
            redundancy = torch.bmm(gap_norm, hist_norm.transpose(1, 2))  # (B, T_gap, T_hist)
            redundancy_score = redundancy.max(dim=2).values              # (B, T_gap)
            importance = log_scores - cross_frame_alpha * redundancy_score
        else:
            importance = log_scores

        # 4. 位置先验
        if gap_frame_indices is not None:
            max_idx = gap_frame_indices.max(dim=1, keepdim=True).values.clamp(min=1)
            pos_normalized = gap_frame_indices.float() / max_idx
            pos_prior = pos_eta * (4 * pos_normalized * (1 - pos_normalized)) ** pos_power
            importance = importance + pos_prior.to(importance.device, importance.dtype)

        # 5. 贪心多样性 top-k
        selected_indices = _greedy_diverse_topk(importance, gap_norm, k_select)

    # 按时间顺序排列
    selected_indices, _ = selected_indices.sort(dim=1)

    # Gather
    idx_t = selected_indices.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # (B, 1, K, 1, 1)
    idx_t = idx_t.expand(B, C, k_select, H, W)
    selected_latents = torch.gather(gap_latents, 2, idx_t)
    selected_frame_indices = torch.gather(gap_frame_indices, 1, selected_indices)

    return selected_latents, selected_frame_indices


def _greedy_diverse_topk(importance, gap_norm, k):
    """贪心选择：每次选最重要的帧，然后降低与它相似的帧的权重。"""
    B, T_gap, D = gap_norm.shape
    selected = torch.zeros(B, k, dtype=torch.long, device=importance.device)
    adj = importance.clone()

    for i in range(k):
        idx = adj.argmax(dim=1)  # (B,)
        selected[:, i] = idx

        chosen = torch.gather(gap_norm, 1, idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, D))  # (B, 1, D)
        sim = torch.bmm(chosen, gap_norm.transpose(1, 2)).squeeze(1)  # (B, T_gap)
        adj = adj - 0.5 * sim
        adj.scatter_(1, idx.unsqueeze(1), float('-inf'))

    return selected
