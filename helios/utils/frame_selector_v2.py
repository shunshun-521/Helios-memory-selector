import torch
import torch.nn.functional as F

def select_key_frames_from_gap(
    gap_latents,            # (B, C, T_gap, H, W)
    gap_frame_indices,      # (B, T_gap)
    history_latents,        # (B, C, T_hist, H, W)
    history_frame_indices,  # (B, T_hist) - 传入已有历史帧的索引
    target_latents,         # (B, C, T_tgt, H, W)
    k_select,               
    alpha=1.0,              
    pos_gamma=0.1,          
    lambda_clip=0.5,        
    cross_frame_alpha=0.85, 
):
    """
    基于 SFI 理论的关键帧选择器，附带位置索引打印功能。
    """
    print(f"[SELECTOR] 进入 select_key_frames_from_gap", flush=True)
    
    B, C, T_gap, H, W = gap_latents.shape
    print(f"[SELECTOR] gap_latents shape: {gap_latents.shape}, gap_frame_indices shape: {gap_frame_indices.shape}", flush=True)
    print(f"[SELECTOR] history_frame_indices shape: {history_frame_indices.shape}", flush=True)

    # 处理样本不足的情况
    if T_gap <= k_select:
        print(f"[SELECTOR] T_gap ({T_gap}) <= k_select ({k_select}), 全选所有 GAP 帧", flush=True)
        _print_selection_stats(history_frame_indices, gap_frame_indices)
        return gap_latents, gap_frame_indices

    with torch.no_grad():
        # --- 1. 特征提取与归一化 ---
        gap_flat = gap_latents.float().flatten(3).permute(0, 2, 1, 3).flatten(2)
        target_flat = target_latents.float().flatten(3).permute(0, 2, 1, 3).flatten(2)
        gap_norm = F.normalize(gap_flat, dim=-1)
        target_norm = F.normalize(target_flat, dim=-1)

        # --- 2. 计算证据分布 f ---
        scores = torch.bmm(gap_norm, target_norm.transpose(1, 2))
        log_scores = torch.logsumexp(scores / alpha, dim=2)
        
        #是否冗余？
        if history_latents is not None and history_latents.shape[2] > 0:
            hist_flat = history_latents.float().flatten(3).permute(0, 2, 1, 3).flatten(2)
            hist_norm = F.normalize(hist_flat, dim=-1)
            redundancy = torch.bmm(gap_norm, hist_norm.transpose(1, 2)).max(dim=2).values
            log_scores = log_scores - cross_frame_alpha * redundancy
        
        f = F.softmax(log_scores, dim=-1)

        # --- 3. 计算先验分布 r (位置偏置 + 范数抑制) ---
        max_gap_idx = gap_frame_indices.max(dim=1, keepdim=True).values
        dist = torch.abs(gap_frame_indices.float() - max_gap_idx)
        pos_bias = torch.exp(-pos_gamma * dist)
        
        latents_l2_norm = torch.norm(gap_flat, p=2, dim=-1).clamp(min=1e-6)
        r_unnorm = pos_bias / latents_l2_norm
        r = F.softmax(r_unnorm, dim=-1)

        # --- 4. SFI Eq.18 动态 Lambda 融合 ---
        f_sq_norm = torch.sum(f**2, dim=-1)
        r_sq_norm = torch.sum(r**2, dim=-1)
        f_dot_r = torch.sum(f * r, dim=-1)
        
        num = f_sq_norm - f_dot_r
        den = f_sq_norm - 2 * f_dot_r + r_sq_norm
        lam = (num / den.clamp(min=1e-8)).clamp(0, lambda_clip)
        
        importance = (1 - lam.unsqueeze(1)) * f + lam.unsqueeze(1) * r

        # --- 5. 贪心 Top-K 选择 ---
        selected_indices = _greedy_diverse_topk(importance, gap_norm, k_select)

    # 排序并提取结果
    selected_indices, _ = selected_indices.sort(dim=1)
    
    idx_t = selected_indices.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand(B, C, k_select, H, W)
    selected_latents = torch.gather(gap_latents, 2, idx_t)
    selected_gap_frame_indices = torch.gather(gap_frame_indices, 1, selected_indices)

    print(f"[SELECTOR] 选择完成，selected_gap_frame_indices shape: {selected_gap_frame_indices.shape}", flush=True)
    
    # --- 6. 打印位置索引功能 ---
    # 我们只打印 Batch 中第一个样本的情况，避免日志刷屏
    print(f"[SELECTOR] 调用 _print_selection_stats", flush=True)
    _print_selection_stats(history_frame_indices[0], selected_gap_frame_indices[0], total_gap_indices=gap_frame_indices[0])

    return selected_latents, selected_gap_frame_indices


def _print_selection_stats(hist_indices, selected_gap_indices, total_gap_indices=None):
    """
    辅助打印函数
    """
    print("\n" + "="*80, flush=True)
    print(" [SFI Frame Selector Statistics] ", flush=True)
    print("="*80, flush=True)
    
    if total_gap_indices is not None:
        print(f"1. GAP 区域总帧数: {len(total_gap_indices)}", flush=True)
        print(f"   所有可选 GAP 帧索引: {total_gap_indices.tolist()}", flush=True)
    
    print(f"\n2. 已有历史帧索引 (History):", flush=True)
    print(f"   {hist_indices.tolist()}", flush=True)
    
    print(f"\n3. 从 GAP 中挑选的关键帧索引 (Selected from GAP):", flush=True)
    print(f"   {selected_gap_indices.tolist()}", flush=True)
    
    # 计算覆盖范围
    all_indices = sorted(hist_indices.tolist() + selected_gap_indices.tolist())
    print(f"\n4. 最终输入模型的完整序列索引 (All frames for training):", flush=True)
    print(f"   History: {hist_indices.tolist()}", flush=True)
    print(f"   Selected GAP: {selected_gap_indices.tolist()}", flush=True)
    print(f"   Combined: {all_indices}", flush=True)
    print("="*80 + "\n", flush=True)


def _greedy_diverse_topk(importance, gap_norm, k):
    B, T_gap, D = gap_norm.shape
    selected = torch.zeros(B, k, dtype=torch.long, device=importance.device)
    adj = importance.clone()
    for i in range(k):
        idx = adj.argmax(dim=1)
        selected[:, i] = idx
        chosen = torch.gather(gap_norm, 1, idx.unsqueeze(1).unsqueeze(2).expand(-1, -1, D))
        sim = torch.bmm(chosen, gap_norm.transpose(1, 2)).squeeze(1)
        adj = adj * (1 - 0.5 * sim)
        adj.scatter_(1, idx.unsqueeze(1), -1e9)
    return selected
