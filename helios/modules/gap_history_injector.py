"""
GAP History Injector — 将 Selector 选出的 GAP 帧注入 history_long
================================================================

【核心思路】
Helios 的 history_long (16帧) 只包含最近的历史帧，随着视频变长，
早期的重要帧（如人物正脸、特定光照）会被挤出窗口而丢失。

本模块将 VLM Selector 选出的 GAP 帧 latent 替换到 history_long 中，
让 DiT 的 self-attention 能看到更有信息量的历史参考帧。

【注入策略】
- 保留 history_long 中最近的 N 帧（保持时间连续性）
- 用 GAP 帧替换最远的 K 帧（这些帧信息量最低）
- 不修改 indices（保持 RoPE 相对位置编码与训练一致）

【重要】
- 不再维护独立的 bank，统一使用 MemoryBank 作为数据源
- Selector 的 toregister_k_indices 直接对应 MemoryBank 的条目索引
"""

import torch
from typing import Optional, Tuple


class GAPHistoryInjector:
    """将 MemoryBank 中 Selector 选出的 GAP 帧注入 history_long。

    用法:
        injector = GAPHistoryInjector(k_inject=4)

        # 每个 chunk 生成前 (selector 选出 top-k 后):
        hook = injector.register_hook(transformer, memory_bank, selected_indices)

        # chunk 生成完毕后:
        hook.remove()
    """

    def __init__(self, k_inject: int = 4):
        self.k_inject = k_inject

    def inject_into_history_long(
        self,
        latents_history_long: torch.Tensor,
        indices_history_long: torch.Tensor,
        gap_latents: list,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """将选中的 GAP 帧 latent 注入 history_long。

        Args:
            latents_history_long: (B, C, 16, H, W)
            indices_history_long: (B, 16) 或 (16,)
            gap_latents: list of (C, T_chunk, H, W) tensors from MemoryBank

        Returns:
            new_latents: (B, C, 16, H, W)
            new_indices: 不修改，原样返回
        """
        if not gap_latents:
            return latents_history_long, indices_history_long

        device = latents_history_long.device
        dtype = latents_history_long.dtype
        B = latents_history_long.shape[0]
        num_long = latents_history_long.shape[2]  # 16

        k = min(self.k_inject, len(gap_latents), num_long // 2)
        if k == 0:
            return latents_history_long, indices_history_long

        # 从每个 chunk latent 中取中间帧: (C, T, H, W) → (C, H, W)
        single_frames = []
        for lat in gap_latents[:k]:
            if lat.ndim == 4:  # (C, T, H, W)
                mid = lat.shape[1] // 2
                single_frames.append(lat[:, mid, :, :])  # (C, H, W)
            elif lat.ndim == 3:  # (C, H, W) — 已经是单帧
                single_frames.append(lat)
            else:
                continue

        if not single_frames:
            return latents_history_long, indices_history_long

        k = len(single_frames)
        new_latents = latents_history_long.clone()
        # (C, H, W) stack on dim=1 → (C, K, H, W)
        gap_stack = torch.stack(single_frames, dim=1).to(device, dtype=dtype)
        gap_stack = gap_stack.unsqueeze(0).expand(B, -1, -1, -1, -1)  # (B, C, K, H, W)
        new_latents[:, :, :k, :, :] = gap_stack #PROBLEM 

        # 不修改 indices — 保持 RoPE 相对位置编码与训练一致
        return new_latents, indices_history_long

    def register_hook(
        self,
        transformer,
        memory_bank,
        selected_indices: Optional[torch.Tensor] = None,
    ):
        """注册 pre-forward hook，在 transformer forward 时自动注入 GAP 帧。

        Args:
            transformer: Helios DiT model
            memory_bank: MemoryBank 实例（唯一数据源）
            selected_indices: (K,) — selector 选出的 MemoryBank 索引

        Returns:
            hook handle (调用 .remove() 清理)
        """
        # 预先从 MemoryBank 取出选中的 latent，避免 hook 中索引变化
        gap_latents = []
        if selected_indices is not None and len(memory_bank) > 0:
            for idx in selected_indices[:self.k_inject]:
                idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
                if 0 <= idx_val < len(memory_bank):
                    gap_latents.append(memory_bank.entries[idx_val]["latent"])

        def pre_hook(module, args, kwargs):
            # ── DEBUG FrameMap 打印（仅第一个 denoising step）──
            _is_first = kwargs.get("is_first_denoising_step", False)
            if _is_first:
                _parts = []
                for _name, _key in [
                    ("target", "indices_hidden_states"),
                    ("short", "indices_latents_history_short"),
                    ("mid", "indices_latents_history_mid"),
                    ("long", "indices_latents_history_long"),
                ]:
                    _idx = kwargs.get(_key)
                    if _idx is not None:
                        _v = _idx[0].tolist() if _idx.ndim > 1 else _idx.tolist()
                        _parts.append(f"{_name}={_v}")
                if _parts:
                    _inj_info = f" | injecting={len(gap_latents)} GAP frames" if gap_latents else ""
                    print(f"[DEBUG FrameMap] {' | '.join(_parts)}{_inj_info}")

            if not gap_latents:
                return args, kwargs

            lat_long = kwargs.get("latents_history_long")
            idx_long = kwargs.get("indices_latents_history_long")
            if lat_long is None or idx_long is None:
                return args, kwargs

            new_lat, new_idx = self.inject_into_history_long(
                lat_long, idx_long, gap_latents,
            )
            kwargs = dict(kwargs)
            kwargs["latents_history_long"] = new_lat
            kwargs["indices_latents_history_long"] = new_idx
            return args, kwargs

        handle = transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
        return handle
