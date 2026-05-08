"""
Reference Attention — Stage 2 模块
====================================

在 DiT block 的 Self-Attn 和 Cross-Attn 之间插入的参考帧注意力层。

【核心设计】
- Q = DiT hidden_states (当前去噪帧的特征)
- K = VLM LHS (选中的 GAP 帧语义特征, 投影到 DiT 空间)
- V = VAE latent (选中的 GAP 帧的 latent, 投影到 DiT 空间)
- 输出 = alpha * Ref-Attn(Q, K, V)，alpha 初始化为 0 (zero-init)

【即插即用】
- zero-init alpha 保证训练初期 Ref-Attn 输出为零，不破坏原有生成质量
- 只插入 DiT 层 10-30 (中间层)，浅层和深层不插

【与 IP-Adapter 的类比】
- IP-Adapter: K=CLIP_image_embed, V=CLIP_image_embed → DiT cross-attn
- 我们: K=VLM_LHS, V=VAE_latent → DiT ref-attn (异构 K/V 空间)
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReferenceAttention(nn.Module):
    """单层 Reference Attention。

    插入位置: Self-Attn 之后, Cross-Attn 之前。
    x = x + alpha * ReferenceAttention(x, gap_lhs, gap_latent)

    Args:
        dit_dim: DiT 的 inner_dim (= num_heads * head_dim = 5120)
        vlm_hidden_dim: VLM 的 hidden_dim (Qwen2.5-VL-3B = 2048)
        latent_channels: VAE latent channels (= 16)
        num_heads: 注意力头数
        dropout: attention dropout
    """

    def __init__(
        self,
        dit_dim: int = 5120,
        vlm_hidden_dim: int = 2048,
        latent_channels: int = 16,
        num_heads: int = 40,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dit_dim = dit_dim
        self.num_heads = num_heads
        self.head_dim = dit_dim // num_heads

        # Q 投影: DiT hidden → attention space
        self.q_proj = nn.Linear(dit_dim, dit_dim, bias=False)

        # K 投影: VLM LHS → DiT attention space (跨空间桥接)
        self.k_proj = nn.Linear(vlm_hidden_dim, dit_dim, bias=False)

        # V 投影: VAE latent → DiT attention space (跨空间桥接)
        # latent 是 per-pixel 的 (C=16, H, W), 需要先 flatten 再投影
        # 实际输入形态: (B, N_selected, latent_token_dim) 
        # latent_token_dim 在 patchify 后 = latent_channels * p_h * p_w = 16 * 2 * 2 = 64
        self.v_proj = nn.Linear(vlm_hidden_dim, dit_dim, bias=False)
        # 注意: V 先用 LHS 作为 fallback (K=V=LHS), 后续可切换为 latent
        # self.v_proj_latent = nn.Linear(latent_patch_dim, dit_dim, bias=False)

        # Output projection
        self.out_proj = nn.Linear(dit_dim, dit_dim, bias=False)

        # Zero-init gate (alpha)
        # 初始为 0, 训练时逐步学习放大
        self.alpha = nn.Parameter(torch.zeros(1))

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.zeros_(self.out_proj.weight)  # zero-init output projection

    def forward(
        self,
        hidden_states: torch.Tensor,    # (B, S, dit_dim) — DiT 当前帧特征
        ref_key: torch.Tensor,          # (B, N_ref, vlm_hidden_dim) — 选中帧的 VLM LHS
        ref_value: Optional[torch.Tensor] = None,  # (B, N_ref, vlm_hidden_dim) — 选中帧的 V
        # 如果 ref_value 为 None, 则 V = ref_key (K=V=LHS fallback)
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: DiT block 中间特征 (B, S, D)
            ref_key: 选中 GAP 帧的 VLM LHS (B, N_ref, vlm_dim)
            ref_value: 选中 GAP 帧的 V 特征 (B, N_ref, vlm_dim), 默认=ref_key

        Returns:
            output: (B, S, D) — 加了 alpha gate 的参考注意力输出
        """
        if ref_key is None or ref_key.shape[1] == 0:
            return torch.zeros_like(hidden_states)

        if ref_value is None:
            ref_value = ref_key  # K=V=LHS fallback

        B, S, D = hidden_states.shape
        N_ref = ref_key.shape[1]

        # Project Q, K, V
        Q = self.q_proj(hidden_states)               # (B, S, D)
        K = self.k_proj(ref_key)                      # (B, N_ref, D)
        V = self.v_proj(ref_value)                    # (B, N_ref, D)

        # Reshape to multi-head: (B, num_heads, seq_len, head_dim)
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_ref, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_ref, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (B, H, S, N_ref)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (B, H, S, head_dim)

        # Merge heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, D)

        # Output projection + zero-init gate
        output = self.out_proj(attn_output)
        output = self.alpha * output

        return output


class ReferenceAttentionLayers(nn.Module):
    """管理多层 Reference Attention 的容器。

    只在 DiT 层 10-30 (中间层) 插入 Ref-Attn。

    【用法】
    ```python
    ref_attn_layers = ReferenceAttentionLayers(
        dit_dim=5120, num_dit_layers=40, active_layers=list(range(10, 31))
    )

    for iidx, block in enumerate(dit_blocks):
        hidden_states = block.self_attn(hidden_states, ...)
        
        # 插入 Ref-Attn
        if ref_attn_layers.is_active(iidx):
            hidden_states = hidden_states + ref_attn_layers(iidx, hidden_states, ref_key, ref_value)
        
        hidden_states = block.cross_attn(hidden_states, ...)
        hidden_states = block.ffn(hidden_states, ...)
    ```
    """

    def __init__(
        self,
        dit_dim: int = 5120,
        vlm_hidden_dim: int = 2048,
        latent_channels: int = 16,
        num_heads: int = 40,
        num_dit_layers: int = 40,
        active_layers: Optional[List[int]] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        if active_layers is None:
            active_layers = list(range(10, 31))  # 默认层 10-30

        self.active_layers = set(active_layers)

        # 只为 active 层创建 Ref-Attn 模块
        self.ref_attn_modules = nn.ModuleDict()
        for layer_idx in active_layers:
            self.ref_attn_modules[str(layer_idx)] = ReferenceAttention(
                dit_dim=dit_dim,
                vlm_hidden_dim=vlm_hidden_dim,
                latent_channels=latent_channels,
                num_heads=num_heads,
                dropout=dropout,
            )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[RefAttnLayers] {len(active_layers)} layers, "
              f"{trainable_params:,} trainable params ({trainable_params/1e6:.1f}M)")

    def is_active(self, layer_idx: int) -> bool:
        return layer_idx in self.active_layers

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        ref_key: torch.Tensor,
        ref_value: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """对指定层应用 Reference Attention。

        Args:
            layer_idx: DiT block 索引
            hidden_states: (B, S, D)
            ref_key: (B, N_ref, vlm_dim)
            ref_value: (B, N_ref, vlm_dim) or None

        Returns:
            ref_attn_output: (B, S, D) — 需要加到 hidden_states 上
        """
        if not self.is_active(layer_idx):
            return torch.zeros_like(hidden_states)

        ref_attn = self.ref_attn_modules[str(layer_idx)]
        return ref_attn(hidden_states, ref_key, ref_value)

    # ─── Hook Management (统一接口，供 train 和 inference 共用) ───

    def _find_self_attn_target(self, block, block_idx: int):
        """在 DiT block 中查找 self-attention 子模块。

        支持多种 block 命名约定: attn1, self_attn, attn 等。
        找不到时 raise RuntimeError 而非静默跳过（Fix #5）。
        """
        for attr_name in ("attn1", "self_attn", "attn"):
            if hasattr(block, attr_name):
                return getattr(block, attr_name)
        # 静默失败是严重 bug，必须报错
        attn_attrs = [k for k in dir(block) if "attn" in k.lower()]
        raise RuntimeError(
            f"Block {block_idx} 没有 attn1/self_attn 属性，"
            f"可用 attn 相关属性: {attn_attrs}。"
            f"请检查 DiT block 结构: {type(block).__name__}"
        )

    def _find_blocks(self, transformer):
        """在 transformer 中查找 block 列表。"""
        for attr_name in ("blocks", "transformer_blocks"):
            if hasattr(transformer, attr_name):
                blocks = getattr(transformer, attr_name)
                if len(blocks) > 0:
                    return blocks
        raise RuntimeError(
            f"Transformer ({type(transformer).__name__}) 没有 blocks 或 "
            f"transformer_blocks 属性。请检查模型结构。"
        )

    def register_hooks(
        self,
        transformer,
        ref_key: Optional[torhuijinh.Tensor] = None,
        ref_value: Optional[torch.Tensor] = None,
        training_mode: bool = False,
    ) -> List:
        """对 transformer 的 blocks 注册 Ref-Attn forward hooks。

        Args:
            transformer: Helios DiT model
            ref_key: (B, N_ref, vlm_dim)
            ref_value: (B, N_ref, vlm_dim) or None
            training_mode: 如果 True，只捕获 hidden_states 不修改 forward。
                          用于 two-pass 训练: 先 no_grad 捕获，再单独计算 ref_attn loss。

        Returns:
            hooks: list of hook handles
        """
        hooks = []
        blocks = self._find_blocks(transformer)

        if training_mode:
            # 初始化捕获字典
            self._captured_states = {}

        registered_count = 0
        for idx, block in enumerate(blocks):
            if not self.is_active(idx):
                continue
            target = self._find_self_attn_target(block, idx)

            if training_mode:
                # Training: 只捕获，不修改 forward
                def make_hook(layer_idx):
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            hs = output[0]
                        else:
                            hs = output
                        self._captured_states[layer_idx] = hs.detach()
                        return output  # 不修改
                    return hook_fn
            else:
                # Inference: 注入 ref_attn
                def make_hook(layer_idx):
                    def hook_fn(module, input, output):
                        if ref_key is None:
                            return output
                        if isinstance(output, tuple):
                            hs = output[0]
                        else:
                            hs = output
                        hs_detached = hs.detach()
                        ref_out = self.forward(layer_idx, hs_detached, ref_key, ref_value)
                        hs_new = hs_detached + ref_out
                        if isinstance(output, tuple):
                            return (hs_new,) + output[1:]
                        return hs_new
                    return hook_fn

            h = target.register_forward_hook(make_hook(idx))
            hooks.append(h)
            registered_count += 1

        if registered_count == 0:
            import warnings
            warnings.warn(
                f"[RefAttnLayers] 没有注册任何 hook！"
                f"active_layers={sorted(self.active_layers)}, "
                f"total blocks={len(blocks)}"
            )
        return hooks

    def compute_training_loss(
        self,
        ref_key: torch.Tensor,
        ref_value: Optional[torch.Tensor],
        target_states: dict,
    ) -> torch.Tensor:
        """Two-pass 训练 loss: hidden_states 空间的 feature alignment。

        对每个 active layer:
          ref_out = ref_attn(baseline_hs, ref_key)
          target_residual = target_hs - baseline_hs  (detached)
          loss += MSE(ref_out, target_residual)

        Args:
            ref_key: (B, N_ref, vlm_dim)
            ref_value: (B, N_ref, vlm_dim) or None
            target_states: dict {layer_idx: target_hs} — 从 GT forward 捕获的 hidden_states

        Returns:
            loss: scalar with grad_fn through ref_attn params
        """
        if not hasattr(self, '_captured_states') or not self._captured_states:
            raise RuntimeError("No captured states.")

        total_loss = torch.tensor(0.0, device=ref_key.device)
        n_layers = 0

        for layer_idx in sorted(self._captured_states.keys()):
            if layer_idx not in target_states:
                continue
            baseline_hs = self._captured_states[layer_idx]  # (B, S, D) detached
            target_hs = target_states[layer_idx]             # (B, S, D) detached
            target_residual = (target_hs - baseline_hs).detach()

            # ref_attn: differentiable
            ref_out = self.forward(layer_idx, baseline_hs, ref_key, ref_value)
            total_loss = total_loss + F.mse_loss(ref_out, target_residual)
            n_layers += 1

        if n_layers > 0:
            total_loss = total_loss / n_layers
        return total_loss

    def clear_captured(self):
        """清除捕获的 hidden_states。"""
        if hasattr(self, '_captured_states'):
            self._captured_states.clear()

    @staticmethod
    def remove_hooks(hooks: List):
        """移除已注册的 hooks。"""
        for h in hooks:
            h.remove()
        hooks.clear()
