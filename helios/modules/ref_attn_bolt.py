"""
ref_attn_bolt.py — BOLT Reference Attention (VAE latent patchify)
==================================================================

基于 ref_attn.py 的 ReferenceAttention 架构，但 K/V 来源从 VLM LHS 改为
VAE latent patchify。用于将 BOLT ITS 选出的 GAP 帧以跨空间注意力形式注入 DiT。

【核心区别】
- ref_attn.py: K=V=VLM LHS (语义空间 2048 维)
- ref_attn_bolt.py: K=V=VAE latent patch (512 维, patch_size=(2,4,4))

【设计】
- Q = DiT hidden_states (5120) × Wq → (5120)
- K = latent patch (16×2×4×4=512) × Wk → (5120)  ← 跨空间桥接
- V = latent patch (512) × Wv → (5120)
- 输出 = gamma * out_proj(Attn(Q, K, V))
  - gamma ∈ R^{dit_dim} 是 LayerScale per-channel 门（初始化为 1e-4，非零）
  - out_proj 使用普通 xavier_uniform_（不再用 gain=0.01）
  - 初始 output 非零 → Wq/Wk/Wv/out_proj 首步即有梯度（避开 zero-gate 死锁）

【历史】
- v1：标量 alpha = zero-init，搭配 out_proj xavier gain=0.01
  实测 50 epoch 训练中 α 只爬到 ~1e-3，Ref-Attn 分支几乎等同未启用
  详见 md/bolt_integration_plan_alpha.md
- v2（当前）：LayerScale per-channel gamma（CaiT 风格），初始 1e-4 + 正常 xavier
  优点：初始非零、无死锁、训练更顺
"""

import math
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════
# Patchify
# ═══════════════════════════════════════════

def patchify_latent(
    latent: torch.Tensor,
    patch_size: tuple = (2, 4, 4),
) -> torch.Tensor:
    """将 VAE latent 切分为 patch 序列。

    Args:
        latent: (B, C, T, H, W) — 一个或多个 GAP 帧的 VAE latent
        patch_size: (p_t, p_h, p_w) — 默认 (2, 4, 4)

    Returns:
        patches: (B, N_patches, C * p_t * p_h * p_w)
                 其中 N_patches = (T//p_t) * (H//p_h) * (W//p_w)
    """
    B, C, T, H, W = latent.shape
    p_t, p_h, p_w = patch_size

    # Pad if not divisible
    pad_t = (p_t - T % p_t) % p_t
    pad_h = (p_h - H % p_h) % p_h
    pad_w = (p_w - W % p_w) % p_w
    if pad_t > 0 or pad_h > 0 or pad_w > 0:
        latent = F.pad(latent, (0, pad_w, 0, pad_h, 0, pad_t), mode="replicate")
        T, H, W = T + pad_t, H + pad_h, W + pad_w

    # Reshape into patches
    latent = latent.reshape(B, C, T // p_t, p_t, H // p_h, p_h, W // p_w, p_w)
    latent = latent.permute(0, 2, 4, 6, 1, 3, 5, 7)  # (B, nT, nH, nW, C, p_t, p_h, p_w)
    patches = latent.reshape(B, -1, C * p_t * p_h * p_w)  # (B, N_patches, patch_dim)
    return patches


def patchify_selected_latents(
    selected_latents: list,
    patch_size: tuple = (2, 4, 4),
) -> torch.Tensor:
    """将多个选中的 GAP 帧 latent 拼接后 patchify。

    Args:
        selected_latents: list of (B, C, T, H, W) tensors
        patch_size: (p_t, p_h, p_w)

    Returns:
        all_patches: (B, total_patches, patch_dim)
    """
    if not selected_latents:
        return None

    patch_list = []
    for lat in selected_latents:
        if lat.ndim == 4:  # (C, T, H, W) → (1, C, T, H, W)
            lat = lat.unsqueeze(0)
        patches = patchify_latent(lat, patch_size)
        patch_list.append(patches)

    return torch.cat(patch_list, dim=1)  # (B, sum_patches, patch_dim)


# ═══════════════════════════════════════════
# Single-layer Bolt Reference Attention
# ═══════════════════════════════════════════

class BoltReferenceAttention(nn.Module):
    """单层 BOLT Reference Attention（LayerScale 版）。

    插入位置: Self-Attn 之后, Cross-Attn 之前。
    x = x + gamma * BoltRefAttn(x, gap_latent_patches)

    Args:
        dit_dim: DiT inner_dim (5120)
        latent_patch_dim: C * p_t * p_h * p_w = 16 * 2 * 4 * 4 = 512
        num_heads: 注意力头数
        attn_dim: bottleneck attention dimension (default 2560)
        dropout: attention dropout
        use_checkpoint: 是否使用 gradient checkpointing 节省显存
        gamma_init: LayerScale γ 的初始值（per-channel 常数）。默认 1e-4。
    """

    def __init__(
        self,
        dit_dim: int = 5120,
        latent_patch_dim: int = 512,
        num_heads: int = 20,
        attn_dim: int = 2560,
        dropout: float = 0.0,
        use_checkpoint: bool = True,
        gamma_init: float = 1e-4,
    ):
        super().__init__()
        self.dit_dim = dit_dim
        self.attn_dim = attn_dim
        self.num_heads = num_heads
        self.head_dim = attn_dim // num_heads
        self.use_checkpoint = use_checkpoint
        self.gamma_init = gamma_init

        # Q: DiT hidden → bottleneck attention space
        self.q_proj = nn.Linear(dit_dim, attn_dim, bias=False)
        # K: latent patch → bottleneck attention space (跨空间桥接)
        self.k_proj = nn.Linear(latent_patch_dim, attn_dim, bias=False)
        # V: latent patch → bottleneck attention space
        self.v_proj = nn.Linear(latent_patch_dim, attn_dim, bias=False)
        # Output projection: bottleneck → DiT
        self.out_proj = nn.Linear(attn_dim, dit_dim, bias=False)

        # LayerScale per-channel gate (CaiT / DeiT-III 风格)
        # 初始 1e-4：output 天然非零，避免"标量 alpha zero-init"导致
        # Wq/Wk/Wv/out_proj 初步梯度为 0 的死锁。
        self.gamma = nn.Parameter(torch.full((dit_dim,), gamma_init))

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

        # ─── 监控用 buffers（不进 state_dict，不影响 resume）─────────
        # r = ‖γ·out_proj(Attn)‖ / ‖hidden_states‖（有效注入比）
        self.register_buffer("_last_r", torch.tensor(0.0), persistent=False)
        # softmax(QK^T/√d) 在 N_ref 维上的平均熵（每次 forward 更新）
        # 收敛信号: 从 log(N_ref) 单调下降到 < log(N_ref) * 0.7 表示注意力变 selective
        self.register_buffer("_attn_entropy", torch.tensor(0.0), persistent=False)
        # softmax(QK^T/√d) 沿 N_ref 维的 max 概率均值
        # 收敛信号: 从 1/N_ref 上升到 > 0.3 表示注意力开始聚焦
        self.register_buffer("_attn_max_prob", torch.tensor(0.0), persistent=False)
        # K/V 投影输出的平均 L2 范数（监控 Wk/Wv 是否真的在塑形输入）
        self.register_buffer("_kv_norm", torch.tensor(0.0), persistent=False)

        # 权重初始快照（用于计算 ‖W - W_init‖_F；persistent=False 避免污染 state_dict）
        # 注意: __init__ 阶段先用 zeros 占位，等 _init_weights() 跑完才能拿到真实的 xavier 初值。
        # 在 _init_weights() 末尾会同步到当前权重；resume 时由外部显式调 snapshot_initial_weights()
        # 重新对齐到"本次训练起点"。
        self.register_buffer("_out_proj_init", torch.zeros_like(self.out_proj.weight), persistent=False)
        self.register_buffer("_q_proj_init", torch.zeros_like(self.q_proj.weight), persistent=False)
        self.register_buffer("_k_proj_init", torch.zeros_like(self.k_proj.weight), persistent=False)
        self.register_buffer("_v_proj_init", torch.zeros_like(self.v_proj.weight), persistent=False)

        self._init_weights()
        # _init_weights 之后立刻 snap，给"未 resume 的新 run"一个合理的 baseline
        self.snapshot_initial_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        # out_proj 恢复正常 xavier_uniform_（不再用 gain=0.01）。
        # LayerScale γ 在前面已经提供了小幅门控（~1e-4），
        # 搭配正常 xavier 的 out_proj 仍保证初始注入幅度足够小。
        nn.init.xavier_uniform_(self.out_proj.weight)

    @torch.no_grad()
    def snapshot_initial_weights(self):
        """把当前 Wq/Wk/Wv/out_proj 权重快照到 _*_init buffer。

        用法:
        - 新 run: __init__ 末尾自动调用一次（捕获 xavier 初值）
        - resume run: 训练循环开始前在 load_state_dict 之后**手动**调一次，
          这样 ‖W - W_init‖_F 度量的是"本次会话的位移"，而不是"距离最初 xavier"
        """
        self._out_proj_init.copy_(self.out_proj.weight.detach())
        self._q_proj_init.copy_(self.q_proj.weight.detach())
        self._k_proj_init.copy_(self.k_proj.weight.detach())
        self._v_proj_init.copy_(self.v_proj.weight.detach())

    def forward(
        self,
        hidden_states: torch.Tensor,       # (B, S, dit_dim)
        ref_patches: torch.Tensor,          # (B, N_ref, latent_patch_dim)
    ) -> torch.Tensor:
        if ref_patches is None or ref_patches.shape[1] == 0:
            return torch.zeros_like(hidden_states)

        B, S, _ = hidden_states.shape
        N_ref = ref_patches.shape[1]
        A = self.attn_dim

        Q = self.q_proj(hidden_states)       # (B, S, A)
        K = self.k_proj(ref_patches)         # (B, N_ref, A)
        V = self.v_proj(ref_patches)         # (B, N_ref, A)

        # Multi-head reshape
        Q = Q.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_ref, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, N_ref, self.num_heads, self.head_dim).transpose(1, 2)

        # Use F.scaled_dot_product_attention (memory efficient)
        attn_output = F.scaled_dot_product_attention(
            Q, K, V, dropout_p=self.dropout.p if self.training else 0.0,
        )  # (B, H, S, head_dim)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, S, A)

        # LayerScale: γ 形状 (dit_dim,)，会 broadcast 到 (B, S, dit_dim)
        output = self.out_proj(attn_output)
        output = self.gamma * output

        # ── 监控指标（不影响梯度，所有计算在 no_grad 下）──
        # 1) r = ‖γ·out_proj(Attn)‖ / ‖hidden_states‖  (S4 有效注入比)
        # 2) attention 熵 / max_prob：从 Q/K 重算 softmax 概率，反映注意力是否变 selective
        # 3) ‖K‖+‖V‖ 平均范数：反映 Wk/Wv 投影输出幅度
        # 注意性能：Q/K 矩阵很大时全量重算开销可观，对 query 维做下采样
        with torch.no_grad():
            ref_norm = output.detach().float().norm(dim=-1).mean()
            hs_norm = hidden_states.detach().float().norm(dim=-1).mean().clamp_min(1e-8)
            self._last_r.copy_((ref_norm / hs_norm).to(self._last_r.dtype))

            # 采样 ≤64 个 query 位置以控制开销
            S_query = Q.shape[2]
            if S_query > 64:
                q_idx = torch.randperm(S_query, device=Q.device)[:64]
                Q_mon = Q[:, :, q_idx, :].float()
            else:
                Q_mon = Q.float()
            K_mon = K.float()
            scores = (Q_mon @ K_mon.transpose(-1, -2)) / self.scale  # (B, H, S', N_ref)
            probs = scores.softmax(dim=-1)                            # (B, H, S', N_ref)
            eps = 1e-12
            entropy = -(probs * (probs + eps).log()).sum(dim=-1)      # (B, H, S')
            self._attn_entropy.copy_(entropy.mean().to(self._attn_entropy.dtype))
            self._attn_max_prob.copy_(probs.max(dim=-1).values.mean().to(self._attn_max_prob.dtype))

            kv_norm = 0.5 * (
                K.detach().float().norm(dim=-1).mean()
                + V.detach().float().norm(dim=-1).mean()
            )
            self._kv_norm.copy_(kv_norm.to(self._kv_norm.dtype))

        return output


# ═══════════════════════════════════════════
# Multi-layer Container
# ═══════════════════════════════════════════

class BoltReferenceAttentionLayers(nn.Module):
    """管理多层 Bolt Ref-Attn 的容器。

    只在 DiT 中间层 (如 layer 15-30) 插入。

    用法:
    ```python
    bolt_layers = BoltReferenceAttentionLayers(active_layers=list(range(15, 31)))

    # 推理: 注册 hooks
    hooks = bolt_layers.register_hooks(transformer, selected_latents)
    output = transformer(...)
    BoltReferenceAttentionLayers.remove_hooks(hooks)

    # 训练: 同上，但 transformer frozen, bolt_layers 有梯度
    ```
    """

    def __init__(
        self,
        dit_dim: int = 5120,
        latent_patch_dim: int = 512,
        num_heads: int = 20,
        attn_dim: int = 2560,
        active_layers: Optional[List[int]] = None,
        dropout: float = 0.0,
        patch_size: tuple = (2, 4, 4),
        gamma_init: float = 1e-4,
    ):
        super().__init__()
        if active_layers is None:
            active_layers = list(range(19, 27))

        self.active_layers = set(active_layers)
        self.patch_size = patch_size

        self.ref_attn_modules = nn.ModuleDict()
        for layer_idx in active_layers:
            self.ref_attn_modules[str(layer_idx)] = BoltReferenceAttention(
                dit_dim=dit_dim,
                latent_patch_dim=latent_patch_dim,
                num_heads=num_heads,
                attn_dim=attn_dim,
                dropout=dropout,
                gamma_init=gamma_init,
            )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[BoltRefAttnLayers] {len(active_layers)} layers, "
              f"{trainable_params:,} trainable params ({trainable_params/1e6:.1f}M)")

    def is_active(self, layer_idx: int) -> bool:
        return layer_idx in self.active_layers

    # ─── Optimizer / Logging Helpers ───

    def gamma_parameters(self):
        """返回所有 LayerScale γ 参数（需要单独用大 lr + weight_decay=0）。"""
        for n, p in self.named_parameters():
            if n.endswith(".gamma") or n.split(".")[-1] == "gamma":
                yield p

    def non_gamma_parameters(self):
        """返回除 γ 之外的所有可训练参数（Wq/Wk/Wv/out_proj）。"""
        for n, p in self.named_parameters():
            if not (n.endswith(".gamma") or n.split(".")[-1] == "gamma"):
                yield p

    @torch.no_grad()
    def snapshot_initial_weights(self):
        """对所有子模块调用 snapshot_initial_weights()。

        在 train_helios_bolt.py 训练循环开始前（特别是 load_state_dict 之后）调用一次，
        让 ‖W - W_init‖_F 度量"本次训练会话的位移"。
        """
        for m in self.ref_attn_modules.values():
            m.snapshot_initial_weights()

    @torch.no_grad()
    def clamp_gamma(self, low: float = -0.4, high: float = 0.4):
        """对所有层的 LayerScale γ 做硬 clamp（防 outlier 通道跑飞）。

        在 optimizer.step() 之后调用。`|γ|_max > 0.4` 区域已实证会破坏 DiT 特征分布。
        """
        for m in self.ref_attn_modules.values():
            m.gamma.data.clamp_(low, high)

    @torch.no_grad()
    def collect_stats(self):
        """汇总各层 γ / 权重范数 / 注意力分布 / 有效注入比 r，用于训练日志。

        Returns:
            dict: {
                # ─ γ ─
                "gamma_abs_mean", "gamma_abs_max",
                # ─ out_proj 绝对范数（与 W_init 无关）─
                "out_proj_fro", "out_proj_fro_max",
                # ─ ‖W - W_init‖_F（位移，相对本次训练起点）─
                "out_proj_delta_fro", "out_proj_rel_change",
                "qkv_delta_fro", "qkv_rel_change",
                # ─ 注意力分布（与 timestep 完全解耦的内在收敛信号）─
                "attn_entropy",     # softmax 熵（越小越 selective）
                "attn_max_prob",    # max attention 概率（越大越聚焦）
                "kv_norm",          # K/V 投影输出平均 L2 范数
                # ─ 有效注入比 r（S4）─
                "r_mean", "r_max",
                "per_layer": { layer_idx: {...各项...} },
            }
        """
        gamma_abs_all = []
        op_fros, op_dW_fros, op_W_init_fros = [], [], []
        qkv_dW_fros, qkv_W_init_fros = [], []
        rs = []
        attn_ents, attn_maxps, kv_norms = [], [], []
        per_layer = {}

        def _fro(t: torch.Tensor) -> float:
            return float(t.detach().float().norm().item())

        def _delta_fro(W: torch.Tensor, W_init: torch.Tensor) -> float:
            # 在 float 下做减法，避免 bf16 大数减大数精度损失
            return float((W.detach().float() - W_init.float()).norm().item())

        for k, m in self.ref_attn_modules.items():
            g_abs = m.gamma.detach().abs()
            op_fro = _fro(m.out_proj.weight)
            op_dW = _delta_fro(m.out_proj.weight, m._out_proj_init)
            op_W_init = _fro(m._out_proj_init)

            qkv_dW = (
                _delta_fro(m.q_proj.weight, m._q_proj_init) ** 2
                + _delta_fro(m.k_proj.weight, m._k_proj_init) ** 2
                + _delta_fro(m.v_proj.weight, m._v_proj_init) ** 2
            ) ** 0.5
            qkv_W_init = (
                _fro(m._q_proj_init) ** 2
                + _fro(m._k_proj_init) ** 2
                + _fro(m._v_proj_init) ** 2
            ) ** 0.5

            r_val = float(m._last_r.detach().item())
            ent = float(m._attn_entropy.detach().item())
            maxp = float(m._attn_max_prob.detach().item())
            kvn = float(m._kv_norm.detach().item())

            gamma_abs_all.append(g_abs)
            op_fros.append(op_fro)
            op_dW_fros.append(op_dW)
            op_W_init_fros.append(op_W_init)
            qkv_dW_fros.append(qkv_dW)
            qkv_W_init_fros.append(qkv_W_init)
            rs.append(r_val)
            attn_ents.append(ent)
            attn_maxps.append(maxp)
            kv_norms.append(kvn)

            per_layer[int(k)] = {
                "gamma_abs_mean": float(g_abs.mean().item()),
                "gamma_abs_max":  float(g_abs.max().item()),
                "out_proj_fro":       op_fro,
                "out_proj_delta_fro": op_dW,
                "out_proj_rel_change": op_dW / max(op_W_init, 1e-8),
                "qkv_delta_fro":   qkv_dW,
                "qkv_rel_change":  qkv_dW / max(qkv_W_init, 1e-8),
                "r": r_val,
                "attn_entropy": ent,
                "attn_max_prob": maxp,
                "kv_norm": kvn,
            }

        if not gamma_abs_all:
            return {
                "gamma_abs_mean": 0.0, "gamma_abs_max": 0.0,
                "out_proj_fro": 0.0, "out_proj_fro_max": 0.0,
                "out_proj_delta_fro": 0.0, "out_proj_rel_change": 0.0,
                "qkv_delta_fro": 0.0, "qkv_rel_change": 0.0,
                "attn_entropy": 0.0, "attn_max_prob": 0.0, "kv_norm": 0.0,
                "r_mean": 0.0, "r_max": 0.0,
                "per_layer": {},
            }
        gamma_cat = torch.cat([g.flatten() for g in gamma_abs_all])
        op_dW_mean = sum(op_dW_fros) / len(op_dW_fros)
        op_W_init_mean = sum(op_W_init_fros) / len(op_W_init_fros)
        qkv_dW_mean = sum(qkv_dW_fros) / len(qkv_dW_fros)
        qkv_W_init_mean = sum(qkv_W_init_fros) / len(qkv_W_init_fros)
        return {
            "gamma_abs_mean": float(gamma_cat.mean().item()),
            "gamma_abs_max":  float(gamma_cat.max().item()),
            "out_proj_fro":       float(sum(op_fros) / len(op_fros)),
            "out_proj_fro_max":   float(max(op_fros)),
            "out_proj_delta_fro": float(op_dW_mean),
            "out_proj_rel_change": float(op_dW_mean / max(op_W_init_mean, 1e-8)),
            "qkv_delta_fro":   float(qkv_dW_mean),
            "qkv_rel_change":  float(qkv_dW_mean / max(qkv_W_init_mean, 1e-8)),
            "attn_entropy":  float(sum(attn_ents) / len(attn_ents)),
            "attn_max_prob": float(sum(attn_maxps) / len(attn_maxps)),
            "kv_norm":       float(sum(kv_norms) / len(kv_norms)),
            "r_mean":        float(sum(rs) / len(rs)),
            "r_max":         float(max(rs)),
            "per_layer": per_layer,
        }

    def forward(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        ref_patches: torch.Tensor,
    ) -> torch.Tensor:
        if not self.is_active(layer_idx):
            return torch.zeros_like(hidden_states)
        module = self.ref_attn_modules[str(layer_idx)]
        if module.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(
                module, hidden_states, ref_patches, use_reentrant=False,
            )
        return module(hidden_states, ref_patches)

    # ─── Hook Management ───

    def _find_blocks(self, transformer):
        for attr_name in ("blocks", "transformer_blocks"):
            if hasattr(transformer, attr_name):
                blocks = getattr(transformer, attr_name)
                if len(blocks) > 0:
                    return blocks
        raise RuntimeError(
            f"Transformer ({type(transformer).__name__}) 没有 blocks 或 "
            f"transformer_blocks 属性。"
        )

    def _find_self_attn_target(self, block, block_idx: int):
        for attr_name in ("attn1", "self_attn", "attn"):
            if hasattr(block, attr_name):
                return getattr(block, attr_name)
        raise RuntimeError(
            f"Block {block_idx} 没有 attn1/self_attn 属性。"
            f"Block type: {type(block).__name__}"
        )

    def register_hooks(
        self,
        transformer,
        selected_latents: Optional[list] = None,
        ref_patches: Optional[torch.Tensor] = None,
        offload_bolt: bool = True,
    ) -> List:
        """注册 forward hooks，在 self-attn 后注入 Bolt Ref-Attn。

        Args:
            transformer: Helios DiT model
            selected_latents: list of (B, C, T, H, W) — 从 select_gap_frames 返回
                              会自动 patchify
            ref_patches: (B, N_ref, patch_dim) — 已经 patchify 好的 patches
                         如果提供则忽略 selected_latents
            offload_bolt: 是否在每层用完后 offload Bolt 参数到 CPU (省显存)

        Returns:
            hooks: list of hook handles
        """
        # Prepare ref_patches (keep on CPU if offloading, will move per-layer)
        if ref_patches is None and selected_latents is not None:
            ref_patches = patchify_selected_latents(selected_latents, self.patch_size)

        # Determine target device/dtype from first available parameter
        target_device = next(self.parameters()).device
        target_dtype = next(self.parameters()).dtype

        if ref_patches is not None:
            if offload_bolt:
                # Keep patches on CPU, each hook will move to GPU on demand
                ref_patches = ref_patches.to(dtype=target_dtype)
            else:
                ref_patches = ref_patches.to(device=target_device, dtype=target_dtype)

        hooks = []
        blocks = self._find_blocks(transformer)

        for idx, block in enumerate(blocks):
            if not self.is_active(idx):
                continue
            target = self._find_self_attn_target(block, idx)

            def make_hook(layer_idx, patches):
                def hook_fn(module, input, output):
                    if patches is None:
                        return output
                    if isinstance(output, tuple):
                        hs = output[0]
                    else:
                        hs = output

                    bolt_module = self.ref_attn_modules[str(layer_idx)]

                    # 在 torch.no_grad() 上下文中，hook 内部需要重新开启梯度
                    # detach: 切断 DiT 反向图，不保留 DiT 激活
                    # enable_grad: 让 bolt 参数的计算有梯度
                    with torch.enable_grad():
                        hs_detached = hs.detach().requires_grad_(False)
                        # 确保 patches 与 hs 在同一 device
                        local_patches = patches.to(device=hs.device) if patches.device != hs.device else patches

                        if bolt_module.use_checkpoint and self.training:
                            ref_out = torch.utils.checkpoint.checkpoint(
                                bolt_module, hs_detached, local_patches, use_reentrant=False,
                            )
                        else:
                            ref_out = bolt_module(hs_detached, local_patches)

                    # hs (no grad) + ref_out (has grad from bolt params)
                    hs_new = hs + ref_out

                    if isinstance(output, tuple):
                        return (hs_new,) + output[1:]
                    return hs_new
                return hook_fn

            h = target.register_forward_hook(make_hook(idx, ref_patches))
            hooks.append(h)

        return hooks

    @staticmethod
    def remove_hooks(hooks: List):
        for h in hooks:
            h.remove()
        hooks.clear()
