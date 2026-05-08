"""
VLM Selector — Stage 1 模块
=============================

使用 Qwen2.5-VL-3B + LoRA 作为 GAP 帧选择器。

【核心思路】
- Q = LHS(prompt + chunk_{t-1})   ← "我需要什么"
- K = LHS(prompt + GAP_i)         ← "这个候选帧有什么"
- Selector Head: cross-attn(Q, K) → top-k idx → 选出最有用的 GAP 帧

【训练】
- Loss = KL(P_pred || P_soft)
- P_soft 由 frozen Helios 的 ΔMSE 计算得到

【推理】
- 每 chunk 调用一次 (不是每 step)
- chunk_{t-1} 生成完毕 → 调用 selector → top-k idx → 查 Memory Bank
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectorHead(nn.Module):
    """Selector Head: 用 cross-attention 计算 Q 和 K 的匹配分数。

    Q = VLM LHS of (prompt + context_chunk)
    K = VLM LHS of (prompt + GAP_i)

    输出: 对所有 GAP 帧的选择概率分布 P_pred
    """

    def __init__(
        self,
        vlm_hidden_dim: int = 2048,   # Qwen2.5-VL-3B hidden_dim
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vlm_hidden_dim = vlm_hidden_dim
        self.num_heads = num_heads
        self.head_dim = vlm_hidden_dim // num_heads

        # Q/K 投影 (从 VLM LHS 空间)
        self.q_proj = nn.Linear(vlm_hidden_dim, vlm_hidden_dim)
        self.k_proj = nn.Linear(vlm_hidden_dim, vlm_hidden_dim)

        self.scale = math.sqrt(self.head_dim)
        self.dropout = nn.Dropout(dropout)

        # 输出: 将多头 attention score 聚合为单个分数
        self.score_proj = nn.Linear(num_heads, 1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.zeros_(self.q_proj.bias)
        nn.init.zeros_(self.k_proj.bias)
        nn.init.xavier_uniform_(self.score_proj.weight)
        nn.init.zeros_(self.score_proj.bias)

    def forward(
        self,
        query_lhs: torch.Tensor,    # (B, S_q, D) — context chunk 的 VLM LHS
        key_lhs: torch.Tensor,      # (B, N_gap, D) — 所有 GAP 帧的 VLM LHS
    ) -> torch.Tensor:
        """
        返回:
            logits: (B, N_gap) — 每个 GAP 帧的选择分数 (未经 softmax)
        """
        B, S_q, D = query_lhs.shape
        _, N_gap, _ = key_lhs.shape

        # 投影
        Q = self.q_proj(query_lhs)   # (B, S_q, D)
        K = self.k_proj(key_lhs)     # (B, N_gap, D)

        # 重塑为多头: (B, num_heads, S, head_dim)
        Q = Q.view(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, N_gap, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, num_heads, S_q, N_gap)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_scores = self.dropout(attn_scores)

        # 聚合: 对 S_q 维度做 mean pool → (B, num_heads, N_gap)
        attn_scores = attn_scores.mean(dim=2)  # (B, num_heads, N_gap)

        # 多头融合: (B, N_gap, num_heads) → (B, N_gap, 1) → (B, N_gap)
        attn_scores = attn_scores.permute(0, 2, 1)  # (B, N_gap, num_heads)
        logits = self.score_proj(attn_scores).squeeze(-1)  # (B, N_gap)

        return logits


class VLMSelector(nn.Module):
    """完整的 VLM Selector 模块。

    包含:
    1. Qwen2.5-VL-3B 视觉-语言模型 (frozen + LoRA)
    2. SelectorHead (trainable)

    【使用方式】
    训练时:
        - 输入: context_images (pixel), gap_images (pixel), prompt (text)
        - 离线预计算 LHS 特征，直接调用 selector_head
        - Loss: KL(P_pred || P_soft)

    推理时:
        - 输入: context_images + prompt → VLM → LHS (Q)
        - Memory Bank 中已有 gap LHS (K)
        - SelectorHead(Q, K) → top-k idx
    """

    def __init__(
        self,
        vlm_model_path: str = "/root/autodl-fs/Qwen2.5-VL-3B-Instruct",
        vlm_hidden_dim: int = 2048,
        num_heads: int = 8,
        k_select: int = 4,
        dropout: float = 0.1,
        use_lora: bool = True,
        lora_rank: int = 16,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.vlm_hidden_dim = vlm_hidden_dim
        self.k_select = k_select
        self.vlm_model_path = vlm_model_path
        self.use_lora = use_lora

        # Selector Head (always trainable)
        self.selector_head = SelectorHead(
            vlm_hidden_dim=vlm_hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        # VLM 模型 (lazy load — 在第一次调用时加载)
        self._vlm = None
        self._vlm_processor = None
        self._vlm_base_model = None  # 指向底层 transformer (不含 LM head)
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha

    def _load_vlm(self, device="cuda"):
        """延迟加载 VLM 模型 + 可选 LoRA"""
        if self._vlm is not None:
            return

        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
        from peft import LoraConfig, get_peft_model

        print(f"[VLMSelector] Loading VLM from {self.vlm_model_path}...")
        self._vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.vlm_model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
        )
        self._vlm_processor = AutoProcessor.from_pretrained(self.vlm_model_path)

        # 保存底层 transformer 引用 (不含 LM head)
        # 在 LoRA wrap 之前获取，确保引用正确
        self._vlm_base_model = self._vlm.model  # Qwen2_5_VLModel

        if self.use_lora:
            lora_config = LoraConfig(
                r=self._lora_rank,
                lora_alpha=self._lora_alpha,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self._vlm = get_peft_model(self._vlm, lora_config)
            # LoRA wrap 后 self._vlm.model 变成了原始 ForConditionalGeneration
            # 底层 transformer 需要通过 self._vlm.model.model 访问
            # 但我们已经在 wrap 前保存了引用，直接用 _vlm_base_model 即可
            trainable_params = sum(p.numel() for p in self._vlm.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self._vlm.parameters())
            print(f"[VLMSelector] LoRA applied: {trainable_params:,} / {total_params:,} trainable ({100*trainable_params/total_params:.2f}%)")
            print(f"[VLMSelector] PeftModel type: {type(self._vlm)}")
            print(f"[VLMSelector] Base model type: {type(self._vlm_base_model)}")
        else:
            # Freeze all VLM parameters
            for param in self._vlm.parameters():
                param.requires_grad = False

    def init_vlm(self, device: str = "cuda"):
        """显式初始化 VLM 模型。建议在训练脚本开头调用一次，避免依赖 lazy load。"""
        self._load_vlm(device)

    def encode_lhs(
        self,
        images: list,          # PIL images list
        prompt: str,           # text prompt
        device: str = "cuda",
    ) -> torch.Tensor:
        """编码图片+文本 → VLM Last Hidden State (LHS)。

        注意:
        - 训练时 (Stage 1 LoRA 需要梯度) 不加 torch.no_grad()
        - 推理时由调用方在外部加 torch.no_grad()
        - 返回保留 seq_len 维度，不做 mean pool，让 SelectorHead 内部处理

        Args:
            images: list of PIL.Image — 要编码的图片 (支持 batch)
            prompt: str — 文本 prompt

        Returns:
            lhs: (N, seq_len, D) — 每张图片的完整 LHS 序列特征
                 N = len(images), seq_len 由 processor padding 对齐
        """
        self._load_vlm(device)

        # 构造 batch 输入 (避免逐张 forward)
        messages_list = [
            [{"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": prompt},
            ]}]
            for img in images
        ]
        texts = [
            self._vlm_processor.apply_chat_template(
                m, tokenize=False, add_generation_prompt=False
            )
            for m in messages_list
        ]
        inputs = self._vlm_processor(
            text=texts, images=images,
            return_tensors="pt", padding=True,
        ).to(device)

        # 使用 _vlm_base_model (底层 transformer，不含 LM head)
        # 这样无论是否经过 LoRA wrap 都能正确访问
        outputs = self._vlm_base_model(
            **{k: v for k, v in inputs.items() if k != "labels"},
            output_hidden_states=True,
        )
        # 返回完整 seq_len，保留位置信息给 SelectorHead
        lhs = outputs.hidden_states[-1]  # (N, seq_len, D)
        return lhs

    @staticmethod
    def pool_lhs(lhs: torch.Tensor) -> torch.Tensor:
        """将 encode_lhs 返回的 (N, seq_len, D) mean-pool 为 (N, D)。

        用于 GAP 帧的 K 编码：每个 GAP 帧压缩为单个向量。
        query 端不需要 pool（保留 seq_len 信息更丰富）。
        """
        return lhs.mean(dim=1)  # (N, D)

    def forward(
        self,
        query_lhs: torch.Tensor,    # (B, S_q, D) — context chunk 的完整 LHS 序列
        key_lhs: torch.Tensor,      # (B, N_gap, D) — GAP 帧的 mean-pooled LHS
        p_soft: Optional[torch.Tensor] = None,  # (B, N_gap) — soft label for training
        temperature: float = 1.0,
    ) -> dict:
        """
        典型调用流程:
            # query: 保留完整序列 (B=1, S_q=seq_len, D)
            query_lhs = selector.encode_lhs([context_img], prompt)   # (1, seq_len, D)

            # keys: 逐帧 mean-pool → (N_gap, D) → unsqueeze(0) → (1, N_gap, D) 如果说batch=1会不会有错误？
            gap_lhs = selector.encode_lhs(gap_imgs, prompt)          # (N_gap, seq_len, D)
            gap_lhs_pooled = VLMSelector.pool_lhs(gap_lhs)           # (N_gap, D)
            key_lhs = gap_lhs_pooled.unsqueeze(0)                    # (1, N_gap, D)

            result = selector(query_lhs, key_lhs, p_soft=p_soft)

        Args:
            query_lhs: (B, S_q, D) — context chunk 完整 LHS（保留 seq_len）
            key_lhs: (B, N_gap, D) — GAP 帧的 pooled LHS（每帧一个向量）
            p_soft: (B, N_gap) — soft label distribution, for KL loss
            temperature: softmax temperature

        Returns:
            dict with logits, p_pred, top_k_indices, [loss]
        """
        # Shape 校验：调用方必须保证 key_lhs 已经 pool 过
        assert key_lhs.dim() == 3, \
            f"key_lhs 必须是 (B, N_gap, D), got dim={key_lhs.dim()}"
        assert key_lhs.shape[-1] == self.vlm_hidden_dim, \
            f"key_lhs.shape[-1]={key_lhs.shape[-1]} != vlm_hidden_dim={self.vlm_hidden_dim}"
        # N_gap 通常 < 100; 如果 shape[1] 很大，可能传入了未 pool 的完整序列
        if key_lhs.shape[1] > 200:
            import warnings
            warnings.warn(
                f"key_lhs.shape[1]={key_lhs.shape[1]} 异常大，"
                f"是否忘记调用 VLMSelector.pool_lhs()？"
            )

        # 统一 dtype: VLM 输出 bf16, SelectorHead 权重 fp32
        head_dtype = next(self.selector_head.parameters()).dtype
        logits = self.selector_head(query_lhs.to(head_dtype), key_lhs.to(head_dtype))  # (B, N_gap)
        p_pred = F.softmax(logits / temperature, dim=-1)  # (B, N_gap)

        # Top-k selection
        _, top_k_indices = torch.topk(logits, k=min(self.k_select, logits.shape[-1]), dim=-1)

        result = {
            "logits": logits,
            "p_pred": p_pred,
            "top_k_indices": top_k_indices,
        }

        # KL Loss (training only)
        if p_soft is not None:
            log_p_pred = F.log_softmax(logits / temperature, dim=-1)
            loss = F.kl_div(log_p_pred, p_soft, reduction="batchmean")
            result["loss"] = loss

        return result
