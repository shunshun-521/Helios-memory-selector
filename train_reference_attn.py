"""
train_reference_attn.py — Stage 2: Reference Attention 训练
===============================================================

冻结 VLM Selector + Helios DiT，只训练 ReferenceAttentionLayers。
将 Ref-Attn 插入 DiT block 的 Self-Attn 和 Cross-Attn 之间。
Loss = Flow Matching MSE（与 Helios 原训练目标一致）。

【与 selector_v2/v3 的区别】
- v2/v3: GAP 帧 concat 进 self-attention，改变了序列长度和注意力分布
- 本方案: 独立 Ref-Attn 层 + zero-init α，不污染原有注意力机制
  - K = VLM LHS (语义空间), V = VLM LHS (fallback, 可切换为 latent)
  - 即插即用: α=0 初始化，训练初期输出为零，不破坏原生成质量

【训练流程 (End-to-End Flow Matching MSE)】
1. 加载 frozen Helios DiT (全量 GPU) + frozen VLM Selector (编码后释放)
2. Selector 选出 top-k GAP 帧 → 获得 ref_key (LHS)
3. DiT forward 时在 active layers 插入 Ref-Attn hooks (detach hs → ref_attn 有 grad)
4. Loss = Flow Matching MSE → backward 只更新 Ref-Attn 参数
5. DiT 权重始终在 GPU 上 → backward 不会遇到 device mismatch

用法:
  python Helios/train_reference_attn.py \
    --meta_json Helios/example/lighting_change/selector_meta_with_psoft.json \
    --transformer_path /root/autodl-fs/BestWishYSH/Helios-Base \
    --selector_ckpt /root/autodl-fs/output/vlm_selector_stage1/checkpoint-epoch003 \
    --output_dir /root/autodl-fs/output/ref_attn_stage2 \
    --num_epochs 30 \
    --lr 5e-5
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

HELIOS_ROOT = os.path.join(os.path.dirname(__file__))
sys.path.insert(0, HELIOS_ROOT)

from helios.modules.selector_vlm import VLMSelector
from helios.modules.ref_attn import ReferenceAttentionLayers

# ── 常量 ──
LATENT_WINDOW_SIZE = 9
HISTORY_SIZES = [16, 2, 1]
HISTORY_WINDOW_SIZE = sum(HISTORY_SIZES)


# ═══════════════════════════════════════════
# Dataset (reuses Stage 1 format + latent .pt)
# ═══════════════════════════════════════════

class RefAttnDataset(Dataset):
    """Stage 2 训练数据集: 需要 latent + pixel + p_soft。"""

    def __init__(self, meta_json: str):
        with open(meta_json, "r", encoding="utf-8") as f:
            meta = json.load(f)

        self.samples = []
        for entry in meta:
            latent_path = entry["latent_pt_path"]
            if not os.path.isabs(latent_path):
                base = os.path.dirname(meta_json)
                latent_path = os.path.join(base, latent_path)
            if not os.path.exists(latent_path):
                continue

            for sample in entry["selector_samples"]:
                if sample.get("p_soft") is None:
                    continue
                ctx_path = sample["context_pixel_path"]
                if not os.path.exists(ctx_path):
                    continue
                valid_gaps = [gp for gp in sample["gap_pixel_paths"] if os.path.exists(gp)]
                if len(valid_gaps) < 2:
                    continue

                self.samples.append({
                    "latent_path": latent_path,
                    "prompt_raw": entry["prompt_raw"],
                    "choice_idx": sample["choice_idx"],
                    "context_pixel_path": ctx_path,
                    "gap_pixel_paths": valid_gaps,
                    "p_soft": sample["p_soft"][:len(valid_gaps)],
                })

        print(f"[RefAttnDataset] {len(self.samples)} valid samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    return batch


# ═══════════════════════════════════════════
# Core: Modified DiT Forward with Ref-Attn
# ═══════════════════════════════════════════

def dit_forward_with_ref_attn(
    transformer,
    ref_attn_layers: ReferenceAttentionLayers,
    hidden_states,           # noisy target (B, C, T, H, W)
    timestep,
    encoder_hidden_states,   # text embed
    ref_key,                 # (B, N_ref, vlm_dim) — selected LHS
    ref_value=None,          # (B, N_ref, vlm_dim) — selected V (or None → K=V)
    **kwargs,
):
    """Helios DiT forward with Ref-Attn injection.

    这是 Stage 2 的核心: 在 DiT 的 forward 过程中，
    在每个 active layer 的 self-attn 后、cross-attn 前插入 Ref-Attn。

    实现方式: 使用 transformer 的 forward hook 机制，
    或者直接修改 forward 函数。这里采用 monkey-patch 方式。
    """
    # 简化实现: 直接调用 transformer 的标准 forward，
    # 但在 transformer 内部通过 register_forward_hook 注入 Ref-Attn。
    # 由于 Helios DiT 的 block 结构复杂 (混合 Self-Attn + Cross-Attn + FFN),
    # 我们在外部封装一层 hook 机制。

    # 存储 ref_attn 输入到 transformer 的 extra_kwargs
    # Helios transformer_v3 支持 gap_latents / gap_frame_indices 参数
    # 我们复用这个接口，但用 ref_attn_layers 替代内部的处理逻辑

    # Fix #5 + #9: 使用 ReferenceAttentionLayers 的统一 hook 接口
    # （内部会验证 block 属性，找不到 self-attn 时会 raise RuntimeError）
    hooks = ref_attn_layers.register_hooks(transformer, ref_key, ref_value)

    # Forward
    try:
        output = transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            **kwargs,
        )
    finally:
        # 清理 hooks
        ReferenceAttentionLayers.remove_hooks(hooks)

    return output


# ═══════════════════════════════════════════
# Training
# ═══════════════════════════════════════════

def train_one_epoch(
    transformer,
    ref_attn_layers: ReferenceAttentionLayers,
    selector: VLMSelector,
    dataloader: DataLoader,
    optimizer,
    epoch: int,
    device="cuda",
    dtype=torch.bfloat16,
    grad_accum_steps=2,
    max_grad_norm=1.0,
    fixed_timestep=500.0,
):
    """Stage 2 单 epoch 训练。"""
    ref_attn_layers.train()
    transformer.eval()  # frozen

    total_loss = 0.0
    num_samples = 0
    num_skipped = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"[Epoch {epoch}]")
    for step, batch in enumerate(pbar):
        sample = batch[0]

        # ── 加载 latent ──
        try:
            feature_data = torch.load(sample["latent_path"], map_location="cpu", weights_only=False)
            vae_latent = feature_data["vae_latent"]  # (num_chunks, C, T, H, W)
            prompt_embed = feature_data["prompt_embed"]  # (1, S, 4096)
        except Exception as e:
            print(f"  [WARN] 跳过: {e}")
            num_skipped += 1
            continue

        choice_idx = sample["choice_idx"]
        if choice_idx >= vae_latent.shape[0]:
            num_skipped += 1
            continue

        # ── Selector: 选出 top-k GAP 帧的 LHS ──
        with torch.no_grad():
            # 加载 context 和 gap 像素帧
            try:
                ctx_img = Image.open(sample["context_pixel_path"]).convert("RGB")
                gap_imgs = [Image.open(gp).convert("RGB") for gp in sample["gap_pixel_paths"]]
            except Exception:
                continue

            # Move VLM back to GPU for encoding
            if hasattr(selector, '_vlm'):
                selector._vlm.to(device)

            query_lhs = selector.encode_lhs([ctx_img], sample["prompt_raw"], device=device)
            gap_lhs = selector.encode_lhs(gap_imgs, sample["prompt_raw"], device=device)
            gap_lhs_pooled = VLMSelector.pool_lhs(gap_lhs)

            result = selector(query_lhs, gap_lhs_pooled.unsqueeze(0))
            top_k_idx = result["top_k_indices"][0]  # (K,)

            # 取出选中帧的 LHS 作为 ref_key 和 ref_value
            ref_key = gap_lhs_pooled[top_k_idx].unsqueeze(0).to(device, dtype=dtype)  # (1, K, D)

            # 释放 VLM 显存给 DiT forward
            if hasattr(selector, '_vlm'):
                selector._vlm.to("cpu")
            torch.cuda.empty_cache()

        # ── Fix #6: 统一用 continue_latent 切分逻辑构建 target 和 history ──
        # vae_latent shape: (num_chunks, C, T, H, W), T=LATENT_WINDOW_SIZE
        # 拼成连续帧序列: (C, total_frames, H, W)
        continue_latent = rearrange(vae_latent, "b c t h w -> c (b t) h w")
        C_lat, total_frames, H_lat, W_lat = continue_latent.shape

        # 在前面补零 (给 history 留空间)
        zero_pad = torch.zeros(C_lat, HISTORY_WINDOW_SIZE, H_lat, W_lat,
                               device=continue_latent.device, dtype=continue_latent.dtype)
        padded_latent = torch.cat([zero_pad, continue_latent], dim=1)

        # target: 第 choice_idx 个 chunk 对应的帧
        target_start = HISTORY_WINDOW_SIZE + choice_idx * LATENT_WINDOW_SIZE
        target_latent = padded_latent[:, target_start:target_start+LATENT_WINDOW_SIZE]
        target_latent = target_latent.unsqueeze(0).to(device, dtype=dtype)  # (1, C, T, H, W)

        # history: target 之前的 HISTORY_WINDOW_SIZE 帧
        hist_start = target_start - HISTORY_WINDOW_SIZE
        history = padded_latent[:, hist_start:hist_start+HISTORY_WINDOW_SIZE]
        history = history.unsqueeze(0).to(device, dtype=dtype)  # (1, C, HISTORY_WINDOW_SIZE, H, W)

        # x0: 第一帧
        x0 = padded_latent[:, HISTORY_WINDOW_SIZE:HISTORY_WINDOW_SIZE+1]
        x0 = x0.unsqueeze(0).to(device, dtype=dtype)  # (1, C, 1, H, W)

        # Prepare inputs
        from helios.utils.utils_helios_base_v3 import prepare_stage1_clean_input_from_latents

        # Fix #7: 打印首次异常而非完全静默
        try:
            clean_inputs = prepare_stage1_clean_input_from_latents(
                history, target_latent, x0,
                device=device, dtype=dtype,
            )
        except Exception as e:
            if step == 0:  # 只打印第一次
                import traceback
                print(f"  [WARN] prepare_stage1_clean_input_from_latents 失败:")
                traceback.print_exc()
            clean_inputs = None

        if clean_inputs is None:
            continue

        (target_clean, idx_hs, idx_short, idx_mid, idx_long,
         lat_short, lat_mid, lat_long) = clean_inputs

        target_clean = target_clean.to(device, dtype=dtype)
        prompt_embed = prompt_embed.to(device, dtype=dtype)
        if prompt_embed.ndim == 2:
            prompt_embed = prompt_embed.unsqueeze(0)  # (S, D) → (1, S, D)

        # ── 加噪 ──
        sigma = fixed_timestep / 1000.0
        sigma_t = torch.tensor([sigma], device=device, dtype=dtype).reshape(1, 1, 1, 1, 1)
        noise = torch.randn_like(target_clean)
        noisy_input = (1.0 - sigma_t) * target_clean + sigma_t * noise
        target_flow = noise - target_clean
        timestep_t = torch.tensor([fixed_timestep], device=device, dtype=dtype)

        # ── Two-pass: no_grad DiT forward + differentiable ref_attn ──
        dit_common_kwargs = dict(
            timestep=timestep_t,
            encoder_hidden_states=prompt_embed,
            indices_hidden_states=idx_hs.to(device) if idx_hs is not None else None,
            indices_latents_history_short=idx_short.to(device) if idx_short is not None else None,
            indices_latents_history_mid=idx_mid.to(device) if idx_mid is not None else None,
            indices_latents_history_long=idx_long.to(device) if idx_long is not None else None,
            latents_history_short=lat_short.to(device, dtype=dtype) if lat_short is not None else None,
            latents_history_mid=lat_mid.to(device, dtype=dtype) if lat_mid is not None else None,
            latents_history_long=lat_long.to(device, dtype=dtype) if lat_long is not None else None,
            return_dict=False,
        )

        # Pass 1: DiT forward (no_grad + group offload) → baseline_pred + captured hs
        ref_attn_layers.clear_captured()
        hooks = ref_attn_layers.register_hooks(transformer, training_mode=True)
        try:
            with torch.no_grad():
                baseline_pred = transformer(hidden_states=noisy_input, **dit_common_kwargs)
        finally:
            ReferenceAttentionLayers.remove_hooks(hooks)
        if isinstance(baseline_pred, tuple):
            baseline_pred = baseline_pred[0]
        baseline_pred = baseline_pred.detach()  # (B, C, T, H, W)
        torch.cuda.empty_cache()

        # Pass 2: differentiable ref_attn → 投影到输出空间 → 加到 baseline
        # 取最后一个 active layer 的 captured hs (最接近输出)
        last_layer_idx = max(ref_attn_layers._captured_states.keys())
        last_hs = ref_attn_layers._captured_states[last_layer_idx]  # (B, S, D) detached

        # ref_attn: differentiable through ref_attn params
        ref_out = ref_attn_layers.forward(last_layer_idx, last_hs, ref_key, ref_value=None)  # (B, S, D)

        # 通过 cached proj_out 投影到 patch 空间: (B, S, D) → (B, S, patch_dim)
        ref_projected = F.linear(ref_out, proj_out_weight, proj_out_bias)  # (B, S, C*p_t*p_h*p_w)

        # Unpatchify: (B, S, patch_dim) → (B, C, T, H, W)
        # S = (T/p_t) * (H/p_h) * (W/p_w), patch_dim = C * p_t * p_h * p_w
        B_sz, C_out, T_out, H_out, W_out = baseline_pred.shape
        p_t, p_h, p_w = 1, 2, 2  # Helios patch size
        ref_correction = ref_projected.reshape(
            B_sz, T_out // p_t, H_out // p_h, W_out // p_w, C_out, p_t, p_h, p_w
        ).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B_sz, C_out, T_out, H_out, W_out)

        # 最终预测 = baseline + ref_attn correction
        pred = baseline_pred + ref_correction

        # ── Loss: Flow Matching MSE ──
        loss = F.mse_loss(pred, target_flow) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(ref_attn_layers.parameters(), max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        num_samples += 1

        # Log: alpha values + grad norm + running avg
        alphas = [ref_attn_layers.ref_attn_modules[k].alpha.item()
                  for k in sorted(ref_attn_layers.ref_attn_modules.keys(), key=int)]
        alpha_mean = sum(alphas) / len(alphas) if alphas else 0
        alpha_max = max(alphas) if alphas else 0
        avg_loss = total_loss / num_samples

        # Grad norm (compute without clipping)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ref_attn_layers.parameters(), float('inf')
        ).item() if num_samples > 0 else 0

        pbar.set_postfix({
            "mse": f"{loss.item()*grad_accum_steps:.4f}",
            "avg": f"{avg_loss:.4f}",
            "α": f"{alpha_mean:.4f}",
            "α_max": f"{alpha_max:.4f}",
            "‖g‖": f"{grad_norm:.2f}",
            "ok/skip": f"{num_samples}/{num_skipped}",
        })

        # 每 50 步打印详细日志
        if (step + 1) % 50 == 0:
            print(f"\n  [Step {step+1}] mse={loss.item()*grad_accum_steps:.6f}, "
                  f"avg_mse={avg_loss:.6f}, α_mean={alpha_mean:.5f}, α_max={alpha_max:.5f}, "
                  f"grad_norm={grad_norm:.3f}, samples={num_samples}, skipped={num_skipped}")

    # Epoch summary
    avg = total_loss / max(num_samples, 1)
    alphas = [ref_attn_layers.ref_attn_modules[k].alpha.item()
              for k in sorted(ref_attn_layers.ref_attn_modules.keys(), key=int)]
    print(f"  [Epoch {epoch} Summary] valid={num_samples}, skipped={num_skipped}, "
          f"α_range=[{min(alphas):.5f}, {max(alphas):.5f}], α_mean={sum(alphas)/len(alphas):.5f}")

    return avg


# ═══════════════════════════════════════════
# Main
# ═══════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Train Reference Attention Layers")
    parser.add_argument("--meta_json", type=str, required=True)
    parser.add_argument("--transformer_path", type=str,
                        default="/root/autodl-fs/Wan-AI/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--selector_ckpt", type=str, required=True,
                        help="Stage 1 checkpoint directory (contains selector_head.pth + vlm_lora_weights.pth)")
    parser.add_argument("--vlm_model_path", type=str, default="/root/autodl-fs/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="/root/autodl-fs/output/ref_attn_stage2")
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--grad_accum_steps", type=int, default=2)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fixed_timestep", type=float, default=500.0)
    parser.add_argument("--dit_dim", type=int, default=5120)
    parser.add_argument("--active_layers", type=str, default="20-30",
                        help="DiT 层范围, e.g. '20-30'")
    parser.add_argument("--k_select", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    # ── Parse active layers ──
    start_l, end_l = map(int, args.active_layers.split("-"))
    active_layers = list(range(start_l, end_l + 1))

    # ── Load frozen DiT ──
    print("[INFO] Loading frozen Helios DiT...")
    from helios.modules.transformer_helios import HeliosTransformer3DModel
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer" if os.path.isdir(os.path.join(args.transformer_path, "transformer")) else None,
        torch_dtype=dtype,
        transformer_additional_kwargs={
            "has_multi_term_memory_patch": True,
        },
    )
    transformer.eval()
    transformer.requires_grad_(False)
    # Group offload: DiT 权重按需从 CPU 加载到 GPU，forward 后释放
    # 配合 torch.no_grad() 使用，不参与 backward
    transformer.enable_group_offload(
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        offload_type="leaf_level",
        use_stream=True,
        record_stream=True,
    )
    # 缓存 proj_out 权重到 GPU (用于 training loss 投影)
    proj_out_weight = transformer.proj_out.weight.detach().clone().to(device, dtype=dtype)
    proj_out_bias = transformer.proj_out.bias.detach().clone().to(device, dtype=dtype) if transformer.proj_out.bias is not None else None
    print(f"[INFO] DiT loaded with group offloading (leaf_level)")

    # ── Load frozen Selector ──
    print("[INFO] Loading frozen VLM Selector...")
    selector = VLMSelector(
        vlm_model_path=args.vlm_model_path,
        vlm_hidden_dim=2048,
        k_select=args.k_select,
        use_lora=True,
    )
    selector.init_vlm(device=device)
    # Load Stage 1 weights
    head_path = os.path.join(args.selector_ckpt, "selector_head.pth")
    if os.path.exists(head_path):
        selector.selector_head.load_state_dict(torch.load(head_path, map_location=device))
    lora_path = os.path.join(args.selector_ckpt, "vlm_lora_weights.pth")
    if os.path.exists(lora_path):
        from peft import set_peft_model_state_dict
        set_peft_model_state_dict(selector._vlm, torch.load(lora_path, map_location=device))
    # Move selector_head to device, then freeze
    selector.selector_head.to(device)
    for p in selector.parameters():
        p.requires_grad = False
    selector.eval()
    print("[INFO] Selector frozen.")

    # ── Create Ref-Attn Layers (trainable) ──
    ref_attn_layers = ReferenceAttentionLayers(
        dit_dim=args.dit_dim,
        vlm_hidden_dim=2048,
        active_layers=active_layers,
    ).to(device, dtype=dtype)

    # ── Optimizer (only Ref-Attn params) ──
    trainable = sum(p.numel() for p in ref_attn_layers.parameters() if p.requires_grad)
    print(f"[INFO] Ref-Attn trainable params: {trainable:,} ({trainable/1e6:.1f}M)")
    optimizer = torch.optim.AdamW(ref_attn_layers.parameters(), lr=args.lr, weight_decay=0.01)

    # ── Dataset ──
    dataset = RefAttnDataset(args.meta_json)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)

    # ── Train ──
    print(f"\n{'='*60}")
    print(f"  Stage 2: Reference Attention Training")
    print(f"  Active layers: {active_layers[0]}-{active_layers[-1]}")
    print(f"  Epochs: {args.num_epochs}, LR: {args.lr}")
    print(f"{'='*60}\n")

    for epoch in range(args.num_epochs):
        t0 = time.time()
        avg_loss = train_one_epoch(
            transformer, ref_attn_layers, selector, dataloader, optimizer, epoch,
            device=device, dtype=dtype,
            grad_accum_steps=args.grad_accum_steps,
            max_grad_norm=args.max_grad_norm,
            fixed_timestep=args.fixed_timestep,
        )
        dt = time.time() - t0
        print(f"[Epoch {epoch}] avg_mse={avg_loss:.6f}, time={dt:.1f}s")

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.output_dir, f"ref_attn_epoch{epoch:03d}.pth")
            torch.save(ref_attn_layers.state_dict(), ckpt_path)
            print(f"  [SAVE] {ckpt_path}")

    # Final save
    final_path = os.path.join(args.output_dir, "ref_attn_final.pth")
    torch.save(ref_attn_layers.state_dict(), final_path)
    print(f"\n[INFO] Training complete. Final: {final_path}")


if __name__ == "__main__":
    main()
