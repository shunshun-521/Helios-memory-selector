"""
test_gap_influence.py — 验证 GAP 帧拼接到 latents_history_long 后是否真正影响 Transformer 输出

测试方法:
  1. 基线: 正常 latents_history_long (16帧 zero padding)
  2. 加 GAP: 在 latents_history_long 前拼接随机 GAP 帧
  3. 加全零 GAP: 拼接全零帧 (应该影响更小)
  4. 加大值 GAP: 拼接大值帧 (应该影响更大)

如果输出完全相同 → GAP 帧没有被处理
如果输出有差异 → GAP 帧确实参与了计算

用法:
  python Helios/scripts/test_gap_influence.py \
    --transformer_path /root/autodl-fs/BestWishYSH/Helios-Base
"""

import argparse
import os
import sys

import torch

HELIOS_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, HELIOS_ROOT)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transformer_path", type=str, default="/root/autodl-fs/BestWishYSH/Helios-Base")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    dtype = torch.bfloat16
    device = args.device

    # 加载 transformer
    from helios.modules.transformer_helios import HeliosTransformer3DModel
    print("[1] 加载 Transformer...")
    transformer = HeliosTransformer3DModel.from_pretrained(
        args.transformer_path,
        subfolder="transformer" if os.path.isdir(os.path.join(args.transformer_path, "transformer")) else None,
        torch_dtype=dtype,
        transformer_additional_kwargs={"has_multi_term_memory_patch": True},
    ).to(device, dtype=dtype).eval()
    transformer.requires_grad_(False)

    # 固定随机种子
    torch.manual_seed(42)

    # 构造输入 (模拟 384x640 → latent 48x80)
    B, C, T, H, W = 1, 16, 9, 48, 80
    hidden_states = torch.randn(B, C, T, H, W, device=device, dtype=dtype)
    timestep = torch.tensor([500.0], device=device, dtype=torch.float32)
    prompt_embed = torch.randn(B, 512, 4096, device=device, dtype=dtype)

    # indices
    indices_hidden_states = torch.arange(20, 29).unsqueeze(0).to(device)
    indices_short = torch.tensor([[0, 19]]).to(device)
    indices_mid = torch.tensor([[17, 18]]).to(device)
    indices_long = torch.arange(1, 17).unsqueeze(0).to(device)  # 16 帧

    # history latents
    latents_short = torch.randn(B, C, 2, H, W, device=device, dtype=dtype)
    latents_mid = torch.randn(B, C, 2, H, W, device=device, dtype=dtype)
    latents_long = torch.randn(B, C, 16, H, W, device=device, dtype=dtype)

    print("\n[2] 测试 1: 基线 (无 GAP)")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out_baseline = transformer(
            hidden_states=hidden_states, timestep=timestep,
            encoder_hidden_states=prompt_embed,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_short,
            indices_latents_history_mid=indices_mid,
            indices_latents_history_long=indices_long,
            latents_history_short=latents_short,
            latents_history_mid=latents_mid,
            latents_history_long=latents_long,
            return_dict=False,
        )
        if isinstance(out_baseline, tuple):
            out_baseline = out_baseline[0]

    print(f"  输出 shape: {out_baseline.shape}")
    print(f"  输出 mean: {out_baseline.mean().item():.6f}, std: {out_baseline.std().item():.6f}")

    # --- 测试 2: 加随机 GAP 帧 ---
    print("\n[3] 测试 2: 加 9 帧随机 GAP (拼接到 long 前面)")
    gap_latent = torch.randn(B, C, 9, H, W, device=device, dtype=dtype)
    gap_indices = torch.arange(10, 19).unsqueeze(0).to(device)  # 模拟 GAP 区域的 indices

    aug_long = torch.cat([gap_latent, latents_long], dim=2)  # (B, C, 25, H, W)
    aug_long_idx = torch.cat([gap_indices, indices_long], dim=1)  # (B, 25)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out_with_gap = transformer(
            hidden_states=hidden_states, timestep=timestep,
            encoder_hidden_states=prompt_embed,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_short,
            indices_latents_history_mid=indices_mid,
            indices_latents_history_long=aug_long_idx,
            latents_history_short=latents_short,
            latents_history_mid=latents_mid,
            latents_history_long=aug_long,
            return_dict=False,
        )
        if isinstance(out_with_gap, tuple):
            out_with_gap = out_with_gap[0]

    diff_random = (out_with_gap - out_baseline).abs()
    print(f"  输出 mean: {out_with_gap.mean().item():.6f}, std: {out_with_gap.std().item():.6f}")
    print(f"  与基线差异 - mean: {diff_random.mean().item():.8f}, max: {diff_random.max().item():.8f}")

    # --- 测试 3: 加全零 GAP 帧 ---
    print("\n[4] 测试 3: 加 9 帧全零 GAP")
    gap_zero = torch.zeros(B, C, 9, H, W, device=device, dtype=dtype)
    aug_long_zero = torch.cat([gap_zero, latents_long], dim=2)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out_with_zero = transformer(
            hidden_states=hidden_states, timestep=timestep,
            encoder_hidden_states=prompt_embed,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_short,
            indices_latents_history_mid=indices_mid,
            indices_latents_history_long=aug_long_idx,
            latents_history_short=latents_short,
            latents_history_mid=latents_mid,
            latents_history_long=aug_long_zero,
            return_dict=False,
        )
        if isinstance(out_with_zero, tuple):
            out_with_zero = out_with_zero[0]

    diff_zero = (out_with_zero - out_baseline).abs()
    print(f"  输出 mean: {out_with_zero.mean().item():.6f}, std: {out_with_zero.std().item():.6f}")
    print(f"  与基线差异 - mean: {diff_zero.mean().item():.8f}, max: {diff_zero.max().item():.8f}")

    # --- 测试 4: 加大值 GAP 帧 ---
    print("\n[5] 测试 4: 加 9 帧大值 GAP (x10)")
    gap_large = torch.randn(B, C, 9, H, W, device=device, dtype=dtype) * 10.0
    aug_long_large = torch.cat([gap_large, latents_long], dim=2)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out_with_large = transformer(
            hidden_states=hidden_states, timestep=timestep,
            encoder_hidden_states=prompt_embed,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_short,
            indices_latents_history_mid=indices_mid,
            indices_latents_history_long=aug_long_idx,
            latents_history_short=latents_short,
            latents_history_mid=latents_mid,
            latents_history_long=aug_long_large,
            return_dict=False,
        )
        if isinstance(out_with_large, tuple):
            out_with_large = out_with_large[0]

    diff_large = (out_with_large - out_baseline).abs()
    print(f"  输出 mean: {out_with_large.mean().item():.6f}, std: {out_with_large.std().item():.6f}")
    print(f"  与基线差异 - mean: {diff_large.mean().item():.8f}, max: {diff_large.max().item():.8f}")

    # --- 测试 5: 不传 long (None) vs 传空 ---
    print("\n[6] 测试 5: latents_history_long=None (完全不传)")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        out_no_long = transformer(
            hidden_states=hidden_states, timestep=timestep,
            encoder_hidden_states=prompt_embed,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_short,
            indices_latents_history_mid=indices_mid,
            indices_latents_history_long=None,
            latents_history_short=latents_short,
            latents_history_mid=latents_mid,
            latents_history_long=None,
            return_dict=False,
        )
        if isinstance(out_no_long, tuple):
            out_no_long = out_no_long[0]

    diff_no_long = (out_no_long - out_baseline).abs()
    print(f"  输出 mean: {out_no_long.mean().item():.6f}, std: {out_no_long.std().item():.6f}")
    print(f"  与基线差异 - mean: {diff_no_long.mean().item():.8f}, max: {diff_no_long.max().item():.8f}")

    # --- 汇总 ---
    print("\n" + "=" * 60)
    print("汇总:")
    print(f"  随机GAP vs 基线:  mean_diff = {diff_random.mean().item():.8f}")
    print(f"  全零GAP vs 基线:  mean_diff = {diff_zero.mean().item():.8f}")
    print(f"  大值GAP vs 基线:  mean_diff = {diff_large.mean().item():.8f}")
    print(f"  无long  vs 基线:  mean_diff = {diff_no_long.mean().item():.8f}")
    print()

    if diff_random.mean().item() < 1e-6:
        print("⚠️  GAP 帧对输出几乎无影响！patch_long 可能未正确处理额外帧。")
    else:
        print("✅ GAP 帧确实影响了输出。")
        if diff_large.mean().item() > diff_random.mean().item() > diff_zero.mean().item():
            print("✅ 影响程度与 GAP 帧幅度正相关（大值 > 随机 > 零），符合预期。")
        else:
            print("⚠️  影响程度不单调，可能存在问题。")


if __name__ == "__main__":
    main()
