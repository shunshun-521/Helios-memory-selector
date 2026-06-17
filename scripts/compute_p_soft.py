"""
compute_p_soft.py  (Step 5)
使用 frozen Helios Transformer 计算 P_soft 软标签。

支持两种指标:
  --metric mse   : P_soft[i] = softmax(ΔMSE_i / τ)     (latent 空间 MSE)
  --metric lpips : P_soft[i] = softmax(ΔLPIPS_i / τ)    (像素空间 LPIPS 感知距离)

LPIPS 流程:
  1. Transformer 1-step 预测 flow → 反推 x0_hat latent
  2. VAE decode x0_hat → predicted pixel
  3. VAE decode GT target → GT pixel
  4. LPIPS(predicted, GT) 作为质量指标
  5. ΔLPIPS = LPIPS_baseline - LPIPS_with_gap (正值 = 该帧有帮助)

用法:
  python Helios/scripts/compute_p_soft.py \
    --meta_json Helios/example/lighting_change/selector_meta.json \
    --transformer_path /root/autodl-fs/BestWishYSH/Helios-Base \
    --output_json Helios/example/lighting_change/selector_meta_with_psoft_3.json \
    --metric lpips --tau 0.1 --fixed_timestep 500
"""

import argparse
import json
import os
import sys
import copy
import traceback

import torch
import torch.nn.functional as F
import lpips
from einops import rearrange
from tqdm import tqdm

# 添加 Helios 根目录到 sys.path
HELIOS_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, HELIOS_ROOT)

# ── 常量 ──
LATENT_WINDOW_SIZE = 9
HISTORY_SIZES = [16, 2, 1]
HISTORY_WINDOW_SIZE = sum(HISTORY_SIZES)  # = 19


def load_vae(vae_path, device="cuda"):
    """加载 VAE (float32) 用于 LPIPS 模式下的 latent→pixel decode"""
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(vae_path, subfolder="vae", torch_dtype=torch.float32)
    vae = vae.to(device).eval()
    vae.requires_grad_(False)
    # 缓存 latent 归一化参数
    latents_mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device)
    latents_std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1).to(device)
    return vae, latents_mean, latents_std


def recover_x0_from_flow(noisy_input, pred_flow, sigma):
    """从 flow matching 1-step 预测恢复 x0_hat。
    noisy = (1-σ)x0 + σ*noise, flow = noise - x0
    → x0_hat = (noisy - σ * pred_flow) / (1 - σ)
    """
    denom = max(1.0 - sigma, 1e-6) if isinstance(sigma, (int, float)) else (1.0 - sigma).clamp(min=1e-6)
    return (noisy_input - sigma * pred_flow) / denom


def denorm_latent(latent, latents_mean, latents_std):
    """反归一化 latent: 从训练空间恢复到 VAE decode 空间"""
    return latent / latents_std + latents_mean


def decode_latent_to_pixel(vae, latent_5d, latents_mean, latents_std):
    """VAE decode latent → pixel tensor [-1, 1], shape (B, C, T, H, W)
    输入 latent_5d: (B, C_latent, T, H, W) 已归一化的 latent
    """
    latent_denorm = denorm_latent(latent_5d.float(), latents_mean, latents_std)
    with torch.no_grad():
        pixel = vae.decode(latent_denorm).sample  # (B, 3, T_pixel, H_pixel, W_pixel)
    return pixel.clamp(-1, 1)


def compute_lpips_distance(loss_fn, pixel_pred, pixel_gt):
    """计算两个 5D pixel tensor 之间的 LPIPS 距离（取所有帧的平均）。
    pixel shape: (B, 3, T, H, W) in [-1, 1]
    """
    B, C, T, H, W = pixel_pred.shape
    # LPIPS 要求 4D (N, 3, H, W)，展开时间维
    pred_2d = pixel_pred.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    gt_2d = pixel_gt.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    with torch.no_grad():
        d = loss_fn(pred_2d, gt_2d)  # (B*T, 1, 1, 1)
    return d.mean().item()


def load_transformer(transformer_path, device="cuda", dtype=torch.bfloat16):
    """加载 frozen Helios Transformer（需要 has_multi_term_memory_patch=True 以启用 patch_short/mid/long）"""
    from helios.modules.transformer_helios import HeliosTransformer3DModel

    print(f"[INFO] 加载 Transformer from {transformer_path} ...")
    transformer = HeliosTransformer3DModel.from_pretrained(
        transformer_path,
        subfolder="transformer" if os.path.isdir(os.path.join(transformer_path, "transformer")) else None,
        torch_dtype=dtype,
        transformer_additional_kwargs={
            "has_multi_term_memory_patch": True,
        },
    )
    # ⚠️ 必须同时指定 device 和 dtype，因为 patch_short/mid/long 等新层
    # 不在预训练权重中，from_pretrained 后默认为 float32，需要显式转为 bfloat16
    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()
    transformer.requires_grad_(False)
    print(f"[INFO] Transformer 加载完成, device={device}, dtype={dtype}")
    return transformer


def prepare_clean_inputs(history_latents, target_latents, x0_latents, device="cuda", dtype=torch.bfloat16):
    """
    简化版 prepare_stage1_clean_input_from_latents，不包含 random_drop 和 corrupt。
    直接从 dataloader_v3 输出的 history_latents / target_latents 构建。

    Args:
        history_latents: (C, 19, H, W)  -- 从 prepare_stage1_latent 获取
        target_latents:  (C, 9, H, W)   -- 从 prepare_stage1_latent 获取
        x0_latents:      (C, 1, H, W)   -- 首帧 latent

    Returns:
        与 prepare_stage1_clean_input_from_latents 同格式的元组
    """
    # 统一确保 5D: (B, C, T, H, W)
    while history_latents.ndim < 5:
        history_latents = history_latents.unsqueeze(0)
    while target_latents.ndim < 5:
        target_latents = target_latents.unsqueeze(0)
    if x0_latents is not None:
        if x0_latents.ndim == 3:        # (C, H, W) → (1, C, 1, H, W)
            x0_latents = x0_latents.unsqueeze(1).unsqueeze(0)
        elif x0_latents.ndim == 4:      # (C, T, H, W) → (1, C, T, H, W)
            x0_latents = x0_latents.unsqueeze(0)
        # ndim == 5: already correct

    history_latents = history_latents.to(device, dtype=dtype)
    target_latents = target_latents.to(device, dtype=dtype)
    if x0_latents is not None:
        x0_latents = x0_latents.to(device, dtype=dtype)

    B = target_latents.shape[0]

    # 构建 indices
    total_len = 1 + HISTORY_WINDOW_SIZE + LATENT_WINDOW_SIZE  # 1(x0) + 19(hist) + 9(target) = 29
    indices = torch.arange(0, total_len).unsqueeze(0).expand(B, -1)

    indices_prefix = indices[:, :1]
    indices_latents_history_long = indices[:, 1: 1 + HISTORY_SIZES[0]]
    indices_latents_history_mid = indices[:, 1 + HISTORY_SIZES[0]: 1 + HISTORY_SIZES[0] + HISTORY_SIZES[1]]
    indices_latents_history_1x = indices[:, 1 + HISTORY_SIZES[0] + HISTORY_SIZES[1]: 1 + HISTORY_WINDOW_SIZE]
    indices_hidden_states = indices[:, 1 + HISTORY_WINDOW_SIZE:]
    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=1)

    # 拆分 history
    latents_history_long = history_latents[:, :, :HISTORY_SIZES[0], :, :]
    latents_history_mid = history_latents[:, :, HISTORY_SIZES[0]:HISTORY_SIZES[0] + HISTORY_SIZES[1], :, :]
    latents_history_1x = history_latents[:, :, HISTORY_SIZES[0] + HISTORY_SIZES[1]:, :, :]

    if x0_latents is not None:
        latents_history_short = torch.cat([x0_latents, latents_history_1x], dim=2)
    else:
        latents_history_short = latents_history_1x

    return (
        target_latents,
        indices_hidden_states.to(device),
        indices_latents_history_short.to(device),
        indices_latents_history_mid.to(device),
        indices_latents_history_long.to(device),
        latents_history_short,
        latents_history_mid,
        latents_history_long,
    )


def add_noise_flow_matching(model_input, sigma):
    """
    流匹配加噪:
        noisy_input = (1 - sigma) * x + sigma * noise
        target_flow = noise - x
    """
    noise = torch.randn_like(model_input)
    noisy_input = (1.0 - sigma) * model_input + sigma * noise
    target_flow = noise - model_input
    return noisy_input, target_flow, noise


def compute_p_soft_for_sample(
    transformer,
    vae_latent,      # (num_chunks, C, 9, H, W) 或 (C, total_T, H, W)
    prompt_embed,    # (1, seq_len, 4096)
    choice_idx,
    fixed_timestep=500.0,
    tau=1.0,
    device="cuda",
    dtype=torch.bfloat16,
    seed=42,
):
    """
    计算单个训练样本的 P_soft 标签。

    复用 dataloader_v3 的 prepare_stage1_latent 切分逻辑。

    Returns:
        p_soft: list of float, 长度 = N_gap（GAP 帧数量）
        如果 GAP 帧不足，返回 None
    """
    from helios.dataset.dataloader_history_latents_dist_v3 import BucketedFeatureDataset

    # 创建一个临时 dataset 对象来复用 prepare_stage1_latent
    # 我们直接手动实现切分逻辑，避免依赖完整的 dataset 初始化

    # vae_latent shape: (num_chunks, C, 9, H, W) -- 与 .pt 文件格式一致
    if vae_latent.ndim == 5:
        source_latent = vae_latent
    else:
        raise ValueError(f"Unexpected vae_latent shape: {vae_latent.shape}")

    total_sections = source_latent.shape[0]
    latent_window_size = source_latent.shape[2]
    history_window_size = HISTORY_WINDOW_SIZE
    section_size = history_window_size + latent_window_size

    # x0 latent
    x0_latent = source_latent[0, :, :1, :, :].clone()

    # 构建 continue_source_latent (与 dataloader_v3 一致)
    temp_source_latent = rearrange(source_latent, "b c t h w -> c (b t) h w")
    zero_padding = torch.zeros(
        temp_source_latent.shape[0],
        history_window_size,
        temp_source_latent.shape[2],
        temp_source_latent.shape[3],
        device=temp_source_latent.device,
        dtype=temp_source_latent.dtype,
    )
    continue_source_latent = torch.cat([zero_padding, temp_source_latent], dim=1)

    # 提取 history 和 target
    start_indice = choice_idx * latent_window_size
    end_indice = start_indice + section_size

    history_latent = continue_source_latent[:, start_indice: start_indice + history_window_size, :, :]
    target_latent = continue_source_latent[:, start_indice + history_window_size: end_indice, :, :]

    # 提取 GAP 帧
    # x0_end_in_continue: continue 坐标中 x0 帧之后的位置（跳过首帧，首帧不作为 GAP）
    x0_end_in_continue = history_window_size + 1  # = 19 + 1 = 20（x0 占 continue[19]，从 20 开始算 GAP）
    gap_end = start_indice  # continue 坐标中 history 窗口起始
    if gap_end <= x0_end_in_continue:
        return None  # 没有 GAP 帧

    gap_latent = continue_source_latent[:, x0_end_in_continue: gap_end, :, :]
    gap_frame_indices = torch.arange(x0_end_in_continue, gap_end) - history_window_size

    N_gap = gap_latent.shape[1]
    if N_gap == 0:
        return None

    # 准备 clean inputs (添加 batch 维)
    (
        target_latents_clean,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
    ) = prepare_clean_inputs(history_latent, target_latent, x0_latent, device=device, dtype=dtype)

    # 准备 prompt
    prompt_embed = prompt_embed.to(device, dtype=dtype)
    if prompt_embed.ndim == 2:
        prompt_embed = prompt_embed.unsqueeze(0)

    # 固定 timestep 和 sigma
    # ⚠️ 近似: sigma = timestep / 1000.0
    # 这假设 Helios 使用线性 noise schedule (t ∈ [0,1000] → σ ∈ [0,1])。
    # 如果 Helios 内部用 logit-normal 或其他非线性 schedule，
    # 应改用 scheduler.get_sigma(timestep) 以保持一致。
    # TODO: 验证此近似对 P_soft 质量的影响（消融实验）。
    sigma = fixed_timestep / 1000.0
    sigma_tensor = torch.tensor([sigma], device=device, dtype=dtype).reshape(1, 1, 1, 1, 1)
    # ⚠️ timestep 必须是 float32，因为 Timesteps (sinusoidal proj) 内部要求 float32
    timestep_tensor = torch.tensor([fixed_timestep], device=device, dtype=torch.float32)

    # 每个 (video_index, choice_idx) 使用独立 seed，避免所有样本噪声完全相同
    # seed 已经由调用方传入 (main 中会为不同样本计算不同 seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(target_latents_clean.shape, generator=generator, device=device, dtype=dtype)
    noisy_input = (1.0 - sigma_tensor) * target_latents_clean + sigma_tensor * noise
    target_flow = noise - target_latents_clean

    # === Baseline MSE（不加任何 GAP 帧）===
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        pred_baseline = transformer(
            hidden_states=noisy_input,
            timestep=timestep_tensor,
            encoder_hidden_states=prompt_embed,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            return_dict=False,
        )
        if isinstance(pred_baseline, tuple):
            pred_baseline = pred_baseline[0]
        mse_baseline = ((pred_baseline - target_flow) ** 2).mean().item()

    # === 逐一加入每个 GAP 帧，计算 MSE ===
    # 方式: 将 GAP 帧拼接到 latents_history_long 中，同时更新 indices
    delta_mse = torch.zeros(N_gap)

    for i in range(N_gap):
        # 提取单个 GAP 帧 (C, 1, H, W) -> (1, C, 1, H, W)
        single_gap = gap_latent[:, i:i + 1, :, :].unsqueeze(0).to(device, dtype=dtype)
        single_idx = gap_frame_indices[i:i + 1].unsqueeze(0).to(device)  # (1, 1)

        # 将 GAP 帧拼接到 latents_history_long 的前面
        aug_latents_history_long = torch.cat([single_gap, latents_history_long], dim=2)
        aug_indices_history_long = torch.cat([single_idx, indices_latents_history_long], dim=1)

        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            pred_with_gap = transformer(
                hidden_states=noisy_input,
                timestep=timestep_tensor,
                encoder_hidden_states=prompt_embed,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=aug_indices_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=aug_latents_history_long,
                return_dict=False,
            )
            if isinstance(pred_with_gap, tuple):
                pred_with_gap = pred_with_gap[0]
            mse_with_gap = ((pred_with_gap - target_flow) ** 2).mean().item()

        delta_mse[i] = mse_baseline - mse_with_gap  # 正值 = 该帧有帮助

    # Softmax 得到 P_soft
    p_soft = torch.softmax(delta_mse / tau, dim=0)
    return p_soft.tolist()


def compute_p_soft_chunk_level(
    transformer,
    vae_latent,
    prompt_embed,
    choice_idx,
    gap_chunk_indices,
    fixed_timestep=500.0,
    tau=1.0,
    device="cuda",
    dtype=torch.bfloat16,
    seed=42,
    metric="mse",
    vae=None,
    latents_mean=None,
    latents_std=None,
    lpips_fn=None,
):
    """
    以 chunk 为粒度计算 P_soft（与 selector_meta.json 中的 gap_chunk_indices 对齐）。

    metric="mse":  ΔMSE (latent 空间)
    metric="lpips": ΔLPIPS (像素空间感知距离，需要 VAE decode)

    Returns:
        p_soft_chunks: list of float, 长度 = len(gap_chunk_indices)
    """

    source_latent = vae_latent  # (num_chunks, C, 9, H, W)
    total_sections = source_latent.shape[0]
    latent_window_size = source_latent.shape[2]
    history_window_size = HISTORY_WINDOW_SIZE

    # x0 latent
    x0_latent = source_latent[0, :, :1, :, :].clone()

    # 构建 continue_source_latent
    temp_source_latent = rearrange(source_latent, "b c t h w -> c (b t) h w")
    zero_padding = torch.zeros(
        temp_source_latent.shape[0], history_window_size,
        temp_source_latent.shape[2], temp_source_latent.shape[3],
        device=temp_source_latent.device, dtype=temp_source_latent.dtype,
    )
    continue_source_latent = torch.cat([zero_padding, temp_source_latent], dim=1)

    # 提取 history 和 target
    start_indice = choice_idx * latent_window_size
    section_size = history_window_size + latent_window_size
    end_indice = start_indice + section_size

    history_latent = continue_source_latent[:, start_indice: start_indice + history_window_size, :, :]
    target_latent = continue_source_latent[:, start_indice + history_window_size: end_indice, :, :]

    # 准备 clean inputs
    (
        target_latents_clean, indices_hidden_states,
        indices_latents_history_short, indices_latents_history_mid,
        indices_latents_history_long, latents_history_short,
        latents_history_mid, latents_history_long,
    ) = prepare_clean_inputs(history_latent, target_latent, x0_latent, device=device, dtype=dtype)

    prompt_embed = prompt_embed.to(device, dtype=dtype)
    if prompt_embed.ndim == 2:
        prompt_embed = prompt_embed.unsqueeze(0)

    # 固定噪声
    # ⚠️ 近似: sigma = timestep / 1000.0 (同 compute_p_soft_for_sample 中的注释)
    sigma = fixed_timestep / 1000.0
    sigma_tensor = torch.tensor([sigma], device=device, dtype=dtype).reshape(1, 1, 1, 1, 1)
    # ⚠️ timestep 必须是 float32，因为 Timesteps (sinusoidal proj) 内部要求 float32
    timestep_tensor = torch.tensor([fixed_timestep], device=device, dtype=torch.float32)

    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(target_latents_clean.shape, generator=generator, device=device, dtype=dtype)
    noisy_input = (1.0 - sigma_tensor) * target_latents_clean + sigma_tensor * noise
    target_flow = noise - target_latents_clean

    # --- 评分函数: 根据 metric 选择 MSE 或 LPIPS ---
    use_lpips = (metric == "lpips" and vae is not None and lpips_fn is not None)

    if use_lpips:
        # 预先 decode GT target pixel (只需一次)
        gt_pixel = decode_latent_to_pixel(vae, target_latents_clean, latents_mean, latents_std)

    def _score_pred(pred_flow):
        """给定 predicted flow，返回 score (越大越差)"""
        if use_lpips:
            x0_hat = recover_x0_from_flow(noisy_input.float(), pred_flow.float(), sigma)
            pred_pixel = decode_latent_to_pixel(vae, x0_hat, latents_mean, latents_std)
            return compute_lpips_distance(lpips_fn, pred_pixel, gt_pixel)
        else:
            return ((pred_flow - target_flow) ** 2).mean().item()

    # Baseline score
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        pred_baseline = transformer(
            hidden_states=noisy_input, timestep=timestep_tensor,
            encoder_hidden_states=prompt_embed,
            indices_hidden_states=indices_hidden_states,
            indices_latents_history_short=indices_latents_history_short,
            indices_latents_history_mid=indices_latents_history_mid,
            indices_latents_history_long=indices_latents_history_long,
            latents_history_short=latents_history_short,
            latents_history_mid=latents_history_mid,
            latents_history_long=latents_history_long,
            return_dict=False,
        )
        if isinstance(pred_baseline, tuple):
            pred_baseline = pred_baseline[0]
        score_baseline = _score_pred(pred_baseline)

    # 逐 chunk 计算 delta
    delta_chunks = torch.zeros(len(gap_chunk_indices))

    x0_end_in_continue = history_window_size + 1
    for ci, chunk_idx in enumerate(gap_chunk_indices):
        chunk_continue_start = history_window_size + chunk_idx * latent_window_size
        chunk_continue_end = chunk_continue_start + latent_window_size

        actual_start = max(chunk_continue_start, x0_end_in_continue)
        actual_end = min(chunk_continue_end, start_indice)

        if actual_end <= actual_start:
            delta_chunks[ci] = 0.0
            continue

        chunk_gap_latent = continue_source_latent[:, actual_start:actual_end, :, :]
        chunk_gap_latent = chunk_gap_latent.unsqueeze(0).to(device, dtype=dtype)
        chunk_gap_indices = (torch.arange(actual_start, actual_end) - history_window_size).unsqueeze(0).to(device)

        aug_latents_history_long = torch.cat([chunk_gap_latent, latents_history_long], dim=2)
        aug_indices_history_long = torch.cat([chunk_gap_indices, indices_latents_history_long], dim=1)

        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            pred_with_chunk = transformer(
                hidden_states=noisy_input, timestep=timestep_tensor,
                encoder_hidden_states=prompt_embed,
                indices_hidden_states=indices_hidden_states,
                indices_latents_history_short=indices_latents_history_short,
                indices_latents_history_mid=indices_latents_history_mid,
                indices_latents_history_long=aug_indices_history_long,
                latents_history_short=latents_history_short,
                latents_history_mid=latents_history_mid,
                latents_history_long=aug_latents_history_long,
                return_dict=False,
            )
            if isinstance(pred_with_chunk, tuple):
                pred_with_chunk = pred_with_chunk[0]
            score_with_chunk = _score_pred(pred_with_chunk)

        delta_chunks[ci] = score_baseline - score_with_chunk  # 正值 = 该 chunk 有帮助

    # Softmax
    p_soft = torch.softmax(delta_chunks / tau, dim=0)
    return p_soft.tolist()


def main():
    parser = argparse.ArgumentParser(description="计算 P_soft 标签")
    parser.add_argument(
        "--meta_json",
        type=str,
        default="/root/autodl-tmp/Helios/example/lighting_change/selector_meta.json",
        help="selector_meta.json 路径",
    )
    parser.add_argument(
        "--transformer_path",
        type=str,
        default="/root/autodl-fs/BestWishYSH/Helios-Base",
        help="Helios Transformer 模型路径",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default="/root/autodl-tmp/Helios/example/lighting_change/selector_meta_with_psoft.json",
        help="输出 JSON（含 P_soft）",
    )
    parser.add_argument("--tau", type=float, default=1.0, help="Softmax 温度")
    parser.add_argument("--fixed_timestep", type=float, default=500.0, help="固定 timestep（建议 500）")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--chunk_level", action="store_true",
                        help="以 chunk 为粒度计算 P_soft（与 selector_meta.json gap_chunk_indices 对齐）。默认开启。")
    parser.add_argument("--no_chunk_level", action="store_true",
                        help="以单帧为粒度计算 P_soft（逐帧逐一加入）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--metric", type=str, default="mse", choices=["mse", "lpips"],
                        help="评分指标: mse (latent空间) 或 lpips (像素空间感知距离)")
    parser.add_argument("--vae_path", type=str, default=None,
                        help="VAE 模型路径 (LPIPS 模式必需，默认与 transformer_path 相同)")
    args = parser.parse_args()

    # 加载 selector_meta.json
    with open(args.meta_json, "r", encoding="utf-8") as f:
        meta = json.load(f)

    print(f"[INFO] 读取 {len(meta)} 个视频的 selector_meta")

    # 统计总样本数
    total_samples = sum(len(e["selector_samples"]) for e in meta)
    print(f"[INFO] 总训练样本数: {total_samples}")

    # 加载 Transformer
    dtype = torch.bfloat16
    transformer = load_transformer(args.transformer_path, device=args.device, dtype=dtype)

    # 加载 VAE + LPIPS (仅 lpips 模式)
    vae_model, l_mean, l_std, lpips_fn = None, None, None, None
    if args.metric == "lpips":
        vae_p = args.vae_path or args.transformer_path
        print(f"[INFO] LPIPS 模式: 加载 VAE from {vae_p}")
        vae_model, l_mean, l_std = load_vae(vae_p, device=args.device)
        lpips_fn = lpips.LPIPS(net="vgg").to(args.device).eval()
        print(f"[INFO] VAE + LPIPS(VGG) 加载完成")

    # 决定是否使用 chunk 粒度（默认 chunk_level，除非显式 --no_chunk_level）
    use_chunk_level = not args.no_chunk_level  # 默认 True
    if args.chunk_level:
        use_chunk_level = True
    print(f"[INFO] 计算粒度: {'chunk_level' if use_chunk_level else 'frame_level'}, 指标: {args.metric}")

    # 逐视频、逐 choice_idx 计算 P_soft
    processed = 0
    skipped = 0

    for vi, entry in enumerate(tqdm(meta, desc="视频")):
        latent_path = entry["latent_pt_path"]
        # 尝试多种路径
        if not os.path.isabs(latent_path):
            base_dir = os.path.dirname(args.meta_json)
            latent_path = os.path.join(base_dir, latent_path)

        if not os.path.exists(latent_path):
            print(f"  [WARN] Latent 文件不存在: {latent_path}, 跳过视频 {entry['video_id']}")
            skipped += len(entry["selector_samples"])
            continue

        # 加载 .pt 文件
        feature_data = torch.load(latent_path, map_location="cpu", weights_only=False)
        vae_latent = feature_data["vae_latent"]  # (num_chunks, C, 9, H, W)
        prompt_embed = feature_data["prompt_embed"]  # (1, seq_len, 4096)

        for si, sample in enumerate(entry["selector_samples"]):
            choice_idx = sample["choice_idx"]
            gap_chunk_indices = sample["gap_chunk_indices"]

            # 每个 (video, choice_idx) 使用独立 seed，避免所有样本噪声完全相同
            sample_seed = args.seed + choice_idx * 1000 + vi * 100000

            try:
                if use_chunk_level:
                    p_soft = compute_p_soft_chunk_level(
                        transformer=transformer,
                        vae_latent=vae_latent,
                        prompt_embed=prompt_embed,
                        choice_idx=choice_idx,
                        gap_chunk_indices=gap_chunk_indices,
                        fixed_timestep=args.fixed_timestep,
                        tau=args.tau,
                        device=args.device,
                        dtype=dtype,
                        seed=sample_seed,
                        metric=args.metric,
                        vae=vae_model,
                        latents_mean=l_mean,
                        latents_std=l_std,
                        lpips_fn=lpips_fn,
                    )
                else:
                    p_soft = compute_p_soft_for_sample(
                        transformer=transformer,
                        vae_latent=vae_latent,
                        prompt_embed=prompt_embed,
                        choice_idx=choice_idx,
                        fixed_timestep=args.fixed_timestep,
                        tau=args.tau,
                        device=args.device,
                        dtype=dtype,
                        seed=sample_seed,
                    )
                sample["p_soft"] = p_soft
                processed += 1
            except Exception as ex:
                print(f"  [WARN] 计算 P_soft 失败: video={entry['video_id']}, choice_idx={choice_idx}: {ex}")
                traceback.print_exc()
                sample["p_soft"] = None
                skipped += 1

            if (processed + skipped) % 50 == 0:
                print(f"  进度: {processed} 完成, {skipped} 跳过")

    # 保存结果
    os.makedirs(os.path.dirname(args.output_json) if os.path.dirname(args.output_json) else ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n[INFO] P_soft 计算完成:")
    print(f"  处理样本: {processed}")
    print(f"  跳过样本: {skipped}")
    print(f"  输出文件: {args.output_json}")

    # 统计 P_soft 分布
    all_p_softs = []
    for entry in meta:
        for sample in entry["selector_samples"]:
            if sample["p_soft"] is not None:
                all_p_softs.append(sample["p_soft"])
    if all_p_softs:
        max_vals = [max(ps) for ps in all_p_softs]
        min_vals = [min(ps) for ps in all_p_softs]
        print(f"\n  === P_soft 分布统计 ===")
        print(f"  样本数: {len(all_p_softs)}")
        print(f"  每样本最大概率 - 均值: {sum(max_vals)/len(max_vals):.4f}, 范围: [{min(max_vals):.4f}, {max(max_vals):.4f}]")
        print(f"  每样本最小概率 - 均值: {sum(min_vals)/len(min_vals):.4f}")
        lens = [len(ps) for ps in all_p_softs]
        print(f"  P_soft 长度范围: {min(lens)} - {max(lens)}")


if __name__ == "__main__":
    main()
