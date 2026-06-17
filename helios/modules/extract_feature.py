"""
extract_feature.py — 推理时在线 CLIP 特征提取
================================================

每生成完一个 chunk，当场提取 CLIP 视觉特征。
不做任何离线预处理，所有计算在推理循环中实时完成。

用途：为 BOLT 选帧提供 visual score 和 text score 的基础特征。
Use cases: online feature extraction for frame selection during inference.
"""

import torch
import numpy as np
from PIL import Image
from transformers import CLIPModel, AutoTokenizer, CLIPProcessor


CLIP_DEFAULT_PATH = "/root/autodl-fs/clip-vit-large-patch14/AI-ModelScope/clip-vit-large-patch14"
DINOv2_DEFAULT_ID = "/root/autodl-fs/dinov2-base"


class CLIP:
    """CLIP ViT-L/14 封装 / wrapper.

    提供视觉/文本特征提取和相似度计算。
    Provides image/text embedding extraction and cosine similarity scoring.
    """

    def __init__(self, device="cuda", model_path=None):
        if model_path is None:
            model_path = CLIP_DEFAULT_PATH
        self.model = CLIPModel.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.device = device
        self.model.eval()

    def extract_visual_features(self, images):
        """视觉特征 / image features.

        输入: list[PIL.Image] → 输出: Tensor (N, 768)
        """
        vision_inputs = self.processor(images=images, return_tensors="pt")
        # 确保输入与模型在同一设备和 dtype
        model_dtype = next(self.model.parameters()).dtype
        vision_inputs = {k: v.to(device=self.device, dtype=model_dtype)
                         if isinstance(v, torch.Tensor) and v.is_floating_point() else
                         v.to(device=self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in vision_inputs.items()}
        with torch.no_grad():
            out = self.model.get_image_features(**vision_inputs)
            if not isinstance(out, torch.Tensor):
                out = out.image_embeds if hasattr(out, "image_embeds") else out.pooler_output
            return out

    def extract_text_features(self, text):
        """文本特征 / text features.

        输入: str → 输出: Tensor (1, 768)
        """
        text_inputs = self.tokenizer(
            [text], truncation=True, padding="max_length", return_tensors="pt",
        )
        text_inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                       for k, v in text_inputs.items()}
        with torch.no_grad():
            out = self.model.get_text_features(**text_inputs)
            if not isinstance(out, torch.Tensor):
                out = out.text_embeds if hasattr(out, "text_embeds") else out.pooler_output
            return out

    def compute_similarity(self, video_features, query_features):
        """
        video_features: (N, 768), query_features: (1, 768)
        输出: Tensor (N,) 余弦相似度分数
        """
        video_features = video_features / video_features.norm(p=2, dim=-1, keepdim=True)
        query_features = query_features / query_features.norm(p=2, dim=-1, keepdim=True)
        logits = torch.matmul(query_features, video_features.t().to(query_features.device))
        logits = logits * self.model.logit_scale.exp().to(query_features.device)
        return logits.detach().cpu().squeeze(0).float()


class DINOv2:
    """DINOv2 封装 / wrapper.

    提供视觉 embedding 提取（适合做 LongMemory 的 codebook key）。
    Provides image embeddings for LongMemory codebook keys.

    依赖 / dependency: transformers Dinov2Model / AutoImageProcessor.
    """

    def __init__(self, device="cuda", model_id_or_path: str | None = None, dtype: str = "fp16"):
        if model_id_or_path is None:
            model_id_or_path = DINOv2_DEFAULT_ID
        # 延迟 import，避免不需要时的依赖开销
        from transformers import AutoImageProcessor, Dinov2Model  # type: ignore

        torch_dtype = {"fp32": torch.float32, "fp16": torch.float16}.get(dtype, torch.bfloat16)
        self.processor = AutoImageProcessor.from_pretrained(model_id_or_path)
        self.model = Dinov2Model.from_pretrained(model_id_or_path, torch_dtype=torch_dtype).to(device)
        self.device = device
        self.model.eval()

    def extract_visual_features(self, images):
        """视觉 embedding / image embedding.

        输入: list[PIL.Image] → 输出: Tensor (N, D) on CPU
        """
        inputs = self.processor(images=images, return_tensors="pt")
        model_dtype = next(self.model.parameters()).dtype
        inputs = {
            k: v.to(device=self.device, dtype=model_dtype)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
            else v.to(device=self.device)
            if isinstance(v, torch.Tensor)
            else v
            for k, v in inputs.items()
        }
        with torch.no_grad():
            out = self.model(**inputs)
            # Dinov2Model outputs last_hidden_state and pooler_output is not always set.
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                feat = out.pooler_output
            else:
                # CLS token as global embedding
                feat = out.last_hidden_state[:, 0]
            return feat.detach().cpu().float()


def decode_middle_frame(chunk_latent, vae, latents_mean, latents_std):
    """解码 chunk 中间帧 / decode middle frame.

    Args:
        chunk_latent: (B, C, T, H, W) 或 (C, T, H, W)
        vae: AutoencoderKLWan
        latents_mean, latents_std: VAE normalization 参数 (与 pipeline 一致)

    Returns:
        PIL Image (RGB)
    """
    if chunk_latent.ndim == 4:
        chunk_latent = chunk_latent.unsqueeze(0)

    T = chunk_latent.shape[2]
    mid_t = T // 2
    mid_latent = chunk_latent[:, :, mid_t:mid_t + 1, :, :]

    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        mid_latent = mid_latent.to(device=vae_device, dtype=vae_dtype)
        latents_std = latents_std.to(device=vae_device, dtype=vae_dtype)
        latents_mean = latents_mean.to(device=vae_device, dtype=vae_dtype)
        normalized = mid_latent / latents_std + latents_mean
        pixel = vae.decode(normalized).sample

    pixel = pixel[0, :, 0].clamp(-1, 1).add(1).div(2)
    pixel = pixel.permute(1, 2, 0).cpu().float().numpy()
    pixel = (pixel * 255).astype(np.uint8)
    return Image.fromarray(pixel)


def decode_last_frame(chunk_latent, vae, latents_mean, latents_std):
    """解码 chunk 最后一帧 / decode last frame (tail frame)."""
    if chunk_latent.ndim == 4:
        chunk_latent = chunk_latent.unsqueeze(0)

    T = int(chunk_latent.shape[2])
    last_t = max(0, T - 1)
    last_latent = chunk_latent[:, :, last_t:last_t + 1, :, :]

    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        last_latent = last_latent.to(device=vae_device, dtype=vae_dtype)
        latents_std = latents_std.to(device=vae_device, dtype=vae_dtype)
        latents_mean = latents_mean.to(device=vae_device, dtype=vae_dtype)
        normalized = last_latent / latents_std + latents_mean
        pixel = vae.decode(normalized).sample

    pixel = pixel[0, :, 0].clamp(-1, 1).add(1).div(2)
    pixel = pixel.permute(1, 2, 0).cpu().float().numpy()
    pixel = (pixel * 255).astype(np.uint8)
    return Image.fromarray(pixel)


def decode_all_frames(chunk_latent, vae, latents_mean, latents_std, max_frames=None):
    """解码 chunk 全部帧 / decode all frames in a chunk.

    Args:
        chunk_latent: (B, C, T, H, W) 或 (C, T, H, W)
        vae: AutoencoderKLWan
        latents_mean, latents_std: VAE normalization 参数 (与 pipeline 一致)
        max_frames: 可选。若设置且小于 T，则按等间隔下采样到 max_frames 帧

    Returns:
        list[PIL.Image]，长度为 T 或 max_frames
    """
    if chunk_latent.ndim == 4:
        chunk_latent = chunk_latent.unsqueeze(0)

    if chunk_latent.shape[0] != 1:
        raise ValueError(f"decode_all_frames expects batch size 1, got {chunk_latent.shape[0]}")

    total_t = int(chunk_latent.shape[2])
    vae_device = next(vae.parameters()).device
    vae_dtype = next(vae.parameters()).dtype
    with torch.no_grad():
        chunk_latent = chunk_latent.to(device=vae_device, dtype=vae_dtype)
        latents_std = latents_std.to(device=vae_device, dtype=vae_dtype)
        latents_mean = latents_mean.to(device=vae_device, dtype=vae_dtype)
        normalized = chunk_latent / latents_std + latents_mean
        pixel = vae.decode(normalized).sample

    # (1, 3, T, H, W) -> (T, H, W, 3)
    pixel = pixel[0].clamp(-1, 1).add(1).div(2)
    pixel = pixel.permute(1, 2, 3, 0).cpu().float().numpy()
    pixel = (pixel * 255).astype(np.uint8)

    if max_frames is not None:
        max_frames = int(max_frames)
        if max_frames <= 0:
            raise ValueError(f"max_frames must be > 0, got {max_frames}")
        if max_frames < total_t:
            sampled_idx = np.linspace(0, total_t - 1, num=max_frames, dtype=int).tolist()
            return [Image.fromarray(pixel[i]) for i in sampled_idx]

    return [Image.fromarray(pixel[i]) for i in range(total_t)]


def extract_chunk_feature(chunk_latent, vae, clip_model, latents_mean, latents_std):
    """CLIP 特征提取 / CLIP feature extraction.

    每生成完一个 chunk 后调用：decode 中间帧 → CLIP 提取特征。
    Called after each chunk: decode middle frame -> CLIP image embedding.

    Returns:
        clip_feat: Tensor (768,) — 存入 history
        middle_frame: PIL Image — 可选调试用
    """
    middle_frame = decode_middle_frame(chunk_latent, vae, latents_mean, latents_std)
    clip_feat = clip_model.extract_visual_features([middle_frame]).squeeze(0).cpu()
    return clip_feat, middle_frame


def extract_tail_embedding(
    chunk_latent,
    vae,
    image_encoder,
    latents_mean,
    latents_std,
    *,
    return_frame: bool = False,
):
    """为 LongMemory 提取“chunk 尾帧”的 embedding / tail-frame embedding for LongMemory.

    Returns:
        emb: Tensor (D,) on CPU — 用于 codebook key / used as codebook key
        tail_frame (optional): PIL Image — 可选返回像素帧 / optionally return the pixel frame
    """
    tail_frame = decode_last_frame(chunk_latent, vae, latents_mean, latents_std)
    emb = image_encoder.extract_visual_features([tail_frame]).squeeze(0).cpu()
    if return_frame:
        return emb, tail_frame
    return emb
