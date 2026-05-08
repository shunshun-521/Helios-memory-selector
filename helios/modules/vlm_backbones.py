"""
vlm_backbones.py — VLM Selector 的实际 backbone 实现
=======================================================

本模块提供零样本的 VLM 打分 backbone，用于在没有训练好的 LoRA/head 权重时，
让 ``--selector_type vlm`` 也能走 **真正的** VLM forward（而不是 CLIP 占位）。

设计原则
---------
1. **可选依赖**：import 本模块不会触发 transformers / Qwen2.5-VL 权重加载；
   加载逻辑集中在 ``QwenVLZeroShotBackbone.__init__``。
2. **统一接口**：任何 backbone 都应暴露::

       score(context_image: PIL.Image | None,
             candidate_images: list[PIL.Image],
             prompt: str) -> np.ndarray  # shape (N_cand,)

3. **不破坏 CLIP+ITS 路径**：backbone 仅在 ``selector_type=vlm`` 且用户显式
   传入 ``--vlm_model_path`` 时构造；失败时 ``VLMFrameSelector`` 会自动回退
   到 CLIP 占位打分（进一步再回退到 CLIP+ITS，由 infer 层控制）。

零样本打分方法
---------------
对每个候选帧，构造一条对话::

    system: "You are a precise video-frame scoring assistant."
    user:   <context_image?> <candidate_image>
            "Does this candidate frame help continue the scene towards:
            '{prompt}'? Answer strictly yes or no."

读取 LM 最后一步的 logits，取 ``P('yes') - P('no')`` 作为分数。
这是社区中经过验证的 VLM zero-shot relevance scoring pattern，
无需任何训练就能跑通推理，并且在有 LoRA/head 权重时可以无缝切换到
"hidden_state → linear head" 路径（见 md/CLIP-base-selector-vlm.md §2.2 A 路径）。
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np


__all__ = [
    "QwenVLZeroShotBackbone",
    "QwenVLHiddenHeadBackbone",
    "load_default_backbone",
]


class QwenVLZeroShotBackbone:
    """Qwen2.5-VL backbone 的零样本打分实现。

    Args:
        model_path: HuggingFace 本地路径，例如 ``/root/autodl-fs/Qwen2.5-VL-3B-Instruct``
        lora_path: 可选 LoRA adapter 路径（训练完成后填入）
        dtype: "bfloat16" / "float16" / "float32"
        device: "cuda" / "cpu"
        image_resize: (h, w)，控制每张图的 visual tokens 数量
        offload_to_cpu: slow step 后是否 ``.to('cpu')`` 腾显存给 DiT
    """

    DEFAULT_SYSTEM = (
        "You are a video continuity assistant. Your job is to decide whether "
        "a past frame from the same video is a useful visual reference for "
        "preserving a character's identity (face, hairstyle, clothing color, "
        "accessories) when generating the next frame, in a video that contains "
        "strong lighting changes (e.g. tunnels, shadow-to-light transitions)."
    )
    YES_TOKENS = ("yes", "Yes", "YES")
    NO_TOKENS = ("no", "No", "NO")

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        image_resize=(256, 448),
        offload_to_cpu: bool = True,
    ):
        import torch
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.torch = torch
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)
        self.device = device
        self.image_resize = tuple(image_resize)
        self.offload_to_cpu = bool(offload_to_cpu)

        print(f"[QwenVLZeroShotBackbone] loading model from {model_path} ...")
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=self.dtype,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(model_path)

        if lora_path is not None:
            try:
                from peft import PeftModel

                self.model = PeftModel.from_pretrained(self.model, lora_path)
                print(f"[QwenVLZeroShotBackbone] merged LoRA adapter from {lora_path}")
            except Exception as exc:  # noqa: BLE001
                print(f"[QwenVLZeroShotBackbone] LoRA load failed ({exc}); running w/o LoRA.")

        self._on_gpu = False
        self._ensure_device(to_gpu=not self.offload_to_cpu)

        tokenizer = self.processor.tokenizer
        self._yes_ids = self._collect_token_ids(tokenizer, self.YES_TOKENS)
        self._no_ids = self._collect_token_ids(tokenizer, self.NO_TOKENS)
        if not self._yes_ids or not self._no_ids:
            raise RuntimeError(
                "QwenVLZeroShotBackbone: cannot locate yes/no token ids in tokenizer vocab."
            )

    # ───────────── 工具 ─────────────
    @staticmethod
    def _collect_token_ids(tokenizer, variants):
        ids = set()
        for v in variants:
            for cand in (v, " " + v):
                try:
                    tids = tokenizer.encode(cand, add_special_tokens=False)
                except Exception:  # noqa: BLE001
                    continue
                if len(tids) == 1:
                    ids.add(int(tids[0]))
        return sorted(ids)

    def _ensure_device(self, to_gpu: bool):
        if to_gpu and not self._on_gpu:
            self.model.to(self.device)
            self._on_gpu = True
        elif (not to_gpu) and self._on_gpu:
            self.model.to("cpu")
            self._on_gpu = False

    def to_cpu(self):
        self._ensure_device(to_gpu=False)
        try:
            self.torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def to_gpu(self):
        self._ensure_device(to_gpu=True)

    # ───────────── 核心打分 ─────────────
    def _resize(self, pil_img):
        if pil_img is None:
            return None
        h, w = self.image_resize
        return pil_img.resize((w, h))

    def _build_messages_from_frames(self, context_frames: List, candidate_frames: List, prompt: str):
        content = []
        for frame in context_frames or []:
            if frame is not None:
                content.append({"type": "image", "image": self._resize(frame)})
        for frame in candidate_frames or []:
            if frame is not None:
                content.append({"type": "image", "image": self._resize(frame)})
        content.append(
            {
                "type": "text",
                "text": (
                    "Image Group 1 is the most recent generated context chunk (anchor video clip).\n"
                    "Image Group 2 is a candidate historical chunk from earlier in the same video.\n"
                    f'The next frame to be generated is described by this prompt: "{prompt}".\n\n'
                    "A candidate is useful ONLY IF all of the following hold:\n"
                    "  (a) it shows the same character as in Image Group 1;\n"
                    "  (b) its lighting condition (bright / dim / dark) matches the lighting "
                    "implied by the upcoming prompt;\n"
                    "  (c) the character's identity details (face, clothing color, hair) are "
                    "clearly visible and not occluded or motion-blurred.\n\n"
                    "Is Image Group 2 a useful identity reference for the next frame?\n"
                    "Answer strictly with a single word: yes or no."
                ),
            }
        )
        return [
            {"role": "system", "content": [{"type": "text", "text": self.DEFAULT_SYSTEM}]},
            {"role": "user", "content": content},
        ]

    def _build_messages(self, context_image, candidate_image, prompt: str):
        context_frames = [context_image] if context_image is not None else []
        candidate_frames = [candidate_image]
        return self._build_messages_from_frames(context_frames, candidate_frames, prompt)

    def _build_video_messages(self, context_frames, candidate_frames, prompt: str):
        return self._build_messages_from_frames(
            context_frames=context_frames or [],
            candidate_frames=candidate_frames or [],
            prompt=prompt,
        )

    def _prepare_inputs(self, messages):
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images = [m for msg in messages for m in msg["content"] if m.get("type") == "image"]
        image_list = [m["image"] for m in images]
        inputs = self.processor(
            text=[text], images=image_list, return_tensors="pt", padding=True
        )
        return {
            k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()
        }

    def _score_one(self, context_image, candidate_image, prompt: str) -> float:
        messages = self._build_messages(context_image, candidate_image, prompt)
        return self._score_one_with_messages(messages)

    def _score_one_video(self, context_frames, candidate_frames, prompt: str) -> float:
        if not candidate_frames:
            raise ValueError("candidate_frames is empty for score_video.")
        messages = self._build_video_messages(context_frames, candidate_frames, prompt)
        return self._score_one_with_messages(messages)

    def _score_one_with_messages(self, messages) -> float:
        torch = self.torch
        inputs = self._prepare_inputs(messages)
        with torch.no_grad():
            out = self.model(**inputs)
        logits = out.logits[:, -1, :]
        yes_logit = logits[0, self._yes_ids].max().float().item()
        no_logit = logits[0, self._no_ids].max().float().item()
        return float(yes_logit - no_logit)

    def score(
        self,
        context_image,
        candidate_images: List,
        prompt: str,
    ) -> np.ndarray:
        """对每个候选帧打一个相关性分数。"""
        if not candidate_images:
            return np.zeros((0,), dtype=np.float32)

        self.to_gpu()
        try:
            scores = [
                self._score_one(context_image, img, prompt)
                for img in candidate_images
            ]
        finally:
            if self.offload_to_cpu:
                self.to_cpu()
        return np.asarray(scores, dtype=np.float32)

    def score_video(
        self,
        context_frames,
        candidate_videos: List[List],
        prompt: str,
    ) -> np.ndarray:
        """对候选视频（每个候选是多帧）打分。"""
        if not candidate_videos:
            return np.zeros((0,), dtype=np.float32)

        self.to_gpu()
        try:
            scores = [
                self._score_one_video(context_frames, cand_frames, prompt)
                for cand_frames in candidate_videos
            ]
        finally:
            if self.offload_to_cpu:
                self.to_cpu()
        return np.asarray(scores, dtype=np.float32)


class QwenVLHiddenHeadBackbone(QwenVLZeroShotBackbone):
    """Qwen2.5-VL + LoRA + Linear(hidden,1) 的 hidden_head 推理 backbone。

    与 QwenVLZeroShotBackbone 的区别：
    - 不再读取 yes/no token logits。
    - 读取最后一个 visual token 的 hidden state，经 selector_head 输出分数。
    """

    def __init__(
        self,
        model_path: str,
        lora_path: Optional[str] = None,
        head_path: Optional[str] = None,
        dtype: str = "bfloat16",
        device: str = "cuda",
        image_resize=(256, 448),
        offload_to_cpu: bool = True,
    ):
        if lora_path is None:
            raise ValueError("QwenVLHiddenHeadBackbone requires lora_path with selector_head.pt.")
        super().__init__(
            model_path=model_path,
            lora_path=lora_path,
            dtype=dtype,
            device=device,
            image_resize=image_resize,
            offload_to_cpu=offload_to_cpu,
        )
        self.image_token_id = int(getattr(self.model.config, "image_token_id", 151655))

        if head_path is None:
            head_path = os.path.join(lora_path, "selector_head.pt")
        if not os.path.exists(head_path):
            raise FileNotFoundError(f"selector_head.pt not found: {head_path}")
        payload = self.torch.load(head_path, map_location="cpu", weights_only=False)
        state_dict = payload.get("state_dict", payload)
        hidden_dim = int(payload.get("hidden_dim", getattr(self.model.config, "hidden_size", 2048)))

        self.selector_head = self.torch.nn.Linear(hidden_dim, 1, bias=True)
        self.selector_head.load_state_dict(state_dict, strict=True)
        self.selector_head.eval()
        self.selector_head.to("cpu", dtype=self.dtype)
        print(
            f"[QwenVLHiddenHeadBackbone] loaded selector_head from {head_path} "
            f"(hidden_dim={hidden_dim}, image_token_id={self.image_token_id})"
        )

    def _ensure_device(self, to_gpu: bool):
        super()._ensure_device(to_gpu=to_gpu)
        if not hasattr(self, "selector_head") or self.selector_head is None:
            return
        if to_gpu:
            self.selector_head.to(self.device, dtype=self.dtype)
        else:
            self.selector_head.to("cpu", dtype=self.dtype)

    def _score_one(self, context_image, candidate_image, prompt: str) -> float:
        messages = self._build_messages(context_image, candidate_image, prompt)
        inputs = self._prepare_inputs(messages)

        return self._score_one_hidden_with_inputs(inputs)

    def _score_one_video(self, context_frames, candidate_frames, prompt: str) -> float:
        messages = self._build_video_messages(context_frames, candidate_frames, prompt)
        inputs = self._prepare_inputs(messages)
        return self._score_one_hidden_with_inputs(inputs)

    def _score_one_hidden_with_inputs(self, inputs) -> float:
        torch = self.torch
        with torch.no_grad():
            out = self.model(**inputs, output_hidden_states=True, return_dict=True)

        input_ids = inputs.get("input_ids", None)
        if input_ids is None:
            raise RuntimeError("input_ids missing; cannot locate visual token for hidden_head.")
        vis_pos = (input_ids[0] == int(self.image_token_id)).nonzero(as_tuple=False)
        if vis_pos.numel() == 0:
            raise RuntimeError(
                f"image_token_id={self.image_token_id} not found in input_ids; cannot score hidden_head."
            )
        last_vis_pos = int(vis_pos[-1].item())
        hidden_last = out.hidden_states[-1][0, last_vis_pos, :]
        score = self.selector_head(hidden_last).squeeze(-1)
        return float(score.float().item())


def load_default_backbone(
    model_path: str,
    lora_path: Optional[str] = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    image_resize=(256, 448),
    offload_to_cpu: bool = True,
    score_mode: str = "yes_no",
    head_path: Optional[str] = None,
) -> Optional[object]:
    """便捷封装：加载失败时返回 None（调用方据此走占位路径）。"""
    if score_mode == "hidden_head":
        try:
            return QwenVLHiddenHeadBackbone(
                model_path=model_path,
                lora_path=lora_path,
                head_path=head_path,
                dtype=dtype,
                device=device,
                image_resize=image_resize,
                offload_to_cpu=offload_to_cpu,
            )
        except Exception as exc:  # noqa: BLE001
            print(
                f"[vlm_backbones] hidden_head load failed ({exc}); "
                "falling back to yes_no backbone."
            )
    try:
        return QwenVLZeroShotBackbone(
            model_path=model_path,
            lora_path=lora_path,
            dtype=dtype,
            device=device,
            image_resize=image_resize,
            offload_to_cpu=offload_to_cpu,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[vlm_backbones] load_default_backbone failed: {exc}")
        return None
