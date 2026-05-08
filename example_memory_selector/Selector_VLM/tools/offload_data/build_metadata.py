#!/usr/bin/env python3
from pathlib import Path
import runpy

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("build_metadata_minimal.py")), run_name="__main__")
    raise SystemExit(0)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build chunk-aligned metadata for Selector_VLM (minimal edition).

Only keeps:
1) Scene segmentation with TransNetV2.
2) Prompt generation with local Qwen3-VL in visual_batch mode.
"""

# from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2

DEFAULT_QWEN_VL_MODEL_PATH = "/root/autodl-fs/Qwen3-VL-8B-Instruct"


@dataclass
class VideoInfo:
    fps: float
    num_frames: int
    height: int
    width: int

    @property
    def duration_sec(self) -> float:
        if self.fps <= 1e-6:
            return 0.0
        return float(self.num_frames) / float(self.fps)


def read_video_info(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 1e-6:
        fps = 24.0
    return VideoInfo(fps=fps, num_frames=num_frames, height=height, width=width)


def chunk_frames(num_latent_frames_per_chunk: int, t_downsample: int) -> int:
    return int((num_latent_frames_per_chunk - 1) * t_downsample + 1)


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def ensure_monotonic_segments(segments: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    last_end = 0.0
    for s, e in segments:
        s = max(0.0, min(float(s), duration))
        e = max(0.0, min(float(e), duration))
        if e <= s:
            continue
        s = max(s, last_end)
        if e <= s:
            continue
        out.append((s, e))
        last_end = e
    if not out:
        return [(0.0, duration)]
    if out[-1][1] < duration:
        out[-1] = (out[-1][0], duration)
    return out


def detect_scenes_transnetv2(video_path: str, threshold: float, device: str, duration_sec: float) -> List[Tuple[float, float]]:
    try:
        from transnetv2_pytorch import TransNetV2  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("Install transnetv2-pytorch first: pip install transnetv2-pytorch") from e

    model = TransNetV2(device=device)
    scenes = model.detect_scenes(str(video_path), threshold=float(threshold))
    if not scenes:
        return [(0.0, duration_sec)]
    segs: List[Tuple[float, float]] = []
    for sc in scenes:
        try:
            s = float(sc.get("start_time", 0.0))
            e = float(sc.get("end_time", 0.0))
        except Exception:
            continue
        if e > s:
            segs.append((s, e))
    if not segs:
        return [(0.0, duration_sec)]
    return ensure_monotonic_segments(segs, duration_sec)


def align_segments_to_chunks(
    segments_sec: List[Tuple[float, float]],
    fps: float,
    num_frames: int,
    ch_frames: int,
    min_len_chunks: int,
) -> List[Dict[str, Any]]:
    duration = float(num_frames) / float(fps)
    segs = ensure_monotonic_segments(segments_sec, duration)
    n_chunks = max(1, int(math.ceil(num_frames / float(ch_frames))))
    out: List[Dict[str, Any]] = []
    for s, e in segs:
        s_chunk = _clamp(int(math.floor((s * fps) / float(ch_frames))), 0, n_chunks - 1)
        e_chunk = _clamp(int(math.ceil((e * fps) / float(ch_frames))), s_chunk + 1, n_chunks)
        if (e_chunk - s_chunk) < int(min_len_chunks):
            continue
        out.append(
            {
                "start_sec": float(s),
                "end_sec": float(e),
                "start_chunk": int(s_chunk),
                "end_chunk": int(e_chunk),
                "prompt": "",
            }
        )
    if not out:
        out = [
            {
                "start_sec": 0.0,
                "end_sec": duration,
                "start_chunk": 0,
                "end_chunk": n_chunks,
                "prompt": "",
            }
        ]
    return out


def load_long_prompts_jsonl(path: Optional[str]) -> Dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        vid = str(obj.get("id", obj.get("sample_id", ""))).strip()
        pr = str(obj.get("prompt", "")).strip()
        if vid:
            out[vid] = pr
    return out


def _pick_rep_times(start_sec: float, end_sec: float, n: int) -> List[float]:
    n = max(1, int(n))
    if end_sec <= start_sec:
        return [max(0.0, float(start_sec))]
    if n == 1:
        return [0.5 * (start_sec + end_sec)]
    span = end_sec - start_sec
    return [start_sec + span * (i + 1) / (n + 1) for i in range(n)]


def _extract_jpeg(video_path: str, t_sec: float) -> bytes:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed frame at {t_sec:.3f}s for {video_path}")
    ok2, enc = cv2.imencode(".jpg", frame)
    if not ok2:
        raise RuntimeError("JPEG encode failed")
    return bytes(enc.tobytes())


def _parse_segments_json(text: str, expected_n: int) -> Optional[List[str]]:
    text = text.strip()
    if text.startswith("```"):
        parts = [x for x in text.split("```") if x.strip()]
        if parts:
            text = parts[0].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
    try:
        obj = json.loads(text)
    except Exception:
        return None
    segs = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segs, list):
        return None
    out = [""] * expected_n
    for item in segs:
        if not isinstance(item, dict):
            continue
        idx = int(item.get("segment_index", -1))
        if 0 <= idx < expected_n:
            out[idx] = str(item.get("prompt", "")).strip()
    if any(not x for x in out):
        return None
    return out


class QwenVLCaptioner:
    def __init__(self, model_path: str, device: str, dtype: str, image_resize: Tuple[int, int]):
        import torch
        from PIL import Image
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch = torch
        self.Image = Image
        self.device = device
        self.image_resize = tuple(image_resize)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = str(getattr(cfg, "model_type", "") or "").lower()
        arch = str((getattr(cfg, "architectures", None) or [""])[0])
        is_qwen3 = ("qwen3" in model_type and "vl" in model_type) or ("Qwen3" in arch and "VL" in arch)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if is_qwen3:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch_dtype, trust_remote_code=True
            ).eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch_dtype, trust_remote_code=True
            ).eval()
        self.model.to(self.device)

    def generate(self, instruction: str, jpeg_bytes_list: List[bytes], max_new_tokens: int) -> str:
        h, w = self.image_resize
        imgs = [self.Image.open(BytesIO(b)).convert("RGB").resize((w, h)) for b in jpeg_bytes_list]
        messages = [{"role": "user", "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": instruction}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def assign_prompts_visual_batch(
    segments: List[Dict[str, Any]],
    video_path: str,
    optional_long_prompt: str,
    captioner: QwenVLCaptioner,
    frames_per_segment: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    flat_jpegs: List[bytes] = []
    per_seg_jpegs: List[List[bytes]] = []
    lines: List[str] = []
    cursor = 0
    for i, seg in enumerate(segments):
        ts = _pick_rep_times(float(seg["start_sec"]), float(seg["end_sec"]), int(frames_per_segment))
        jpegs = [_extract_jpeg(video_path, t) for t in ts]
        per_seg_jpegs.append(jpegs)
        lo, hi = cursor, cursor + len(jpegs) - 1
        lines.append(f"- Segment {i+1}/{len(segments)} -> images [{lo}..{hi}], time [{seg['start_sec']:.3f}s, {seg['end_sec']:.3f}s)")
        cursor += len(jpegs)
        flat_jpegs.extend(jpegs)

    instruction = (
        f"You are writing prompts for {len(segments)} temporal segments of one continuous video.\n"
        "Images are ordered by time and grouped by segment.\n"
        + "\n".join(lines)
        + "\n\nOptional long prompt hint:\n"
        + (optional_long_prompt.strip() or "(none)")
        + "\n\nReturn STRICT JSON only:\n"
        + '{"segments":[{"segment_index":0,"prompt":"..."},{"segment_index":1,"prompt":"..."}]}\n'
        + "Constraints: concise English (1-2 sentences), prompts must be different, keep identity continuity.\n"
    )

    raw = captioner.generate(instruction=instruction, jpeg_bytes_list=flat_jpegs, max_new_tokens=int(max_new_tokens))
    parsed = _parse_segments_json(raw, expected_n=len(segments))
    if parsed is None:
        for i, seg in enumerate(segments):
            txt = captioner.generate(
                instruction="Describe only this segment in concise English (1-2 sentences).",
                jpeg_bytes_list=per_seg_jpegs[i],
                max_new_tokens=min(256, int(max_new_tokens)),
            )
            seg["prompt"] = txt.strip()
        return segments

    for seg, p in zip(segments, parsed):
        seg["prompt"] = p
    return segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chunk-aligned metadata with TransNetV2 + Qwen3-VL.")
    p.add_argument("--videos_dir", required=True)
    p.add_argument("--out_json", required=True)
    p.add_argument("--long_prompts_jsonl", default=None)

    p.add_argument("--seg_method", choices=["scenedetect", "transnetv2"], default="scenedetect")
    p.add_argument("--prompt_mode", choices=["qwen_vl"], default="qwen_vl")
    p.add_argument("--qwen_vl_prompt_source", choices=["visual_batch"], default="visual_batch")

    p.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    p.add_argument("--t_downsample", type=int, default=4)
    p.add_argument("--min_len_chunks", type=int, default=1)

    p.add_argument("--transnetv2_threshold", type=float, default=0.5)
    p.add_argument("--transnetv2_device", type=str, default="auto")

    p.add_argument("--qwen_vl_model_path", type=str, default=DEFAULT_QWEN_VL_MODEL_PATH)
    p.add_argument("--qwen_vl_device", type=str, default="cuda")
    p.add_argument("--qwen_vl_dtype", type=str, default="bfloat16")
    p.add_argument("--qwen_vl_image_resize", type=int, nargs=2, default=[256, 448])
    p.add_argument("--qwen_vl_frames_per_segment", type=int, default=2)
    p.add_argument("--qwen_vl_batch_max_new_tokens", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    if not videos_dir.exists():
        raise SystemExit(f"videos_dir not found: {videos_dir}")

    video_paths = sorted([p for p in videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".webm"}])
    if not video_paths:
        raise SystemExit(f"No videos found in: {videos_dir}")

    long_prompt_map = load_long_prompts_jsonl(args.long_prompts_jsonl)
    chf = chunk_frames(int(args.num_latent_frames_per_chunk), int(args.t_downsample))
    captioner = QwenVLCaptioner(
        model_path=str(args.qwen_vl_model_path),
        device=str(args.qwen_vl_device),
        dtype=str(args.qwen_vl_dtype),
        image_resize=(int(args.qwen_vl_image_resize[0]), int(args.qwen_vl_image_resize[1])),
    )

    records: List[Dict[str, Any]] = []
    for vp in video_paths:
        info = read_video_info(str(vp))
        seg_sec = detect_scenes_transnetv2(
            str(vp),
            threshold=float(args.transnetv2_threshold),
            device=str(args.transnetv2_device),
            duration_sec=float(info.duration_sec),
        )
        segments = align_segments_to_chunks(
            seg_sec,
            fps=float(info.fps),
            num_frames=max(1, int(info.num_frames)),
            ch_frames=int(chf),
            min_len_chunks=int(args.min_len_chunks),
        )
        segments = assign_prompts_visual_batch(
            segments=segments,
            video_path=str(vp),
            optional_long_prompt=str(long_prompt_map.get(vp.stem, "")),
            captioner=captioner,
            frames_per_segment=int(args.qwen_vl_frames_per_segment),
            max_new_tokens=int(args.qwen_vl_batch_max_new_tokens),
        )
        records.append(
            {
                "id": vp.stem,
                "path": f"videos/{vp.name}",
                "fps": float(info.fps),
                "num_frames": int(info.num_frames),
                "resolution": {"height": int(info.height), "width": int(info.width)},
                "chunking": {
                    "num_latent_frames_per_chunk": int(args.num_latent_frames_per_chunk),
                    "t_downsample": int(args.t_downsample),
                    "chunk_frames": int(chf),
                },
                "segments": segments,
                "notes": {
                    "segmentation_method": str(args.seg_method),
                    "segmentation_used": "transnetv2",
                    "prompt_mode": str(args.prompt_mode),
                    "qwen_vl_prompt_source": str(args.qwen_vl_prompt_source),
                },
            }
        )
        print(f"[OK] {vp.stem}: {len(segments)} segments")

    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {len(records)} records -> {out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build chunk-aligned metadata for Selector_VLM (minimal edition).

This simplified script keeps ONLY:
1) Scene segmentation with TransNetV2.
2) Prompt generation with local Qwen3-VL (visual_batch mode).

Quick run:
  conda activate helios
  python /root/autodl-tmp/Helios/example_memory_selector/Selector_VLM/tools/offload_data/build_metadata.py \
    --videos_dir /root/autodl-tmp/Helios/example_memory_selector/seedance/video_light_change/videos \
    --out_json /root/autodl-tmp/Helios/example_memory_selector/seedance/video_light_change/metadata.json \
    --seg_method scenedetect \
    --prompt_mode qwen_vl \
    --qwen_vl_prompt_source visual_batch \
    --qwen_vl_model_path /root/autodl-fs/Qwen3-VL-8B-Instruct \
    --qwen_vl_batch_max_new_tokens 1024
"""

# from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

DEFAULT_QWEN_VL_MODEL_PATH = "/root/autodl-fs/Qwen3-VL-8B-Instruct"


@dataclass
class VideoInfo:
    fps: float
    num_frames: int
    height: int
    width: int

    @property
    def duration_sec(self) -> float:
        if self.fps <= 1e-6:
            return 0.0
        return float(self.num_frames) / float(self.fps)


def read_video_info(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    if fps <= 1e-6:
        fps = 24.0

    return VideoInfo(fps=fps, num_frames=num_frames, height=height, width=width)


def chunk_frames(num_latent_frames_per_chunk: int, t_downsample: int) -> int:
    return int((num_latent_frames_per_chunk - 1) * t_downsample + 1)


def _clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def sec_to_chunk_floor(t_sec: float, fps: float, ch_frames: int) -> int:
    return int(math.floor((t_sec * fps) / float(ch_frames)))


def sec_to_chunk_ceil(t_sec: float, fps: float, ch_frames: int) -> int:
    return int(math.ceil((t_sec * fps) / float(ch_frames)))


def ensure_monotonic_segments(segments: List[Tuple[float, float]], duration: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    last_end = 0.0
    for s, e in segments:
        s = max(0.0, min(float(s), duration))
        e = max(0.0, min(float(e), duration))
        if e <= s:
            continue
        s = max(s, last_end)
        if e <= s:
            continue
        out.append((s, e))
        last_end = e

    if not out:
        return [(0.0, duration)]
    if out[-1][1] < duration:
        out[-1] = (out[-1][0], duration)
    return out


def detect_scenes_transnetv2(
    video_path: str,
    *,
    threshold: float,
    device: str,
    duration_sec: float,
) -> List[Tuple[float, float]]:
    try:
        from transnetv2_pytorch import TransNetV2  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            "Missing dependency transnetv2-pytorch. Install with: pip install transnetv2-pytorch"
        ) from e

    model = TransNetV2(device=device)
    scenes = model.detect_scenes(str(video_path), threshold=float(threshold))
    if not scenes:
        return [(0.0, duration_sec)]

    segs: List[Tuple[float, float]] = []
    for sc in scenes:
        try:
            s = float(sc.get("start_time", 0.0))
            e = float(sc.get("end_time", 0.0))
        except Exception:
            continue
        if e > s:
            segs.append((s, e))

    if not segs:
        return [(0.0, duration_sec)]
    return ensure_monotonic_segments(segs, duration_sec)


def align_segments_to_chunks(
    segments_sec: List[Tuple[float, float]],
    *,
    fps: float,
    num_frames: int,
    ch_frames: int,
    min_len_chunks: int = 1,
) -> List[Dict[str, Any]]:
    duration = float(num_frames) / float(fps)
    segs = ensure_monotonic_segments(segments_sec, duration)
    n_chunks = max(1, int(math.ceil(num_frames / float(ch_frames))))

    out: List[Dict[str, Any]] = []
    for s, e in segs:
        s_chunk = _clamp(sec_to_chunk_floor(s, fps, ch_frames), 0, n_chunks - 1)
        e_chunk = _clamp(sec_to_chunk_ceil(e, fps, ch_frames), s_chunk + 1, n_chunks)
        if (e_chunk - s_chunk) < int(min_len_chunks):
            continue
        out.append(
            {
                "start_sec": float(s),
                "end_sec": float(e),
                "start_chunk": int(s_chunk),
                "end_chunk": int(e_chunk),
                "prompt": "",
            }
        )

    if not out:
        out = [
            {
                "start_sec": 0.0,
                "end_sec": duration,
                "start_chunk": 0,
                "end_chunk": n_chunks,
                "prompt": "",
            }
        ]
    return out


def load_long_prompts_jsonl(path: str) -> Dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    out: Dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        vid = str(obj.get("id", obj.get("sample_id", ""))).strip()
        pr = str(obj.get("prompt", "")).strip()
        if vid:
            out[vid] = pr
    return out


def _pick_rep_times(start_sec: float, end_sec: float, n: int) -> List[float]:
    n = max(1, int(n))
    if end_sec <= start_sec:
        return [max(0.0, float(start_sec))]
    if n == 1:
        return [0.5 * (start_sec + end_sec)]
    span = end_sec - start_sec
    return [start_sec + span * (i + 1) / (n + 1) for i in range(n)]


def _extract_jpeg_at_time(video_path: str, t_sec: float) -> bytes:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t_sec)) * 1000.0)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to extract frame at t={t_sec:.3f}s from {video_path}")
    ok2, enc = cv2.imencode(".jpg", frame)
    if not ok2:
        raise RuntimeError(f"Failed to JPEG-encode frame at t={t_sec:.3f}s")
    return bytes(enc.tobytes())


def _parse_json_prompts(text: str, expected_n: int) -> Optional[List[str]]:
    text = text.strip()
    if not text:
        return None
    # strip markdown fence
    if text.startswith("```"):
        parts = [x for x in text.split("```") if x.strip()]
        if parts:
            text = parts[0].strip()
            if text.lower().startswith("json"):
                text = text[4:].strip()
    try:
        obj = json.loads(text)
    except Exception:
        return None

    segs = obj.get("segments") if isinstance(obj, dict) else None
    if not isinstance(segs, list):
        return None

    out = [""] * expected_n
    for it in segs:
        if not isinstance(it, dict):
            continue
        idx = int(it.get("segment_index", -1))
        if 0 <= idx < expected_n:
            out[idx] = str(it.get("prompt", "")).strip()
    if any(not x for x in out):
        return None
    return out


class QwenVLCaptioner:
    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        dtype: str,
        image_resize: Tuple[int, int],
    ) -> None:
        import torch
        from PIL import Image
        from transformers import (
            AutoConfig,
            AutoModelForImageTextToText,
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        self._torch = torch
        self._Image = Image
        self.device = device
        self.image_resize = tuple(image_resize)

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        torch_dtype = dtype_map.get(dtype.lower(), torch.bfloat16)

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        model_type = str(getattr(cfg, "model_type", "") or "").lower()
        arch_list = getattr(cfg, "architectures", None) or []
        arch0 = str(arch_list[0]) if arch_list else ""
        is_qwen3 = ("qwen3" in model_type and "vl" in model_type) or ("Qwen3" in arch0 and "VL" in arch0)

        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        if is_qwen3:
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_path, torch_dtype=torch_dtype, trust_remote_code=True
            ).eval()
        else:
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_path, torch_dtype=torch_dtype, trust_remote_code=True
            ).eval()
        self.model.to(self.device)

    def _decode_images(self, jpeg_bytes_list: List[bytes]) -> List[Any]:
        imgs = []
        h, w = self.image_resize
        for b in jpeg_bytes_list:
            img = self._Image.open(BytesIO(b)).convert("RGB")
            imgs.append(img.resize((w, h)))
        return imgs

    def generate(self, *, instruction: str, jpeg_bytes_list: List[bytes], max_new_tokens: int) -> str:
        imgs = self._decode_images(jpeg_bytes_list)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im} for im in imgs]
                + [{"type": "text", "text": str(instruction)}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self._torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=int(max_new_tokens))
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        return self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def assign_prompts_visual_batch(
    *,
    segments: List[Dict[str, Any]],
    video_path: str,
    optional_long_prompt: str,
    captioner: QwenVLCaptioner,
    frames_per_segment: int,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    all_jpegs: List[bytes] = []
    image_map_lines: List[str] = []
    cursor = 0

    per_segment_jpegs: List[List[bytes]] = []
    for i, seg in enumerate(segments):
        ts = _pick_rep_times(float(seg["start_sec"]), float(seg["end_sec"]), int(frames_per_segment))
        jpegs = [_extract_jpeg_at_time(video_path, t) for t in ts]
        per_segment_jpegs.append(jpegs)
        lo = cursor
        hi = cursor + len(jpegs) - 1
        image_map_lines.append(
            f"- Segment {i+1}/{len(segments)}: images [{lo}..{hi}], time [{seg['start_sec']:.3f}s, {seg['end_sec']:.3f}s)"
        )
        cursor += len(jpegs)
        all_jpegs.extend(jpegs)

    instruction = (
        "You are writing segment-level prompts for video generation.\n"
        f"The clip is split into {len(segments)} temporal segments.\n"
        "Images are in chronological order, grouped by segment.\n\n"
        + "\n".join(image_map_lines)
        + "\n\n"
        "Optional long prompt hint (may be empty):\n"
        + (optional_long_prompt.strip() or "(none)")
        + "\n\n"
        "Return STRICT JSON only:\n"
        '{"segments":[{"segment_index":0,"prompt":"..."},{"segment_index":1,"prompt":"..."}]}\n'
        "Constraints:\n"
        "- prompt must be concise English (1-2 sentences).\n"
        "- each segment prompt must be different.\n"
        "- keep identity continuity when the same person appears.\n"
        f"- segment_index must cover 0..{len(segments)-1} in order.\n"
    )

    raw = captioner.generate(instruction=instruction, jpeg_bytes_list=all_jpegs, max_new_tokens=max_new_tokens)
    parsed = _parse_json_prompts(raw, expected_n=len(segments))

    if parsed is None:
        # fallback per-segment
        for i, seg in enumerate(segments):
            local_instruction = (
                "Describe only this segment in concise English (1-2 sentences). "
                "If same person appears, mention continuity like 'The same woman/man ...'."
            )
            txt = captioner.generate(
                instruction=local_instruction,
                jpeg_bytes_list=per_segment_jpegs[i],
                max_new_tokens=min(256, max_new_tokens),
            )
            seg["prompt"] = txt.strip()
        return segments

    for seg, p in zip(segments, parsed):
        seg["prompt"] = p
    return segments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chunk-aligned metadata from generated videos.")
    p.add_argument("--videos_dir", required=True, help="Directory containing *.mp4 videos.")
    p.add_argument("--out_json", required=True, help="Output metadata JSON path.")
    p.add_argument("--long_prompts_jsonl", default=None, help='Optional JSONL: {"id"/"sample_id","prompt"}')

    # Keep only requested pipeline modes.
    p.add_argument("--seg_method", choices=["scenedetect", "transnetv2"], default="scenedetect")
    p.add_argument("--prompt_mode", choices=["qwen_vl"], default="qwen_vl")
    p.add_argument("--qwen_vl_prompt_source", choices=["visual_batch"], default="visual_batch")

    p.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    p.add_argument("--t_downsample", type=int, default=4)
    p.add_argument("--min_len_chunks", type=int, default=1)

    p.add_argument("--transnetv2_threshold", type=float, default=0.5)
    p.add_argument("--transnetv2_device", type=str, default="auto")

    p.add_argument("--qwen_vl_model_path", type=str, default=DEFAULT_QWEN_VL_MODEL_PATH)
    p.add_argument("--qwen_vl_device", type=str, default="cuda")
    p.add_argument("--qwen_vl_dtype", type=str, default="bfloat16")
    p.add_argument("--qwen_vl_image_resize", type=int, nargs=2, default=[256, 448], help="H W")
    p.add_argument("--qwen_vl_frames_per_segment", type=int, default=2)
    p.add_argument("--qwen_vl_batch_max_new_tokens", type=int, default=1024)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    videos_dir = Path(args.videos_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    if not videos_dir.exists():
        raise SystemExit(f"videos_dir not found: {videos_dir}")

    video_paths = sorted([p for p in videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".webm"}])
    if not video_paths:
        raise SystemExit(f"No videos found in: {videos_dir}")

    long_prompt_map: Dict[str, str] = {}
    if args.long_prompts_jsonl:
        long_prompt_map = load_long_prompts_jsonl(args.long_prompts_jsonl)

    captioner = QwenVLCaptioner(
        model_path=str(args.qwen_vl_model_path),
        device=str(args.qwen_vl_device),
        dtype=str(args.qwen_vl_dtype),
        image_resize=(int(args.qwen_vl_image_resize[0]), int(args.qwen_vl_image_resize[1])),
    )

    ch_frames = chunk_frames(int(args.num_latent_frames_per_chunk), int(args.t_downsample))
    records: List[Dict[str, Any]] = []

    for vp in video_paths:
        vid = vp.stem
        info = read_video_info(str(vp))
        segments_sec = detect_scenes_transnetv2(
            str(vp),
            threshold=float(args.transnetv2_threshold),
            device=str(args.transnetv2_device),
            duration_sec=float(info.duration_sec),
        )
        segments = align_segments_to_chunks(
            segments_sec,
            fps=float(info.fps),
            num_frames=max(int(info.num_frames), 1),
            ch_frames=int(ch_frames),
            min_len_chunks=int(args.min_len_chunks),
        )
        segments = assign_prompts_visual_batch(
            segments=segments,
            video_path=str(vp),
            optional_long_prompt=str(long_prompt_map.get(vid, "")),
            captioner=captioner,
            frames_per_segment=int(args.qwen_vl_frames_per_segment),
            max_new_tokens=int(args.qwen_vl_batch_max_new_tokens),
        )

        records.append(
            {
                "id": vid,
                "path": f"videos/{vp.name}",
                "fps": float(info.fps),
                "num_frames": int(info.num_frames),
                "resolution": {"height": int(info.height), "width": int(info.width)},
                "chunking": {
                    "num_latent_frames_per_chunk": int(args.num_latent_frames_per_chunk),
                    "t_downsample": int(args.t_downsample),
                    "chunk_frames": int(ch_frames),
                },
                "segments": segments,
                "notes": {
                    "segmentation_method": str(args.seg_method),
                    "segmentation_used": "transnetv2",
                    "prompt_mode": str(args.prompt_mode),
                    "qwen_vl_prompt_source": str(args.qwen_vl_prompt_source),
                    "has_long_prompt": bool(long_prompt_map.get(vid, "").strip()),
                },
            }
        )
        print(f"[OK] {vid}: {len(segments)} segments")

    out_json.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[DONE] wrote {len(records)} records -> {out_json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build chunk-aligned metadata from closed-source generated videos.

Goal:
- Input: videos/*.mp4 (+ optional long prompt per video)
- Output: metadata.json (list of objects) with:
    - fps / num_frames / resolution
    - chunking (Helios-side definition)
    - segments: [{prompt, start_chunk, end_chunk, start_sec, end_sec}]

Segmentation:
- Preferred: TransNetV2 (PyTorch; `pip install transnetv2-pytorch`, weights ship with the package)
- Fallback: simple content-change detection via OpenCV HSV histogram diffs

Prompt:
- Optional: provide a JSONL mapping id->long_prompt. We will split it into segment prompts.

Download Qwen3-VL-8B-Instruct from ModelScope (recommended on CN mirrors):
  conda activate helios
  pip install -U modelscope
  python -c "from modelscope import snapshot_download; snapshot_download(
    'Qwen/Qwen3-VL-8B-Instruct', local_dir='/root/autodl-fs/Qwen3-VL-8B-Instruct'
  )"

Shot boundary: TransNetV2 may return a single shot on many AIGC clips; metadata still valid (one segment covering the whole video).

conda activate helios
python /root/autodl-tmp/Selector_VLM/tools/offload_data/build_metadata.py \
  --videos_dir /root/autodl-tmp/seedance/video_light_change/videos \
  --out_json /root/autodl-tmp/seedance/video_light_change/metadata.json \
  --seg_method scenedetect \
  --prompt_mode qwen_vl \
  --qwen_vl_prompt_source visual_batch \
  --qwen_vl_model_path /root/autodl-fs/Qwen3-VL-8B-Instruct \
  --qwen_vl_batch_max_new_tokens 1024

Example (enable LLM semantic prompt split with local Qwen2.5-VL as text-only LLM):
python /root/autodl-tmp/Selector_VLM/tools/offload_data/build_metadata.py \
  --videos_dir /root/autodl-tmp/seedance/video_light_change/videos \
  --out_json /root/autodl-tmp/seedance/video_light_change/metadata_qwenvl_ppl7.json \
  --long_prompts_jsonl /root/autodl-tmp/seedance/video_light_change/manifest.jsonl \
  --seg_method scenedetect \
  --prompt_mode qwen_vl \
  --prompt_split_method llm \
  --prompt_split_llm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
  --prompt_split_llm_device cuda \
  --prompt_split_llm_dtype bfloat16 \
  --prompt_split_llm_max_new_tokens 256 \
  --qwen_vl_model_path /root/autodl-fs/Qwen3-VL-8B-Instruct \
  --qwen_vl_frames_per_segment 2

Example (TransNetV2 + one Qwen-VL call for all segments, coherent prompts; long_prompts_jsonl optional):
python /root/autodl-tmp/Selector_VLM/tools/offload_data/build_metadata.py \
  --videos_dir /path/to/videos \
  --out_json /path/to/metadata.json \
  --seg_method scenedetect \
  --prompt_mode qwen_vl \
  --qwen_vl_prompt_source visual_batch \
  --qwen_vl_batch_max_new_tokens 1024 \
  --qwen_vl_model_path /root/autodl-fs/Qwen3-VL-8B-Instruct \
  --qwen_vl_frames_per_segment 2

"""

# from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from io import BytesIO

try:
    import cv2  # type: ignore
except Exception as e:  # noqa: BLE001
    raise RuntimeError(
        "Missing dependency: OpenCV (cv2).\n"
        "Install one of:\n"
        "  - pip install opencv-python\n"
        "  - pip install opencv-python-headless\n"
        "Note: this script uses OpenCV for video IO and/or frame processing."
    ) from e
import numpy as np

# Default local path for Qwen3-VL-8B (create by downloading; see module docstring).
DEFAULT_QWEN_VL_MODEL_PATH = "/root/autodl-fs/Qwen3-VL-8B-Instruct"

_QWEN_VL_WEIGHTS_CACHE: Dict[Tuple[str, str, str], Tuple[Any, Any]] = {}


def load_qwen_vl_model_and_processor(
    model_path: str,
    *,
    torch_dtype: Any,
    device: str,
    trust_remote_code: bool = True,
) -> Tuple[Any, Any]:
    """
    Load Qwen2.5-VL or Qwen3-VL from disk or Hub.

    Qwen3-VL uses Transformers' AutoModelForImageTextToText; Qwen2.5-VL keeps
    Qwen2_5_VLForConditionalGeneration. Detection is by config.model_type / architectures.

    Callers that share the same (path, dtype, device) reuse one loaded model (e.g. prompt-split LLM + Qwen-VL path).
    """
    mp = Path(model_path)
    cache_path = str(mp.resolve()) if mp.exists() else str(model_path)
    cache_key = (cache_path, str(torch_dtype), str(device))
    hit = _QWEN_VL_WEIGHTS_CACHE.get(cache_key)
    if hit is not None:
        return hit

    from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
    from transformers import Qwen2_5_VLForConditionalGeneration

    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    model_type = str(getattr(cfg, "model_type", "") or "").lower()
    arch_list = getattr(cfg, "architectures", None) or []
    arch0 = str(arch_list[0]) if arch_list else ""

    is_qwen3 = "qwen3" in model_type and "vl" in model_type
    if not is_qwen3 and arch0:
        is_qwen3 = "Qwen3" in arch0 and "VL" in arch0

    if is_qwen3:
        print(f"[Qwen-VL] loading Qwen3-VL: model_type={model_type!r} arch={arch0!r} path={model_path}")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).eval()
    else:
        print(f"[Qwen-VL] loading Qwen2.5-VL: model_type={model_type!r} arch={arch0!r} path={model_path}")
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        ).eval()
    model.to(device)
    _QWEN_VL_WEIGHTS_CACHE[cache_key] = (model, processor)
    return model, processor


@dataclass
class VideoInfo:
    fps: float
    num_frames: int
    height: int
    width: int

    @property
    def duration_sec(self) -> float:
        if self.fps <= 0:
            return 0.0
        return float(self.num_frames) / float(self.fps)


def read_video_info(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()

    # Some encodes are VFR-ish; cv2 may return 0 fps. Fallback: assume 24.
    if fps <= 1e-6:
        fps = 24.0

    return VideoInfo(fps=fps, num_frames=num_frames, height=height, width=width)


# Text-only prompt splitter cache (LLM semantic split).
_QWEN_TEXT_SPLITTER_CACHE: Dict[Tuple[str, str, str, int], "_QwenTextSplitter"] = {}

# TransNetV2 (shot boundary) — one model instance per device string.
_TRANSNETV2_MODEL_CACHE: Dict[str, Any] = {}


def chunk_frames(num_latent_frames_per_chunk: int, t_downsample: int) -> int:
    # Helios convention: visible frames per chunk = (latent_window_size - 1) * t_downsample + 1
    return int((num_latent_frames_per_chunk - 1) * t_downsample + 1)


def sec_to_chunk(t_sec: float, fps: float, ch_frames: int, *, mode: str) -> int:
    x = (t_sec * fps) / float(ch_frames)
    if mode == "floor":
        return int(math.floor(x))
    if mode == "ceil":
        return int(math.ceil(x))
    raise ValueError(f"Unknown mode={mode}")


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def _clamp_segments_count(n: int, *, min_segments: int, max_segments: int) -> int:
    min_segments = int(max(1, min_segments))
    max_segments = int(max(min_segments, max_segments))
    return clamp_int(int(n), min_segments, max_segments)


def ensure_monotonic_segments(segments: List[Tuple[float, float]], duration_sec: float) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    last_end = 0.0
    for s, e in segments:
        s = float(max(0.0, min(s, duration_sec)))
        e = float(max(0.0, min(e, duration_sec)))
        if e <= s:
            continue
        s = max(s, last_end)
        if e <= s:
            continue
        out.append((s, e))
        last_end = e
    if not out:
        return [(0.0, duration_sec)]
    # Ensure covers tail
    if out[-1][1] < duration_sec:
        out[-1] = (out[-1][0], duration_sec)
    return out


def _get_transnetv2_model(device: str) -> Any:
    """Lazy-load TransNetV2 (weights bundled in transnetv2-pytorch)."""
    key = str(device or "auto")
    if key not in _TRANSNETV2_MODEL_CACHE:
        try:
            from transnetv2_pytorch import TransNetV2  # type: ignore
        except Exception as e:  # noqa: BLE001
            raise ImportError(
                "transnetv2-pytorch is not installed. Install with: pip install transnetv2-pytorch"
            ) from e
        _TRANSNETV2_MODEL_CACHE[key] = TransNetV2(device=key)
    return _TRANSNETV2_MODEL_CACHE[key]


def try_detect_scenes_transnetv2(
    video_path: str,
    *,
    threshold: float = 0.5,
    device: str = "auto",
    duration_sec_fallback: Optional[float] = None,
    debug: bool = False,
) -> Optional[List[Tuple[float, float]]]:
    """
    Returns list of (start_sec, end_sec) using TransNetV2 shot-boundary model.
    """
    try:
        model = _get_transnetv2_model(device)
        scenes = model.detect_scenes(str(video_path), threshold=float(threshold))
    except Exception as e:  # noqa: BLE001
        if debug:
            print(f"[WARN] TransNetV2 failed for {video_path}: {e}")
        return None

    if not scenes:
        dur = float(duration_sec_fallback or 0.0)
        if dur <= 0:
            return None
        return [(0.0, dur)]

    duration = float(duration_sec_fallback or 0.0)
    segments: List[Tuple[float, float]] = []
    for sc in scenes:
        try:
            s = float(sc.get("start_time", 0.0))
            e = float(sc.get("end_time", 0.0))
        except Exception:
            continue
        if e <= s:
            continue
        segments.append((s, e))

    if not segments:
        if duration > 0:
            return [(0.0, duration)]
        return None

    if duration > 0:
        segments = ensure_monotonic_segments(segments, duration)
    return segments


def _frame_hist_hsv(frame_bgr: np.ndarray, hist_bins: int = 16) -> np.ndarray:
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    # Use H/S/V jointly: lighting changes are primarily reflected in V (value).
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        None,
        [hist_bins, hist_bins, hist_bins],
        [0, 180, 0, 256, 0, 256],
    )
    hist = cv2.normalize(hist, None).flatten()
    return hist


def detect_scenes_opencv(
    video_path: str,
    fps: float,
    num_frames: int,
    *,
    sample_every_n_frames: int = 6,
    hist_bins: int = 16,
    diff_threshold: float = 0.55,
    min_scene_sec: float = 1.0,
) -> List[Tuple[float, float]]:
    """
    Lightweight fallback:
    - sample frames every N
    - compute HSV hist
    - boundary when cosine distance > threshold
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return [(0.0, float(num_frames) / float(fps))]

    frame_indices = list(range(0, max(num_frames, 1), max(1, sample_every_n_frames)))
    if len(frame_indices) <= 1:
        cap.release()
        return [(0.0, float(num_frames) / float(fps))]

    hists: List[np.ndarray] = []
    times: List[float] = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        hists.append(_frame_hist_hsv(frame, hist_bins=hist_bins))
        times.append(float(fi) / float(fps))
    cap.release()

    if len(hists) <= 1:
        return [(0.0, float(num_frames) / float(fps))]

    def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(1.0 - (float(np.dot(a, b)) / denom))

    diffs = [cos_dist(hists[i], hists[i - 1]) for i in range(1, len(hists))]
    boundary_times = [times[i] for i, d in enumerate([0.0] + diffs) if d >= diff_threshold]

    # Always include start
    boundary_times = [0.0] + [t for t in boundary_times if t > 0.0]
    boundary_times = sorted(set(boundary_times))

    duration = float(num_frames) / float(fps)
    boundary_times = [t for t in boundary_times if t < duration]
    boundary_times.append(duration)
    boundary_times = sorted(set(boundary_times))

    segments: List[Tuple[float, float]] = []
    for i in range(len(boundary_times) - 1):
        s, e = boundary_times[i], boundary_times[i + 1]
        if e - s >= min_scene_sec:
            segments.append((s, e))

    if not segments:
        segments = [(0.0, duration)]
    return ensure_monotonic_segments(segments, duration)


def _peak_split_segments_from_video(
    video_path: str,
    *,
    fps: float,
    num_frames: int,
    ch_frames: int,
    n_parts: int,
    ffmpeg_scale: str = "256:-1",
    min_gap_chunks: int = 1,
) -> Optional[List[Tuple[float, float]]]:
    """
    When shot detection (e.g. TransNetV2) returns a single scene but prompt suggests multiple phases,
    find (n_parts-1) boundaries by picking peaks of visual change between chunk-mid frames.
    """
    n_parts = int(n_parts)
    if n_parts < 2 or fps <= 0 or ch_frames <= 0:
        return None

    total_chunks = int(math.ceil(float(max(num_frames, 1)) / float(ch_frames)))
    total_chunks = max(total_chunks, 1)
    if total_chunks < 2:
        return None

    # sample one representative frame per chunk: midpoint time
    times: List[float] = []
    hists: List[np.ndarray] = []
    for k in range(total_chunks):
        mid_frame = int(min(max(k * ch_frames + (ch_frames // 2), 0), max(num_frames - 1, 0)))
        t = float(mid_frame) / float(fps)
        try:
            jpeg = _ffmpeg_extract_jpeg(video_path, t, scale=str(ffmpeg_scale))
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                continue
            hists.append(_frame_hist_hsv(frame, hist_bins=16))
            times.append(t)
        except Exception:
            continue

    if len(hists) < 3:
        return None

    def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(1.0 - (float(np.dot(a, b)) / denom))

    diffs = [cos_dist(hists[i], hists[i - 1]) for i in range(1, len(hists))]
    # smooth with a small window to reduce noise
    w = 3
    smoothed: List[float] = []
    for i in range(len(diffs)):
        lo = max(0, i - w)
        hi = min(len(diffs), i + w + 1)
        smoothed.append(float(np.mean(diffs[lo:hi])))

    need = n_parts - 1
    min_gap = int(max(1, min_gap_chunks))

    # greedy peak picking with suppression by min_gap
    candidates = sorted(range(len(smoothed)), key=lambda i: smoothed[i], reverse=True)
    chosen: List[int] = []
    for idx in candidates:
        # boundary between chunk idx and idx+1 -> place at chunk (idx+1)
        b = idx + 1
        if b <= 0 or b >= len(times):
            continue
        if any(abs(b - c) < min_gap for c in chosen):
            continue
        chosen.append(b)
        if len(chosen) >= need:
            break

    if len(chosen) < need:
        return None

    chosen = sorted(set(chosen))
    # convert chunk boundary indices (in sampled list space) to seconds
    duration = float(num_frames) / float(fps)
    boundary_secs = [0.0]
    for b in chosen:
        boundary_secs.append(float(times[b]))
    boundary_secs.append(duration)
    boundary_secs = sorted(set([t for t in boundary_secs if 0.0 <= t <= duration]))
    if len(boundary_secs) < 3:
        return None

    segs: List[Tuple[float, float]] = []
    for i in range(len(boundary_secs) - 1):
        s, e = boundary_secs[i], boundary_secs[i + 1]
        if e > s:
            segs.append((float(s), float(e)))
    return ensure_monotonic_segments(segs, duration)


_SPLIT_PATTERNS = [
    r"\bThen\b",
    r"\bNext\b",
    r"\bAfter that\b",
    r"\bSuddenly\b",
    r"\bFinally\b",
    # temporal/causal transitions
    r"\bAs\s+(?:he|she|it|they)\b",
    r"\bWhen\s+(?:he|she|it|they)\b",
    r"\bMeanwhile\b",
    r"\bGradually\b",
    r"\bEventually\b",
    r"\bMoments?\s+later\b",
    # video prompt cues
    r"\bThe\s+(?:scene|camera|shot|light)\s+(?:shifts?|changes?|cuts?|moves?|transitions?)\b",
    r"[—–]\s*",  # em-dash / en-dash separators
    r"然后",
    r"接着",
    r"随后",
    r"之后",
    r"最后",
    r"突然",
    r"与此同时",
    # 中文增强
    r"此时",
    r"紧接着",
    r"当(?:他|她|它|他们|她们|它们)",
    r"渐渐地",
    r"慢慢地",
]


def _strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    # Common model outputs: ```json\n{...}\n``` or ```\n{...}\n```
    if t.startswith("```"):
        # Remove the first line (``` or ```json) and the trailing ```
        lines = t.splitlines()
        if len(lines) >= 2 and lines[0].lstrip().startswith("```"):
            # drop first fence line
            lines = lines[1:]
            # drop last fence line if present
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            t2 = "\n".join(lines).strip()
            return t2
    return t


def _parse_qwen_vl_batch_prompts_json(text: str, *, expected_n: int) -> Optional[List[str]]:
    """
    Parse batch visual caption output: JSON object with key 'segments' (preferred) or 'prompts',
    or a JSON list of objects with 'prompt' / string entries.
    Returns exactly expected_n strings when possible.
    """
    t = _strip_code_fences(text or "").strip()
    if not t:
        return None

    def _coerce_list(obj: Any) -> Optional[List[str]]:
        if obj is None:
            return None
        if isinstance(obj, list):
            out_ls: List[str] = []
            for item in obj:
                if isinstance(item, dict):
                    p = item.get("prompt", item.get("segment_prompt", ""))
                    out_ls.append(str(p).strip())
                else:
                    out_ls.append(str(item).strip())
            out_ls = [x for x in out_ls if x]
            return out_ls if out_ls else None
        return None

    try:
        root = json.loads(t)
    except Exception:
        return None

    payload: Optional[Any] = None
    if isinstance(root, dict):
        if isinstance(root.get("segments"), list):
            payload = root.get("segments")
        elif isinstance(root.get("prompts"), list):
            payload = root.get("prompts")
    elif isinstance(root, list):
        payload = root

    if isinstance(payload, list) and payload and isinstance(payload[0], dict):
        if any("segment_index" in x for x in payload if isinstance(x, dict)):
            try:
                payload = sorted(payload, key=lambda x: int(x.get("segment_index", 0)) if isinstance(x, dict) else 0)
            except Exception:
                pass

    parts = _coerce_list(payload)
    if not parts:
        return None

    if len(parts) == expected_n:
        return parts
    if len(parts) > expected_n:
        return parts[:expected_n]
    # Too few: pad by repeating last (better than empty for training pipelines).
    if parts:
        parts = parts + [parts[-1]] * (expected_n - len(parts))
        return parts[:expected_n]
    return None


def _normalize_prompt_for_dup_check(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _segment_prompts_have_duplicates(prompts: List[str]) -> bool:
    """True if any two segment prompts are identical after whitespace-normalization (case-insensitive)."""
    seen: set[str] = set()
    for p in prompts:
        key = _normalize_prompt_for_dup_check(str(p))
        if not key:
            continue
        if key in seen:
            return True
        seen.add(key)
    return False


def _extract_segment_prompt_from_text(text: str) -> Optional[str]:
    """
    Robustly extract segment_prompt from model output.
    Supports:
    - strict JSON object: {"segment_prompt": "...", ...}
    - partial/broken JSON containing a "segment_prompt": "..." field
    - plain text (returns None)
    """
    t = _strip_code_fences(text or "").strip()
    if not t:
        return None

    # 1) Strict JSON object.
    if t.startswith("{") and t.endswith("}"):
        try:
            obj = json.loads(t)
            if isinstance(obj, dict) and isinstance(obj.get("segment_prompt", None), str):
                s = str(obj["segment_prompt"]).strip()
                return s if s else None
        except Exception:
            pass

    # 2) Broken JSON: regex extract a JSON string value.
    #    This handles escaped quotes and backslashes inside the string.
    m = re.search(r'"segment_prompt"\s*:\s*"((?:\\.|[^"\\])*)"', t, flags=re.DOTALL)
    if m:
        raw = str(m.group(1))
        # Safely unescape like a JSON string.
        try:
            s = json.loads('"' + raw.replace('"', '\\"') + '"')
        except Exception:
            s = raw
        s = str(s).strip()
        return s if s else None

    return None


def _text_jaccard(a: str, b: str) -> float:
    def norm_tokens(x: str) -> List[str]:
        x = (x or "").lower()
        x = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", x)
        toks = [t for t in x.split() if t]
        return toks

    ta = set(norm_tokens(a))
    tb = set(norm_tokens(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return float(inter) / float(max(union, 1))


def _trim_prefix_overlap(prev: str, cur: str, *, min_overlap_chars: int = 24) -> str:
    """
    If cur starts with a suffix of prev (verbatim), trim that overlapping prefix from cur.
    This is a deterministic postprocess to enforce non-overlap even when the LLM repeats.
    """
    prev = str(prev or "")
    cur = str(cur or "")
    if not prev or not cur:
        return cur.strip()

    max_k = min(len(prev), len(cur))
    best = 0
    # Find the longest k such that prev[-k:] == cur[:k]
    for k in range(min(max_k, 512), min_overlap_chars - 1, -1):
        if prev[-k:] == cur[:k]:
            best = k
            break
    if best >= min_overlap_chars:
        return cur[best:].lstrip()
    return cur.strip()


def _dedupe_prompt_parts(parts: List[str]) -> List[str]:
    """
    Enforce non-overlap between adjacent parts by trimming exact prefix overlaps.
    """
    out: List[str] = []
    for p in parts:
        s = str(p or "").strip()
        if not s:
            continue
        if out:
            s = _trim_prefix_overlap(out[-1], s)
        if s:
            out.append(s)
    return out


def split_long_prompt_llm(
    long_prompt: str,
    *,
    max_parts: int = 5,
    model_path: str,
    device: str,
    dtype: str,
    max_new_tokens: int,
) -> Tuple[List[str], str]:
    """
    LLM semantic splitting (text-only).

    Reuse Qwen2.5-VL Instruct as a text-only chat model (no images) to split the prompt
    into semantic phases. Output should be a strict JSON list of strings.
    """
    p = (long_prompt or "").strip()
    if not p:
        return [], "empty"

    key = (str(model_path), str(device), str(dtype), int(max_new_tokens))
    splitter = _QWEN_TEXT_SPLITTER_CACHE.get(key)
    if splitter is None:
        splitter = _QwenTextSplitter(
            model_path=str(model_path),
            device=str(device),
            dtype=str(dtype),
            max_new_tokens=int(max_new_tokens),
        )
        _QWEN_TEXT_SPLITTER_CACHE[key] = splitter

    instruction = splitter.split_instruction.format(full_prompt=p, max_parts=int(max_parts))
    out_text = splitter.generate(instruction).strip()
    out_text = _strip_code_fences(out_text)

    try:
        obj = json.loads(out_text)
        parts: List[str] = []
        if isinstance(obj, list):
            parts = [str(x).strip() for x in obj if isinstance(x, str) and str(x).strip()]
        elif isinstance(obj, dict):
            v = obj.get("parts", None)
            if isinstance(v, list):
                parts = [str(x).strip() for x in v if isinstance(x, str) and str(x).strip()]
        parts = [x for x in parts if x]
        if parts:
            if len(parts) > int(max_parts):
                head = parts[: int(max_parts) - 1]
                tail = " ".join(parts[int(max_parts) - 1 :]).strip()
                parts = head + ([tail] if tail else [])
            return parts, "llm"
    except Exception:
        pass

    return [], "llm_parse_failed"


class _QwenTextSplitter:
    def __init__(
        self,
        *,
        model_path: str,
        device: str,
        dtype: str,
        max_new_tokens: int,
    ) -> None:
        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.torch = torch
        self.device = str(device)
        self.dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)
        self.max_new_tokens = int(max_new_tokens)

        print(f"[PromptSplitLLM] loading model from {model_path} ...")
        self.model, self.processor = load_qwen_vl_model_and_processor(
            str(model_path), torch_dtype=self.dtype, device=str(self.device)
        )

        self.split_instruction = (
            "You are given ONE long prompt describing an entire ~10s video.\n"
            "Task: split it into at most {max_parts} semantic phases that correspond to visual state changes "
            "(location change, lighting change, new action begins, subject enters/exits).\n"
            "\n"
            "HARD RULES (must follow exactly):\n"
            "1) Output MUST be STRICT JSON.\n"
            "2) Output MUST be a JSON array of strings.\n"
            "3) Each output string MUST be copied VERBATIM from FULL PROMPT (exact original wording).\n"
            "   - You may ONLY delete text; you MUST NOT paraphrase, summarize, translate, or rewrite.\n"
            "   - Do NOT introduce any new words not present in FULL PROMPT.\n"
            "   - Each string should be a contiguous span (substring) taken from FULL PROMPT.\n"
            "4) PRIORITY: Prefer splitting at LIGHTING/VISIBILITY transitions.\n"
            "   - Common pattern: bright/sunlit/daylight → dark/dim/shadow/tunnel/underpass/corridor → bright/sunlit/daylight.\n"
            "   - Use exact boundaries around clauses that mention lighting change or visibility loss/return.\n"
            "   - Keywords that often indicate a boundary (use ONLY if present in FULL PROMPT):\n"
            "     * bright/sunlit/afternoon light/overcast/midday light\n"
            "     * dark/dim/shadow/black/tunnel/underpass/corridor/lights flicker and die\n"
            "4) Non-overlap: do not include the same sentence/phrase in multiple parts.\n"
            "5) Coverage: concatenating all parts in order should reconstruct the FULL PROMPT content (up to deletions of minor filler).\n"
            "6) No extra keys, no markdown, no commentary.\n"
            "\n"
            "FULL PROMPT:\n{full_prompt}\n"
        )

    def generate(self, user_text: str) -> str:
        messages = [{"role": "user", "content": [{"type": "text", "text": str(user_text)}]}]
        chat = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[chat], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        decoded = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return str(decoded).strip()


def split_long_prompt_with_method(
    long_prompt: str,
    *,
    max_parts: int = 8,
    method: str = "auto",
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], str]:
    p = (long_prompt or "").strip()
    if not p:
        return [], "empty"

    method = str(method or "auto").strip().lower()
    if method not in {"auto", "regex", "llm"}:
        method = "auto"

    # Prefer (optional) LLM splitting if requested.
    if method in {"auto", "llm"}:
        if llm_cfg:
            parts_llm, used = split_long_prompt_llm(
                p,
                max_parts=min(int(max_parts), 8),
                model_path=str(llm_cfg.get("model_path")),
                device=str(llm_cfg.get("device", "cuda")),
                dtype=str(llm_cfg.get("dtype", "bfloat16")),
                max_new_tokens=int(llm_cfg.get("max_new_tokens", 256)),
            )
            if parts_llm:
                return parts_llm, used

    # Normalize separators to a token.
    sep_regex = "(" + "|".join(_SPLIT_PATTERNS) + ")"
    parts = re.split(sep_regex, p, flags=re.IGNORECASE)

    # re.split keeps separators; merge them back to the following clause.
    merged: List[str] = []
    buf = ""
    for token in parts:
        if not token:
            continue
        if re.fullmatch(sep_regex, token, flags=re.IGNORECASE):
            # start a new clause marker
            if buf.strip():
                merged.append(buf.strip().strip(".，,;；"))
            buf = token.strip() + " "
        else:
            buf += token
    if buf.strip():
        merged.append(buf.strip().strip(".，,;；"))

    # Fallback: if split failed, keep whole prompt
    if len(merged) <= 1:
        # Secondary fallback: sentence-based split.
        # Many long prompts describe multiple phases without explicit "Then/Next" markers.
        # We split into sentences and group them into a few segments.
        # Split on sentence-ending punctuation, allowing missing whitespace after it
        # (common in prompts like "... light.He passes ...").
        sentences = re.split(r"(?<=[\.\!\?\。\！\？])\s*", p)
        sentences = [s.strip() for s in sentences if s and s.strip()]
        if len(sentences) <= 1:
            return [p], "whole"

        # Heuristic: choose a small number of parts (2-4) based on sentence count.
        if len(sentences) >= 9:
            target_parts = 4
        elif len(sentences) >= 6:
            target_parts = 3
        else:
            target_parts = 2
        target_parts = min(target_parts, max_parts, len(sentences))

        # Group sentences roughly equally.
        per = int(math.ceil(len(sentences) / float(target_parts)))
        out = []
        for i in range(target_parts):
            chunk = sentences[i * per : (i + 1) * per]
            if not chunk:
                continue
            out.append(" ".join(chunk).strip().strip(".，,;；"))
        return (out if out else [p]), "sentence_group"

    # Truncate extremely many parts
    if len(merged) > max_parts:
        head = merged[: max_parts - 1]
        tail = " ".join(merged[max_parts - 1 :])
        return head + [tail], "marker_regex"
    return merged, "marker_regex"


def split_long_prompt(long_prompt: str, max_parts: int = 8) -> List[str]:
    parts, _ = split_long_prompt_with_method(long_prompt, max_parts=max_parts, method="auto")
    return parts


def align_segments_to_chunks(
    segments_sec: List[Tuple[float, float]],
    *,
    fps: float,
    num_frames: int,
    ch_frames: int,
    min_len_chunks: int = 1,
    snap_sec_to_chunk: bool = False,
) -> List[Dict[str, Any]]:
    duration = float(num_frames) / float(fps)
    segments_sec = ensure_monotonic_segments(segments_sec, duration)

    total_chunks = int(math.ceil(float(num_frames) / float(ch_frames))) if ch_frames > 0 else 1
    total_chunks = max(total_chunks, 1)

    def chunk_to_sec(c: int) -> float:
        return float(c * ch_frames) / float(fps) if fps > 0 else 0.0

    aligned: List[Dict[str, Any]] = []
    last_end_chunk = 0
    for s, e in segments_sec:
        sc = sec_to_chunk(s, fps, ch_frames, mode="floor")
        ec = sec_to_chunk(e, fps, ch_frames, mode="ceil")
        sc = clamp_int(sc, 0, total_chunks)
        ec = clamp_int(ec, 0, total_chunks)
        # enforce monotonic non-overlapping chunk ranges
        sc = max(sc, last_end_chunk)
        if ec <= sc:
            continue
        if (ec - sc) < min_len_chunks:
            continue

        if snap_sec_to_chunk:
            s2 = chunk_to_sec(int(sc))
            e2 = chunk_to_sec(int(ec))
            s = float(max(0.0, min(s2, duration)))
            e = float(max(0.0, min(e2, duration)))

        aligned.append(
            {
                "start_sec": float(s),
                "end_sec": float(e),
                "start_chunk": int(sc),
                "end_chunk": int(ec),
            }
        )
        last_end_chunk = int(ec)

    if not aligned:
        aligned = [
            {
                "start_sec": 0.0,
                "end_sec": duration,
                "start_chunk": 0,
                "end_chunk": total_chunks,
            }
        ]

    # Ensure last ends at total_chunks (and cover duration)
    aligned[-1]["end_chunk"] = max(int(aligned[-1]["end_chunk"]), total_chunks)
    aligned[-1]["end_sec"] = duration
    return aligned


def _segments_dict_to_sec_list(segments: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    for s in segments:
        try:
            a = float(s.get("start_sec", 0.0) or 0.0)
            b = float(s.get("end_sec", 0.0) or 0.0)
        except Exception:
            continue
        if b > a:
            out.append((a, b))
    return out


def enforce_final_segment_bounds(
    segments: List[Dict[str, Any]],
    *,
    min_segments: int,
    max_segments: int,
    video_path: str,
    fps: float,
    num_frames: int,
    ch_frames: int,
    peak_split_min_gap_chunks: int,
    snap_sec_to_chunk: bool,
    min_len_chunks: int,
) -> List[Dict[str, Any]]:
    """
    Enforce min/max *on the final chunk-aligned segments*.
    Chunk alignment and chunkwise postprocess may merge/drop segments, so we enforce again here.
    """
    min_segments = int(max(1, min_segments))
    max_segments = int(max(min_segments, max_segments))
    cur_n = len(segments)
    if min_segments <= cur_n <= max_segments:
        return segments

    segs_sec = _segments_dict_to_sec_list(segments)
    if not segs_sec:
        return segments

    if cur_n > max_segments:
        segs_sec = merge_segments_by_similarity(
            segs_sec,
            target_n=int(max_segments),
            video_path=str(video_path),
            fps=float(fps),
        )
    elif cur_n < min_segments:
        segs_floor = _peak_split_segments_from_video(
            str(video_path),
            fps=float(fps),
            num_frames=max(int(num_frames), 1),
            ch_frames=int(ch_frames),
            n_parts=int(min_segments),
            min_gap_chunks=int(peak_split_min_gap_chunks),
        )
        if segs_floor is not None and len(segs_floor) >= min_segments:
            segs_sec = segs_floor
        else:
            duration = float(max(int(num_frames), 1)) / float(fps) if float(fps) > 0 else 0.0
            if duration > 0 and min_segments >= 2:
                step = duration / float(min_segments)
                segs_sec = [(i * step, (i + 1) * step) for i in range(int(min_segments))]

    return align_segments_to_chunks(
        segs_sec,
        fps=float(fps),
        num_frames=max(int(num_frames), 1),
        ch_frames=int(ch_frames),
        min_len_chunks=int(min_len_chunks),
        snap_sec_to_chunk=bool(snap_sec_to_chunk),
    )


def merge_segments_by_similarity(
    segments_sec: List[Tuple[float, float]],
    *,
    target_n: int,
    video_path: str,
    fps: float,
    ffmpeg_scale: str = "256:-1",
) -> List[Tuple[float, float]]:
    """
    Merge visually most-similar adjacent segments until len == target_n.
    Similarity: HSV hist cosine distance on each segment's midpoint representative frame.
    """
    target_n = int(target_n)
    if target_n < 1:
        return segments_sec
    if len(segments_sec) <= target_n:
        return segments_sec

    segs: List[Tuple[float, float]] = [(float(s), float(e)) for s, e in segments_sec if float(e) > float(s)]
    if len(segs) <= target_n:
        return segs

    def rep_hist(seg: Tuple[float, float]) -> Optional[np.ndarray]:
        s, e = seg
        t = float((s + e) * 0.5)
        try:
            jpeg = _ffmpeg_extract_jpeg(str(video_path), t, scale=str(ffmpeg_scale))
            arr = np.frombuffer(jpeg, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                return None
            return _frame_hist_hsv(frame, hist_bins=16)
        except Exception:
            return None

    def cos_dist(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
        return float(1.0 - (float(np.dot(a, b)) / denom))

    # Cache hists per current segment index (rebuilt after merges).
    hists: List[Optional[np.ndarray]] = [rep_hist(s) for s in segs]

    while len(segs) > target_n and len(segs) >= 2:
        best_i = None
        best_d = None
        for i in range(len(segs) - 1):
            ha = hists[i]
            hb = hists[i + 1]
            if ha is None or hb is None:
                continue
            d = cos_dist(ha, hb)
            if best_d is None or d < best_d:
                best_d = d
                best_i = i

        if best_i is None:
            # If we can't compute similarity reliably, fallback to merging shortest segment.
            lens = [float(e - s) for s, e in segs]
            j = int(min(range(len(segs)), key=lambda k: lens[k]))
            best_i = max(0, min(j, len(segs) - 2))

        i = int(best_i)
        a = segs[i]
        b = segs[i + 1]
        merged = (float(a[0]), float(b[1]))
        segs[i : i + 2] = [merged]
        # rebuild affected hist entries locally
        hists[i : i + 2] = [rep_hist(merged)]

    return segs


def adaptive_segmentation(
    segments_sec: List[Tuple[float, float]],
    *,
    target_n: int,
    video_path: str,
    fps: float,
    num_frames: int,
    ch_frames: int,
    peak_split_min_gap_chunks: int = 1,
    ffmpeg_scale: str = "256:-1",
) -> List[Tuple[float, float]]:
    """
    Prompt-guided adaptive segmentation:
    - If M≈N: keep
    - If M>>N: merge by visual similarity to N
    - If M<<N: try peak-split to N, else keep original
    """
    target_n = int(target_n)
    if target_n < 2:
        return segments_sec

    M = len(segments_sec)
    if int(0.7 * target_n) <= M <= int(1.3 * target_n):
        return segments_sec

    if M > target_n:
        return merge_segments_by_similarity(
            segments_sec,
            target_n=target_n,
            video_path=str(video_path),
            fps=float(fps),
            ffmpeg_scale=str(ffmpeg_scale),
        )

    # M < target_n
    segs_peak = _peak_split_segments_from_video(
        str(video_path),
        fps=float(fps),
        num_frames=max(int(num_frames), 1),
        ch_frames=int(ch_frames),
        n_parts=int(target_n),
        min_gap_chunks=int(peak_split_min_gap_chunks),
    )
    if segs_peak is not None and len(segs_peak) >= 2:
        return segs_peak
    return segments_sec


def assign_prompts_to_segments(
    segments: List[Dict[str, Any]],
    *,
    long_prompt: Optional[str],
    prompt_mode: str,
    prompt_split_method: str = "auto",
    llm_cfg: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    n = len(segments)
    if prompt_mode == "none":
        for seg in segments:
            seg["prompt"] = ""
        return segments

    if prompt_mode == "gemini":
        raise RuntimeError(
            "prompt_mode=gemini requires calling assign_prompts_to_segments_gemini() "
            "from main() with video context."
        )

    if (long_prompt is None) or (not str(long_prompt).strip()):
        # No text provided: keep empty or a generic placeholder.
        base = "" if prompt_mode == "empty_if_missing" else "same scene continuation"
        for seg in segments:
            seg["prompt"] = base
        return segments

    lp = str(long_prompt).strip()
    if prompt_mode == "whole":
        for seg in segments:
            seg["prompt"] = lp
        return segments

    # prompt_mode == split
    parts, _ = split_long_prompt_with_method(lp, method=str(prompt_split_method), llm_cfg=llm_cfg)
    if not parts:
        for seg in segments:
            seg["prompt"] = lp
        return segments

    if len(parts) == n:
        for seg, p in zip(segments, parts):
            seg["prompt"] = p
        return segments

    if len(parts) > n:
        # Merge tail
        head = parts[: n - 1] if n >= 2 else []
        tail = " ".join(parts[n - 1 :]) if n >= 1 else " ".join(parts)
        parts2 = head + ([tail] if n >= 1 else [])
        for seg, p in zip(segments, parts2):
            seg["prompt"] = p
        return segments

    # len(parts) < n: repeat last part
    parts2 = parts + [parts[-1]] * (n - len(parts))
    for seg, p in zip(segments, parts2):
        seg["prompt"] = p
    return segments


def _pick_rep_times(start_sec: float, end_sec: float, n: int) -> List[float]:
    s = float(start_sec)
    e = float(end_sec)
    if n <= 1 or e <= s:
        return [float((s + e) * 0.5)]
    # Prefer stable, human-interpretable fractions.
    # n=3: 25/50/75% (often more robust than 1/6,3/6,5/6 for gradual transitions)
    if n == 3:
        fracs = [0.25, 0.5, 0.75]
        return [float(s + (e - s) * f) for f in fracs]
    if n == 2:
        fracs = [1.0 / 3.0, 2.0 / 3.0]
        return [float(s + (e - s) * f) for f in fracs]

    # Otherwise: evenly spaced centers inside (s,e)
    out: List[float] = []
    for i in range(n):
        t = s + (i + 0.5) * (e - s) / float(n)
        out.append(float(t))
    return out


def _postprocess_segments_chunkwise(
    segments: List[Dict[str, Any]],
    *,
    min_scene_chunks: int,
    max_scenes: int,
) -> List[Dict[str, Any]]:
    """
    Postprocess chunk-aligned segments:
    - Merge segments shorter than min_scene_chunks with neighbors (chunk-length based)
    - Cap total number of segments by repeatedly merging the shortest segment
    """
    if not segments:
        return segments

    min_scene_chunks = int(max(1, min_scene_chunks))
    max_scenes = int(max(1, max_scenes))

    def seg_len(seg: Dict[str, Any]) -> int:
        try:
            return int(seg.get("end_chunk", 0)) - int(seg.get("start_chunk", 0))
        except Exception:
            return 0

    def merge_pair(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        # Merge into a single segment spanning [a.start, b.end]
        return {
            "start_sec": float(a.get("start_sec", 0.0) or 0.0),
            "end_sec": float(b.get("end_sec", 0.0) or 0.0),
            "start_chunk": int(a.get("start_chunk", 0) or 0),
            "end_chunk": int(b.get("end_chunk", 0) or 0),
        }

    # 1) Merge short segments (< min_scene_chunks)
    segs = [
        {
            "start_sec": float(s.get("start_sec", 0.0) or 0.0),
            "end_sec": float(s.get("end_sec", 0.0) or 0.0),
            "start_chunk": int(s.get("start_chunk", 0) or 0),
            "end_chunk": int(s.get("end_chunk", 0) or 0),
        }
        for s in segments
    ]

    i = 0
    while i < len(segs):
        if seg_len(segs[i]) >= min_scene_chunks:
            i += 1
            continue
        # choose neighbor: prefer merging with the shorter neighbor if both exist
        if len(segs) == 1:
            break
        if i == 0:
            segs[0] = merge_pair(segs[0], segs[1])
            del segs[1]
            continue
        if i == len(segs) - 1:
            segs[i - 1] = merge_pair(segs[i - 1], segs[i])
            del segs[i]
            i = max(i - 1, 0)
            continue

        left_len = seg_len(segs[i - 1])
        right_len = seg_len(segs[i + 1])
        if left_len <= right_len:
            segs[i - 1] = merge_pair(segs[i - 1], segs[i])
            del segs[i]
            i = max(i - 1, 0)
        else:
            segs[i] = merge_pair(segs[i], segs[i + 1])
            del segs[i + 1]

    # 2) Cap max_scenes by merging the shortest segment repeatedly
    while len(segs) > max_scenes:
        lens = [seg_len(s) for s in segs]
        j = int(min(range(len(segs)), key=lambda k: lens[k]))
        if len(segs) == 1:
            break
        if j == 0:
            segs[0] = merge_pair(segs[0], segs[1])
            del segs[1]
        elif j == len(segs) - 1:
            segs[j - 1] = merge_pair(segs[j - 1], segs[j])
            del segs[j]
        else:
            # Merge with the neighbor yielding smaller merged length (heuristic)
            left_merge_len = seg_len(merge_pair(segs[j - 1], segs[j]))
            right_merge_len = seg_len(merge_pair(segs[j], segs[j + 1]))
            if left_merge_len <= right_merge_len:
                segs[j - 1] = merge_pair(segs[j - 1], segs[j])
                del segs[j]
            else:
                segs[j] = merge_pair(segs[j], segs[j + 1])
                del segs[j + 1]

    return segs


def _ffmpeg_extract_jpeg(video_path: str, t_sec: float, *, scale: Optional[str]) -> bytes:
    vf = []
    if scale:
        vf.append(f"scale={scale}")
    vf_arg = ",".join(vf) if vf else "null"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-ss",
        f"{float(t_sec):.6f}",
        "-i",
        video_path,
        "-frames:v",
        "1",
        "-vf",
        vf_arg,
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "pipe:1",
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    if p.returncode != 0 or not p.stdout:
        raise RuntimeError(
            f"ffmpeg failed extracting frame at t={t_sec}s for {video_path}\n"
            f"cmd={' '.join(cmd)}\n"
            f"stderr={p.stderr.decode('utf-8', errors='ignore')}"
        )
    return p.stdout


class _GeminiVision:
    def __init__(self, *, model: str, api_key: Optional[str]) -> None:
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Gemini API key missing. Set env GEMINI_API_KEY (or GOOGLE_API_KEY), "
                "or pass --gemini_api_key."
            )

        self._client = None
        self._model = str(model)
        self._mode = None

        # Prefer google-genai
        try:
            from google import genai  # type: ignore

            self._client = genai.Client(api_key=api_key)
            self._mode = "google_genai"
        except Exception:
            try:
                import google.generativeai as genai_legacy  # type: ignore

                genai_legacy.configure(api_key=api_key)
                self._client = genai_legacy
                self._mode = "google_generativeai"
            except Exception as e:
                raise RuntimeError(
                    "Missing Gemini SDK. Install one of:\n"
                    "  - pip install google-genai\n"
                    "  - pip install google-generativeai"
                ) from e

    def generate(self, *, instruction: str, jpeg_bytes_list: List[bytes]) -> str:
        if self._mode == "google_genai":
            from google.genai import types  # type: ignore

            parts: List[Any] = [types.Part.from_text(text=str(instruction))]
            for b in jpeg_bytes_list:
                parts.append(types.Part.from_bytes(data=b, mime_type="image/jpeg"))
            resp = self._client.models.generate_content(model=self._model, contents=[types.Content(parts=parts)])
            text = getattr(resp, "text", None)
            return str(text or "").strip()

        # legacy google-generativeai
        model = self._client.GenerativeModel(self._model)
        contents: List[Any] = [str(instruction)]
        for b in jpeg_bytes_list:
            try:
                contents.append(b)
            except Exception:
                contents.append({"mime_type": "image/jpeg", "data": base64.b64encode(b).decode("utf-8")})
        resp = model.generate_content(contents)
        return str(getattr(resp, "text", "")).strip()


def assign_prompts_to_segments_gemini(
    segments: List[Dict[str, Any]],
    *,
    video_path: str,
    long_prompt: Optional[str],
    gemini_model: str,
    gemini_api_key: Optional[str],
    frames_per_segment: int,
    ffmpeg_scale: str,
    sleep_sec: float,
    instruction_template: str,
) -> List[Dict[str, Any]]:
    """
    Use Gemini Vision to map each already-segmented scene to a sub-prompt, given the full prompt.
    """
    if (long_prompt is None) or (not str(long_prompt).strip()):
        # Behave like "empty_if_missing": do not invent prompts if user didn't supply one.
        for seg in segments:
            seg["prompt"] = ""
        return segments

    gv = _GeminiVision(model=gemini_model, api_key=gemini_api_key)
    lp = str(long_prompt).strip()

    for i, seg in enumerate(segments):
        s = float(seg.get("start_sec", 0.0) or 0.0)
        e = float(seg.get("end_sec", 0.0) or 0.0)
        rep_times = _pick_rep_times(s, e, int(frames_per_segment))
        jpegs = [_ffmpeg_extract_jpeg(video_path, t, scale=str(ffmpeg_scale)) for t in rep_times]

        instruction = instruction_template.format(
            full_prompt=lp,
            segment_index=i,
            segment_index_1based=i + 1,
            segment_count=len(segments),
            start_sec=f"{s:.3f}",
            end_sec=f"{e:.3f}",
        )
        text = gv.generate(instruction=instruction, jpeg_bytes_list=jpegs)
        seg["prompt"] = text.strip()
        if float(sleep_sec) > 0:
            time.sleep(float(sleep_sec))
    return segments


class _QwenVLPromptExtractor:
    """
    Local Qwen-VL prompt extractor.

    Given: full_prompt + a few frames from ONE segment,
    generate the sub-prompt corresponding to this segment.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 120,
        image_resize: Tuple[int, int] = (256, 448),
    ) -> None:
        try:
            from PIL import Image  # noqa: F401
        except Exception as e:  # noqa: BLE001
            raise RuntimeError("Missing dependency: pillow. Install with: pip install pillow") from e

        import torch

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        self.torch = torch
        self.device = str(device)
        self.dtype = dtype_map.get(str(dtype).lower(), torch.bfloat16)
        self.max_new_tokens = int(max_new_tokens)
        self.image_resize = tuple(image_resize)

        print(f"[QwenVLPromptExtractor] loading model from {model_path} ...")
        self.model, self.processor = load_qwen_vl_model_and_processor(
            str(model_path), torch_dtype=self.dtype, device=str(self.device)
        )

    def _decode_images(self, jpeg_bytes_list: List[bytes]):
        from PIL import Image

        imgs = []
        for b in jpeg_bytes_list:
            img = Image.open(BytesIO(b)).convert("RGB")
            # Keep consistent token budget
            h, w = self.image_resize
            imgs.append(img.resize((w, h)))
        return imgs

    def extract(
        self,
        *,
        instruction: str,
        jpeg_bytes_list: List[bytes],
        max_new_tokens: Optional[int] = None,
    ) -> str:
        # Build a simple chat with images then instruction.
        imgs = self._decode_images(jpeg_bytes_list)
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im} for im in imgs]
                + [{"type": "text", "text": str(instruction)}],
            }
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=imgs, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        mt = int(max_new_tokens) if max_new_tokens is not None else int(self.max_new_tokens)
        with self.torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=mt)

        # Strip the prompt part: keep only newly generated tokens
        gen_ids = out[0][inputs["input_ids"].shape[-1] :]
        decoded = self.processor.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return str(decoded).strip()


_QWEN_EXTRACTOR_CACHE: Dict[Tuple[str, str, str, Tuple[int, int], int], _QwenVLPromptExtractor] = {}

# Default instruction for visual-only batch captioning (placeholders: segment_count, segment_count_minus_one,
# image_index_map, optional_long_prompt). Literal JSON braces are doubled.
_DEFAULT_QWEN_VL_VISUAL_BATCH_INSTRUCTION = (
    "You write short English prompts for video-generation training data.\n"
    "You see ONE continuous video (same cast / same take) split into {segment_count} temporal segments. "
    "Images are in chronological order: all frames of segment 1 first, then segment 2, and so on.\n\n"
    "{image_index_map}\n\n"
    "Optional reference text from the dataset (may be empty; must not contradict the visuals):\n"
    "{optional_long_prompt}\n\n"
    "CRITICAL — UNIQUE PROMPTS:\n"
    "- Each segment MUST get a DIFFERENT prompt string. Copy-pasting the same paragraph across segments is INVALID.\n"
    "- Even if two segments look similar, describe what changes in time: action, pose, gaze, camera distance, "
    "framing (wide vs close-up), props, lighting, or background.\n\n"
    "CRITICAL — SAME PEOPLE (identity continuity):\n"
    "- Treat all visible people as THE SAME individuals throughout this clip unless the visuals unmistakably "
    "introduce a new person.\n"
    "- From segment 1: briefly anchor recognizable traits (clothing colors/cut, hair, approximate age, notable facial cues).\n"
    "- In segment 2 and later: START by tying identity with phrases like 'The same woman ...', 'The same man ...', "
    "or 'The same two people ...', then RESTATE 1–2 signature outfit/face/hair anchors, and ONLY THEN describe "
    "what is NEW in THIS segment (never reuse another segment's wording verbatim).\n\n"
    "Task: For EACH segment, write ONE prompt (1–3 sentences) for THAT segment only.\n"
    "Do not mention segment indices or image numbers inside the prompt strings.\n\n"
    "Return STRICT JSON only (no markdown fences):\n"
    '{{"segments":[{{"segment_index":0,"prompt":"..."}},{{"segment_index":1,"prompt":"..."}}]}}\n'
    "segment_index must be 0..{segment_count_minus_one} in order; every prompt must be non-empty and pairwise distinct."
)


def _build_visual_batch_image_index_map(
    segments: List[Dict[str, Any]],
    segment_jpeg_lists: List[List[bytes]],
) -> str:
    lines: List[str] = []
    cursor = 0
    for i, (seg, jpegs) in enumerate(zip(segments, segment_jpeg_lists)):
        n = len(jpegs)
        s = float(seg.get("start_sec", 0.0) or 0.0)
        e = float(seg.get("end_sec", 0.0) or 0.0)
        if n <= 0:
            lines.append(f"- Segment {i + 1}/{len(segments)}: (no frames) time [{s:.3f}s, {e:.3f}s)")
            continue
        lo = cursor
        hi = cursor + n - 1
        lines.append(
            f"- Segment {i + 1}/{len(segments)}: images [{lo}..{hi}] ({n} frames), "
            f"time [{s:.3f}s, {e:.3f}s)"
        )
        cursor += n
    return "\n".join(lines)


def assign_prompts_to_segments_qwen_vl_visual_batch(
    segments: List[Dict[str, Any]],
    *,
    video_path: str,
    optional_long_prompt: Optional[str],
    qwen_vl_model_path: str,
    qwen_vl_device: str,
    qwen_vl_dtype: str,
    qwen_vl_image_resize: Tuple[int, int],
    qwen_vl_max_new_tokens: int,
    qwen_vl_batch_max_new_tokens: int,
    frames_per_segment: int,
    ffmpeg_scale: str,
    instruction_template: str,
) -> List[Dict[str, Any]]:
    """
    One forward pass: all segments' frames (concatenated in time order) + strict JSON array of prompts.
    Does not require a long_prompt; optional_long_prompt is only a weak textual hint.
    """
    n = len(segments)
    if n == 0:
        return segments

    key = (
        str(qwen_vl_model_path),
        str(qwen_vl_device),
        str(qwen_vl_dtype),
        tuple(qwen_vl_image_resize),
        int(qwen_vl_max_new_tokens),
    )
    extractor = _QWEN_EXTRACTOR_CACHE.get(key)
    if extractor is None:
        extractor = _QwenVLPromptExtractor(
            model_path=str(qwen_vl_model_path),
            device=str(qwen_vl_device),
            dtype=str(qwen_vl_dtype),
            max_new_tokens=int(qwen_vl_max_new_tokens),
            image_resize=tuple(qwen_vl_image_resize),
        )
        _QWEN_EXTRACTOR_CACHE[key] = extractor

    segment_jpeg_lists: List[List[bytes]] = []
    for seg in segments:
        s = float(seg.get("start_sec", 0.0) or 0.0)
        e = float(seg.get("end_sec", 0.0) or 0.0)
        rep_times = _pick_rep_times(s, e, int(frames_per_segment))
        segment_jpeg_lists.append([_ffmpeg_extract_jpeg(video_path, t, scale=str(ffmpeg_scale)) for t in rep_times])

    flat: List[bytes] = [b for group in segment_jpeg_lists for b in group]
    if not flat:
        for seg in segments:
            seg["prompt"] = ""
        return segments

    image_map = _build_visual_batch_image_index_map(segments, segment_jpeg_lists)
    opt_lp = str(optional_long_prompt or "").strip() or "(none)"
    instruction = str(instruction_template).format(
        segment_count=n,
        segment_count_minus_one=max(n - 1, 0),
        image_index_map=image_map,
        optional_long_prompt=opt_lp,
    )

    raw = extractor.extract(
        instruction=instruction,
        jpeg_bytes_list=flat,
        max_new_tokens=int(qwen_vl_batch_max_new_tokens),
    )
    parsed = _parse_qwen_vl_batch_prompts_json(raw, expected_n=n)
    if parsed is not None:
        if n >= 2 and _segment_prompts_have_duplicates(parsed):
            prev_json = json.dumps(
                [{"segment_index": i, "prompt": parsed[i]} for i in range(n)],
                ensure_ascii=False,
                indent=2,
            )
            repair_instruction = (
                "The previous JSON is INVALID: at least two segments have the SAME prompt text (copy-paste). "
                "Regenerate the COMPLETE answer with the SAME JSON schema and segment order, but:\n"
                "- Every 'prompt' string must be DISTINCT across segments.\n"
                "- This is one continuous clip with the SAME people; in segment 1 establish look (outfit/face/hair). "
                "In segments 2+ start with 'The same woman/man/people ...', repeat 1–2 signature visual anchors, "
                "then describe ONLY what changes in THAT segment (shot scale, action, lighting).\n"
                "Return STRICT JSON only (no markdown).\n\n"
                "Invalid previous output:\n"
                + prev_json
            )
            raw2 = extractor.extract(
                instruction=repair_instruction,
                jpeg_bytes_list=flat,
                max_new_tokens=int(qwen_vl_batch_max_new_tokens),
            )
            parsed2 = _parse_qwen_vl_batch_prompts_json(raw2, expected_n=n)
            if parsed2 is not None:
                parsed = parsed2
        for seg, p in zip(segments, parsed):
            seg["prompt"] = str(p).strip()
        return segments

    # Fallback: per-segment short captions (no cross-segment coreference guarantee).
    for i, seg in enumerate(segments):
        sj = segment_jpeg_lists[i] if i < len(segment_jpeg_lists) else []
        if not sj:
            seg["prompt"] = ""
            continue
        fb_inst = (
            "Describe ONLY what is visible in THESE frames in 1–2 English sentences for video generation training. "
            "If a person appears, give concrete outfit/face/hair cues; if this is a later part of a clip, you may "
            "prefix 'The same woman/man ...' with those anchors. Describe what is NEW in this shot. "
            "Output plain text only (no JSON)."
        )
        mt = min(256, int(max(qwen_vl_batch_max_new_tokens, 128)))
        t2 = extractor.extract(instruction=fb_inst, jpeg_bytes_list=sj, max_new_tokens=mt)
        seg["prompt"] = _strip_code_fences(str(t2)).strip()
    return segments


def assign_prompts_to_segments_qwen_vl(
    segments: List[Dict[str, Any]],
    *,
    video_path: str,
    long_prompt: Optional[str],
    qwen_vl_model_path: str,
    qwen_vl_device: str,
    qwen_vl_dtype: str,
    qwen_vl_image_resize: Tuple[int, int],
    qwen_vl_max_new_tokens: int,
    frames_per_segment: int,
    ffmpeg_scale: str,
    sleep_sec: float,
    instruction_template: str,
    fallback_prompt_parts: Optional[List[str]] = None,
    repeat_sim_threshold: float = 0.82,
) -> List[Dict[str, Any]]:
    if (long_prompt is None) or (not str(long_prompt).strip()):
        for seg in segments:
            seg["prompt"] = ""
        return segments

    key = (
        str(qwen_vl_model_path),
        str(qwen_vl_device),
        str(qwen_vl_dtype),
        tuple(qwen_vl_image_resize),
        int(qwen_vl_max_new_tokens),
    )
    extractor = _QWEN_EXTRACTOR_CACHE.get(key)
    if extractor is None:
        extractor = _QwenVLPromptExtractor(
            model_path=str(qwen_vl_model_path),
            device=str(qwen_vl_device),
            dtype=str(qwen_vl_dtype),
            max_new_tokens=int(qwen_vl_max_new_tokens),
            image_resize=tuple(qwen_vl_image_resize),
        )
        _QWEN_EXTRACTOR_CACHE[key] = extractor
    lp = str(long_prompt).strip()

    used_prompts: List[str] = []
    for i, seg in enumerate(segments):
        s = float(seg.get("start_sec", 0.0) or 0.0)
        e = float(seg.get("end_sec", 0.0) or 0.0)
        rep_times = _pick_rep_times(s, e, int(frames_per_segment))
        jpegs = [_ffmpeg_extract_jpeg(video_path, t, scale=str(ffmpeg_scale)) for t in rep_times]

        instruction = instruction_template.format(
            full_prompt=lp,
            segment_index=i,
            segment_index_1based=i + 1,
            segment_count=len(segments),
            start_sec=f"{s:.3f}",
            end_sec=f"{e:.3f}",
            used_prompts="\n".join([f"- {p}" for p in used_prompts]) if used_prompts else "(none)",
        )
        prompt_out: str = ""
        last_raw: str = ""
        # Retry a couple of times if output repeats/overlaps too much.
        for attempt in range(2):
            extra = ""
            if attempt >= 1:
                extra = (
                    "\n\nIMPORTANT: Your previous output overlapped/repeated earlier segments. "
                    "Rewrite to be DISJOINT from used_prompts while staying faithful."
                )
            text = extractor.extract(instruction=str(instruction) + extra, jpeg_bytes_list=jpegs).strip()
            last_raw = text
            maybe = _extract_segment_prompt_from_text(text)
            prompt_candidate = maybe if maybe is not None else _strip_code_fences(text).strip()
            prompt_candidate = str(prompt_candidate).strip()

            # If empty, retry.
            if not prompt_candidate:
                continue

            # Reject highly overlapping repeats.
            if used_prompts:
                max_sim = max(_text_jaccard(prompt_candidate, up) for up in used_prompts)
                if max_sim >= float(repeat_sim_threshold):
                    continue
                # Also reject containment repeats (often happens with copied appearance clauses).
                for up in used_prompts:
                    a = prompt_candidate.strip()
                    b = str(up).strip()
                    if len(a) >= 40 and a in b:
                        max_sim = 1.0
                        break
                    if len(b) >= 40 and b in a:
                        max_sim = 1.0
                        break
                if max_sim >= 1.0:
                    continue

            prompt_out = prompt_candidate
            break

        # Ensure prompt_out is plain text, never a JSON blob.
        if prompt_out and prompt_out.lstrip().startswith("{"):
            maybe2 = _extract_segment_prompt_from_text(prompt_out)
            if maybe2:
                prompt_out = maybe2

        if not prompt_out:
            # Prefer falling back to the LLM-split prompt parts (guarantees non-overlap coverage).
            if fallback_prompt_parts and 0 <= i < len(fallback_prompt_parts):
                prompt_out = str(fallback_prompt_parts[i]).strip()
            else:
                # Worst-case: keep a cleaned version of raw text (avoid fenced blocks).
                prompt_out = _strip_code_fences(last_raw).strip()
                maybe3 = _extract_segment_prompt_from_text(prompt_out)
                if maybe3:
                    prompt_out = maybe3

        seg["prompt"] = prompt_out
        if prompt_out:
            used_prompts.append(prompt_out)
        if float(sleep_sec) > 0:
            time.sleep(float(sleep_sec))
    return segments


def load_long_prompts_jsonl(path: str) -> Dict[str, str]:
    """
    Accepts multiple formats:
    - JSONL: one JSON object per line: {"id": "...", "prompt": "..."} (also accepts {"sample_id": "...", ...})
    - Single JSON object file (pretty-printed allowed): {"id": "...", "prompt": "..."} (also accepts {"sample_id": "...", ...})
    - JSON list file: [{"id": "...", "prompt": "..."}, ...] (also accepts sample_id)
    """
    out: Dict[str, str] = {}

    raw = Path(path).read_text(encoding="utf-8").strip()
    if not raw:
        return out

    # 1) Try parse as whole-file JSON (dict or list). This supports pretty-printed JSON.
    try:
        payload = json.loads(raw)
        items: List[Dict[str, Any]]
        if isinstance(payload, dict):
            items = [payload]
        elif isinstance(payload, list):
            items = [x for x in payload if isinstance(x, dict)]
        else:
            items = []
        for obj in items:
            vid = str(obj.get("sample_id", obj.get("id", ""))).strip()
            pr = str(obj.get("prompt", "")).strip()
            if vid:
                out[vid] = pr
        if out:
            return out
    except Exception:
        pass

    # 2) Fallback: JSONL or multi-line concatenated JSON objects.
    #    We accumulate lines until a valid JSON object parses.
    buf_lines: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            buf_lines.append(line)
            candidate = "".join(buf_lines).strip()
            try:
                obj = json.loads(candidate)
            except Exception:
                continue

            if isinstance(obj, dict):
                vid = str(obj.get("id", obj.get("sample_id", ""))).strip()
                pr = str(obj.get("prompt", "")).strip()
                if vid:
                    out[vid] = pr
            elif isinstance(obj, list):
                for x in obj:
                    if not isinstance(x, dict):
                        continue
                    vid = str(x.get("id", x.get("sample_id", ""))).strip()
                    pr = str(x.get("prompt", "")).strip()
                    if vid:
                        out[vid] = pr
            buf_lines = []

    # If buffer remains but didn't parse, ignore silently.
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build chunk-aligned metadata.json from videos.")
    p.add_argument("--videos_dir", required=True, help="Directory containing *.mp4 videos.")
    p.add_argument("--out_json", required=True, help="Output metadata JSON path.")
    p.add_argument(
        "--long_prompts_jsonl",
        default=None,
        help='Optional JSONL mapping: {"id": "...", "prompt": "..."}; id defaults to video stem.',
    )
    p.add_argument(
        "--prompt_mode",
        choices=["split", "whole", "none", "empty_if_missing", "gemini", "qwen_vl"],
        default="split",
        help="How to produce per-segment prompt from long prompt.",
    )
    p.add_argument(
        "--prompt_split_method",
        choices=["auto", "regex", "llm"],
        default="auto",
        help="How to split a long prompt into semantic phases (used for split prompt mode and adaptive segmentation).",
    )
    p.add_argument(
        "--prompt_split_llm_model_path",
        default="/root/autodl-fs/Qwen2.5-VL-3B-Instruct",
        help=(
            "Local HF path for LLM prompt splitting. Supports Qwen2.5-VL or Qwen3-VL (text-only). "
            "If you only downloaded Qwen3-VL-8B, set this to the same path as --qwen_vl_model_path to reuse weights."
        ),
    )
    p.add_argument("--prompt_split_llm_device", default="cuda", help="cuda/cpu for prompt-splitting LLM.")
    p.add_argument("--prompt_split_llm_dtype", default="bfloat16", help="bfloat16/float16/float32 for LLM.")
    p.add_argument("--prompt_split_llm_max_new_tokens", type=int, default=256, help="Max new tokens for LLM split.")
    p.add_argument(
        "--fallback_equal_split_from_prompt",
        action="store_true",
        help=(
            "If scene segmentation returns a single segment but the long prompt splits into multiple parts, "
            "fallback to equally split the video duration into N parts (N = #prompt parts)."
        ),
    )
    p.add_argument(
        "--fallback_peak_split_from_prompt",
        action="store_true",
        help=(
            "If segmentation returns a single scene but the long prompt splits into multiple parts, "
            "split into N parts by picking visual-change peaks between chunk-mid frames (better than equal split)."
        ),
    )

    # Helios chunking definition (user-side)
    p.add_argument("--num_latent_frames_per_chunk", type=int, default=9)
    p.add_argument("--t_downsample", type=int, default=4)
    p.add_argument("--min_len_chunks", type=int, default=1, help="Drop segments shorter than this many chunks.")
    p.add_argument(
        "--snap_sec_to_chunk",
        action="store_true",
        help="Snap start_sec/end_sec to chunk boundary seconds after chunk alignment to avoid time gaps.",
    )

    # Segmentation
    p.add_argument(
        "--seg_method",
        choices=["auto", "scenedetect", "opencv"],
        default="scenedetect",
        help='Scene segmentation: "scenedetect" uses TransNetV2 (not PySceneDetect); "auto" tries TransNetV2 then OpenCV.',
    )
    p.add_argument(
        "--transnetv2_threshold",
        type=float,
        default=0.5,
        help="TransNetV2 boundary probability threshold in [0,1]. Lower => more cuts (more false positives).",
    )
    p.add_argument(
        "--transnetv2_device",
        type=str,
        default="auto",
        help="Device for TransNetV2: auto|cuda|cpu|mps (passed to transnetv2-pytorch).",
    )
    p.add_argument("--opencv_sample_every", type=int, default=6)
    p.add_argument("--opencv_diff_threshold", type=float, default=0.55)
    p.add_argument("--opencv_min_scene_sec", type=float, default=1.0)
    p.add_argument(
        "--peak_split_min_gap_chunks",
        type=int,
        default=1,
        help="Peak-split: enforce at least this many chunks between boundaries.",
    )
    p.add_argument(
        "--transnetv2_debug",
        action="store_true",
        help="Print TransNetV2 failure reasons (when seg_method uses scenedetect/auto).",
    )
    p.add_argument(
        "--post_min_scene_chunks",
        type=int,
        default=2,
        help="Postprocess: merge scenes shorter than this many chunks (chunk-aligned).",
    )
    p.add_argument(
        "--post_max_scenes",
        type=int,
        default=5,
        help="Postprocess: cap max number of scenes per video by merging shortest scenes.",
    )
    p.add_argument(
        "--min_segments",
        type=int,
        default=1,
        help="Hard minimum number of segments per video (applied before chunk alignment).",
    )
    p.add_argument(
        "--max_segments",
        type=int,
        default=8,
        help="Hard maximum number of segments per video (applied before chunk alignment).",
    )

    # Gemini (prompt refinement per segment)
    p.add_argument(
        "--gemini_model",
        default="gemini-1.5-pro",
        help="Gemini model name (must support vision). Used when --prompt_mode=gemini.",
    )
    p.add_argument(
        "--gemini_api_key",
        default=None,
        help="Optional. Otherwise use env GEMINI_API_KEY/GOOGLE_API_KEY. Used when --prompt_mode=gemini.",
    )
    p.add_argument(
        "--gemini_frames_per_segment",
        type=int,
        default=1,
        help="How many representative frames to send per segment (1-4 recommended).",
    )
    p.add_argument(
        "--gemini_ffmpeg_scale",
        default="512:-1",
        help='ffmpeg scale for extracted frames, e.g. "512:-1" or "-1:512".',
    )
    p.add_argument("--gemini_sleep_sec", type=float, default=0.0, help="Sleep between Gemini requests.")
    p.add_argument(
        "--gemini_instruction",
        default=(
            "You are given a full prompt describing an entire 10s video and a set of frames from ONE segment.\\n"
            "Full prompt:\\n{full_prompt}\\n\\n"
            "This is segment {segment_index_1based}/{segment_count}, time [{start_sec}s, {end_sec}s).\\n"
            "Task: Extract ONLY the sub-prompt that best corresponds to this segment, using details consistent with the frames.\\n"
            "- Keep it faithful to the full prompt; do not invent new entities/attributes.\\n"
            "- Prefer a concise, training-friendly description (1-2 sentences).\\n"
            "- If the segment matches a portion of the full prompt, rewrite that portion cleanly.\\n"
            "Return only the segment prompt text."
        ),
        help=(
            "Instruction template for Gemini. Available fields: "
            "{full_prompt},{segment_index},{segment_index_1based},{segment_count},{start_sec},{end_sec}."
        ),
    )

    # Qwen-VL (local prompt extraction per segment)
    p.add_argument(
        "--qwen_vl_model_path",
        default=DEFAULT_QWEN_VL_MODEL_PATH,
        help="Local HF path for Qwen2.5-VL or Qwen3-VL Instruct. Default: Qwen3-VL-8B-Instruct directory.",
    )
    p.add_argument("--qwen_vl_device", default="cuda", help="cuda/cpu for Qwen-VL inference.")
    p.add_argument("--qwen_vl_dtype", default="bfloat16", help="bfloat16/float16/float32")
    p.add_argument("--qwen_vl_image_resize", type=int, nargs=2, default=[256, 448], help="(h, w) for images.")
    p.add_argument("--qwen_vl_max_new_tokens", type=int, default=120)
    p.add_argument("--qwen_vl_frames_per_segment", type=int, default=3, help="Frames per segment for Qwen-VL.")
    p.add_argument("--qwen_vl_ffmpeg_scale", default="512:-1")
    p.add_argument("--qwen_vl_sleep_sec", type=float, default=0.0)
    p.add_argument(
        "--qwen_vl_instruction",
        default=(
            "You are an expert video-segment prompt aligner. Your job is NOT to invent a new story, "
            "but to EXTRACT the part of the full prompt that matches this segment's frames.\\n\\n"
            "FULL PROMPT (entire video):\\n{full_prompt}\\n\\n"
            "SEGMENT: {segment_index_1based}/{segment_count}, time [{start_sec}s, {end_sec}s).\\n"
            "ALREADY-USED segment prompts from earlier segments (must avoid overlap/reuse):\\n{used_prompts}\\n\\n"
            "Hard constraints:\\n"
            "1) Output must be STRICT JSON with keys: segment_prompt, quotes.\\n"
            "2) segment_prompt must be 1-2 sentences, describing ONLY what is visible in THIS segment "
            "(lighting/shadows, subject appearance details, action).\\n"
            "3) segment_prompt must be faithful to FULL PROMPT; do not introduce new entities, objects, colors, "
            "or actions not supported by both the frames and the full prompt.\\n"
            "4) Non-overlap: do NOT reuse phrases already used above; if unavoidable, rewrite to be disjoint.\\n"
            "5) quotes must be a JSON list of 1-3 short exact phrases copied from FULL PROMPT that you used "
            "as evidence for this segment.\\n\\n"
            "Return JSON only, no extra text."
        ),
        help=(
            "Instruction template for Qwen-VL. Available fields: "
            "{full_prompt},{segment_index},{segment_index_1based},{segment_count},{start_sec},{end_sec},{used_prompts}."
        ),
    )
    p.add_argument(
        "--qwen_vl_prompt_source",
        choices=["long_prompt", "visual_batch"],
        default="long_prompt",
        help=(
            "long_prompt: per-segment extraction using FULL PROMPT from JSONL (original behavior). "
            "visual_batch: caption all segments in ONE Qwen-VL call from frames only (optional JSONL as hint); "
            "best for TransNetV2 segments without reliable long prompts."
        ),
    )
    p.add_argument(
        "--qwen_vl_batch_max_new_tokens",
        type=int,
        default=1024,
        help="Max new tokens for visual_batch single-pass JSON output (use more when many segments).",
    )
    p.add_argument(
        "--qwen_vl_visual_batch_instruction",
        default=_DEFAULT_QWEN_VL_VISUAL_BATCH_INSTRUCTION,
        help=(
            "Instruction for visual_batch mode. Placeholders: {segment_count}, {segment_count_minus_one}, "
            "{image_index_map}, {optional_long_prompt}."
        ),
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    videos_dir = Path(args.videos_dir)
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    long_prompt_map: Dict[str, str] = {}
    if args.long_prompts_jsonl:
        long_prompt_map = load_long_prompts_jsonl(args.long_prompts_jsonl)

    ch_frames = chunk_frames(args.num_latent_frames_per_chunk, args.t_downsample)

    video_paths = sorted([p for p in videos_dir.iterdir() if p.suffix.lower() in {".mp4", ".mov", ".avi", ".webm"}])
    if not video_paths:
        raise SystemExit(f"No videos found in: {videos_dir}")

    records: List[Dict[str, Any]] = []
    for vp in video_paths:
        vid = vp.stem
        info = read_video_info(str(vp))
        long_prompt = long_prompt_map.get(vid)
        llm_cfg = {
            "model_path": str(args.prompt_split_llm_model_path),
            "device": str(args.prompt_split_llm_device),
            "dtype": str(args.prompt_split_llm_dtype),
            "max_new_tokens": int(args.prompt_split_llm_max_new_tokens),
        }
        prompt_split_method = "missing"
        prompt_parts_n = 0
        if long_prompt is not None and str(long_prompt).strip():
            _parts, prompt_split_method = split_long_prompt_with_method(
                str(long_prompt).strip(),
                method=str(args.prompt_split_method),
                llm_cfg=llm_cfg,
            )
            prompt_parts_n = len(_parts)

        # Step: detect segments in seconds
        segments_sec: Optional[List[Tuple[float, float]]] = None
        segmentation_used = None
        equal_split_from_prompt_used = False
        if args.seg_method in {"auto", "scenedetect"}:
            segments_sec = try_detect_scenes_transnetv2(
                str(vp),
                threshold=float(args.transnetv2_threshold),
                device=str(args.transnetv2_device),
                duration_sec_fallback=float(info.duration_sec),
                debug=bool(args.transnetv2_debug),
            )
            if args.seg_method == "scenedetect" and segments_sec is None:
                raise SystemExit(
                    "TransNetV2 segmentation requested but failed. "
                    "Install: pip install transnetv2-pytorch  (requires PyTorch). "
                    "Use --transnetv2_debug to print the error, or use --seg_method opencv."
                )
            if segments_sec is not None:
                segmentation_used = "transnetv2"

        if segments_sec is None:
            if args.seg_method == "scenedetect":
                # Should not happen due to the explicit error above, but keep it safe.
                raise SystemExit(
                    "TransNetV2 segmentation requested but produced no valid scenes. "
                    "Try lowering --transnetv2_threshold (e.g. 0.35), or use --seg_method opencv."
                )
            segments_sec = detect_scenes_opencv(
                str(vp),
                fps=info.fps,
                num_frames=max(info.num_frames, 1),
                sample_every_n_frames=int(args.opencv_sample_every),
                diff_threshold=float(args.opencv_diff_threshold),
                min_scene_sec=float(args.opencv_min_scene_sec),
            )
            segmentation_used = "opencv_hsv_hist"

        # Decide desired segment count bounds.
        min_seg = int(max(1, args.min_segments))
        max_seg = int(max(min_seg, args.max_segments))
        desired_n = None
        if prompt_parts_n >= 1:
            desired_n = _clamp_segments_count(prompt_parts_n, min_segments=min_seg, max_segments=max_seg)
        else:
            desired_n = _clamp_segments_count(len(segments_sec or []), min_segments=min_seg, max_segments=max_seg)

        # Prompt-Guided adaptive segmentation (before chunk alignment).
        if desired_n is not None and desired_n >= 2 and segments_sec is not None and len(segments_sec) >= 1:
            segs2 = adaptive_segmentation(
                segments_sec,
                target_n=int(desired_n),
                video_path=str(vp),
                fps=float(info.fps),
                num_frames=max(int(info.num_frames), 1),
                ch_frames=int(ch_frames),
                peak_split_min_gap_chunks=int(args.peak_split_min_gap_chunks),
            )
            if segs2 is not None and len(segs2) >= 1 and len(segs2) != len(segments_sec):
                segments_sec = segs2
                segmentation_used = f"{segmentation_used or 'unknown'}+adaptive"

        # Hard cap / floor on segment count (seconds-space, before chunk alignment).
        if segments_sec is not None:
            if len(segments_sec) > max_seg:
                segments_sec = merge_segments_by_similarity(
                    segments_sec,
                    target_n=int(max_seg),
                    video_path=str(vp),
                    fps=float(info.fps),
                )
                segmentation_used = f"{segmentation_used or 'unknown'}+cap{max_seg}"
            elif len(segments_sec) < min_seg:
                segs_floor = _peak_split_segments_from_video(
                    str(vp),
                    fps=float(info.fps),
                    num_frames=max(int(info.num_frames), 1),
                    ch_frames=int(ch_frames),
                    n_parts=int(min_seg),
                    min_gap_chunks=int(args.peak_split_min_gap_chunks),
                )
                if segs_floor is not None and len(segs_floor) >= min_seg:
                    segments_sec = segs_floor
                    segmentation_used = f"{segmentation_used or 'unknown'}+floor{min_seg}"

        # Fallback: if we only got one segment but prompt has multiple parts,
        # split video into N parts (N=#prompt parts) using either peak-split or equal-split.
        if long_prompt is not None and str(long_prompt).strip():
            parts, _ = split_long_prompt_with_method(
                str(long_prompt).strip(),
                method=str(args.prompt_split_method),
                llm_cfg=llm_cfg,
            )
            if len(parts) >= 2 and (segments_sec is not None) and len(segments_sec) <= 1:
                n = min(len(parts), 8)
                if args.fallback_peak_split_from_prompt:
                    segs_peak = _peak_split_segments_from_video(
                        str(vp),
                        fps=float(info.fps),
                        num_frames=max(int(info.num_frames), 1),
                        ch_frames=int(ch_frames),
                        n_parts=int(n),
                        min_gap_chunks=int(args.peak_split_min_gap_chunks),
                    )
                    if segs_peak is not None and len(segs_peak) >= 2:
                        segments_sec = segs_peak
                        equal_split_from_prompt_used = True
                        segmentation_used = "peak_split_from_prompt"
                if (
                    (
                        segmentation_used is None
                        or str(segmentation_used).startswith("transnetv2")
                    )
                    and args.fallback_equal_split_from_prompt
                ):
                    duration = info.duration_sec
                    if duration > 0 and n >= 2:
                        step = duration / float(n)
                        segments_sec = [(i * step, (i + 1) * step) for i in range(n)]
                        equal_split_from_prompt_used = True
                        segmentation_used = "equal_split_from_prompt"

        # Align to chunk boundaries
        segments = align_segments_to_chunks(
            segments_sec,
            fps=info.fps,
            num_frames=max(info.num_frames, 1),
            ch_frames=ch_frames,
            min_len_chunks=int(args.min_len_chunks),
            snap_sec_to_chunk=bool(args.snap_sec_to_chunk),
        )

        # Postprocess TransNetV2-based outputs: merge short scenes (by chunk) and cap max scene count.
        # This improves stability for downstream prompt extraction (e.g. Qwen-VL).
        if segmentation_used is not None and str(segmentation_used).startswith("transnetv2"):

            segments = _postprocess_segments_chunkwise(
                segments,
                min_scene_chunks=int(args.post_min_scene_chunks),
                max_scenes=int(args.post_max_scenes),
            )

        # Second-stage fallback: if chunk-alignment collapses to a single segment but prompt has multiple parts,
        # split into N parts and re-align (peak-split preferred, else equal-split).
        if long_prompt is not None and str(long_prompt).strip():
            parts, _ = split_long_prompt_with_method(
                str(long_prompt).strip(),
                method=str(args.prompt_split_method),
                llm_cfg=llm_cfg,
            )
            if len(parts) >= 2 and len(segments) <= 1:
                n = min(len(parts), 8)
                segments_sec2: Optional[List[Tuple[float, float]]] = None
                if args.fallback_peak_split_from_prompt:
                    segments_sec2 = _peak_split_segments_from_video(
                        str(vp),
                        fps=float(info.fps),
                        num_frames=max(int(info.num_frames), 1),
                        ch_frames=int(ch_frames),
                        n_parts=int(n),
                        min_gap_chunks=int(args.peak_split_min_gap_chunks),
                    )
                    if segments_sec2 is not None and len(segments_sec2) >= 2:
                        segmentation_used = "peak_split_from_prompt"
                if segments_sec2 is None and args.fallback_equal_split_from_prompt:
                    duration = info.duration_sec
                    if duration > 0 and n >= 2:
                        step = duration / float(n)
                        segments_sec2 = [(i * step, (i + 1) * step) for i in range(n)]
                        segmentation_used = "equal_split_from_prompt"
                if segments_sec2 is not None:
                    segments = align_segments_to_chunks(
                        segments_sec2,
                        fps=info.fps,
                        num_frames=max(info.num_frames, 1),
                        ch_frames=ch_frames,
                        min_len_chunks=int(args.min_len_chunks),
                        snap_sec_to_chunk=bool(args.snap_sec_to_chunk),
                    )
                    equal_split_from_prompt_used = True

        if segmentation_used is not None and str(segmentation_used).startswith("transnetv2"):
            segments = _postprocess_segments_chunkwise(
                segments,
                min_scene_chunks=int(args.post_min_scene_chunks),
                max_scenes=int(args.post_max_scenes),
            )

        # Enforce min/max on the final chunk-aligned segments (postprocess may reduce count).
        segments = enforce_final_segment_bounds(
            segments,
            min_segments=int(min_seg),
            max_segments=int(max_seg),
            video_path=str(vp),
            fps=float(info.fps),
            num_frames=max(int(info.num_frames), 1),
            ch_frames=int(ch_frames),
            peak_split_min_gap_chunks=int(args.peak_split_min_gap_chunks),
            snap_sec_to_chunk=bool(args.snap_sec_to_chunk),
            min_len_chunks=int(args.min_len_chunks),
        )

        # Assign prompts
        if args.prompt_mode == "gemini":
            segments = assign_prompts_to_segments_gemini(
                segments,
                video_path=str(vp),
                long_prompt=long_prompt,
                gemini_model=str(args.gemini_model),
                gemini_api_key=args.gemini_api_key,
                frames_per_segment=int(args.gemini_frames_per_segment),
                ffmpeg_scale=str(args.gemini_ffmpeg_scale),
                sleep_sec=float(args.gemini_sleep_sec),
                instruction_template=str(args.gemini_instruction),
            )
        elif args.prompt_mode == "qwen_vl":
            if str(args.qwen_vl_prompt_source) == "visual_batch":
                segments = assign_prompts_to_segments_qwen_vl_visual_batch(
                    segments,
                    video_path=str(vp),
                    optional_long_prompt=long_prompt,
                    qwen_vl_model_path=str(args.qwen_vl_model_path),
                    qwen_vl_device=str(args.qwen_vl_device),
                    qwen_vl_dtype=str(args.qwen_vl_dtype),
                    qwen_vl_image_resize=tuple(args.qwen_vl_image_resize),
                    qwen_vl_max_new_tokens=int(args.qwen_vl_max_new_tokens),
                    qwen_vl_batch_max_new_tokens=int(args.qwen_vl_batch_max_new_tokens),
                    frames_per_segment=int(args.qwen_vl_frames_per_segment),
                    ffmpeg_scale=str(args.qwen_vl_ffmpeg_scale),
                    instruction_template=str(args.qwen_vl_visual_batch_instruction),
                )
            else:
                # Provide a deterministic fallback mapping from segment index -> LLM-split prompt part.
                fallback_parts: Optional[List[str]] = None
                if long_prompt is not None and str(long_prompt).strip():
                    fallback_parts, _ = split_long_prompt_with_method(
                        str(long_prompt).strip(),
                        method=str(args.prompt_split_method),
                        llm_cfg=llm_cfg,
                    )
                    if fallback_parts:
                        fallback_parts = _dedupe_prompt_parts(list(fallback_parts))
                    if fallback_parts:
                        if len(fallback_parts) > len(segments):
                            fallback_parts = fallback_parts[: len(segments) - 1] + [
                                " ".join(fallback_parts[len(segments) - 1 :]).strip()
                            ]
                        elif len(fallback_parts) < len(segments):
                            fallback_parts = fallback_parts + [fallback_parts[-1]] * (len(segments) - len(fallback_parts))

                segments = assign_prompts_to_segments_qwen_vl(
                    segments,
                    video_path=str(vp),
                    long_prompt=long_prompt,
                    qwen_vl_model_path=str(args.qwen_vl_model_path),
                    qwen_vl_device=str(args.qwen_vl_device),
                    qwen_vl_dtype=str(args.qwen_vl_dtype),
                    qwen_vl_image_resize=tuple(args.qwen_vl_image_resize),
                    qwen_vl_max_new_tokens=int(args.qwen_vl_max_new_tokens),
                    frames_per_segment=int(args.qwen_vl_frames_per_segment),
                    ffmpeg_scale=str(args.qwen_vl_ffmpeg_scale),
                    sleep_sec=float(args.qwen_vl_sleep_sec),
                    instruction_template=str(args.qwen_vl_instruction),
                    fallback_prompt_parts=fallback_parts,
                    repeat_sim_threshold=0.82,
                )
        else:
            segments = assign_prompts_to_segments(
                segments,
                long_prompt=long_prompt,
                prompt_mode=args.prompt_mode,
                prompt_split_method=str(args.prompt_split_method),
                llm_cfg=llm_cfg,
            )

        rec: Dict[str, Any] = {
            "id": vid,
            "path": str(vp.relative_to(videos_dir.parent)) if videos_dir.parent in vp.parents else str(vp),
            "fps": float(info.fps),
            "num_frames": int(info.num_frames),
            "resolution": {"height": int(info.height), "width": int(info.width)},
            "chunking": {
                "num_latent_frames_per_chunk": int(args.num_latent_frames_per_chunk),
                "t_downsample": int(args.t_downsample),
                "chunk_frames": int(ch_frames),
            },
            "segments": segments,
            "notes": {
                "segmentation_method": str(args.seg_method),
                "segmentation_used": segmentation_used or "unknown",
                "equal_split_from_prompt_used": bool(equal_split_from_prompt_used),
                "has_long_prompt": bool(long_prompt is not None and str(long_prompt).strip()),
                "prompt_mode": str(args.prompt_mode),
                "qwen_vl_prompt_source": str(args.qwen_vl_prompt_source)
                if str(args.prompt_mode) == "qwen_vl"
                else "",
                "prompt_split_method": prompt_split_method,
                "prompt_parts": int(prompt_parts_n),
            },
        }
        records.append(rec)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {len(records)} record(s) to: {out_json}")


if __name__ == "__main__":
    main()

