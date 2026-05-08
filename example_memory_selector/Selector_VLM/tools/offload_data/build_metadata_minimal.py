#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

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
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "image": im} for im in imgs] + [{"type": "text", "text": instruction}],
            }
        ]
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
        lines.append(
            f"- Segment {i+1}/{len(segments)} -> images [{lo}..{hi}], time [{seg['start_sec']:.3f}s, {seg['end_sec']:.3f}s)"
        )
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

