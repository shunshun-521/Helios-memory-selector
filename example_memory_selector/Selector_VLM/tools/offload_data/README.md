# offload_data (简版)

本目录当前建议只关注一个入口：

- `build_metadata_minimal.py`：实际执行脚本（精简版）

精简版只保留两条能力：

1. **TransNetV2** 做分镜（`--seg_method scenedetect`）
2. **Qwen3-VL visual_batch** 生成 segment prompt（`--prompt_mode qwen_vl --qwen_vl_prompt_source visual_batch`）

---

## 1) 依赖安装

```bash
conda activate helios
pip install -U opencv-python-headless pillow
pip install -U transnetv2-pytorch
pip install -U transformers accelerate sentencepiece modelscope
```

---

## 2) 模型准备

### Qwen3-VL-8B-Instruct（ModelScope）

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='/root/autodl-fs/Qwen3-VL-8B-Instruct')"
```

### TransNetV2

通过 `transnetv2-pytorch` 直接调用，不需要额外手工传权重路径。

---

## 3) 运行命令（推荐）

```bash
conda activate helios
python /root/autodl-tmp/Helios/example_memory_selector/Selector_VLM/tools/offload_data/build_metadata_minimal.py \
  --videos_dir /root/autodl-tmp/Helios/example_memory_selector/seedance/video_light_change/videos \
  --out_json /root/autodl-tmp/Helios/example_memory_selector/seedance/video_light_change/metadata.json \
  --seg_method scenedetect \
  --prompt_mode qwen_vl \
  --qwen_vl_prompt_source visual_batch \
  --qwen_vl_model_path /root/autodl-fs/Qwen3-VL-8B-Instruct \
  --qwen_vl_batch_max_new_tokens 1024
```

---

## 4) 输出格式

`metadata.json` 每条记录包含：

- `id`, `path`
- `fps`, `num_frames`, `resolution`
- `chunking`（包含 `chunk_frames`）
- `segments`（`start_sec/end_sec/start_chunk/end_chunk/prompt`）
- `notes`

---

## 5) 关于 `build_metadata_minimal.py`

- 这是当前唯一保留的元数据脚本。
- 可以直接替代你们之前的冗余版本。
- 覆盖你现在的主链路：**TransNetV2 + Qwen3-VL**。

