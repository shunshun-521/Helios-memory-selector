# Selector_VLM: Metadata Processing Notes

本目录主要放两类内容：

- `example/`、`example_long/`：处理好的示例产物
- `tools/offload_data/build_metadata_minimal.py`：从 `videos/*.mp4` 构建 `metadata.json` 的脚本

说明：

- 目前只保留 `build_metadata_minimal.py`；
- 流程固定为“TransNetV2 分镜 + Qwen3-VL visual_batch 写段落 prompt”。

当前脚本保留的唯一路径：

- 分镜：`TransNetV2`
- 分段 prompt：`Qwen3-VL-8B-Instruct`（`visual_batch`）

---

## 1. 环境依赖

```bash
conda activate helios
pip install -U modelscope transformers accelerate sentencepiece
pip install -U opencv-python-headless pillow transnetv2-pytorch
```

> 说明：`transnetv2-pytorch` 通常自带/自动处理权重加载，因此一般不需要你手动指定权重文件路径。

---

## 2. ModelScope 下载（Qwen-VL）

```bash
python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen3-VL-8B-Instruct', local_dir='/root/autodl-fs/Qwen3-VL-8B-Instruct')"
```

下载后在命令里传：

```bash
--qwen_vl_model_path /root/autodl-fs/Qwen3-VL-8B-Instruct
```

---

## 3. ModelScope 与 TransNetV2 说明

TransNetV2 在本流程里通过 `transnetv2-pytorch` 调用，核心是：

```python
from transnetv2_pytorch import TransNetV2
```

如果你所在环境必须统一走 ModelScope 镜像管理，建议做法是：

1. 先确保 `transnetv2-pytorch` 可安装；
2. 将该依赖所需文件缓存到你们统一存储；
3. 用固定环境镜像复用，避免每台机器重复下载。

---

## 4. 运行命令（当前推荐）

```bash
conda activate helios
python /root/autodl-tmp/Helios/example_memory_selector/Selector_VLM/tools/offload_data/build_metadata_minimal.py \
  --videos_dir /root/autodl-tmp/Helios/example_memory_selector/seedance/video_aba/videos \
  --out_json /root/autodl-tmp/Helios/example_memory_selector/seedance/video_aba/metadata.json \
  --seg_method scenedetect \
  --prompt_mode qwen_vl \
  --qwen_vl_prompt_source visual_batch \
  --qwen_vl_model_path /root/autodl-fs/Qwen3-VL-8B-Instruct \
  --qwen_vl_batch_max_new_tokens 1024
```

---

## 5. 输出说明

输出 `metadata.json` 每条样本包含：

- `id`, `path`
- `fps`, `num_frames`, `resolution`
- `chunking`
- `segments`（含 chunk 对齐后的边界和段落 prompt）
- `notes`

