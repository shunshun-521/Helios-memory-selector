# Example Memory Selector Pipeline

这个目录是我们在 `Helios` 基础上整理的**数据生成 + 元数据构建**最小可复现流程，目标是让协作者快速理解：

1. 如何用 Seedance 2.0 批量生成原始视频数据；
2. 如何把视频处理成 `metadata.json`（分镜 + 分段 prompt）；
3. 如何对接后续 `Selector_VLM / LongMemory / Ref-Attn` 训练与验证。

---

## 1. 目录组织

```text
example_memory_selector/
├── seedance/
│   ├── create_video.py
│   ├── run_all.sh
│   ├── prompt/
│   │   ├── 360.csv
│   │   ├── aba.csv
│   │   └── light_change.csv
│   ├── video_light_change/
│   │   ├── videos/*.mp4
│   │   ├── failed.jsonl
│   │   └── manifest.jsonl
│   └── ... (其它 case 输出目录)
└── Selector_VLM/
    ├── example_long/            # 处理后的范例数据
    ├── tools/offload_data/
    │   └── build_metadata_minimal.py
    └── README.md
```

---

## 2. Step A: 生成视频数据（Seedance）

`seedance/` 下是 Seedance 2.0 的批量生成脚本和原生长 prompt。

你可以直接在该目录执行（对应 `create_video.py` 的说明）：

```bash
export ARK_API_KEY=""
python create_video.py --csv prompt/light_change.csv --duration 10 --resolution 480p --ratio 16:9 --limit 50 --out-dir video_light_change
```

执行后会在 `video_light_change/` 下得到：

- `videos/`：生成的 mp4
- `failed.jsonl`：失败任务
- `manifest.jsonl`：成功任务与原始 prompt

> `metadata.json` / `metadata.jsonl` 不在这一步产出，它来自下一步 `build_metadata_minimal.py`。

---

## 3. Step B: 构建 metadata（TransNetV2 + Qwen3-VL）

`Selector_VLM/tools/offload_data/build_metadata_minimal.py` 为当前唯一元数据构建脚本：

- 分镜：仅 `TransNetV2`
- prompt 生成：仅 `Qwen-VL visual_batch`

### 推荐命令（与你当前流程对齐）

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

产出的 `metadata.json` 每条记录包含：

- 视频信息：`fps / num_frames / resolution`
- chunk 定义：`num_latent_frames_per_chunk / t_downsample / chunk_frames`
- 分段结果：`segments[{start_sec,end_sec,start_chunk,end_chunk,prompt}]`
- 备注：`segmentation_used=transnetv2`, `prompt_mode=qwen_vl`

---

## 4. Step C: 把MP4文件生成 `latents_short`（用于训练）

当 `metadata.json` 准备好后，可以用 `Helios/tools/offload_data/get_short-latents.sh/.py` 把视频离线编码成 `latents_short/*.pt`。

### 4.1 直接用 shell 脚本（推荐）

```bash
cd /root/autodl-tmp/Helios
CONDA_ENV_NAME=helios \
JSON_FILE=/root/autodl-tmp/Helios/data_memory_selector/apdcephfs_qy2/share_302508595/xiaodayang/seedance/seedance/video_360/metadata.json \
VIDEO_FOLDER=/root/autodl-tmp/Helios/data_memory_selector/apdcephfs_qy2/share_302508595/xiaodayang/seedance/seedance/video_360 \
OUTPUT_LATENT_FOLDER=/root/autodl-tmp/Helios/data_memory_selector/apdcephfs_qy2/share_302508595/xiaodayang/seedance/seedance/video_360/latents_short \
STRIDE=1 BATCH_SIZE=4 RESOLUTION=640 \
bash /root/autodl-tmp/Helios/tools/offload_data/get_short-latents.sh
```

> 关键点：`VIDEO_FOLDER` 要指向包含 `videos/` 子目录的根目录，因为 `metadata.json` 里 `path` 是 `videos/*.mp4` 相对路径。


输出结果：

- `latents_short/*.pt`，每条包含 `vae_latent`，以及 `prompt_raw/prompt_embed`
- 如果元数据含 `segments`，脚本也会额外写入按段信息（用于 chunk 对齐训练）

---

## 5. Step D: 训练与推理对接说明

有了 `metadata.json + latents_short` 后，你可以继续用于：

- Ref-Attn 训练与评估
- LongMemory / Codebook 推理检索验证

当前项目约定：

- **训练路径**：通过 `scripts/training/train_bolt.sh` 调用 `scripts/training/configs/bolt_ref_attn.yaml`。
- **训练粒度**：每个 step 优化一个 `chunk_idx` 对应 chunk，使用该 chunk 对应的 caption/prompt。
- **LongMemory/Codebook**：仅在推理中启用，不参与训练参数更新。

具体模型下载与依赖请看：`example_memory_selector/Selector_VLM/README.md`。

