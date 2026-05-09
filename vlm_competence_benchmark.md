# VLM 选帧能力小实验指南（vlm_competence_benchmark）

这份文档给做小实验的同学，目标是两件事：

1. 对比“一个 chunk 怎么喂给 VLM 更合理”：  
   - 单帧（middle）  
   - 单帧（first）  
   - 单帧（last）  
   - 多帧（整段 chunk）
2. 给出一个可复现的**定量 benchmark**，而不只靠人眼看 ID 是否一致。

---

## 1. 先准备模型

实验前请先下载这两个模型：

- `Qwen2.5-VL-3B-Instruct`
- `BestWishYSH/Helios-Base`（至少需要其 `vae`）

推荐（ModelScope）：

```bash
conda activate helios
pip install -U modelscope

python -c "from modelscope import snapshot_download; snapshot_download('Qwen/Qwen2.5-VL-3B-Instruct', local_dir='/root/autodl-fs/Qwen2.5-VL-3B-Instruct')"
python -c "from modelscope import snapshot_download; snapshot_download('BestWishYSH/Helios-Base', local_dir='/root/autodl-fs/BestWishYSH/Helios-Base')"
```

---

## 2. 实验脚本与输入

脚本：`tools/compare_selector_clipits_vs_vlm_zeroshot.py`

输入是 `latents_short/*.pt`，通常来自 `tools/offload_data/get_short-latents.py`。

---

## 3. 运行命令（核心）

以下命令都在仓库根目录执行。

公共参数（建议）：

- `--feature_folder /root/autodl-tmp/Selector_VLM/example_long/latents_short`
- `--vae_path /root/autodl-fs/BestWishYSH/Helios-Base`
- `--vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct`
- `--choice_mode all`
- `--k_select 1`
- `--alpha 0.6 --power 2.0 --min_chunk_distance 2`
- `--print_topn 8`（输出每种方法 top8 排序）

### 3.1 VLM 单帧（middle，默认）

```bash
python tools/compare_selector_clipits_vs_vlm_zeroshot.py \
  --feature_folder /root/autodl-tmp/Selector_VLM/example_long/latents_short \
  --vae_path /root/autodl-fs/BestWishYSH/Helios-Base \
  --vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
  --device cuda --vlm_device cuda \
  --choice_mode all --k_select 1 \
  --alpha 0.6 --power 2.0 --min_chunk_distance 2 \
  --vlm_single_frame_position middle \
  --vlm_rank_mode topk --print_topn 8
```

### 3.2 VLM 单帧（first）

```bash
python tools/compare_selector_clipits_vs_vlm_zeroshot.py \
  --feature_folder /root/autodl-tmp/Selector_VLM/example_long/latents_short \
  --vae_path /root/autodl-fs/BestWishYSH/Helios-Base \
  --vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
  --device cuda --vlm_device cuda \
  --choice_mode all --k_select 1 \
  --alpha 0.6 --power 2.0 --min_chunk_distance 2 \
  --vlm_single_frame_position first \
  --vlm_rank_mode topk --print_topn 8
```

### 3.3 VLM 单帧（last）

```bash
python tools/compare_selector_clipits_vs_vlm_zeroshot.py \
  --feature_folder /root/autodl-tmp/Selector_VLM/example_long/latents_short \
  --vae_path /root/autodl-fs/BestWishYSH/Helios-Base \
  --vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
  --device cuda --vlm_device cuda \
  --choice_mode all --k_select 1 \
  --alpha 0.6 --power 2.0 --min_chunk_distance 2 \
  --vlm_single_frame_position last \
  --vlm_rank_mode topk --print_topn 8
```

### 3.4 VLM 多帧（chunk video）

```bash
python tools/compare_selector_clipits_vs_vlm_zeroshot.py \
  --feature_folder /root/autodl-tmp/Selector_VLM/example_long/latents_short \
  --vae_path /root/autodl-fs/BestWishYSH/Helios-Base \
  --vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
  --device cuda --vlm_device cuda \
  --choice_mode all --k_select 1 \
  --alpha 0.6 --power 2.0 --min_chunk_distance 2 \
  --vlm_use_chunk_video --vlm_video_max_frames 9 \
  --vlm_rank_mode topk --print_topn 8
```

> `CLIP+ITS` 基线已在同一脚本里同时输出，无需单独跑。

---

## 4. 你关心的问题 1：首帧/尾帧输入怎么做

现在脚本已支持参数：

- `--vlm_single_frame_position middle|first|last`

逻辑是：

- `middle`：原逻辑（每个 chunk 用中间帧）
- `first`：每个 chunk 取首帧代表
- `last`：每个 chunk 取尾帧代表

不需要再手改代码即可跑三种单帧策略。

---

## 5. 你关心的问题 2：定量 benchmark 怎么做

你的建议非常合理：用 DINOv2 embedding 做相似度打分。

### 5.1 统一评估口径

对每个 target chunk（`choice_idx`）和每种方法 `m`：

1. 拿该方法输出的候选排序 top8（脚本 `--print_topn 8`）
2. 取前4个候选 chunk：`c1..c4`
3. 取 target chunk 的**中间帧**做 DINOv2 embedding，记为 `e_tgt`
4. 对每个候选 chunk 取其代表帧（建议也用中间帧）得 `e_ci`
5. 相似度：`sim_i = dot(norm(e_tgt), norm(e_ci))`
6. 方法分数：`Score_m = mean(sim_1..sim_4)`

分数越高，说明该方法在“语义/身份一致性”上更接近 target。

### 5.2 推荐比较方法集合

- CLIP+ITS
- VLM single-middle
- VLM single-first
- VLM single-last
- VLM multi-frame

### 5.3 最终汇总

按样本和全局输出：

- 每方法 `Score_m` 的均值、方差
- 各方法 pairwise 差值（例如 `multi-frame - single-last`）
- 排名表（高到低）

---

## 6. 实验记录模板（建议）

| method | top8_source | top4_dino_mean | notes |
|---|---|---:|---|
| clip_its | combined score rank | 0.xxx | baseline |
| vlm_single_middle | vlm score rank | 0.xxx | single frame |
| vlm_single_first | vlm score rank | 0.xxx | single frame |
| vlm_single_last | vlm score rank | 0.xxx | single frame |
| vlm_multi_frame | vlm score rank | 0.xxx | chunk video |

---

## 7. 结论建议（如何读结果）

- 如果 `vlm_multi_frame` 稳定最高，说明多帧上下文确实提供有效信息。
- 如果 `single-last` 高于 `single-middle`，说明“尾帧更贴近 target chunk 过渡状态”。
- 如果 `single-first` 最差，说明首帧在这个任务里代表性较弱（常见于动作连续变化场景）。

## 注意的点

1./root/autodl-tmp/Selector_VLM/example_long/latents_short下的数据只有3条。暂时不够，预计需要300条左右的数据，会稍后上传

