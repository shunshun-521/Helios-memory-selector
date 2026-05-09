# lm_tau_cut 交接文档（给执行同学）

本文用于指导你们基于约 300 条视频样本，**数据驱动**地给出 `lm_tau_cut` 的候选范围。  
目标是不再“拍脑袋猜阈值”，而是先统计切镜处相邻 chunk 的视觉距离分布，再从分布中取区间。

---

## 1. 你手里的数据和它们分别做什么

我们目前的300条数据集，注意360度旋转一般没有切镜，只有aba和light_change会有，huggingface链接如下：
https://huggingface.co/datasets/shunshun-521/seedance-zip

- 视频目录（示例）：`/root/autodl-tmp/seedance/video_aba/videos/sample_00002.mp4`
- 元数据：`/root/autodl-tmp/Selector_VLM/example_long/metadata_sample.jsonl`
- 预处理后的 latent（`.pt`）

你们要做的是：
- 从 metadata 找到“切镜边界”的相邻 chunk 对；
- 对每个边界取两帧（你们当前定义：**前一个 chunk 的中间帧** vs **后一个 chunk 的最后一帧**）；
- 用与现有代码一致的 DINOv2 编码方式提 embedding；
- 计算距离 `dist`，汇总所有切镜样本；
- 给出 `lm_tau_cut` 的候选区间。

---

## 2. 必须先统一的基础定义（避免口径不一致）

### 2.1 chunk 是什么

`metadata_sample.jsonl` 中 `chunking` 字段已给出：

- `num_latent_frames_per_chunk = 9`
- `t_downsample = 4`
- `chunk_frames = 33`
- `fps = 24`

所以一个 chunk 的时长约为：

- `chunk_seconds = chunk_frames / fps = 33 / 24 = 1.375 秒`

### 2.2 segment 和 chunk 的关系

`segments` 里的 `start_chunk`、`end_chunk` 建议按左闭右开理解：`[start_chunk, end_chunk)`。

例如：
- 段1：`start_chunk=0, end_chunk=2` -> chunk `0,1`
- 段2：`start_chunk=2, end_chunk=4` -> chunk `2,3`

那么边界在 `2`，跨边界相邻对是 `(1, 2)`。

### 2.3 “切镜边界的相邻 chunk 对”怎么取

对每条样本，遍历相邻 segment：

- 边界 `b = segments[i].end_chunk`（也应等于 `segments[i+1].start_chunk`）
- 取 pair：`(prev_chunk, next_chunk) = (b-1, b)`

只要 `prev_chunk >= 0` 且 `next_chunk < num_chunks` 就纳入统计。

---

## 3. 与现有代码一致的 DINO 编码口径

你们需要和训练/验证里 LongMemory 的方式一致：

1) 从 latent 解码帧（先 VAE decode 到 RGB 图）  
2) 把 PIL 图送入 DINOv2 提 embedding  
3) embedding 在 CPU 上做相似度

代码对应位置（供核对）：
- `Helios/helios/modules/extract_feature.py`
  - `decode_middle_frame(...)`
  - `decode_last_frame(...)`
  - `extract_tail_embedding(...)`
  - `DINOv2.extract_visual_features(...)`
- `Helios/train_helios_bolt.py`
  - `dist_prev = 1.0 - cosine_similarity(prev_tail_emb, tail_emb)`

> 注意：当前线上 gate 用的是“tail vs tail”。  
> 你们这次实验定义是“prev-middle vs next-last”，可以做，但报告里要明确这是**实验口径**。

---

## 4. 距离 dist 的定义（建议）

建议按与现有 gate 一致的量纲定义：

- 先做余弦相似度：`sim = cosine(emb_a, emb_b)`
- 再转距离：`dist = 1 - sim`

如果你们说“点积”，请确保先 L2 归一化后再点积，这样点积就等于余弦相似度：

- `sim = dot(norm(emb_a), norm(emb_b))`
- `dist = 1 - sim`

这样才能和现有 `lm_tau_cut` 的语义一致。

---

## 5. 每个边界样本的记录格式（建议）

建议每条记录至少包含：

- `sample_id`
- `boundary_chunk`（即 `b`）
- `prev_chunk = b-1`
- `next_chunk = b`
- `prev_seg_idx`
- `next_seg_idx`
- `dist`
- `sim`
- `fps`
- `chunk_frames`

最后导出一个 `cut_dist_records.csv/jsonl`，用于画图和算统计量。

---

## 6. 如何从分布给出 lm_tau_cut 候选范围

你可以同时给 3 套“候选区间”，交给训练同学网格试验：

### A. 保守覆盖法（不做离群剔除）

- 下界：`P10`
- 上界：`P25`

用途：尽量只在明显切镜触发 slow。

### B. IQR 去离群后区间（推荐）

1) 计算 `Q1, Q3, IQR = Q3-Q1`
2) 保留 `[Q1-1.5*IQR, Q3+1.5*IQR]` 内样本
3) 在保留样本上取：
   - 候选下界：`Q1`
   - 候选中位：`median`
   - 候选上界：`Q3`

### C. 极值参考（仅辅助，不直接用）

- `min(dist)`、`max(dist)` 仅作观察，不建议直接当阈值。

---

## 7. 建议最终交付物（给训练同学）

你们跑完后请交付：

1) `cut_dist_records.csv`（每个切镜边界一条）  
2) `cut_dist_summary.json`（统计量）：
   - `count, mean, std, min, p10, p25, p50, p75, p90, max`
   - `iqr_filtered_count`
   - `iqr_filtered_p25/p50/p75`
3) 1 张分布图（直方图 + 箱线图）  
4) 一行建议（示例）：
   - `建议将 lm_tau_cut 搜索在 [x, y]，中心值 z`

---

## 8. 常见坑（一定看）

- 不要把“秒”直接映射成 chunk 索引，优先用 metadata 的 `start_chunk/end_chunk`。
- `segments` 的 `end_chunk` 不是该段最后一个 chunk 下标，而是右边界（建议按左闭右开）。
- 必须统一 embedding 和距离口径，否则不同人结果不可比较。
- 统计时只用“切镜边界 pair”，不要混入普通连续 chunk。

---

## 9. 你们这次实验口径（一句话版）

“在每个切镜边界 `(b-1, b)`，取 `chunk b-1` 的中间帧和 `chunk b` 的最后一帧，按 Helios 现有 DINOv2 解码+编码方式提 embedding，计算 `dist=1-cos`，汇总分布后给出 `lm_tau_cut` 的候选范围。”

