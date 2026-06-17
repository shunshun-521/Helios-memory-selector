# 5_16 stage1_init 代码与训练链路核查

## 结论（先看）

- `stage1_init.yaml` 对应的数据加载链路和 prompt 条件输入主流程是通的，整体没有发现“明显跑错数据/跑错 prompt”的硬错误。
- 训练中你看到的 `[DEBUG][seg-prompt] ... choice_idx=... bounds=...` 与代码逻辑一致，说明当前 batch 的 chunk 级 prompt 选择在生效。
- 目前日志并不能支持“已经不收敛”的结论，更准确是“loss 抖动较大、样本少、还在早期段”；你提到 500 step validation 明显变好，这与当前观测不冲突。
- 建议按你的计划继续跑到 1000 step，但在 1000 step 做一次结构化检查（见文末 checklist），再决定继续还是改参。

---

## 1) 数据加载链路是否正确

配置来自 `scripts/training/configs/stage_1_init.yaml`：

- `data_config.use_stage1_dataset: true`
- `data_config.instance_data_root` 指向 `.../video_light_change/latents_short`
- `training_config.train_batch_size: 2`

训练入口 `train_helios.py` 在 `use_stage1_dataset=true` 时会走：

- `helios/dataset/dataloader_history_latents_dist.py`
- `BucketedFeatureDataset` + `BucketedSampler` + `collate_fn`

核查结果：

- `latents_short` 目录存在 100 个 `.pt` 文件，文件名格式与解析逻辑一致（`*_numframe_h_w.pt`）。
- 数据集构建时会过滤 `num_frame < 121`，你当前数据是 `241`，满足条件。
- 单分辨率过滤按 `(384,640)/(192,320)/(96,160)` 允许，你当前样本是 `384x640`，匹配。

结论：**数据路径和采样桶逻辑正常**。

---

## 2) prompt 调用与训练条件是否正确

### 2.1 训练 prompt（真正参与反向传播）

`dataloader_history_latents_dist.py` 的 `__getitem__` 逻辑：

- 先按 `choice_idx` 采样 chunk；
- 若样本内存在 `prompt_embeds_by_segment + segments`，则按 `start_chunk <= choice_idx < end_chunk` 选对应 segment 的 `prompt_embed`；
- 将其放入 batch 的 `prompt_embeds`，训练主循环直接使用 `batch["prompt_embeds"]`。

抽样检查单个 `.pt`（`sample_00042...`）：

- `vae_latent`: `(7,16,9,48,80)`
- `prompt_embed`: `(512,4096)`
- `prompt_embeds_by_segment`: `(2,512,4096)`
- `segments`: 2 段，含 `start_chunk/end_chunk/prompt_raw`

与日志中的调试输出一致（`[DEBUG][seg-prompt] ... choice_idx ... bounds ...`）。

结论：**训练时用的是分段 prompt embedding，且映射逻辑已生效**。

### 2.2 validation prompt（用于看效果）

`train_helios.py::_build_validation_jobs` 使用的是 `validation_config.validation_prompts`：

- 你的配置里是 3 段文本；
- `use_interpolate_prompt: true`，因此 validation 走“多段 prompt 插值”路径；
- 这条路径与训练的 `prompt_embeds_by_segment` 是两条不同来源，但都合理（训练靠预编码 embedding，验证靠文本推理）。

结论：**训练 prompt 与验证 prompt 来源不同是设计使然，不是错误**。

---

## 3) 当前配置下的风险点（不是硬错误）

1. **LoRA 排除未生效**
   - 日志有警告：`exclude_modules={'up','down'} but no modules were excluded`
   - 说明你希望排除的模块名没有匹配上实际层名。
   - 影响：不是 crash，但会让可训练范围比预期更大。

2. **训练样本量仅 100**
   - 在大模型+高维条件下，step 级 loss 抖动大非常常见；
   - 这时更应看固定 validation 的视频趋势，而不是只盯单点 loss。

3. **`missing keys` 提示**
   - `patch_short/mid/long` missing 在当前配置下大概率是可预期初始化项，不像是灾难性加载失败；
   - 需要结合输出视频与是否出现 NaN/爆炸来判断是否异常。目前未见 NaN/inf 报错。

---

## 4) 是否现在停？还是跑到 1000 step？

结合你补充“500 step 的 validation 逐步变正确”，建议：

- **继续跑到 1000 step（不必立刻停）**。
- 但 1000 step 到点后，按以下 checklist 判断：

### 1000 step 检查清单

- 固定同一组 validation prompt + seed，对比 `step=500` 与 `step=1000`：
  - 时序连贯性是否继续提升；
  - 主体一致性是否提升；
  - 是否出现模式坍塌（总生成同质画面）。
- loss 观察用“窗口均值”而非单点（例如最近 100 step 均值）。
- 检查 `grad_norm` 是否长期极低且无变化，或偶发异常尖峰。
- 如 1000 step 效果仍无改善，再优先处理：
  1) LoRA `exclude_modules` 命名匹配；
  2) 学习率（可对比 `5e-5` vs `2e-5`）；
  3) 数据清洗（caption-视频一致性）。

---

## 5) 关于你提到的“另一数据集 800 step 到 0.0010”

这个对比很有价值，但需要同口径比较，否则容易误判：

- loss 定义是否一致（flow loss / 其他项是否混合）；
- batch size、分辨率、帧数、LoRA 可训练范围是否一致；
- 数据规模与难度是否一致。

如果你把那次训练的日志路径给我，我可以在这个文档后续加一节“同口径对照结论”，直接给出是否需要改参的判定阈值。

