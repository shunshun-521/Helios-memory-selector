# Helios-LMC: 基于 Helios 的长视频生成改进版

本仓库基于 [Helios 原版](https://github.com/PKU-YuanGroup/Helios)，我们在其基础上实现了一个面向长视频一致性的改进方案：

- `VLM Selector`：用于语义级参考帧选择（替代/增强 CLIP+ITS）
- `LongMemory`：按需唤醒的长程记忆机制
- `Codebook`：在线更新的视觉原型库（cut-gated 检索）

目标是让模型在长时段生成中更稳定地保持人物、场景和语义连续性，同时控制推理开销。

---

## 1. 项目定位

- **上游基线**：Helios 官方实现（14B 实时长视频生成）
- **我们的增量改动**：不改主干 DiT 训练范式，重点改造参考检索与记忆路径
- **核心价值**：
  - 长视频段间切换时减少漂移
  - 在语义突变场景下提升参考帧命中质量
  - 保持工程可插拔：可回退到原始 CLIP+ITS

---

## 2. 方法概览

### 2.1 VLM Selector（语义精排）

- 在候选历史帧上，先做粗筛，再用 VLM 打分精排
- 输出接口保持与原 `select_gap_frames` 一致：`(selected_latents, selected_indices)`
- 支持两种策略：
  - `its`：概率采样（与原 CLIP+ITS 习惯一致）
  - `topk`：确定性选择（更利于复现）

对应设计文档：`md/CLIP-base-selector-vlm.md`

### 2.2 LongMemory + Codebook（按需检索）

- 用 chunk 边界信号判断是否发生“明显切换”
- 非切换时走短程记忆，切换时才触发全局 codebook 检索
- codebook 按相似度进行 `merge / insert` 在线更新，支持容量控制

对应设计文档：`md/LongMemory.md`

---

## 3. 仓库中的关键文件

- 推理入口：`infer_helios_bolt.py`
- Ref-Attn 训练：`train_helios_bolt.py`
- Selector 训练：`train_helios_selector_vlm.py`
- VLM 选帧模块：`helios/modules/select_frames_vlm.py`
- 长记忆模块：`helios/modules/long_memory.py`
- codebook/记忆管理：`helios/modules/memory_bank.py`
- 训练配置示例：`scripts/training/configs/bolt_ref_attn.yaml`

---

## 4. 快速开始（团队协作最小流程）

### 4.1 环境准备

```bash
conda create -n helios_lmc python=3.11 -y
conda activate helios_lmc
bash install.sh
```

### 4.2 基础推理（先确认主干可跑）

```bash
bash scripts/inference/helios-base_t2v.sh
```

### 4.3 开启 VLM + LongMemory 路径（示例）

请参考：

- `infer_helios_bolt.py` 的 selector / long memory 参数
- `scripts/training/configs/bolt_ref_attn.yaml` 中 `selector_training` 与 `selector_inference` 配置

建议先用小样本视频做 smoke test，再逐步扩展到完整评估集。

---

## 5. 训练建议

### 5.1 VLM Selector 蒸馏训练

- 先离线准备 selector 数据（候选帧 + soft/hard 标签）
- 训练目标建议采用 soft + hard 组合（KL + CE）
- 先用 `topk` 路径验证稳定性，再尝试 `its` 做多样性对比

### 5.2 Bolt Ref-Attn 训练

- 冻结 DiT 主干，仅训练参考注意力模块
- 先在较小 `k_select` 和较低 candidate 数量下验证收敛
- 每轮固定验证脚本记录关键指标，避免“看起来变好但不可复现”

---

## 6. 协作规范（建议）

- 分支命名：
  - `feat/vlm-selector-*`
  - `feat/long-memory-*`
  - `exp/*`（临时实验）
- 提交信息建议：
  - `feat: add cut-gated codebook retrieval`
  - `fix: align selector output signature with clip_its`
  - `chore: clean unused logs and tmp artifacts`
- PR 必须包含：
  - 改动动机
  - 关键实验设置
  - 最小可复现命令
  - 对比结果（至少 1 个 baseline）

---

## 7. 上传 GitHub 前检查清单

- [ ] 删除/忽略训练日志、大体积中间产物、临时脚本
- [ ] 路径去本地化（不要包含个人机器绝对路径）
- [ ] 配置里敏感信息脱敏
- [ ] `README` 中提供可直接运行的最小命令
- [ ] 至少完成一次从空环境到推理成功的自测

---

## 8. 致谢

感谢 Helios 原作者团队开源高质量基线。我们在其基础上进行研究性改进，欢迎 issue / PR 交流与共建。

