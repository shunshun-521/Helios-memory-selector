# Helios-LMC: 基于 Helios 的长视频生成改进版

本仓库基于 [Helios 原版](https://github.com/PKU-YuanGroup/Helios)，我们在其基础上实现了一个面向长视频一致性的改进方案：

- `VLM Selector`：用于选择参考的GAP历史帧作为condition注入
- `LongMemory`：按需唤醒的长程记忆机制，会从codebook中利用VLM selector来选帧
- `Codebook`：根据最近生成的chunk的最后一帧，与上一个chunk的中间帧的DINOv2 embedding的点积，决定相似度。

目标是让模型在Helios的stage 1阶段（将双向变化为单向AR diffusion模型并学习历史依赖）的训练能够训练出ref-attn，使得长时段推理生成中利用参考的condition更稳定地保持人物、场景和语义连续性，同时利用codebook控制推理开销。值得注意的是，**vlm选帧**、**codebook的refresh和insert**和**longmemory检索机制的启动**都不是训练得来的，而是通过各种方式验证其有效得来的。

---

## 1. 仓库中的关键文件


- Ref-Attn 训练：`train_helios_bolt.py`
- Ref-Attn 训练脚本： `scripts/training/train_bolt.sh`
- Ref-Attn 训练配置示例：`scripts/training/configs/bolt_ref_attn.yaml`

- VLM 选帧模块：`helios/modules/select_frames_vlm.py`
- VLM 选帧模块的设计（包含CLIP+ITS的粗粒度选帧方法）：`md/CLIP-base-selector-vlm.md` （待修改查验）

- LongMemory记忆检索启动模块：`helios/modules/long_memory.py`
- codebook：`helios/modules/memory_bank.py`
- LongMemory 和 codebook 的模块设计： `/root/autodl-tmp/Helios/md/LongMemory.md`（待修改查验）

- 推理入口（利用已训好的Ref-Attn权重和Helios stage 1的权重推理）：`infer_helios_bolt.py`

- 我们构建训练集的pipeline全流程：`example_memory_selector/README.md`

- 【治宇】：`lm_tau_cut.md`
- 【虎威】：`vlm_competence_benchmark.md`


---

## 1. 快速开始
数据集在https://huggingface.co/datasets/shunshun-521/seedance-zip
### 1.1 环境准备

```bash
# 1. Create conda environment
conda create -n helios python=3.11.2 -y
conda activate helios
bash install.sh

# 2. Install PyTorch (adjust for your CUDA version)
# CUDA 12.6
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu126
# CUDA 12.8
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128
# CUDA 13.0
pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130

# 3. Install dependencies
bash install.sh

# 4. Download models 由于我们的训练只用到stage1，所以只下载base的权重
pip install modelscope
modelscope download BestWishYSH/Helios-Base --local_dir BestWishYSH/Helios-Base
```

### 1.2 基础Helios推理（先确认主干可跑）

```bash
bash scripts/inference/helios-base_t2v.sh
```

interactive式推理
```bash
bash scripts/inference/experiment_interactive/helios-base_t2v.sh
```

### 1.3 Ref-Attn 的训练与推理 （小实验不涉及）

```bash
bash scripts/training/train_bolt.sh
```

请参考：

- `infer_helios_bolt.py` 的 selector / long memory 参数
- `scripts/training/configs/bolt_ref_attn.yaml` 中 `selector_training` 与 `selector_inference` 配置



---




