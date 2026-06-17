#!/bin/bash
# =============================================================================
# BOLT Reference Attention 训练启动脚本
# =============================================================================
# 作用 / What:
#   启动 train_helios_bolt.py：冻结 Helios DiT，只训练 BoltReferenceAttentionLayers，
#   用预计算 VAE latent（.pt）做 Flow Matching MSE。
#
# 默认配置 / Config:
#   scripts/training/configs/bolt_ref_attn.yaml
#   可在 yaml 里改 data_paths、bolt 层范围、验证 CSV、validation_epochs 等；
#   命令行参数会覆盖 yaml 中同名项（见 train_helios_bolt.py --help）。
#
# 选帧方式 / Selector (训练时):
#   - 默认 CLIP+ITS（与历史行为一致）
#   - 使用 VLM 选帧训练 Ref-Attn 时追加例如:
#       --selector_type vlm \
#       --vlm_model_path /path/to/Qwen2.5-VL-3B-Instruct \
#       --vlm_lora_path /path/to/lora_epochXX
#
# 验证 / Validation:
#   由 yaml 中 validation_config 控制（每 N epoch 跑 interactive 推理）。
#   selector_type=vlm 时验证默认走 LongMemory+codebook（与 infer 对齐），
#   详见 train_helios_bolt.py 中 validate_epoch。
#
# 用法 / Usage:
#   cd /root/autodl-tmp/Helios
#   bash scripts/training/train_bolt.sh
#   bash scripts/training/train_bolt.sh --num_epochs 30 --output_dir /path/out
#   bash scripts/training/train_bolt.sh --selector_type vlm --vlm_model_path ...   # 任意额外 CLI 透传
#
# 日志 / Log:
#   默认 tee 到 ${OUTPUT_DIR}/train_MMDD_HHMM.log；请先改 OUTPUT_DIR 或 yaml 里 output_dir 一致。
# =============================================================================

set -euo pipefail

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"
export PYTHONUNBUFFERED=1

# 项目根（与仓库布局一致；若迁移请改此路径）
HELIOS_ROOT="/root/autodl-tmp/Helios"
cd "${HELIOS_ROOT}"

# 训练配置文件（可通过环境变量 CONFIG_YAML 覆盖）
CONFIG_YAML="${CONFIG_YAML:-${HELIOS_ROOT}/scripts/training/configs/bolt_ref_attn_5_17_stage1.yaml}"

# 日志输出目录：应与 yaml 里 output_dir 一致，或训练时用 --output_dir 覆盖
# 若仅改此处而不改 yaml，请同时传 --output_dir "$OUTPUT_DIR" 或在 yaml 中同步修改
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/output/5_20_ref}"
mkdir -p "${OUTPUT_DIR}"

LOG_FILE="${OUTPUT_DIR}/train_$(date +%m%d_%H%M%S).log"
echo "[INFO] Logging to: ${LOG_FILE}"
echo "[INFO] Config: ${CONFIG_YAML}"
echo "[INFO] Extra args: $*"

# 启动训练；"$@" 透传所有命令行参数给 train_helios_bolt.py
python train_helios_bolt.py \
    --config "${CONFIG_YAML}" \
    "$@" \
    2>&1 | tee "${LOG_FILE}"
