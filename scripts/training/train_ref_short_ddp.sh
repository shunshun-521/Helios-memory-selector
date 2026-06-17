#!/bin/bash
# =============================================================================
# Ref-Short Stage1 post 训练启动脚本（DDP / accelerate）
# =============================================================================
# 作用:
#   启动 train_helios_ref_short.py → train_helios.main
#   - patch_ref 序列注入 + SelectorRuntime（训练 force_slow / 全 slow 选帧）
#   - 训练配方与 stage_1_post 对齐（抗漂移、guidance_cross_attn、LoRA 等）
#   - 训练期 validation 与训练 forward 一致（build_validation_ctx）
#
# 默认配置:
#   scripts/training/configs/stage_1_ref_short.yaml
#
# 数据:
#   先用 tools/offload_data/get_short-latents.sh 生成 latents_short，
#   并在 yaml 的 data_config.instance_data_root 指向该目录。
#
# 注意:
#   - ref-short 训练路径当前要求 train_batch_size=1（每卡）
#   - 需配置 selector_training.vlm_model_path
#
# 用法:
#   cd /root/autodl-tmp/Helios
#   bash scripts/training/train_ref_short_ddp.sh
#
#   # 覆盖配置 / 输出目录
#   CONFIG_YAML=... OUTPUT_DIR=... bash scripts/training/train_ref_short_ddp.sh
#
#   # 多卡（示例 2 卡，总进程数=2，仍须 yaml 里 train_batch_size=1）
#   NUM_PROCESSES_PER_MACHINE=2 bash scripts/training/train_ref_short_ddp.sh
# =============================================================================

set -euo pipefail

export HELIOS_LOG_CHOICE_IDX_EVERY="${HELIOS_LOG_CHOICE_IDX_EVERY:-10}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-true}"
export PYTHONUNBUFFERED=1
export ACCELERATE_LOG_LEVEL="${ACCELERATE_LOG_LEVEL:-info}"

export OMNISTORE_LOAD_STRICT_MODE=0
export OMNISTORE_LOGGING_LEVEL=ERROR
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-yes}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-1}"

#################################################################
## Torch
#################################################################
export TORCH_LOGS="${TORCH_LOGS:-+dynamo,recompiles,graph_breaks}"
export TORCHDYNAMO_VERBOSE="${TORCHDYNAMO_VERBOSE:-1}"
export TORCH_NCCL_ENABLE_MONITORING="${TORCH_NCCL_ENABLE_MONITORING:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,garbage_collection_threshold:0.9}"

#################################################################
## NCCL（多机多卡时按需修改）
#################################################################
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_SOCKET_TIMEOUT=3600000
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_SHM_DISABLE=0
export NCCL_P2P_LEVEL=NVL
export NCCL_PXN_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22

#################################################################
## Accelerate
#################################################################
HELIOS_ROOT="/root/autodl-tmp/Helios"
cd "${HELIOS_ROOT}"

CONFIG_YAML="${CONFIG_YAML:-${HELIOS_ROOT}/scripts/training/configs/stage_1_ref_short.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-fs/output/ref_short_post_6_1}"

MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29532}"
NUM_MACHINES="${NUM_MACHINES:-1}"
MACHINE_RANK="${MACHINE_RANK:-0}"
NUM_PROCESSES_PER_MACHINE="${NUM_PROCESSES_PER_MACHINE:-1}"

ACCELERATE_ARGS="--num_machines ${NUM_MACHINES} --machine_rank ${MACHINE_RANK} --num_processes $((NUM_PROCESSES_PER_MACHINE * NUM_MACHINES)) --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT}"

mkdir -p "${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_FILE:-${OUTPUT_DIR}/logs/train_$(date +%m%d_%H%M%S).log}"

echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"
echo "[INFO] Config: ${CONFIG_YAML}"
echo "[INFO] Output: ${OUTPUT_DIR}"
echo "[INFO] Logging to: ${LOG_FILE}"

/root/miniconda3/bin/conda run --no-capture-output -n helios accelerate launch \
    ${ACCELERATE_ARGS} \
    train_helios_ref_short.py \
    --config "${CONFIG_YAML}" \
    2>&1 | tee -a "${LOG_FILE}"
