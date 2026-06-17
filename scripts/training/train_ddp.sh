#!/bin/bash
export HELIOS_LOG_CHOICE_IDX_EVERY=10   # 每 10 个 step 打一次 choice_idx；改成 1 则每步都打
export OMP_NUM_THREADS=20
export WANDB_MODE="offline"
export WANDB_API_KEY=""
export TOKENIZERS_PARALLELISM=true
export HELIOS_DEBUG_SEGMENT_PROMPT=1
export PYTHONUNBUFFERED=1
export ACCELERATE_LOG_LEVEL=info

export OMNISTORE_LOAD_STRICT_MODE=0
export OMNISTORE_LOGGING_LEVEL=ERROR
export HF_ENABLE_PARALLEL_LOADING="${HF_ENABLE_PARALLEL_LOADING:-yes}"
export HF_PARALLEL_LOADING_WORKERS="${HF_PARALLEL_LOADING_WORKERS:-1}"
#################################################################
## Torch
#################################################################
export TOKENIZERS_PARALLELISM=false
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export TORCH_NCCL_ENABLE_MONITORING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,garbage_collection_threshold:0.9"
#################################################################


#################################################################
## NCCL
#################################################################
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_TIMEOUT=3600000

export NCCL_DEBUG=WARN
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


#################################################################
## ACCELERATE CONFIG (单机单卡)
#################################################################
MASTER_ADDR="127.0.0.1"
export MASTER_PORT=29531
NUM_MACHINES=1
MACHINE_RANK=0
NUM_PROCESSES_PER_MACHINE=1

ACCELERATE_ARGS="--num_machines $NUM_MACHINES --machine_rank $MACHINE_RANK --num_processes $((NUM_PROCESSES_PER_MACHINE*NUM_MACHINES)) --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"

echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"

cd /root/autodl-tmp/Helios

LOG_DIR="/root/autodl-fs/output/6_1_post"
mkdir -p "${LOG_DIR}/logs"
LOG_FILE="${LOG_FILE:-/root/autodl-fs/output/6_1_post/logs/1.log}"
echo "[INFO] Logging to: ${LOG_FILE}"

# NOTE: Use helios env's accelerate to avoid falling back to base.
/root/miniconda3/bin/conda run --no-capture-output -n helios accelerate launch \
    $ACCELERATE_ARGS \
    train_helios.py \
    --config /root/autodl-tmp/Helios/scripts/training/configs/stage_1_post.yaml \
    2>&1 | tee -a "${LOG_FILE}"
