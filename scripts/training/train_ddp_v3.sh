#!/bin/bash
export OMP_NUM_THREADS=15
export WANDB_MODE="offline"
export WANDB_API_KEY=""
export TOKENIZERS_PARALLELISM=true

export OMNISTORE_LOAD_STRICT_MODE=0
export OMNISTORE_LOGGING_LEVEL=ERROR
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
export MASTER_PORT=29500
NUM_MACHINES=1
MACHINE_RANK=0
NUM_PROCESSES_PER_MACHINE=1

ACCELERATE_ARGS="--num_machines $NUM_MACHINES --machine_rank $MACHINE_RANK --num_processes $((NUM_PROCESSES_PER_MACHINE*NUM_MACHINES)) --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"

echo -e "\033[31mACCELERATE_ARGS: ${ACCELERATE_ARGS}\033[0m"

cd /root/autodl-tmp/Helios

accelerate launch \
    $ACCELERATE_ARGS \
    train_helios_v3.py \
    --config /root/autodl-tmp/Helios/scripts/training/configs/stage_1_init_v3.yaml \
    2>&1 | tee /root/autodl-tmp/log/v3_selector_v2_try_2026_03_31_0114.log
