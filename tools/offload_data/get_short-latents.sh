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
export NCCL_IB_HCA=$ARNOLD_RDMA_DEVICE
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_TIMEOUT=3600000

export NCCL_DEBUG=WARN  # disable the verbose NCCL logs
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0  # was 1
export NCCL_SHM_DISABLE=0  # was 1
export NCCL_P2P_LEVEL=NVL

export NCCL_PXN_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_IB_TIMEOUT=22
#################################################################

#################################################################
## DIST (单机单卡)
#################################################################
MASTER_ADDR="127.0.0.1"
MASTER_PORT=29500
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo -e "\033[31mDISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}\033[0m"

#################################################################

cd /root/autodl-tmp/Helios
export PYTHONPATH=/root/autodl-tmp/Helios:$PYTHONPATH
export OMP_NUM_THREADS=4

CONDA_ENV_NAME="${CONDA_ENV_NAME:-helios}"
ENV_PREFIX="/root/miniconda3/envs/${CONDA_ENV_NAME}"
TORCHRUN_BIN="${ENV_PREFIX}/bin/torchrun"

if [ ! -x "$TORCHRUN_BIN" ]; then
  echo "[ERROR] torchrun not found at: $TORCHRUN_BIN"
  echo "Set CONDA_ENV_NAME or check your conda env path."
  exit 1
fi

# Optional: override these for a single-sample run
JSON_FILE="${JSON_FILE:-}"
VIDEO_FOLDER="${VIDEO_FOLDER:-}"
OUTPUT_LATENT_FOLDER="${OUTPUT_LATENT_FOLDER:-}"

$TORCHRUN_BIN $DISTRIBUTED_ARGS \
    tools/offload_data/get_short-latents.py \
    --pretrained_model_name_or_path /root/autodl-fs/BestWishYSH/Helios-Base \
    --dataloader_num_workers 20 \
    ${JSON_FILE:+--json_file "$JSON_FILE"} \
    ${VIDEO_FOLDER:+--video_folder "$VIDEO_FOLDER"} \
    ${OUTPUT_LATENT_FOLDER:+--output_latent_folder "$OUTPUT_LATENT_FOLDER"} \
    ${STRIDE:+--stride "$STRIDE"} \
    ${BATCH_SIZE:+--batch_size "$BATCH_SIZE"} \
    ${RESOLUTION:+--resolution "$RESOLUTION"}
