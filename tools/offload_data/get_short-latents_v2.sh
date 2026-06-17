export LD_LIBRARY_PATH=/root/miniconda3/envs/helios/lib:$LD_LIBRARY_PATH
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
## NCCL (Optimized for single node, single device)
#################################################################
export NCCL_SOCKET_IFNAME=eth0
export NCCL_SOCKET_TIMEOUT=3600000

export NCCL_DEBUG=WARN  # disable the verbose NCCL logs
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0  # was 1
export NCCL_SHM_DISABLE=0  # was 1

export NCCL_PXN_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
#################################################################

#################################################################
## DIST (Single GPU setup)
#################################################################
export CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=127.0.0.1
MASTER_PORT=12345
NNODES=1
NODE_RANK=0
GPUS_PER_NODE=1

WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo -e "\033[31mDISTRIBUTED_ARGS: ${DISTRIBUTED_ARGS}\033[0m"

#################################################################
export OMP_NUM_THREADS=1
export PYTHONPATH=$PWD/Helios:$PYTHONPATH

# Execute the v2 Python script
torchrun $DISTRIBUTED_ARGS \
    /root/autodl-tmp/Helios/tools/offload_data/get_short-latents_v2.py \
    --pretrained_model_name_or_path /root/autodl-tmp/mova-weight \
    --audio_vae_path /root/autodl-tmp/mova-weight/audio_vae \
    --json_file /root/autodl-tmp/Helios/data/audiocaps/train/train1_1.json \
    --audio_folder /root/autodl-tmp/Helios/data/audiocaps/train/train1 \
    --output_latent_folder /root/autodl-tmp/Helios/data/audiocaps/train/latent \
    --batch_size 4
