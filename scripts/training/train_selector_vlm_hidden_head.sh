#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# 训练 VLM Selector（hidden_head: last visual token + Linear head）
# ═══════════════════════════════════════════════════════════════
# 前置：先跑 tools/build_vlm_selector_dataset.py 生成 jsonl + frames_mid
# 产物：epoch_xxx/adapter_* + selector_head.pt + score_mode.json
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export TMPDIR="/root/autodl-fs/tmp"
export HF_ENDPOINT=https://hf-mirror.com

source /root/miniconda3/etc/profile.d/conda.sh && conda activate helios

CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/train_helios_selector_vlm.py \
    --data_jsonl /root/autodl-fs/selector_vlm_data/train.jsonl \
    --frames_dir /root/autodl-fs/selector_vlm_data/frames_mid \
    --vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
    --vlm_dtype bfloat16 \
    --vlm_image_resize 256 448 \
    --score_mode hidden_head \
    --lora_r 16 --lora_alpha 32 --lora_dropout 0.05 \
    --temperature 0.7 --temperature_teacher 0.5 \
    --kl_lambda 1.0 --ce_lambda 0.3 \
    --min_candidates 2 \
    --vlm_lr 2e-4 --head_lr 5e-4 --weight_decay 0.0 \
    --num_epochs 10 --batch_size 1 --grad_accum_steps 1 \
    --max_grad_norm 1.0 \
    --save_every 5 --log_every 1 \
    --output_dir /root/autodl-fs/output/selector_vlm_hidden_head
