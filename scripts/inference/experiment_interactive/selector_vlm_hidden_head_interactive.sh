#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# VLM Selector 推理脚本（hidden_head）
# ═══════════════════════════════════════════════════════════════
# 要求：--vlm_lora_path 指向包含 selector_head.pt 的 epoch 目录
# 例如：/root/autodl-fs/output/selector_vlm_hidden_head/epoch_009
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export TMPDIR="/root/autodl-fs/tmp"
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios_bolt.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --transformer_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --enable_bolt_injection \
    --selector_type vlm \
    --vlm_model_path /root/autodl-fs/Qwen2.5-VL-3B-Instruct \
    --vlm_lora_path /root/autodl-fs/output/selector_vlm_hidden_head/epoch_009 \
    --vlm_score_mode hidden_head \
    --vlm_k_select 4 \
    --vlm_power 2.0 \
    --vlm_min_chunk_distance 3 \
    --vlm_max_candidates 16 \
    --vlm_prefilter_alpha 0.6 \
    --vlm_temperature 0.7 \
    --vlm_fallback_to_clip \
    --mb_max_history_chunks 32 \
    --mb_keep_recent_k 8 \
    --mb_evict_strategy farthest_lowclip \
    --bolt_ckpt /root/autodl-fs/output/bolt_ref_attn2_420_resume_from_419/bolt_ref_attn_epoch049.pth \
    --bolt_active_layers "33-39" \
    --bolt_attn_dim 1280 \
    --bolt_num_heads 10 \
    --interactive_prompt_csv_path /root/autodl-tmp/Helios/example/prompt_interactive_helios_2.csv \
    --use_interpolate_prompt \
    --interpolation_steps 3 \
    --interpolate_time 7 \
    --num_frames 99 \
    --height 384 --width 640 --fps 24 \
    --output_folder /root/autodl-tmp/output_selector_vlm_hidden_head \
    --enable_low_vram_mode \
    --group_offloading_type leaf_level \
    --bolt_log_every_chunk \
    --bolt_selection_dump_path /root/autodl-tmp/output_selector_vlm_hidden_head/selection.json
