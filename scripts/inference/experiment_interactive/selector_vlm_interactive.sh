#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# VLM Selector 推理脚本（新增分支，不替代 CLIP+ITS）
# 对应 md/CLIP-base-selector-vlm.md §2.5
# ═══════════════════════════════════════════════════════════════
# 切换回 CLIP+ITS：把 --selector_type vlm 改成 --selector_type clip_its 即可，
# 其余参数与 train_helios_bolt / infer_helios_bolt 完全一致。
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
    --output_folder /root/autodl-tmp/output_selector_vlm \
    --enable_low_vram_mode \
    --group_offloading_type leaf_level \
    --bolt_log_every_chunk \
    --bolt_selection_dump_path /root/autodl-tmp/output_selector_vlm/selection.json
