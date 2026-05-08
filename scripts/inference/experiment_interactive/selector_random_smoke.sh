#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
# Smoke test: 验证 --selector_type=random 分流、memory bank 驱逐、
# slow/fast cache、Bolt Ref-Attn hook 重挂全链路正常。
# ═══════════════════════════════════════════════════════════════
# 注意：RandomSelector 不具备任何选帧质量，仅用于跑通 pipeline。
# 用于业务推理请改用 selector_vlm_interactive.sh（selector_type=vlm）。
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

export TMPDIR="/root/autodl-fs/tmp"
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios_bolt.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --transformer_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --enable_bolt_injection \
    --selector_type random \
    --bolt_ckpt /root/autodl-fs/output/bolt_ref_attn2_420_resume_from_419/bolt_ref_attn_epoch049.pth \
    --bolt_active_layers "33-39" \
    --bolt_attn_dim 1280 \
    --bolt_num_heads 10 \
    --vlm_k_select 4 \
    --vlm_power 2.0 \
    --vlm_min_chunk_distance 3 \
    --mb_max_history_chunks 32 \
    --mb_keep_recent_k 8 \
    --mb_evict_strategy farthest_lowclip \
    --interactive_prompt_csv_path /root/autodl-tmp/Helios/example/prompt_interactive_helios_2.csv \
    --use_interpolate_prompt \
    --interpolation_steps 3 \
    --interpolate_time 7 \
    --num_frames 99 \
    --height 384 --width 640 --fps 24 \
    --output_folder /root/autodl-tmp/output_selector_smoke_random \
    --enable_low_vram_mode \
    --group_offloading_type leaf_level \
    --bolt_log_every_chunk \
    --bolt_selection_dump_path /root/autodl-tmp/output_selector_smoke_random/selection.json
