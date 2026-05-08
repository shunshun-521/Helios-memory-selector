#!/bin/bash
# ═══════════════════════════════════════════════════════════
# GAP History Injection — Interactive 推理
# ═══════════════════════════════════════════════════════════
#
# 用法:
#   bash Helios/scripts/inference/experiment_interactive/gap_inject_interactive.sh
#
# CSV 格式 (同 Helios interactive):
#   id, prompt_index, prompt
#   8, 1, "A woman walks down a sidewalk..."
#   8, 2, "She enters a dark tunnel..."
#   8, 3, "She emerges back into light..."

SELECTOR_CKPT="/root/autodl-fs/output/vlm_selector_stage1/checkpoint-epoch003"
CSV_PATH="Helios/example/prompt_interactive_helios_3.csv"
OUTPUT_DIR="output_gap_inject/interactive"

python Helios/inference_with_gap_injection.py \
    --selector_ckpt ${SELECTOR_CKPT} \
    --interactive_prompt_csv_path ${CSV_PATH} \
    --use_interpolate_prompt \
    --interpolation_steps 3 \
    --interpolate_time 7 \
    --k_select 4 \
    --k_inject 4 \
    --max_bank_size 64 \
    --num_frames 726 \
    --num_inference_steps 50 \
    --guidance_scale 5.0 \
    --height 384 \
    --width 640 \
    --latent_window_size 9 \
    --seed 42 \
    --fps 24 \
    --enable_low_vram_mode \
    --group_offloading_type "leaf_level" \
    --output_folder ${OUTPUT_DIR}
