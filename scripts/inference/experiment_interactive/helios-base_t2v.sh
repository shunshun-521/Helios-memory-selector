#!/usr/bin/env bash
set -euo pipefail

export TMPDIR="/root/autodl-fs/tmp"
export HF_ENDPOINT="https://hf-mirror.com"
export HELIOS_SKIP_FLASH_KERNEL_DOWNLOAD=1

PYTHON_BIN="python"
INFER_PY="/root/autodl-tmp/Helios/infer_helios.py"

BASE_MODEL="/root/autodl-fs/BestWishYSH/Helios-Base"
WAN_TRANSFORMER="/root/autodl-fs/BestWishYSH/Helios-Base"
LORA_CKPT="/root/autodl-fs/output/5_17/checkpoint-1500/pytorch_lora_weights.safetensors"

OUT_ROOT="/root/autodl-fs/output/5_17/infer_compare_interactive"
PROMPT_CSV="${OUT_ROOT}/interactive_stage1_init_prompts.csv"

mkdir -p "${OUT_ROOT}"

# Keep prompts / duration / interpolation settings aligned with scripts/training/configs/stage_1_init.yaml
cat > "${PROMPT_CSV}" <<'EOF'
id,prompt_index,prompt
0,0,"A skateboarder in a white t-shirt with a red lightning bolt walks down a suburban street, holding his board as the sun casts long shadows."
0,1,"The skateboarder walks into a graffiti-covered tunnel, his shadow stretching on the concrete walls as he moves deeper into the dim space."
0,2,"Emerging from the tunnel, the skateboarder rides his board out into the bright daylight, silhouetted against the open sky."
EOF

COMMON_ARGS=(
  --base_model_path "${BASE_MODEL}"
  --transformer_path "${WAN_TRANSFORMER}"
  --sample_type "t2v"
  --seed 43
  --height 384
  --width 640
  --num_frames 99
  --fps 24
  --num_inference_steps 50
  --guidance_scale 5.0
  --num_latent_frames_per_chunk 9
  --use_interpolate_prompt
  --interpolation_steps 1
  --interactive_prompt_csv_path "${PROMPT_CSV}"
  --interpolate_time_list 2 3 4

)

echo "[Run] helios-base baseline (no LoRA)"
CUDA_VISIBLE_DEVICES=0 "${PYTHON_BIN}" "${INFER_PY}" \
  "${COMMON_ARGS[@]}" \
  --output_folder "${OUT_ROOT}/Helios-Base"