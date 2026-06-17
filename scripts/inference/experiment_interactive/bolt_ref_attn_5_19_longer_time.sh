#!/usr/bin/env bash
set -euo pipefail

# 实验目标：
# 1) 使用你指定的 merged transformer + bolt_ref_attn_epoch049 做推理
# 2) 把时长拉长（num_frames / interpolate_time）观察是否仍然频繁切镜
# 3) 同时支持：交互式 CSV 推理 + 非交互式单 prompt 推理

export TMPDIR="/root/autodl-fs/tmp"
export HF_ENDPOINT="https://hf-mirror.com"

# ====== 你关心的核心路径 ======
BASE_MODEL_PATH="/root/autodl-fs/BestWishYSH/Helios-Base"
TRANSFORMER_PATH="/root/autodl-fs/output/5_19_ref/merged"
BOLT_CKPT="/root/autodl-fs/output/5_19_ref/bolt_ref_attn_continue/bolt_ref_attn_epoch049.pth"
PROMPT_CSV="/root/autodl-fs/output/5_17/infer_compare_interactive/interactive_stage1_init_prompts.csv"

# ====== 输出目录 ======
OUT_ROOT="/root/autodl-fs/output/5_20_ref/infer_epoch049_longer_time"
OUT_INTERACTIVE_DIR="${OUT_ROOT}/interactive"
OUT_SINGLE_DIR="${OUT_ROOT}/single_prompt"

# ====== 可调参数（先用长时长配置）======
# 原来常见是 99 帧 + 每段 2/3/4；这里先拉长到 161 帧 + 每段 7
NUM_FRAMES=161
INTERPOLATE_TIME=7
INTERPOLATION_STEPS=1
HEIGHT=384
WIDTH=640
FPS=24
NUM_INFERENCE_STEPS=50
GUIDANCE_SCALE=5.0
SEED=43

# ====== 执行开关 ======
RUN_INTERACTIVE=1
RUN_SINGLE_PROMPT=1

# ====== 非交互式单 prompt（请你填写）======
# 例如：SINGLE_PROMPT="A boy is walking on the street..."
SINGLE_PROMPT="A skateboarder in a white t-shirt with a red lightning bolt walks down a suburban street, holding his board as the sun casts long shadows.The skateboarder walks into a graffiti-covered tunnel, his shadow stretching on the concrete walls as he moves deeper into the dim space.Emerging from the tunnel, the skateboarder rides his board out into the bright daylight, silhouetted against the open sky."

# ====== 共享参数 ======
COMMON_ARGS=(
  --base_model_path "${BASE_MODEL_PATH}"
  --transformer_path "${TRANSFORMER_PATH}"
  --enable_bolt_injection
  --selector_type clip_its
  --bolt_ckpt "${BOLT_CKPT}"
  --bolt_active_layers "33-39"
  --bolt_attn_dim 1280
  --bolt_num_heads 10
  --bolt_k_select 4
  --bolt_alpha 0.6
  --bolt_power 2.0
  --bolt_min_chunk_distance 3
  --num_frames "${NUM_FRAMES}"
  --height "${HEIGHT}"
  --width "${WIDTH}"
  --num_inference_steps "${NUM_INFERENCE_STEPS}"
  --guidance_scale "${GUIDANCE_SCALE}"
  --fps "${FPS}"
  --seed "${SEED}"
  --enable_low_vram_mode
  --group_offloading_type leaf_level
  --bolt_log_every_chunk
)

mkdir -p "${OUT_INTERACTIVE_DIR}" "${OUT_SINGLE_DIR}"

if [[ "${RUN_INTERACTIVE}" == "1" ]]; then
  echo "[RUN] interactive csv inference..."
  CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios_bolt.py \
    "${COMMON_ARGS[@]}" \
    --interactive_prompt_csv_path "${PROMPT_CSV}" \
    --use_interpolate_prompt \
    --interpolation_steps "${INTERPOLATION_STEPS}" \
    --interpolate_time "${INTERPOLATE_TIME}" \
    --output_folder "${OUT_INTERACTIVE_DIR}" \
    --bolt_selection_dump_path "${OUT_INTERACTIVE_DIR}/selection_longer_time.json"
  echo "[DONE] interactive output: ${OUT_INTERACTIVE_DIR}"
fi

if [[ "${RUN_SINGLE_PROMPT}" == "1" ]]; then
  if [[ -z "${SINGLE_PROMPT}" ]]; then
    echo "[SKIP] single prompt inference is enabled, but SINGLE_PROMPT is empty."
    echo "       请先在脚本里填写 SINGLE_PROMPT 后重跑。"
    exit 1
  fi

  echo "[RUN] non-interactive single prompt inference..."
  CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios_bolt.py \
    "${COMMON_ARGS[@]}" \
    --prompt "${SINGLE_PROMPT}" \
    --output_folder "${OUT_SINGLE_DIR}" \
    --bolt_selection_dump_path "${OUT_SINGLE_DIR}/selection_longer_time.json"
  echo "[DONE] single-prompt output: ${OUT_SINGLE_DIR}"
fi

echo "[ALL DONE] outputs root: ${OUT_ROOT}"
