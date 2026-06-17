set -e

# 1) merge LoRA checkpoint into base transformer
python /root/autodl-tmp/Helios/tools/merge_lora_ref_short_ckpt.py \
  --base_model_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
  --ckpt_dir "/root/autodl-fs/output/ref_short_post_5_27_cro/checkpoint-1500" \
  --out_dir "/root/autodl-fs/output/ref_short_post_5_27_cro/checkpoint-1500/merged"

# 2) infer with merged transformer
CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios_ref_short.py \
  --config "/root/autodl-tmp/Helios/scripts/inference/ref_short.yaml" \
  --pretrained_model_name_or_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
  --transformer_model_name_or_path "/root/autodl-fs/output/ref_short_post_5_27_cro/checkpoint-1500/merged/transformer" \
  --vlm_model_path "/root/autodl-fs/Qwen2.5-VL-3B-Instruct" \
  --prompt "A sharp-dressed businessman in a charcoal three-piece suit walks briskly across a sun-bleached plaza." \
  --output_path "/root/autodl-fs/output/5_28/helios-ref_short_merged.mp4"

