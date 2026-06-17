CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios_ref_short.py \
    --config "/root/autodl-tmp/Helios/scripts/inference/ref_short.yaml" \
    --pretrained_model_name_or_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --transformer_model_name_or_path "/root/autodl-fs/output/ref_short_post_5_27_cro/checkpoint-2000/merged/transformer" \
    --vlm_model_path "/root/autodl-fs/Qwen2.5-VL-3B-Instruct" \
    --vlm_k_select 2 \
    --ref_frames_per_chunk 3 \
    --prompt "A sharp-dressed businessman in a charcoal three-piece suit walks briskly across a sun-bleached plaza." \
    --height 384 \
    --width 640 \
    --num_frames 81 \
    --num_inference_steps 30 \
    --guidance_scale 5.0 \
    --seed 42 \
    --output_path "/root/autodl-fs/output/5_28/helios-ref_short_merged.mp4"

