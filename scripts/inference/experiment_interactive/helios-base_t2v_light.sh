# 确保在执行前已经设置了必要的临时文件夹
mkdir -p /root/autodl-fs/tmp
export TMPDIR="/root/autodl-fs/tmp"
export HF_ENDPOINT=https://hf-mirror.com

# 使用单卡推理
CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --transformer_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --sample_type "t2v" \
    --num_frames 81 \
    --fps 24 \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs." \
    --guidance_scale 5.0 \
    --use_interpolate_prompt \
    --interpolation_steps 3 \
    --interactive_prompt_csv_path "/root/autodl-tmp/Helios/example/prompt_interactive_helios.csv" \
    --interpolate_time 1 \
    --output_folder "./output_helios/test_run" \
    --enable_low_vram_mode \
    --group_offloading_type "leaf_level" \
    --num_blocks_per_group 4
