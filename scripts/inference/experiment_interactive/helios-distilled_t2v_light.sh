# 确保在执行前已经设置了必要的临时文件夹
mkdir -p /root/autodl-fs/tmp
export TMPDIR="/root/autodl-fs/tmp"


# 使用单卡推理
CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/HeliosDistillede" \
    --transformer_path "/root/autodl-fs/BestWishYSH/HeliosDistillede" \
    --sample_type "t2v" \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs." \
    --num_frames 1452 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_amplify_first_chunk \
    --interpolation_steps 3 \
    --interactive_prompt_csv_path "/root/autodl-tmp/Helios/example/prompt_interactive_helios_2.csv" \
    --use_interpolate_prompt \
    --interpolate_time 7 \
    --output_folder "/root/autodl-tmp/output_4_1/baseline_interact_ABAB" \
    --enable_low_vram_mode \
    --group_offloading_type "leaf_level" \
    --num_blocks_per_group 4
