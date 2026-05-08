# 确保在执行前已经设置了必要的临时文件夹
mkdir -p /root/autodl-fs/tmp
export TMPDIR="/root/autodl-fs/tmp"
export HF_ENDPOINT=https://hf-mirror.com

# 使用单卡推理
CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/Helios-Distilled" \
    --transformer_path "/root/autodl-fs/BestWishYSH/Helios-Distilled" \
    --sample_type "t2v" \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs." \
    --num_frames 81 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_amplify_first_chunk \
    --output_folder "./output_helios/test_run" \
    #--enable_low_vram_mode \
    #--group_offloading_type "leaf_level" \
    #--num_blocks_per_group 4