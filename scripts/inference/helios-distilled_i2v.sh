# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/HeliosDistillede" \
    --transformer_path "/root/autodl-fs/BestWishYSH/HeliosDistillede" \
    --sample_type "i2v" \
    --image_path "/root/autodl-tmp/Helios/scripts/inference/image/tokyo street.png" \
    --image_noise_sigma_min 0.111 \
    --image_noise_sigma_max 0.135 \
    --prompt "A dynamic video sequence featuring 4 scenes of Shibuya Crossing in Tokyo, Japan. The video starts with the first scene: clear blue sky, colorful giant billboards, TSUTAYA building, Starbucks, and crowded crosswalks. Then, the camera rotates 90 degrees around itself, and the second scene appears: dense building facades with UC, DHC, IKEA signs, black cars, and passing pedestrians. Next, the camera continues to rotate 90 degrees around itself, and the third scene emerges: open brick-paved square, yellow guide lines, super high-rise glass office buildings, red tower cranes, and pedestrians walking. And then, the camera rotates another 90 degrees around itself, and the fourth scene shows up: street lamp posts covered with stickers, Shibuya Ekimae traffic signs, blue cars, bus traffic, and colorful billboards. Finally,the camera rotates another 90 degrees around itself.The entire video maintains a realistic street view style, with bright daylight, vivid colors, and smooth rotation transitions between each scene." \
    --num_frames 2160 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_amplify_first_chunk \
    --output_folder "/root/autodl-tmp/output_4_1/baseline_rotate" \
    --enable_low_vram_mode \
    --group_offloading_type "leaf_level" \
    --num_blocks_per_group 4
    # --pyramid_num_inference_steps_list 1 1 1 \