# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]
export TMPDIR="/root/autodl-fs/tmp"
export HF_ENDPOINT=https://hf-mirror.com

CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/infer_helios.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/HeliosDistillede" \
    --transformer_path "/root/autodl-fs/BestWishYSH/HeliosDistillede" \
    --sample_type "t2v" \
    --prompt "A vibrant tropical fish swimming gracefully among colorful coral reefs in a clear, turquoise ocean. The fish has bright blue and yellow scales with a small, distinctive orange spot on its side, its fins moving fluidly. The coral reefs are alive with a variety of marine life, including small schools of colorful fish and sea turtles gliding by. The water is crystal clear, allowing for a view of the sandy ocean floor below. The reef itself is adorned with a mix of hard and soft corals in shades of red, orange, and green. The photo captures the fish from a slightly elevated angle, emphasizing its lively movements and the vivid colors of its surroundings. A close-up shot with dynamic movement." \
    --num_frames 1452 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_amplify_first_chunk \
    --interpolation_steps 3 \
    --interactive_prompt_csv_path "/root/autodl-tmp/Helios/example/prompt_interactive_helios.csv" \
    --interpolate_time 7 \
    --output_folder "/root/autodl-tmp/Helios/output_helios/helios-distilled" \
    --enable_low_vram_mode \
    --group_offloading_type "leaf_level" 
    