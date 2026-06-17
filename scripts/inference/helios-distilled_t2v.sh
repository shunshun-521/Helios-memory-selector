# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/HeliosDistillede" \
    --transformer_path "/root/autodl-fs/BestWishYSH/HeliosDistillede" \
    --sample_type "t2v" \
    --prompt "A man in a charcoal pinstripe suit, white dress shirt, and burgundy tie walks into a brightly lit office holding a manila folder. He sits down at the desk, leaning forward with elbows resting on it, gesturing calmly as he speaks. Then he suddenly grips the armrests and stands up abruptly." \
    --num_frames 1440 \
    --guidance_scale 1.0 \
    --is_enable_stage2 \
    --pyramid_num_inference_steps_list 2 2 2 \
    --is_amplify_first_chunk \
    --output_folder "/root/autodl-tmp/output_4_2/sudden_motion" \
    --enable_low_vram_mode \
    --group_offloading_type "leaf_level" \
    --num_blocks_per_group 4
    # --pyramid_num_inference_steps_list 1 1 1 \
# prompt 1：A man in a charcoal pinstripe suit, white dress shirt, and burgundy tie walks into a brightly lit office holding a manila folder. He sits down at the desk, leaning forward with elbows resting on it, gesturing calmly as he speaks. Then he suddenly grips the armrests and stands up abruptly.
# prompt 2：A disheveled young woman with smudged mascara, a wrinkled white linen blouse with collar askew, and a silver necklace shuffles slowly down a sun-drenched sidewalk. She enters a long urban tunnel — her silhouette fades into shadow, all details obscured. She emerges at the far end into pale afternoon light: the blouse, the necklace, the mascara tracks all reappear exactly as before. She continues walking, the same exhausted shuffle.
# prompt 3：On a shallow tropical seafloor, a coral formation rises in vivid tangerine orange, electric violet, and pale ivory, its branching polyps surrounded by translucent blue fish. Suddenly an underwater current churns the sand into a dense, swirling cloud that swallows the coral entirely. As the silt settles and the water clears, the coral re-emerges — colors, geometry, and fish all unchanged.