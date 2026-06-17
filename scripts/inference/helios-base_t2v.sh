# Example: Running inference with 2-GPU parallelism
# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node 2 infer_helios.py \
#     --enable_parallelism \
#     --cp_backend "ulysses" \   #  ["ring", "ulysses", "unified", "ulysses_anything"]

CUDA_VISIBLE_DEVICES=0 python infer_helios.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --transformer_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --sample_type "t2v" \
    --num_frames 824 \
    --fps 24 \
    --prompt "A sharp-dressed businessman in a charcoal three-piece suit, a pale blue pocket square folded into a perfect triangle, and a gold tie bar pinning a striped tie walks briskly across a sun-bleached plaza. The camera frames him from head to mid-thigh, his full upper body clearly visible — sharp jawline, slicked hair, confident expression. The pocket square is vivid. The gold tie bar glints sharply. The pinstripe of the suit is crisp in the hard light. He passes through a narrow concrete parking structure corridor. The fluorescent lights flicker and die midway — his silhouette is swallowed entirely, the suit's fine texture and pocket square dissolving into black, the gold tie bar gone. He strides out the other side into hard afternoon sun. The camera holds the same head-to-mid-thigh framing — his face catches the light first, then the pale blue pocket square blazes back into view. The gold tie bar glints sharply. The pinstripe of the suit snaps back into focus. He adjusts his cufflinks and walks on." \
    --guidance_scale 5.0 \
    --output_folder "/root/autodl-fs/output/5_28/helios-base" \
    --enable_low_vram_mode \
    --group_offloading_type "leaf_level" \
    --num_blocks_per_group 4
    # --use_cfg_zero_star \
    # --use_zero_init \
    # --zero_steps 1 \
    # prompt 1:"A man in a tailored charcoal gray suit with faint pinstripes, white dress shirt, and a burgundy tie walks into a brightly lit office. His leather oxford shoes click against the hardwood floor as he strides in with a measured, confident pace, his left hand slightly swinging while his right holds a manila folder. He pulls out the chair and settles into it, placing the folder on the desk. He remains seated, leaning slightly forward with his elbows resting on the desk, gesturing occasionally with his right hand as he speaks calmly. Then he suddenly grips the armrests, shifts his weight, and stands up abruptly." \
    # prompt 2:A disheveled young woman with smudged mascara, a wrinkled white linen blouse with collar askew, and a silver necklace shuffles slowly down a sun-drenched sidewalk. She enters a long urban tunnel — her silhouette fades into shadow, all details obscured. She emerges at the far end into pale afternoon light: the blouse, the necklace, the mascara tracks all reappear exactly as before. She continues walking, the same exhausted shuffle.
    # prompt 3:On a shallow tropical seafloor, a coral formation rises in vivid tangerine orange, electric violet, and pale ivory, its branching polyps surrounded by translucent blue fish. Suddenly an underwater current churns the sand into a dense, swirling cloud that swallows the coral entirely. As the silt settles and the water clears, the coral re-emerges — colors, geometry, and fish all unchanged.