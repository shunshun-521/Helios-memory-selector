CUDA_VISIBLE_DEVICES=0 python /root/autodl-tmp/Helios/tools/merge_lora_ref_short_ckpt.py \
    --base_model_path "/root/autodl-fs/BestWishYSH/Helios-Base" \
    --ckpt_dir "/root/autodl-fs/output/ref_short_post_5_27_cro/checkpoint-1500" \
    --out_dir "/root/autodl-fs/output/ref_short_post_5_27_cro/checkpoint-1500/merged"

