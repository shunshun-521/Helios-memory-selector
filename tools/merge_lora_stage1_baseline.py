"""
Merge LoRA weights for v3_baseline_2 checkpoint into base transformer.
Usage: cd Helios && python tools/merge_lora_stage1_baseline.py
"""
import sys
from argparse import Namespace

sys.path.append("./")
from helios.modules.transformer_helios_v3 import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.utils.utils_base import load_extra_components


# ---- Config ----
CHECKPOINT_DIR = "/root/autodl-fs/output/v3_baseline_2/checkpoint-1000"
BASE_MODEL_PATH = "/root/autodl-fs/BestWishYSH/HeliosDistillede"
TRANSFORMER_PATH = "/root/autodl-fs/Wan-AI/Wan2.1-T2V-14B-Diffusers"
OUTPUT_DIR = "/root/autodl-fs/output/v3_baseline_2/merged/transformer"

transformer_additional_kwargs = {
    "has_multi_term_memory_patch": True,
    "zero_history_timestep": True,
    "guidance_cross_attn": True,
    "restrict_self_attn": False,
    "is_train_restrict_lora": False,
    "restrict_lora": False,
    "restrict_lora_rank": 128,
}

transformer = HeliosTransformer3DModel.from_pretrained(
    TRANSFORMER_PATH,
    subfolder="transformer",
    transformer_additional_kwargs=transformer_additional_kwargs,
)
pipe = HeliosPipeline.from_pretrained(
    BASE_MODEL_PATH,
    transformer=transformer,
)

pipe.load_lora_weights(
    f"{CHECKPOINT_DIR}/pytorch_lora_weights.safetensors",
    adapter_name="default",
)
pipe.set_adapters(["default"], adapter_weights=[1.0])

args = Namespace()
args.training_config = Namespace()
args.training_config.is_enable_stage1 = True
args.training_config.restrict_self_attn = False
args.training_config.is_amplify_history = False
args.training_config.is_use_gan = False
args.training_config.use_selector = False
load_extra_components(args, transformer, f"{CHECKPOINT_DIR}/transformer_partial.pth")

pipe.fuse_lora()
pipe.unload_lora_weights()
pipe.transformer.save_pretrained(OUTPUT_DIR)
print(f"Merged transformer saved to {OUTPUT_DIR}")
