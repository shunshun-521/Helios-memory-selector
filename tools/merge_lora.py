import sys
from argparse import Namespace


sys.path.append("../")
from helios.modules.transformer_helios import HeliosTransformer3DModel
from helios.pipelines.pipeline_helios import HeliosPipeline
from helios.utils.utils_base import load_extra_components


transformer_additional_kwargs = {
    "has_multi_term_memory_patch": True,
    "zero_history_timestep": True,
    "guidance_cross_attn": True,
    "restrict_self_attn": False,
    "is_train_restrict_lora": False,
    "restrict_lora": False,
    "restrict_lora_rank": 128,
}

# Stage 3 ODE: 基础 transformer 来自 Stage 2 合并后的权重
transformer = HeliosTransformer3DModel.from_pretrained(
    "/root/autodl-fs/output/ablation_stage_2_init_smoke_test/merged",
    subfolder="transformer",
    transformer_additional_kwargs=transformer_additional_kwargs,
)
pipe = HeliosPipeline.from_pretrained(
    "/root/autodl-fs/BestWishYSH/Helios-Base",
    transformer=transformer,
)

# 使用 EMA 权重（model_ema/ 目录）
pipe.load_lora_weights(
    "/root/autodl-fs/output/ablation_stage_3_ode_smoke_test/checkpoint-100/model_ema/pytorch_lora_weights.safetensors",
    adapter_name="default",
)
pipe.set_adapters(["default"], adapter_weights=[1.0])


args = Namespace()
if not hasattr(args, "training_config"):
    args.training_config = Namespace()
args.training_config.is_enable_stage1 = True
args.training_config.restrict_self_attn = False
args.training_config.is_amplify_history = False
args.training_config.is_use_gan = False
load_extra_components(
    args,
    transformer,
    "/root/autodl-fs/output/ablation_stage_3_ode_smoke_test/checkpoint-100/model_ema/transformer_partial.pth",
)

pipe.fuse_lora()
pipe.unload_lora_weights()
pipe.transformer.save_pretrained(
    "/root/autodl-fs/output/ablation_stage_3_ode_smoke_test/merged/transformer"
)
