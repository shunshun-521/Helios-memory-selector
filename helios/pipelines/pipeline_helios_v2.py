# Copyright 2025 The Helios Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import html
import math
from enum import Enum
from itertools import accumulate
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import regex as re
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor

from ..dataset.dac_vae import DAC
from ..modules.transformer_helios_v2 import HeliosTransformer1DModel
from ..scheduler.scheduling_helios import HeliosScheduler
from ..utils.utils_base import AdaptiveAntiDrifting, apply_schedule_shift
from ..utils.utils_helios_post import add_noise, convert_flow_pred_to_x0
from .pipeline_output import HeliosPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from transformers import AutoTokenizer, UMT5EncoderModel
        >>> from helios.dataset.dac_vae import DAC
        >>> from helios.modules.transformer_helios_v2 import HeliosTransformer1DModel
        >>> from helios.scheduler.scheduling_helios import HeliosScheduler
        >>> from helios.pipelines.pipeline_helios_v2 import HeliosPipeline
        >>>
        >>> tokenizer = AutoTokenizer.from_pretrained("your-model", subfolder="tokenizer")
        >>> text_encoder = UMT5EncoderModel.from_pretrained("your-model", subfolder="text_encoder")
        >>> vae = DAC.from_pretrained("your-audio-vae")
        >>> transformer = HeliosTransformer1DModel.from_pretrained("your-model", subfolder="transformer")
        >>> scheduler = HeliosScheduler.from_pretrained("your-model", subfolder="scheduler")
        >>> pipe = HeliosPipeline(tokenizer, text_encoder, vae, scheduler, transformer).to("cuda")
        >>> output = pipe(prompt="ocean waves with distant birds", duration=8.0, output_type="pt")
        >>> audio = output.frames
        ```
"""


@torch.amp.autocast("cuda", dtype=torch.float32)
def optimized_scale(positive_flat, negative_flat):
    dot_product = torch.sum(positive_flat * negative_flat, dim=1, keepdim=True)
    squared_norm = torch.sum(negative_flat**2, dim=1, keepdim=True) + 1e-8
    st_star = dot_product / squared_norm
    return st_star


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


class VAEDecodeType(str, Enum):
    DEFAULT = "default"


class HeliosPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-audio generation using Helios T2A.

    Args:
        tokenizer:
            Tokenizer for text prompts.
        text_encoder:
            Text encoder used to build prompt embeddings.
        vae:
            Audio VAE/codec used to decode audio latents.
        scheduler:
            Scheduler used during denoising.
        transformer:
            1D Helios transformer operating on audio latents of shape `(B, C, L)`.
    """

    model_cpu_offload_seq = "text_encoder->transformer->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: DAC,
        scheduler: UniPCMultistepScheduler | HeliosScheduler,
        transformer: HeliosTransformer1DModel,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
        )
        self.vae_scale_factor_temporal = getattr(self.vae, "hop_length", 1)
        self.audio_sample_rate = getattr(self.vae, "sample_rate", 48000)

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds, text_inputs.attention_mask.bool()

    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds, prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        else:
            prompt_attention_mask = None

        negative_prompt_attention_mask = None
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} != {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`: {prompt} has batch size {batch_size}."
                )

            negative_prompt_embeds, negative_prompt_attention_mask = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
    ):
        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError("Cannot forward both `prompt` and `prompt_embeds`.")
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError("Cannot forward both `negative_prompt` and `negative_prompt_embeds`.")
        elif prompt is None and prompt_embeds is None:
            raise ValueError("Provide either `prompt` or `prompt_embeds`.")
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int,
        latent_length: int,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        shape = (batch_size, num_channels_latents, latent_length)
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested batch size {batch_size}."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    def interpolate_prompt_embeds(
        self,
        prompt_embeds_1: torch.Tensor,
        prompt_embeds_2: torch.Tensor,
        interpolation_steps: int = 4,
    ):
        x = torch.lerp(
            prompt_embeds_1,
            prompt_embeds_2,
            torch.linspace(0, 1, steps=interpolation_steps).unsqueeze(1).unsqueeze(2).to(prompt_embeds_1),
        )
        return list(x.chunk(interpolation_steps, dim=0))

    def sample_block_noise(
        self,
        batch_size,
        channel,
        seq_len,
        patch_size: int | tuple[int, ...] = 1,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
    ):
        if generator is None:
            generator = torch.Generator(device=device)
        elif isinstance(generator, list):
            generator = generator[0]

        gamma = self.scheduler.config.gamma
        block_size = patch_size if isinstance(patch_size, int) else math.prod(patch_size)
        block_size = max(int(block_size), 1)

        cov = torch.eye(block_size, device=device) * (1 + gamma) - torch.ones(block_size, block_size, device=device) * gamma
        cov += torch.eye(block_size, device=device) * 1e-8
        cov = cov.float()

        L = torch.linalg.cholesky(cov)
        num_blocks = math.ceil(seq_len / block_size)
        block_number = batch_size * channel * num_blocks
        z = torch.randn(block_number, block_size, generator=generator, device=generator.device).to(device=device)
        noise = z @ L.T
        noise = noise.view(batch_size, channel, num_blocks, block_size).reshape(batch_size, channel, num_blocks * block_size)
        return noise[:, :, :seq_len]

    def stage1_sample(
        self,
        latents: torch.Tensor = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        timesteps: torch.Tensor = None,
        guidance_scale: Optional[float] = 5.0,
        indices_hidden_states: torch.Tensor = None,
        indices_latents_history_short: torch.Tensor = None,
        indices_latents_history_mid: torch.Tensor = None,
        indices_latents_history_long: torch.Tensor = None,
        latents_history_short: torch.Tensor = None,
        latents_history_mid: torch.Tensor = None,
        latents_history_long: torch.Tensor = None,
        attention_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        transformer_dtype: torch.dtype = None,
        generator: Optional[torch.Generator] = None,
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        use_dmd: bool = False,
        dmd_sigmas: torch.Tensor = None,
        dmd_timesteps: torch.Tensor = None,
        callback_on_step_end: Optional[callable] = None,
        callback_on_step_end_tensor_inputs: list = None,
        progress_bar=None,
    ):
        batch_size = latents.shape[0]

        for i, t in enumerate(timesteps):
            is_first_step = i == 0

            if self.interrupt:
                continue

            self._current_timestep = t
            timestep = t.expand(latents.shape[0])

            latent_model_input = latents.to(transformer_dtype)
            with self.transformer.cache_context("cond"):
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep,
                    encoder_hidden_states=prompt_embeds,
                    indices_hidden_states=indices_hidden_states,
                    indices_latents_history_short=indices_latents_history_short,
                    indices_latents_history_mid=indices_latents_history_mid,
                    indices_latents_history_long=indices_latents_history_long,
                    latents_history_short=latents_history_short.to(transformer_dtype),
                    latents_history_mid=latents_history_mid.to(transformer_dtype),
                    latents_history_long=latents_history_long.to(transformer_dtype),
                    is_first_denoising_step=is_first_step,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]

            if self.do_classifier_free_guidance and not use_dmd:
                with self.transformer.cache_context("uncond"):
                    noise_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=negative_prompt_embeds,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short.to(transformer_dtype),
                        latents_history_mid=latents_history_mid.to(transformer_dtype),
                        latents_history_long=latents_history_long.to(transformer_dtype),
                        is_first_denoising_step=is_first_step,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                if use_cfg_zero_star:
                    noise_pred_text = noise_pred
                    positive_flat = noise_pred_text.view(batch_size, -1)
                    negative_flat = noise_uncond.view(batch_size, -1)

                    alpha = optimized_scale(positive_flat, negative_flat)
                    alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                    alpha = alpha.to(noise_pred_text.dtype)

                    if (i <= zero_steps) and use_zero_init:
                        noise_pred = noise_pred_text * 0.0
                    else:
                        noise_pred = noise_uncond * alpha + guidance_scale * (noise_pred_text - noise_uncond * alpha)
                else:
                    noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            if use_dmd:
                pred_audio_latent = convert_flow_pred_to_x0(
                    flow_pred=noise_pred,
                    xt=latent_model_input,
                    timestep=t * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                    sigmas=dmd_sigmas,
                    timesteps=dmd_timesteps,
                )
                if i < len(timesteps) - 1:
                    latents = add_noise(
                        pred_audio_latent,
                        randn_tensor(pred_audio_latent.shape, generator=generator, device=device),
                        timesteps[i + 1] * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                        sigmas=dmd_sigmas,
                        timesteps=dmd_timesteps,
                    )
                else:
                    latents = pred_audio_latent
            else:
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

            if callback_on_step_end is not None:
                callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)
                latents = callback_outputs.pop("latents", latents)
                prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

            progress_bar.update()

            if XLA_AVAILABLE:
                xm.mark_step()

        return latents

    def stage2_sample(
        self,
        latents: torch.Tensor = None,
        stage2_num_stages: int = None,
        stage2_num_inference_steps_list: List[int] = None,
        prompt_embeds: torch.Tensor = None,
        negative_prompt_embeds: torch.Tensor = None,
        guidance_scale: Optional[float] = 5.0,
        indices_hidden_states: torch.Tensor = None,
        indices_latents_history_short: torch.Tensor = None,
        indices_latents_history_mid: torch.Tensor = None,
        indices_latents_history_long: torch.Tensor = None,
        latents_history_short: torch.Tensor = None,
        latents_history_mid: torch.Tensor = None,
        latents_history_long: torch.Tensor = None,
        attention_kwargs: Optional[dict] = None,
        device: Optional[torch.device] = None,
        transformer_dtype: torch.dtype = None,
        scheduler_type: str = "unipc",
        use_dynamic_shifting: bool = False,
        generator: torch.Generator | list[torch.Generator] | None = None,
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        use_dmd: bool = False,
        is_amplify_first_chunk: bool = False,
        callback_on_step_end: Optional[callable] = None,
        callback_on_step_end_tensor_inputs: list = None,
        progress_bar=None,
    ):
        seq_len = latents.shape[-1]
        for _ in range(stage2_num_stages - 1):
            seq_len //= 2
            latents = F.interpolate(latents, size=seq_len, mode="linear") * 2

        batch_size = latents.shape[0]
        if use_dmd:
            start_point_list = [latents]

        step_count = 0
        for i_s in range(stage2_num_stages):
            if use_dmd:
                if is_amplify_first_chunk:
                    self.scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] * 2 + 1, i_s, device=device)
                else:
                    self.scheduler.set_timesteps(stage2_num_inference_steps_list[i_s] + 1, i_s, device=device)
                self.scheduler.timesteps = self.scheduler.timesteps[:-1]
                self.scheduler.sigmas = torch.cat([self.scheduler.sigmas[:-2], self.scheduler.sigmas[-1:]])
            else:
                self.scheduler.set_timesteps(stage2_num_inference_steps_list[i_s], i_s, device=device)

            if i_s > 0:
                seq_len = latents.shape[-1] * 2
                latents = F.interpolate(latents, size=seq_len, mode="nearest")
                ori_sigma = 1 - self.scheduler.ori_start_sigmas[i_s]
                gamma = self.scheduler.config.gamma
                alpha = 1 / (math.sqrt(1 + (1 / gamma)) * (1 - ori_sigma) + ori_sigma)
                beta = alpha * (1 - ori_sigma) / math.sqrt(gamma)

                batch_size, channel, cur_seq_len = latents.shape
                patch_size = self.transformer.config.patch_size
                noise = self.sample_block_noise(batch_size, channel, cur_seq_len, patch_size, device, generator)
                noise = noise.to(device=device, dtype=transformer_dtype)
                latents = alpha * latents + beta * noise

                if use_dmd:
                    start_point_list.append(latents)

            if use_dynamic_shifting:
                temp_sigmas = apply_schedule_shift(
                    self.scheduler.sigmas,
                    latents,
                    base_seq_len=self.scheduler.config.get("base_image_seq_len", 256),
                    max_seq_len=self.scheduler.config.get("max_image_seq_len", 4096),
                    base_shift=self.scheduler.config.get("base_shift", 0.5),
                    max_shift=self.scheduler.config.get("max_shift", 1.15),
                )
                temp_timesteps = self.scheduler.timesteps_per_stage[i_s].min() + temp_sigmas[:-1] * (
                    self.scheduler.timesteps_per_stage[i_s].max() - self.scheduler.timesteps_per_stage[i_s].min()
                )
                self.scheduler.sigmas = temp_sigmas
                self.scheduler.timesteps = temp_timesteps

            timesteps = self.scheduler.timesteps

            for idx, t in enumerate(timesteps):
                is_first_step = i_s == 0 and idx == 0
                timestep = t.expand(latents.shape[0]).to(torch.int64)

                with self.transformer.cache_context("cond"):
                    noise_pred = self.transformer(
                        hidden_states=latents.to(transformer_dtype),
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short.to(transformer_dtype),
                        latents_history_mid=latents_history_mid.to(transformer_dtype),
                        latents_history_long=latents_history_long.to(transformer_dtype),
                        is_first_denoising_step=is_first_step,
                    )[0]

                if self.do_classifier_free_guidance:
                    with self.transformer.cache_context("cond_uncond"):
                        noise_uncond = self.transformer(
                            hidden_states=latents.to(transformer_dtype),
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=indices_latents_history_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=latents_history_short.to(transformer_dtype),
                            latents_history_mid=latents_history_mid.to(transformer_dtype),
                            latents_history_long=latents_history_long.to(transformer_dtype),
                            is_first_denoising_step=is_first_step,
                        )[0]

                    if use_cfg_zero_star:
                        noise_pred_text = noise_pred
                        positive_flat = noise_pred_text.view(batch_size, -1)
                        negative_flat = noise_uncond.view(batch_size, -1)

                        alpha = optimized_scale(positive_flat, negative_flat)
                        alpha = alpha.view(batch_size, *([1] * (len(noise_pred_text.shape) - 1)))
                        alpha = alpha.to(noise_pred_text.dtype)

                        if (i_s == 0 and idx <= zero_steps) and use_zero_init:
                            noise_pred = noise_pred_text * 0.0
                        else:
                            noise_pred = noise_uncond * alpha + guidance_scale * (noise_pred_text - noise_uncond * alpha)
                    else:
                        noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

                if use_dmd:
                    pred_audio_latent = convert_flow_pred_to_x0(
                        flow_pred=noise_pred,
                        xt=latents,
                        timestep=timestep,
                        sigmas=self.scheduler.sigmas,
                        timesteps=self.scheduler.timesteps,
                    )
                    if idx < len(timesteps) - 1:
                        latents = add_noise(
                            pred_audio_latent,
                            start_point_list[i_s],
                            timesteps[idx + 1] * torch.ones(batch_size, dtype=torch.long, device=noise_pred.device),
                            sigmas=self.scheduler.sigmas,
                            timesteps=self.scheduler.timesteps,
                        )
                    else:
                        latents = pred_audio_latent
                else:
                    if scheduler_type == "unipc":
                        latents = self.scheduler.step_unipc(noise_pred.float(), t, latents, return_dict=False)[0]
                    else:
                        latents = self.scheduler.step(noise_pred.float(), t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {k: locals()[k] for k in callback_on_step_end_tensor_inputs}
                    callback_outputs = callback_on_step_end(self, step_count, t, callback_kwargs)
                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()
                step_count += 1

            if return_ode_stages:
                if not hasattr(self, "ode_stages_tensor_all"):
                    self.ode_stages_tensor_all = []
                self.ode_stages_tensor_all.append(self._last_ode_stages)

        return latents

    def get_ode_stages(self, *args, **kwargs):
        kwargs["return_ode_stages"] = True
        return self(*args, **kwargs)

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        duration: float = 8.0,
        sample_rate: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        return_ode_stages: bool = False,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        image: Optional[PipelineImageInput] = None,
        video: Optional[PipelineImageInput] = None,
        use_interpolate_prompt: bool = False,
        interpolate_time_list: list = [7, 7, 7],
        interpolation_steps: int = 3,
        history_sizes: list = [16, 2, 1],
        latent_window_size: int = 9,
        use_dynamic_shifting: bool = False,
        is_keep_x0: bool = True,
        is_enable_stage2: bool = False,
        stage2_num_stages: int = 3,
        stage2_num_inference_steps_list: list = [10, 10, 10],
        scheduler_type: str = "unipc",
        use_cfg_zero_star: Optional[bool] = False,
        use_zero_init: Optional[bool] = True,
        zero_steps: Optional[int] = 1,
        use_dmd: bool = False,
        is_skip_first_section: bool = False,
        is_amplify_first_chunk: bool = False,
        use_adaptive_anti_drifting: bool = False,
        anti_drift_rho_mu: float = 0.9,
        anti_drift_rho_sigma: float = 0.9,
        anti_drift_delta_mu: float = 0.15,
        anti_drift_delta_sigma: float = 0.15,
        anti_drift_corruption_strength: float = 0.1,
        use_kv_cache: bool = False,
        vae_decode_type: VAEDecodeType = "default",
        height: int = 0,
        width: int = 0,
    ):
        r"""
        The call function to the audio pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide audio generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to avoid during generation. Ignored when not using guidance.
            duration (`float`, defaults to `8.0`):
                Target audio duration in seconds.
            sample_rate (`int`, *optional*):
                Output sample rate. Defaults to the VAE sample rate.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps.
            guidance_scale (`float`, defaults to `5.0`):
                Classifier-free guidance scale. Higher values strengthen prompt adherence.
            output_type (`str`, *optional*, defaults to `"pt"`):
                Output format. Supported values include `"pt"`, `"np"`, and `"latent"`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether to return a [`HeliosPipelineOutput`] instead of a tuple.

        Returns:
            [`~HeliosPipelineOutput`] or `tuple`:
                Generated audio in the requested format.
        
        Examples: 
        
        """
        if image is not None or video is not None:
            raise NotImplementedError("Audio pipeline does not support image/video conditioning.")
        if vae_decode_type != VAEDecodeType.DEFAULT:
            raise NotImplementedError("Only `vae_decode_type=default` is supported in audio pipeline.")

        if use_kv_cache:
            self.transformer.enable_kv_cache()

        if use_interpolate_prompt:
            assert num_videos_per_prompt == 1, f"num_videos_per_prompt must be 1, got {num_videos_per_prompt}"
            assert isinstance(prompt, list), "prompt must be a list"
            assert len(prompt) == len(interpolate_time_list), (
                f"Length mismatch: {len(prompt)} vs {len(interpolate_time_list)}"
            )
            assert min(interpolate_time_list) > interpolation_steps, (
                f"Minimum value {min(interpolate_time_list)} must be greater than {interpolation_steps}"
            )
            interpolate_interval_idx = None
            interpolate_embeds = None
            interpolate_cumulative_list = list(accumulate(interpolate_time_list))

        anti_drifting_helper = None
        if use_adaptive_anti_drifting:
            anti_drifting_helper = AdaptiveAntiDrifting(
                rho_mu=anti_drift_rho_mu,
                rho_sigma=anti_drift_rho_sigma,
                delta_mu=anti_drift_delta_mu,
                delta_sigma=anti_drift_delta_sigma,
                device=self._execution_device,
                dtype=torch.float32,
            )

        history_sizes = sorted(history_sizes, reverse=True)
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        self.check_inputs(
            prompt,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
        )

        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device
        vae_dtype = self.vae.dtype
        sample_rate = sample_rate or self.audio_sample_rate
        total_audio_samples = max(1, int(round(duration * sample_rate)))
        total_latent_length = math.ceil(total_audio_samples / self.vae_scale_factor_temporal)
        num_latent_sections = max(1, math.ceil(total_latent_length / latent_window_size))

        if use_interpolate_prompt or (prompt is not None and isinstance(prompt, str)):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        all_prompt_embeds, _, negative_prompt_embeds, _ = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype
        all_prompt_embeds = all_prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            if use_interpolate_prompt:
                negative_prompt_embeds = negative_prompt_embeds[0].unsqueeze(0)
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        num_channels_latents = self.transformer.config.in_channels
        history_latents = torch.zeros(
            batch_size,
            num_channels_latents,
            sum(history_sizes),
            device=device,
            dtype=torch.float32,
        )
        x0_latent = None
        total_generated_latent_frames = 0

        if use_interpolate_prompt and num_latent_sections < max(interpolate_cumulative_list):
            num_latent_sections = sum(interpolate_cumulative_list)
            print(f"Update num_latent_sections to: {num_latent_sections}")

        for k in range(num_latent_sections):
            if use_interpolate_prompt:
                current_interval_idx = 0
                for idx, cumulative_val in enumerate(interpolate_cumulative_list):
                    if k < cumulative_val:
                        current_interval_idx = idx
                        break

                if current_interval_idx == 0:
                    cur_prompt_embeds = all_prompt_embeds[0].unsqueeze(0)
                else:
                    interval_start = interpolate_cumulative_list[current_interval_idx - 1]
                    position_in_interval = k - interval_start

                    if position_in_interval < interpolation_steps:
                        if interpolate_embeds is None or interpolate_interval_idx != current_interval_idx:
                            interpolate_embeds = self.interpolate_prompt_embeds(
                                prompt_embeds_1=all_prompt_embeds[current_interval_idx - 1].unsqueeze(0),
                                prompt_embeds_2=all_prompt_embeds[current_interval_idx].unsqueeze(0),
                                interpolation_steps=interpolation_steps,
                            )
                            interpolate_interval_idx = current_interval_idx
                        cur_prompt_embeds = interpolate_embeds[position_in_interval]
                    else:
                        cur_prompt_embeds = all_prompt_embeds[current_interval_idx].unsqueeze(0)
            else:
                cur_prompt_embeds = all_prompt_embeds

            is_first_section = k == 0
            is_second_section = k == 1

            if is_keep_x0:
                indices = torch.arange(0, sum([1, *history_sizes, latent_window_size]), device=device)
                (
                    indices_prefix,
                    indices_latents_history_long,
                    indices_latents_history_mid,
                    indices_latents_history_1x,
                    indices_hidden_states,
                ) = indices.split([1, *history_sizes, latent_window_size], dim=0)
                indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=0)

                latents_history_long, latents_history_mid, latents_history_1x = history_latents[:, :, -sum(history_sizes):].split(
                    history_sizes, dim=2
                )
                if x0_latent is None:
                    latents_prefix = torch.zeros(batch_size, num_channels_latents, 1, device=device, dtype=torch.float32)
                else:
                    latents_prefix = x0_latent
                latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
            else:
                indices = torch.arange(0, sum([*history_sizes, latent_window_size]), device=device)
                (
                    indices_latents_history_long,
                    indices_latents_history_mid,
                    indices_latents_history_short,
                    indices_hidden_states,
                ) = indices.split([*history_sizes, latent_window_size], dim=0)
                latents_history_long, latents_history_mid, latents_history_short = history_latents[:, :, -sum(history_sizes):].split(
                    history_sizes, dim=2
                )

            indices_hidden_states = indices_hidden_states.unsqueeze(0).expand(batch_size, -1)
            indices_latents_history_short = indices_latents_history_short.unsqueeze(0).expand(batch_size, -1)
            indices_latents_history_mid = indices_latents_history_mid.unsqueeze(0).expand(batch_size, -1)
            indices_latents_history_long = indices_latents_history_long.unsqueeze(0).expand(batch_size, -1)

            cur_latents = self.prepare_latents(
                batch_size=batch_size,
                num_channels_latents=num_channels_latents,
                latent_length=latent_window_size,
                dtype=torch.float32,
                device=device,
                generator=generator,
                latents=latents,
            )

            if not is_enable_stage2:
                try:
                    self.scheduler.set_timesteps(num_inference_steps, mu=1, device=device)
                except TypeError:
                    self.scheduler.set_timesteps(num_inference_steps, device=device)

                if use_dynamic_shifting:
                    sigmas = torch.linspace(0.999, 0.0, steps=num_inference_steps + 1, dtype=torch.float32, device=device)[:-1]
                    sigmas = apply_schedule_shift(
                        sigmas=sigmas,
                        noise=cur_latents,
                        base_seq_len=self.scheduler.config.get("base_image_seq_len", 256),
                        max_seq_len=self.scheduler.config.get("max_image_seq_len", 4096),
                        base_shift=self.scheduler.config.get("base_shift", 0.5),
                        max_shift=self.scheduler.config.get("max_shift", 1.15),
                    )
                    timesteps = sigmas * 1000.0
                    self.scheduler.timesteps = timesteps.to(device)
                    self.scheduler.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])

                timesteps = self.scheduler.timesteps
                dmd_sigmas = self.scheduler.sigmas.to(self.transformer.device) if use_dmd else None
                dmd_timesteps = self.scheduler.timesteps.to(self.transformer.device) if use_dmd else None
                self._num_timesteps = len(timesteps)
                total_steps = num_inference_steps
            else:
                total_steps = (
                    sum(stage2_num_inference_steps_list) * 2
                    if is_amplify_first_chunk and use_dmd and is_first_section
                    else sum(stage2_num_inference_steps_list)
                )

            with self.progress_bar(total=total_steps) as progress_bar:
                if is_enable_stage2:
                    cur_latents = self.stage2_sample(
                        latents=cur_latents,
                        stage2_num_stages=stage2_num_stages,
                        stage2_num_inference_steps_list=stage2_num_inference_steps_list,
                        prompt_embeds=cur_prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds,
                        guidance_scale=guidance_scale,
                        indices_hidden_states=indices_hidden_states,
                        indices_latents_history_short=indices_latents_history_short,
                        indices_latents_history_mid=indices_latents_history_mid,
                        indices_latents_history_long=indices_latents_history_long,
                        latents_history_short=latents_history_short,
                        latents_history_mid=latents_history_mid,
                        latents_history_long=latents_history_long,
                        attention_kwargs=attention_kwargs,
                        device=device,
                        transformer_dtype=transformer_dtype,
                        scheduler_type=scheduler_type,
                        use_dynamic_shifting=use_dynamic_shifting,
                        generator=generator,
                        use_cfg_zero_star=use_cfg_zero_star,
                        use_zero_init=use_zero_init,
                        zero_steps=zero_steps,
                        use_dmd=use_dmd,
                        is_amplify_first_chunk=is_amplify_first_chunk and is_first_section,
                        callback_on_step_end=callback_on_step_end,
                        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                        progress_bar=progress_bar,
                    )
                else:
                    if return_ode_stages:
                        raise NotImplementedError("return_ode_stages is only supported for stage 2")
                    else:
                        cur_latents = self.stage1_sample(
                            latents=cur_latents,
                            prompt_embeds=cur_prompt_embeds,
                            negative_prompt_embeds=negative_prompt_embeds,
                            timesteps=timesteps,
                            guidance_scale=guidance_scale,
                            indices_hidden_states=indices_hidden_states,
                            indices_latents_history_short=indices_latents_history_short,
                            indices_latents_history_mid=indices_latents_history_mid,
                            indices_latents_history_long=indices_latents_history_long,
                            latents_history_short=latents_history_short,
                            latents_history_mid=latents_history_mid,
                            latents_history_long=latents_history_long,
                            attention_kwargs=attention_kwargs,
                            device=device,
                            transformer_dtype=transformer_dtype,
                            generator=generator,
                            use_cfg_zero_star=use_cfg_zero_star,
                            use_zero_init=use_zero_init,
                            zero_steps=zero_steps,
                            use_dmd=use_dmd,
                            dmd_sigmas=dmd_sigmas,
                            dmd_timesteps=dmd_timesteps,
                            callback_on_step_end=callback_on_step_end,
                            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                            progress_bar=progress_bar,
                        )

            if use_kv_cache:
                self.transformer.clear_kv_cache()

            if use_adaptive_anti_drifting:
                current_mean, current_var = anti_drifting_helper.compute_latent_statistics(cur_latents)
                anti_drifting_helper.update_global_statistics(current_mean, current_var)
                has_drift = anti_drifting_helper.detect_drift(current_mean, current_var)
                if has_drift and k < num_latent_sections - 1:
                    logger.info(f"Drift detected at chunk {k + 1}/{num_latent_sections}. Applying corruption.")
                    cur_latents = anti_drifting_helper.apply_frame_aware_corruption(
                        cur_latents,
                        corruption_strength=anti_drift_corruption_strength,
                        generator=generator,
                    )

            if is_keep_x0 and ((is_first_section and x0_latent is None) or (is_skip_first_section and is_second_section)):
                x0_latent = cur_latents[:, :, :1]

            total_generated_latent_frames += cur_latents.shape[2]
            history_latents = torch.cat([history_latents, cur_latents], dim=2)

        real_history_latents = history_latents[:, :, -total_generated_latent_frames:]
        real_history_latents = real_history_latents[:, :, :total_latent_length]
        self._current_timestep = None

        if output_type == "latent":
            output = real_history_latents
        else:
            audio = self.vae.decode(real_history_latents.to(vae_dtype))
            audio = audio[..., :total_audio_samples]
            if output_type == "np":
                output = audio.float().cpu().numpy()
            else:
                output = audio

        self.maybe_free_model_hooks()

        if not return_dict:
            return (output,)

        return HeliosPipelineOutput(frames=output)
