#!/usr/bin/env python3
"""Smoke test: patch_ref increases sequence length in transformer."""

import sys

import torch

sys.path.insert(0, "/root/autodl-tmp/Helios")

from helios.modules.transformer_helios import HeliosTransformer3DModel


def main():
    device = "cpu"
    dtype = torch.float32
    model = HeliosTransformer3DModel(
        in_channels=16,
        out_channels=16,
        num_layers=2,
        num_attention_heads=4,
        attention_head_dim=32,
        has_multi_term_memory_patch=True,
        zero_history_timestep=True,
    ).to(device=device, dtype=dtype)
    model.eval()

    b, c, t, h, w = 1, 16, 9, 8, 8
    target = torch.randn(b, c, t, h, w, device=device, dtype=dtype)
    short = torch.randn(b, c, 2, h, w, device=device, dtype=dtype)
    mid = torch.randn(b, c, 2, h, w, device=device, dtype=dtype)
    long_h = torch.randn(b, c, 16, h, w, device=device, dtype=dtype)
    ref = torch.randn(b, c, 3, h, w, device=device, dtype=dtype)

    with torch.no_grad():
        hs0, _, _, _, _, seq0 = model.process_input_hidden_states(
            latents=target,
            indices_hidden_states=torch.arange(t).unsqueeze(0),
            latents_history_short=short,
            indices_latents_history_short=torch.tensor([[0, 1]]),
            latents_history_mid=mid,
            indices_latents_history_mid=torch.tensor([[2, 3]]),
            latents_history_long=long_h,
            indices_latents_history_long=torch.arange(16).unsqueeze(0),
        )
        hs1, _, _, _, _, seq1 = model.process_input_hidden_states(
            latents=target,
            indices_hidden_states=torch.arange(t).unsqueeze(0),
            latents_history_short=short,
            indices_latents_history_short=torch.tensor([[0, 1]]),
            latents_history_mid=mid,
            indices_latents_history_mid=torch.tensor([[2, 3]]),
            latents_history_long=long_h,
            indices_latents_history_long=torch.arange(16).unsqueeze(0),
            latents_history_ref=ref,
            indices_latents_history_ref=torch.tensor([[-5, -3, -1]]),
        )

    delta = hs1.shape[1] - hs0.shape[1]
    ref_tokens = ref.shape[2] * (h // 2) * (w // 2)
    assert delta == ref_tokens, f"expected +{ref_tokens} tokens, got +{delta}"
    print(f"patch_ref added {delta} tokens (expected {ref_tokens})")
    print("test_patch_ref_transformer: PASSED")


if __name__ == "__main__":
    main()
