import os

import torch
try:
    from kernels import get_kernel
except Exception:
    get_kernel = None


SKIP_FLASH_KERNEL_DOWNLOAD = os.getenv("HELIOS_SKIP_FLASH_KERNEL_DOWNLOAD", "0") == "1"

try:
    if get_kernel is None:
        raise RuntimeError("python package `kernels` is not available")
    if SKIP_FLASH_KERNEL_DOWNLOAD:
        raise RuntimeError("Skip flash kernel download by HELIOS_SKIP_FLASH_KERNEL_DOWNLOAD=1")
    try:
        flash_attn3 = get_kernel("kernels-community/flash-attn3")
        flash_attn_func = flash_attn3.flash_attn_func
        flash_attn_varlen_func = flash_attn3.flash_attn_varlen_func
        print("Flash Attn 3 is installed!")
    except Exception:
        flash_attn2 = get_kernel("kernels-community/flash-attn2")
        flash_attn_func = flash_attn2.flash_attn_func
        flash_attn_varlen_func = flash_attn2.flash_attn_varlen_func
        print("Flash Attn 2 is installed!")
except Exception as e:
    print(f"Flash Attn 2 / 3 is unavailable, fallback to torch attention. reason={e}")
    flash_attn_varlen_func = None
    flash_attn_func = None

try:
    # raise NotImplementedError
    from sageattention import sageattn, sageattn_varlen

    print("Sage Attn is installed!")
except ImportError:
    print("Sage Attn is not installed!")
    sageattn_varlen = None
    sageattn = None

try:
    # raise NotImplementedError
    from xformers.ops import memory_efficient_attention as xformers_attn_func

    print("Xformers is installed!")
except ImportError:
    print("Xformers is not installed!")
    xformers_attn_func = None


def create_navit_attention_masks(
    batch_size: int,
    original_context_length_list: list,
    history_context_length: int,
    encoder_hidden_states_seq_len: int,
    device: torch.device,
    restrict_self_attn: bool = False,
    guidance_cross_attn: bool = False,
):
    # For navit_hidden_attention_mask
    if restrict_self_attn:
        cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_q.append(cu_seqlens_q[-1] + length)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, device=device, dtype=torch.int32)
        max_seqlen_q = max(original_context_length_list)

        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + length + history_context_length)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(original_context_length_list) + history_context_length
    else:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + length + history_context_length)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = max(original_context_length_list) + history_context_length
        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv
    navit_hidden_attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    # For navit_history_hidden_attention_mask
    navit_history_hidden_attention_mask = None
    if restrict_self_attn:
        cu_seqlens_kv = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cu_seqlens_kv.append(cu_seqlens_kv[-1] + history_context_length)
        cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
        max_seqlen_kv = history_context_length
        cu_seqlens_q = cu_seqlens_kv
        max_seqlen_q = max_seqlen_kv
        navit_history_hidden_attention_mask = cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv

    # For navit_encoder_attention_mask
    if guidance_cross_attn:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cross_cu_seqlens_q.append(cross_cu_seqlens_q[-1] + length)
        cross_cu_seqlens_q = torch.tensor(cross_cu_seqlens_q, device=device, dtype=torch.int32)
        cross_max_seqlen_q = max(original_context_length_list)
    else:
        cross_cu_seqlens_q = [0]
        for _ in range(batch_size):
            for length in original_context_length_list:
                cross_cu_seqlens_q.append(cross_cu_seqlens_q[-1] + length + history_context_length)
        cross_cu_seqlens_q = torch.tensor(cross_cu_seqlens_q, device=device, dtype=torch.int32)
        cross_cu_seqlens_q[0] = 0
        cross_max_seqlen_q = max(original_context_length_list) + history_context_length

    cu_seqlens_kv = [0]
    for _ in range(batch_size):
        for length in original_context_length_list:
            cu_seqlens_kv.append(cu_seqlens_kv[-1] + encoder_hidden_states_seq_len)
    cu_seqlens_kv = torch.tensor(cu_seqlens_kv, device=device, dtype=torch.int32)
    max_seqlen_kv = encoder_hidden_states_seq_len
    navit_encoder_attention_mask = cross_cu_seqlens_q, cu_seqlens_kv, cross_max_seqlen_q, max_seqlen_kv

    return navit_hidden_attention_mask, navit_encoder_attention_mask, navit_history_hidden_attention_mask


@torch.compiler.disable
def _flash_attn_wrapper(q, k, v):
    return flash_attn_func(q, k, v)


@torch.compiler.disable
def _flash_attn_varlen_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv):
    return flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)


def _torch_sdpa_varlen_fallback(q, k, v, cu_seqlens_q, cu_seqlens_kv):
    """Fallback varlen attention using torch SDPA per segment."""
    out = torch.empty_like(q)
    num_segments = cu_seqlens_q.numel() - 1
    for i in range(num_segments):
        q_start = int(cu_seqlens_q[i].item())
        q_end = int(cu_seqlens_q[i + 1].item())
        k_start = int(cu_seqlens_kv[i].item())
        k_end = int(cu_seqlens_kv[i + 1].item())
        if q_end <= q_start or k_end <= k_start:
            continue

        q_seg = q[q_start:q_end]  # [Lq, H, C]
        k_seg = k[k_start:k_end]  # [Lk, H, C]
        v_seg = v[k_start:k_end]  # [Lk, H, C]

        # SDPA expects [B, H, L, C]
        q_b = q_seg.permute(1, 0, 2).unsqueeze(0)
        k_b = k_seg.permute(1, 0, 2).unsqueeze(0)
        v_b = v_seg.permute(1, 0, 2).unsqueeze(0)
        o_b = torch.nn.functional.scaled_dot_product_attention(q_b, k_b, v_b)
        out[q_start:q_end] = o_b.squeeze(0).permute(1, 0, 2)
    return out


def attn_varlen_func(q, k, v, attention_mask=None):
    if attention_mask is None:
        if flash_attn_func is not None:
            x = _flash_attn_wrapper(q, k, v)
            return x

        if sageattn is not None:
            x = sageattn(q, k, v, tensor_layout="NHD")
            return x

        if xformers_attn_func is not None:
            x = xformers_attn_func(q, k, v)
            return x

        x = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
        return x

    B, L, H, C = q.shape

    q = q.flatten(0, 1)
    k = k.flatten(0, 1)
    v = v.flatten(0, 1)

    cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv = attention_mask
    if flash_attn_varlen_func is not None:
        x = _flash_attn_varlen_wrapper(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    elif sageattn_varlen is not None:
        x = sageattn_varlen(q, k, v, cu_seqlens_q, cu_seqlens_kv, max_seqlen_q, max_seqlen_kv)
    else:
        x = _torch_sdpa_varlen_fallback(q, k, v, cu_seqlens_q, cu_seqlens_kv)

    x = x.unflatten(0, (B, L))

    return x
