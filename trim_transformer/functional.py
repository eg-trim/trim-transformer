import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True

except Exception:
    # Triton not available on this system – fall back to PyTorch ops.
    # If the sequence length is long, and mask is not None, attention will be memory intensive.
    _TRITON_AVAILABLE = False

def set_triton_available(available: bool):
    global _TRITON_AVAILABLE
    _TRITON_AVAILABLE = available   


def _multi_linear_attn_no_mask(query, key, value, kv_cache, dropout_p):
    key_value_store = key.transpose(-2, -1) @ value + kv_cache  # [..., d_k, d_v]
    key_value_store_view = torch.dropout(key_value_store, dropout_p, train=True)
    out = query @ key_value_store_view
    return out, key_value_store

def _multi_linear_attn_mask_torch(query, key, value, kv_cache, mask, dropout_p):
    key = key.unsqueeze(-1)
    value = value.unsqueeze(-2)
    key_value_store = key @ value

    safe_mask = mask.clone()
    neg_mask = safe_mask < 0
    safe_mask[neg_mask] = 0
    key_value_store = key_value_store.cumsum(dim=-3)[..., safe_mask, :, :]
    if neg_mask.any():
        key_value_store[..., neg_mask, :, :] = 0

    key_value_store = key_value_store + kv_cache.unsqueeze(-3)
    key_value_store = torch.dropout(key_value_store, dropout_p, train=True)
    out = (query.unsqueeze(-2) @ key_value_store).squeeze(-2)
    key_value_store = key_value_store[..., -1, :, :]
    return out, key_value_store

def pad_to_pow2(x: torch.Tensor, dims=None) -> torch.Tensor:
    shape = torch.as_tensor(x.shape, device=x.device, dtype=torch.int32)

    if dims is None:
        mask = torch.ones_like(shape, dtype=torch.bool)
    else:
        idx = torch.remainder(
            torch.as_tensor(dims, device=x.device, dtype=torch.int32), x.dim()
        )
        mask = torch.zeros_like(shape, dtype=torch.bool)
        mask[idx] = True

    next_pow2 = (2 ** torch.ceil(torch.log2(shape.float()))).to(torch.int32)
    target = torch.where(mask, next_pow2, shape)

    out = x.new_zeros(tuple(target.tolist()))
    out[tuple(slice(0, int(s)) for s in shape)] = x
    return out

def get_seed() -> int:
    return int(torch.randint(
            2 ** 31, (1,), dtype=torch.int32
    ).item())

def build_reverse_mask(mask : torch.Tensor, num_keys: int) -> torch.Tensor:
    if mask.ndim != 1 or not torch.all(mask[:-1] <= mask[1:]):
        raise ValueError("mask must be a 1-D non-decreasing tensor")

    key_indices = torch.arange(num_keys, device=mask.device, dtype=mask.dtype)

    attend_to_ge = torch.searchsorted(mask, key_indices, right=False).to(torch.int32)
    mask_le = attend_to_ge - 1
    mask_g_flipped_q = mask.shape[0] - mask_le - 1
    mask_ge_flipped_q = mask_g_flipped_q - 1
    mask_ge_flipped_q_k = torch.flip(mask_ge_flipped_q, (0,))
    return mask_ge_flipped_q_k

if _TRITON_AVAILABLE:

    @triton.jit
    def _mla_mask_kernel(
        Q_ptr, K_ptr, V_ptr, kv_cache_ptr,
        MASK_ptr,
        OUT_ptr, OUT_KV_cache_ptr,
        S: tl.constexpr,
        R: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        dropout_p: tl.constexpr,
        accumulate_dropout: tl.constexpr,
        transpose_dropout: tl.constexpr,
        reverse_counter: tl.constexpr,
        seed,
    ):
        pid = tl.program_id(axis=0)
        stride_qb = S * DK
        stride_kb = R * DK
        stride_vb = R * DV
        stride_ob = S * DV

        Q_ptr  = Q_ptr  + pid * stride_qb
        K_ptr  = K_ptr  + pid * stride_kb
        V_ptr  = V_ptr  + pid * stride_vb
        OUT_ptr = OUT_ptr + pid * stride_ob

        kv_block_ptr = tl.make_block_ptr(
            base=kv_cache_ptr + pid * DK * DV,
            shape=(DK, DV),
            strides=(DV, 1),
            offsets=(0, 0),
            block_shape=(DK, DV),
            order=(0, 1),
        )
        OUT_KV_cache_block_ptr = tl.make_block_ptr(
            base=OUT_KV_cache_ptr + pid * DK * DV,
            shape=(DK, DV),
            strides=(DV, 1),
            offsets=(0, 0),
            block_shape=(DK, DV),
            order=(0, 1),
        )
        running_kv = tl.load(kv_block_ptr)

        prev_idx = tl.full((), -1, tl.int32)

        for pos in tl.range(0, S):  # type: ignore
            tgt_idx = tl.load(MASK_ptr + pos)
            tgt_idx = tl.minimum(tgt_idx, R-1)
            delta   = tgt_idx - prev_idx
            delta   = tl.maximum(delta, 0)
            for step in tl.range(delta):  # type: ignore
                idx  = prev_idx + 1 + step
                k_vec = tl.load(K_ptr + idx * DK + tl.arange(0, DK))
                v_vec = tl.load(V_ptr + idx * DV + tl.arange(0, DV))
                next_kv = k_vec[:, None] * v_vec[None, :]

                if dropout_p > 0.0 and accumulate_dropout:
                    dropout_scale = 1.0 - dropout_p

                    if reverse_counter:
                        rng_seed = seed + pid * R + (R - 1 - idx)
                    else:
                        rng_seed = seed + pid * S + pos

                    if transpose_dropout:
                        offs_k = tl.arange(0, DV)[:, None] * DK
                        offs_v = tl.arange(0, DK)[None, :]
                        offset = offs_k + offs_v
                        rnd = tl.rand(rng_seed, offset)
                        rnd = tl.trans(rnd, (1, 0))
                    else:
                        offs_k = tl.arange(0, DK)[:, None] * DV
                        offs_v = tl.arange(0, DV)[None, :]
                        offset = offs_k + offs_v
                        rnd = tl.rand(rng_seed, offset)
                    dropout_mask = rnd >= dropout_p
                    running_kv += tl.where(dropout_mask, next_kv, 0.0) / dropout_scale
                else:
                    running_kv += next_kv

            if dropout_p > 0.0 and not accumulate_dropout:
                dropout_scale = 1.0 - dropout_p
                rng_seed = seed + pid * S + pos
                if transpose_dropout:
                    offs_k = tl.arange(0, DV)[:, None] * DK
                    offs_v = tl.arange(0, DK)[None, :]
                    offset = offs_k + offs_v
                    rnd = tl.rand(rng_seed, offset)
                    rnd = tl.trans(rnd, (1, 0))
                else:
                    offs_k = tl.arange(0, DK)[:, None] * DV
                    offs_v = tl.arange(0, DV)[None, :]
                    offset = offs_k + offs_v
                    rnd = tl.rand(rng_seed, offset)
                dropout_mask = rnd >= dropout_p
                kv_view = tl.where(dropout_mask, running_kv, 0.0) / dropout_scale
            else:
                kv_view = running_kv

            q_vec   = tl.load(Q_ptr + pos * DK + tl.arange(0, DK))
            out_vec = tl.sum(kv_view * q_vec[:, None], axis=0)
            tl.store(OUT_ptr + pos * DV + tl.arange(0, DV), out_vec)
            prev_idx = tl.maximum(prev_idx, tgt_idx)
        tl.store(OUT_KV_cache_block_ptr, running_kv)

def _run_mla_mask_triton_kernel(query, key, value, kv_cache, mask, dropout_p, accumulate_dropout=False, transpose_dropout=False, reverse_counter=False, seed=None):
    B, H, S, DK = query.shape
    R, DV = value.shape[-2:]

    q = query.contiguous()
    k = key.contiguous()
    v = value.contiguous()
    kv_c = kv_cache.contiguous()
    m = mask.contiguous()

    out = torch.empty((B, H, S, DV), device=query.device, dtype=query.dtype)
    kv_out = torch.empty((B, H, DK, DV), device=query.device, dtype=query.dtype)

    grid = (B * H,)

    if seed is None:
        seed = get_seed()

    _mla_mask_kernel[grid](
        q,
        k,
        v,
        kv_c,
        m,
        out,
        kv_out,
        S=S,
        R=R,
        DK=DK,
        DV=DV,
        dropout_p=tl.constexpr(dropout_p),
        accumulate_dropout=tl.constexpr(accumulate_dropout),
        transpose_dropout=tl.constexpr(transpose_dropout),
        reverse_counter=tl.constexpr(reverse_counter),
        seed=seed,
    )

    return out, kv_out

class _MLAMaskAutogradFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query, key, value, kv_cache, mask, dropout_p: float, seed=None):
        if seed is None:
            seed = get_seed()

        ctx.save_for_backward(query, key, value, kv_cache, mask)
        ctx.dropout_p = dropout_p
        ctx.seed = seed

        out, kv_out = _run_mla_mask_triton_kernel(query, key, value, kv_cache, mask, dropout_p, seed=seed)

        return out, kv_out

    @staticmethod
    def backward(ctx, grad_out, grad_kv_out):
        query, key, value, kv_cache, mask = ctx.saved_tensors
        seed = ctx.seed
        reverse_kv_cache = kv_cache.transpose(-2, -1)
        reverse_mask = build_reverse_mask(mask, key.shape[-2])
        reverse_query = query.flip(-2)
        reverse_key = key.flip(-2)
        reverse_value = value.flip(-2)
        reverse_grad_out = grad_out.flip(-2)
        empty_query = torch.zeros_like(query)
        empty_kv = torch.zeros_like(kv_cache)
        causal_mask = torch.arange(mask.shape[0], device=mask.device, dtype=mask.dtype)

        dQ, _ = _run_mla_mask_triton_kernel(grad_out, value, key, reverse_kv_cache,
                                            mask, ctx.dropout_p,
                                            accumulate_dropout=False, transpose_dropout=True, reverse_counter=False,
                                            seed=seed)
        dK, _ = _run_mla_mask_triton_kernel(reverse_value, reverse_grad_out, reverse_query, empty_kv,
                                            reverse_mask, ctx.dropout_p,
                                            accumulate_dropout=True, transpose_dropout=True, reverse_counter=True, 
                                            seed=seed)
        dV, _ = _run_mla_mask_triton_kernel(reverse_key, reverse_query, reverse_grad_out, empty_kv,
                                            reverse_mask, ctx.dropout_p,
                                            accumulate_dropout=True, transpose_dropout=False, reverse_counter=True,
                                            seed=seed)
        _, dKv = _run_mla_mask_triton_kernel(empty_query, query, grad_out, empty_kv,
                                            causal_mask, ctx.dropout_p,
                                            accumulate_dropout=True, transpose_dropout=False, reverse_counter=False,
                                            seed=seed)
        dK = dK.flip(-2)
        dV = dV.flip(-2)

        max_active_idx = mask[-1]
        kv_grad_mask = (torch.arange(key.shape[-2], device=key.device, dtype=torch.int32) <= max_active_idx)
        kv_grad_mask = kv_grad_mask.view(1, 1, -1, 1)

        dK = dK + (value @ grad_kv_out.transpose(-2, -1)) * kv_grad_mask
        dV = dV + (key @ grad_kv_out) * kv_grad_mask
        dKv = dKv + grad_kv_out

        return dQ, dK, dV, dKv, None, None, None

def _multi_linear_attn_mask_triton(query, key, value, kv_cache, mask, dropout_p, seed=None):
    out, kv_cache = _MLAMaskAutogradFunction.apply(
        query, key, value, kv_cache, mask, dropout_p, seed
    )
    return out, kv_cache

def multi_linear_attn(
    query,
    key,
    value,
    kv_cache=None,
    mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    enable_gqa: bool = False,
    seed=None,
):
    seq_len = query.size(-2)
    B, H, dict_size, DK = key.shape
    DV = value.shape[-1]
    scale_factor = 1 / dict_size if scale is None else scale

    received_mask = mask is not None
    if received_mask:
        assert mask.shape == (seq_len,)
        mask = mask.to(torch.int32)
        sorted_mask, perm = torch.sort(mask)
        inv_perm = torch.argsort(perm)
        query = query[..., perm, :]
        mask = sorted_mask

    if is_causal:
        assert mask is None
        mask = torch.arange(dict_size-seq_len, dict_size, dtype=torch.int32)
        mask = mask.to(query.device)

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    if kv_cache is None:
        kv_cache = torch.zeros(B, H, DK, DV, device=query.device, dtype=query.dtype)

    query = query * scale_factor
    use_triton = mask is not None and _TRITON_AVAILABLE and query.device.type == 'cuda'
    if use_triton:
        padding_needed = triton.next_power_of_2(DK) != DK or triton.next_power_of_2(DV) != DV
        if padding_needed:
            query = pad_to_pow2(query, dims=[-1])
            key = pad_to_pow2(key, dims=[-1])
            value = pad_to_pow2(value, dims=[-1])
            kv_cache = pad_to_pow2(kv_cache, dims=[-2, -1])
            out, kv_cache = _multi_linear_attn_mask_triton(
                query,
                key,
                value,
                kv_cache,
                mask,
                dropout_p,
                seed=seed,
            )
            out = out[..., :DV]
            kv_cache = kv_cache[:, :, :DK, :DV]
        else:
            out, kv_cache = _multi_linear_attn_mask_triton(
                query,
                key,
                value,
                kv_cache,
                mask,
                dropout_p,
                seed=seed,
            )
    elif mask is not None:
        out, kv_cache = _multi_linear_attn_mask_torch(
            query,
            key,
            value,
            kv_cache,
            mask,
            dropout_p,
        )
    else:
        out, kv_cache = _multi_linear_attn_no_mask(
            query,
            key,
            value,
            kv_cache,
            dropout_p,
        )
    if received_mask:
        out = out[..., inv_perm, :]
    return out, kv_cache