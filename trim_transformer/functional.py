import torch

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True

except Exception:
    # Triton not available on this system – fall back to PyTorch ops.
    # If the mask has many unique values, attention will be slow.
    _TRITON_AVAILABLE = False
_TRITON_AVAILABLE = False


def _multi_linear_attn_no_mask(query, key, value, dropout_p, kv_cache):
    key_value_store = key.transpose(-2, -1) @ value + kv_cache  # [..., d_k, d_v]
    key_value_store_view = torch.dropout(key_value_store, dropout_p, train=True)
    out = query @ key_value_store_view
    return out, key_value_store


def _multi_linear_attn_mask_torch(query, key, value, mask, dropout_p, kv_cache):
    key = key.unsqueeze(-1)  # [..., S, d_k, 1]
    value = value.unsqueeze(-2)  # [..., S, 1, d_v]
    key_value_store = key @ value  # [..., S, d_k, d_v]
    key_value_store = key_value_store.cumsum(dim=-3)[..., mask, :, :] + kv_cache.unsqueeze(-3)
    key_value_store = torch.dropout(key_value_store, dropout_p, train=True)
    out = (query.unsqueeze(-2) @ key_value_store).squeeze(-2)
    key_value_store = key_value_store[..., -1, :, :]
    return out, key_value_store


def pad_to_pow2(x: torch.Tensor, dims=None) -> torch.Tensor:
    shape = torch.as_tensor(x.shape, device=x.device, dtype=torch.long)

    if dims is None:
        mask = torch.ones_like(shape, dtype=torch.bool)
    else:
        idx = torch.remainder(
            torch.as_tensor(dims, device=x.device, dtype=torch.long), x.dim()
        )
        mask = torch.zeros_like(shape, dtype=torch.bool)
        mask[idx] = True

    next_pow2 = (2 ** torch.ceil(torch.log2(shape.float()))).long()
    target = torch.where(mask, next_pow2, shape)

    out = x.new_zeros(tuple(target.tolist()))
    out[tuple(slice(0, int(s)) for s in shape)] = x
    return out

if _TRITON_AVAILABLE:
    def _multi_linear_attn_mask_triton(query, key, value, mask, dropout_p, kv_cache):
        B, H, S, DK = query.shape
        R, DV          = value.shape[-2:]

        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        mask = mask.contiguous()
        kv_cache = kv_cache.contiguous()

        out = torch.empty((B, H, S, DV), device=query.device, dtype=query.dtype)
        kv_out = torch.empty((B, H, DK, DV), device=query.device, dtype=query.dtype)

        grid = (B * H,)

        base_seed = torch.randint(2**31, (1,), device=query.device, dtype=torch.int32).item()

        _mla_mask_kernel[grid](
            q, k, v, mask, kv_cache,
            out, kv_out,
            S=S,
            R=R,
            DK=DK,
            DV=DV,
            dropout_p=dropout_p,
            seed=base_seed,
        )

        return out, kv_out


    @triton.jit
    def _mla_mask_kernel(
        Q_ptr, K_ptr, V_ptr, MASK_ptr, kv_cache_ptr,
        OUT_ptr, OUT_KV_cache_ptr,
        S: tl.constexpr,
        R: tl.constexpr,
        DK: tl.constexpr,
        DV: tl.constexpr,
        dropout_p,
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
            delta   = tgt_idx - prev_idx
            delta   = tl.maximum(delta, 0)
            for step in tl.range(delta):  # type: ignore
                idx  = prev_idx + 1 + step
                k_vec = tl.load(K_ptr + idx * DK + tl.arange(0, DK))
                v_vec = tl.load(V_ptr + idx * DV + tl.arange(0, DV))
                running_kv += k_vec[:, None] * v_vec[None, :]

            prev_idx = tgt_idx
            q_vec   = tl.load(Q_ptr + pos * DK + tl.arange(0, DK))

            if dropout_p > 0.0:
                dropout_scaling = 1.0 - dropout_p
                rng_seed = seed + pid * S + pos
                offs_k = tl.arange(0, DK)[:, None] * DV
                offs_v = tl.arange(0, DV)[None, :]
                offset = offs_k + offs_v
                rnd = tl.rand(rng_seed, offset)
                dropout_mask = rnd >= dropout_p
                kv_view = running_kv * dropout_mask / dropout_scaling
            else:
                kv_view = running_kv

            out_vec = tl.sum(kv_view * q_vec[:, None], axis=0)
            tl.store(OUT_ptr + pos * DV + tl.arange(0, DV), out_vec)
        tl.store(OUT_KV_cache_block_ptr, running_kv)

def multi_linear_attn(
    query,
    key,
    value,
    mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale=None,
    enable_gqa: bool = False,
    kv_cache=None,
):
    seq_len = query.size(-2)
    B, H, dict_size, DK = key.shape
    DV = value.shape[-1]
    scale_factor = 1 / dict_size if scale is None else scale

    if mask is not None:
        assert mask.shape == (seq_len,)
        assert torch.all(0 <= mask) & torch.all(mask < dict_size)
        mask = mask.to(torch.int32)

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
                mask,
                dropout_p,
                kv_cache,
            )
            out = out[..., :DV]
            kv_cache = kv_cache[:, :, :DK, :DV]
        else:
            out, kv_cache = _multi_linear_attn_mask_triton(
                query,
                key,
                value,
                mask,
                dropout_p,
                kv_cache,
            )
        return out, kv_cache
    elif mask is not None:
        return _multi_linear_attn_mask_torch(
            query,
            key,
            value,
            mask,
            dropout_p,
            kv_cache,
        )
    else:
        return _multi_linear_attn_no_mask(
            query,
            key,
            value,
            dropout_p,
            kv_cache
        )