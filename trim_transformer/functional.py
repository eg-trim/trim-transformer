import torch
import math

def cumulative_linear_attn(query, key, value, mask = None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False, kv_cache=None):
    seq_len = query.size(-2)
    dict_size = key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

    if is_causal:
        assert mask is None
        mask = torch.arange(dict_size-seq_len, dict_size, dtype=torch.int32)
        mask = mask.to(query.device)
    elif mask is None:
        mask = torch.full((seq_len,), dict_size-1, dtype=torch.int32)
        mask = mask.to(query.device)
    assert torch.all(0 <= mask) & torch.all(mask < dict_size)
    assert mask.shape == (seq_len,)

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    if kv_cache is None:
        kv_cache = torch.zeros(key.size(-1), value.size(-1), device=query.device)

    query = query * scale_factor
    key = key.unsqueeze(-1)  # [..., S, d_k, 1]
    value = value.unsqueeze(-2)  # [..., S, 1, d_v]
    key_value_store = key @ value  # [..., S, d_k, d_v]
    key_value_store = key_value_store.cumsum(dim=-3)[..., mask, :, :] + kv_cache
    key_value_store = torch.dropout(key_value_store, dropout_p, train=True)
    return (query.unsqueeze(-2) @ key_value_store).squeeze(-2), key_value_store