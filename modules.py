import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from torch.nn.functional import _in_projection, _in_projection_packed
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from typing import Optional

from functional import cumulative_linear_attn

class CumulativeLinearMultiheadAttentionKV(Module):
    """
    MultiheadAttention module using cumulative linear attention instead of 
    standard scaled dot product attention.
    
    This implementation follows the same interface as PyTorch's MultiheadAttention
    but uses cumulative linear attention for more efficient computation.

    Additionally implements a key-value cache for inference.
    """
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        kdim=None,
        vdim=None,
        batch_first=False,
        device=None,
        dtype=None,
    ) -> None:
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError(
                f"embed_dim and num_heads must be greater than 0,"
                f" got embed_dim={embed_dim} and num_heads={num_heads} instead"
            )
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = Parameter(
                torch.empty((embed_dim, embed_dim), **self.factory_kwargs)
            )
            self.k_proj_weight = Parameter(
                torch.empty((embed_dim, self.kdim), **self.factory_kwargs)
            )
            self.v_proj_weight = Parameter(
                torch.empty((embed_dim, self.vdim), **self.factory_kwargs)
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = Parameter(
                torch.empty((3 * embed_dim, embed_dim), **self.factory_kwargs)
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **self.factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)
        self.out_proj = NonDynamicallyQuantizableLinear(
            embed_dim, embed_dim, bias=bias, **self.factory_kwargs
        )

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **self.factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **self.factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.kv_cache = None
        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.0)
            constant_(self.out_proj.bias, 0.0)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old checkpoints
        if "_qkv_same_embed_dim" not in state:
            state["_qkv_same_embed_dim"] = True
        super().__setstate__(state)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        is_causal: bool = False,
        mask_after: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        use_kv_cache: bool = False,
        update_kv_cache: bool = False,
    ) -> tuple[Tensor, Optional[Tensor]]:
        is_batched = query.dim() == 3

        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)
        assert query.dim() == key.dim() == value.dim() == 3

        if self.batch_first and is_batched:
            # Convert from batch_first to seq_first
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        # Get dimensions
        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape
        
        # Input projections
        if not self._qkv_same_embed_dim:
            q, k, v = _in_projection(
                query, key, value,
                self.q_proj_weight, self.k_proj_weight, self.v_proj_weight,
                self.in_proj_bias[0:embed_dim] if self.in_proj_bias is not None else None,
                self.in_proj_bias[embed_dim:2*embed_dim] if self.in_proj_bias is not None else None,
                self.in_proj_bias[2*embed_dim:] if self.in_proj_bias is not None else None,
            )
        else:
            q, k, v = _in_projection_packed(query, key, value, self.in_proj_weight, self.in_proj_bias)

        # Add bias vectors if specified
        if self.bias_k is not None and self.bias_v is not None:
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            src_len += 1

        # Reshape for multi-head attention: (seq_len, batch, embed_dim) -> (batch, num_heads, seq_len, head_dim)
        q = q.view(tgt_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        k = k.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)
        v = v.view(src_len, bsz, self.num_heads, self.head_dim).transpose(0, 1).transpose(1, 2)

        if src_key_padding_mask is not None:
            mask_expanded = src_key_padding_mask.unsqueeze(1).unsqueeze(3)
            mask_expanded = mask_expanded.expand(bsz, self.num_heads, -1, self.head_dim)
            q = q.masked_fill(mask_expanded[:, :, :q.size(2), :], 0.0)
            k = k.masked_fill(mask_expanded[:, :, :k.size(2), :], 0.0)
            v = v.masked_fill(mask_expanded[:, :, :v.size(2), :], 0.0)

        # Apply cumulative linear attention
        dropout_p = self.dropout if self.training else 0.0

        if use_kv_cache:
            if self.kv_cache is None:
                self.kv_cache = torch.zeros(bsz, self.num_heads, 1, self.kdim//self.num_heads, self.vdim//self.num_heads, **self.factory_kwargs)
            kv_cache = self.kv_cache
            assert kv_cache.shape == (bsz, self.num_heads, 1, self.kdim//self.num_heads, self.vdim//self.num_heads)
        else:
            kv_cache = None

        attn_output, key_value_store = cumulative_linear_attn(
            q, k, v, 
            mask_after=mask_after,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=None,
            enable_gqa=False,
            kv_cache=kv_cache
        )

        if update_kv_cache:
            self.kv_cache = key_value_store[:, :, -1:, :, :]

        # Reshape back: (batch, num_heads, tgt_len, head_dim) -> (tgt_len, batch, embed_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, embed_dim)
        attn_output = attn_output.transpose(0, 1)

        # Output projection
        attn_output = torch.nn.functional.linear(attn_output.contiguous().view(-1, embed_dim), 
                                               self.out_proj.weight, self.out_proj.bias)
        attn_output = attn_output.view(tgt_len, bsz, embed_dim)

        # Convert back to batch_first if needed
        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        # Remove batch dimension if input was unbatched
        if not is_batched:
            attn_output = attn_output.squeeze(1)

        return attn_output

    def clear_kv_cache(self):
        self.kv_cache = None