import copy
import warnings
from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.init import xavier_uniform_
from torch.nn import Module, ModuleList, Dropout, Linear, LayerNorm

from .modules import CumulativeLinearMultiheadAttentionKV


def _get_clones(module, N):
    return ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation: Union[str, Callable[[Tensor], Tensor]]) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "tanh":
        return torch.tanh
    elif activation == "sigmoid":
        return torch.sigmoid
    
    return activation


class CumulativeTransformerEncoderLayerKV(Module):
    r"""CumulativeTransformerEncoderLayerKV is made up of cumulative self-attention and feedforward network.
    
    This is a modified version of PyTorch's TransformerEncoderLayer that uses cumulative linear attention
    instead of standard scaled dot product attention for more efficient computation.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectively. Otherwise it's done after. Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.
        pos_emb: the positional encoding module. Defaults to None.

    Examples::
        >>> encoder_layer = CumulativeTransformerEncoderLayerKV(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = CumulativeTransformerEncoderLayerKV(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """


    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        bias: bool = True,
        pos_emb: Optional[Module] = None,
        norm_q: Optional[Module] = None,
        norm_k: Optional[Module] = None,
        norm_v: Optional[Module] = None,
        q_weight_init: Optional[Callable[[Tensor], Tensor]] | Optional[Callable[[Tensor], None]] = None,
        k_weight_init: Optional[Callable[[Tensor], Tensor]] | Optional[Callable[[Tensor], None]] = None,
        v_weight_init: Optional[Callable[[Tensor], Tensor]] | Optional[Callable[[Tensor], None]] = None,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.self_attn = CumulativeLinearMultiheadAttentionKV(
            d_model,
            nhead,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            norm_q=norm_q,
            norm_k=norm_k,
            norm_v=norm_v,
            q_weight_init=q_weight_init,
            k_weight_init=k_weight_init,
            v_weight_init=v_weight_init,
            **factory_kwargs,
        )
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)
        self.batch_first = batch_first
        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.pos_emb = pos_emb
        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        use_kv_cache: bool = False,
        update_kv_cache: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            mask: the mask tensor for cumulative attention (optional).
                Should be of shape (seq_len,) with integer values indicating 
                the last position each token can attend to.
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask. Will override mask.
                Default: ``False``.
            use_kv_cache: Whether to use key-value caching for efficiency.
                Default: ``False``.
            update_kv_cache: Whether to update the key-value cache.
                Default: ``False``.

        Shape:
            - src: (N, S, E) if batch_first=True or (S, N, E) if batch_first=False
            - mask: (S,) where S is sequence length
        """
        x = src
        assert x.ndim == 3, "Input must have three axes."
        if not self.batch_first:
            x = x.permute(1, 0, 2)
        if self.pos_emb is not None:
            x = self.pos_emb(x)
        if not self.batch_first:
            x = x.permute(1, 0, 2)

        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x), mask, src_key_padding_mask, is_causal=is_causal,
                use_kv_cache=use_kv_cache, update_kv_cache=update_kv_cache
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, mask, src_key_padding_mask, is_causal=is_causal,
                                 use_kv_cache=use_kv_cache, update_kv_cache=update_kv_cache)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        mask: Optional[Tensor],
        src_key_padding_mask: Optional[Tensor],
        is_causal: bool = False,
        use_kv_cache: bool = False,
        update_kv_cache: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            mask=mask,
            src_key_padding_mask=src_key_padding_mask,
            is_causal=is_causal,
            use_kv_cache=use_kv_cache,
            update_kv_cache=update_kv_cache,
        )
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

    def clear_kv_cache(self):
        """Clear the key-value cache in the attention layer."""
        self.self_attn.clear_kv_cache()


class CumulativeTransformerEncoderKV(Module):
    r"""CumulativeTransformerEncoderKV is a stack of N encoder layers using cumulative attention.

    Users can build models with cumulative linear attention for improved efficiency.

    Args:
        encoder_layer: an instance of the CumulativeTransformerEncoderLayerKV() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).

    Examples::
        >>> encoder_layer = CumulativeTransformerEncoderLayerKV(d_model=512, nhead=8)
        >>> transformer_encoder = CumulativeTransformerEncoderKV(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    def __init__(
        self,
        encoder_layer: "CumulativeTransformerEncoderLayerKV",
        num_layers: int,
        norm: Optional[Module] = None,
    ) -> None:
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: Optional[bool] = None,
        use_kv_cache: bool = False,
        update_kv_cache: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layers in turn.

        Args:
            src: the sequence to the encoder (required).
            mask: the mask tensor for cumulative attention (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            is_causal: If specified, applies a causal mask. Will override mask.
                Default: ``None``; try to detect a causal mask.
            use_kv_cache: Whether to use key-value caching for efficiency.
                Default: ``False``.
            update_kv_cache: Whether to update the key-value cache.
                Default: ``False``.

        Shape:
            - src: (N, S, E) if batch_first=True or (S, N, E) if batch_first=False
            - mask: (S,) where S is sequence length
        """
        output = src

        # Default is_causal to False if not specified
        if is_causal is None:
            is_causal = False

        for mod in self.layers:
            output = mod(
                output,
                mask=mask,
                is_causal=is_causal,
                src_key_padding_mask=src_key_padding_mask,
                use_kv_cache=use_kv_cache,
                update_kv_cache=update_kv_cache,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output

    def clear_kv_cache(self):
        """Clear the key-value cache in all encoder layers."""
        for layer in self.layers:
            layer.clear_kv_cache()