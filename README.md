# Trim Transformer

`trim-transformer` is a lightweight PyPI package that replicates the familiar interface of `torch.nn.TransformerEncoder`. The attention kernel has the form Attn(Q,K,V) = QK^TV. This implementation has complexity O(nd^2), where n is the sequence length and d is the model dimension.

Additionally, this implementation supports key-value caching for inference that is also linear in the number of tokens generated.

![PyPI](https://img.shields.io/pypi/v/trim-transformer?color=%2334D058&logo=pypi) ![License](https://img.shields.io/github/license/emanuel-nuclearsoftware/trim-transformer)

---

## Installation

The package is published on PyPI and only depends on PyTorch:

```bash
pip install trim-transformer
```

Alternatively, install the latest commit from GitHub:

```bash
pip install git+https://github.com/emanuel-nuclearsoftware/trim-transformer.git
```

---

## Quickstart

```python
import torch
from trim_transformer.transformer_layers import CumulativeTransformerKV

layer = CumulativeTransformerEncoderLayerKV(d_model=EMBED_DIM, nhead=NUM_HEADS, batch_first=True)
model = CumulativeTransformerEncoderKV(layer, num_layers=NUM_LAYERS)

x = torch.randn(8, 2048, 512)  # (batch, seq_len, dim)

# Standard forward pass (causal mask optional)
out = model(x, is_causal=True)  # (batch, seq_len, dim)
```

## Masking

The most significant departure from PyTorch syntax is the structure of the mask. Multi-linear attention with arbitrary boolean masks cannot be computed in linear time. Instead, this package supports masks such that the i-th query attends to all keys up to index m_i. Such masks can be specified by an integer array of length n, with values in [0, n-1], where n is the sequence length.

For example, a causal mask of length n is given by torch.arange(n). And the one dimensional mask [2, 0, 1] corresponds to the two dimensional mask [[False, False, False], [False, True, True], [False, False, True]], following the PyTorch convention that True means to set that element of the attention matrix to 0.

---

## License

`trim-transformer` is released under the MIT License.  See [LICENSE](LICENSE) for the full text.