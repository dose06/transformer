import torch
import torch.nn as nn
import math
from util.utils import subsequent_mask, clones


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn



class MultiHeadedAttention(nn.Module):
    """
    Multi-Headed Attention mechanism.
    """
    def __init__(self, h, d_model, dropout=0.1):
        """
        Initialize the multi-head attention module.

        Args:
            h: Number of attention heads.
            d_model: Dimensionality of the model.
            dropout: Dropout rate.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0, "d_model must be divisible by h"
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        Compute the attention output.

        Args:
            query: Query tensor.
            key: Key tensor.
            value: Value tensor.
            mask: Attention mask.

        Returns:
            Output tensor after applying multi-head attention.
        """
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # Apply linear projections and split into heads.
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # Compute attention using scaled dot-product.
        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        # Concatenate and apply final linear layer.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

