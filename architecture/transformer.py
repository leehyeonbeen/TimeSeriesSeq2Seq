import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
from typing import Optional

import os, sys

sys.path.append(os.getcwd())
from architecture.skeleton import Skeleton
from architecture.init import initialize
from architecture.attention import create_square_mask

"""
References:
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., ... & Polosukhin, I. (2017). Attention is all you need. Advances in neural information processing systems, 30.
https://github.com/hyunwoongko/transformer
https://github.com/UdbhavPrasad072300/Transformer-Implementations
"""


class LayerNorm(nn.Module):
    def __init__(self, d_model: int = 512, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.__dict__.update(locals())
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        # '-1' means last dimension.

        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super(MultiHeadAttention, self).__init__()
        self.__dict__.update(locals())
        self.d_k = d_model // n_heads
        assert round(self.d_k * self.n_heads) == d_model
        self.w_q = nn.Linear(d_model, d_model)  # parallel projection and divide
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.softmax = nn.Softmax(dim=-1)

    def split(self, q, k, v):
        batch_size_q, length_q, d_model_q = q.size()
        batch_size_k, length_k, d_model_k = k.size()
        batch_size_v, length_v, d_model_v = v.size()
        # Parallel projections d_model -> d_k
        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        # Divide into batch, head, length, d_k
        q = q.reshape(batch_size_q, length_q, self.n_heads, self.d_k).transpose(1, 2)
        k = k.reshape(batch_size_k, length_k, self.n_heads, self.d_k).transpose(1, 2)
        v = v.reshape(batch_size_v, length_v, self.n_heads, self.d_k).transpose(1, 2)
        return q, k, v

    def forward(self, q, k, v, mask=None):
        batch_size, length, d_model = q.size()
        q, k, v = self.split(q, k, v)
        batch_size, n_heads, length, d_k = q.size()
        attn_score = torch.einsum("bhLd,bhld->bhLl", q, k)  # matmul
        attn_score /= math.sqrt(d_k)  # scale

        # Mask
        if mask is not None:
            mask = mask.expand(batch_size, n_heads, length, length)
            attn_score = attn_score.masked_fill_(mask == 0, -1e12)

        # Attend values
        attn_score = self.softmax(attn_score)
        attn_v = torch.matmul(attn_score, v)

        # Concat heads and project
        attn_v = attn_v.reshape(batch_size, length, d_model)
        attn_v = self.w_o(attn_v)

        return attn_v, attn_score


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int = 512, d_ff: int = 2048):
        super(PositionwiseFeedForward, self).__init__()
        self.__dict__.update(locals())
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        # d_model -> d_ff -> d_model
        x = self.ff(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int = 512, max_len: int = 1024):
        super(PositionalEncoding, self).__init__()
        self.__dict__.update(locals())
        pos = torch.arange(0, max_len).float().unsqueeze(dim=1)
        self.pe = torch.zeros(max_len, d_model)
        evens = torch.arange(0, d_model, 2).float()
        self.pe[:, 0::2] = torch.sin(pos / (10000 ** (evens / d_model)))
        self.pe[:, 1::2] = torch.cos(pos / (10000 ** (evens / d_model)))
        assert torch.unique(self.pe, dim=0).shape[0] == max_len

    def forward(self, ebd_x):
        batch_size, length, d_model = ebd_x.size()
        pe = self.pe[:length, :].unsqueeze(0).repeat(batch_size, 1, 1).to(ebd_x.device)
        return pe


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super(TransformerEncoder, self).__init__()
        self.__dict__.update(locals())
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, ebd_x: torch.Tensor):
        batch_size, length, d_model = ebd_x.size()
        res1 = ebd_x
        attn_v, attn_scores = self.attn(q=ebd_x, k=ebd_x, v=ebd_x, mask=None)
        attn_v = self.norm1(attn_v + self.dropout(res1))
        res2 = attn_v
        attn_v = self.ff(attn_v)
        attn_v = self.norm2(attn_v + self.dropout(res2))
        return attn_v


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super(TransformerDecoder, self).__init__()
        self.__dict__.update(locals())
        self.attn_dec = MultiHeadAttention(d_model, n_heads)
        self.attn_enc_dec = MultiHeadAttention(d_model, n_heads)
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        attn_v_enc: torch.Tensor,
        ebd_y: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        batch_size, length_x, d_model = attn_v_enc.size()
        batch_size, length_y, d_model = ebd_y.size()
        # decoder masked self attention
        mask = create_square_mask(length_y, device=ebd_y.device)
        res1 = ebd_y
        attn_v_dec, attn_scores_dec = self.attn_dec(
            q=ebd_y, k=ebd_y, v=ebd_y, mask=mask
        )
        attn_v_dec = self.norm1(attn_v_dec + self.dropout(res1))
        # encoder-decoder attention
        res2 = attn_v_dec
        attn_v_dec, attn_scores_enc_dec = self.attn_enc_dec(
            q=attn_v_dec, k=attn_v_enc, v=attn_v_enc
        )
        attn_v_dec = self.norm2(attn_v_dec + self.dropout(res2))
        # positionwise feedforward
        res3 = attn_v_dec
        attn_v_dec = self.ff(attn_v_dec)
        attn_v_dec = self.norm3(attn_v_dec + self.dropout(res3))
        return attn_v_dec



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 16
    length_x = 100
    input_size = 27
    length_y = 20
    output_size = 6
    x = torch.rand(batch_size, length_x, input_size, device=device)
    y = torch.rand(batch_size, length_y, output_size, device=device)

    model = Transformer(input_size=input_size, output_size=output_size).to(device)
    out = model.forward_auto(x, trg_len=10)  # autoregressive inference
    out.mean().backward()
    out = model.forward_masked(x, y)  # masked inference with label data
    out.mean().backward()
