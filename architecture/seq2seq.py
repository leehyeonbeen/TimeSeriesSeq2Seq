import os, sys

sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from architecture.rnn import LSTMEncoder, LSTMDecoder
from architecture.mlp import MLP
from architecture.attention import create_square_mask, BahdanauAttention
from architecture.skeleton import Skeleton
from architecture.transformer import (
    TransformerEncoder,
    TransformerDecoder,
    PositionalEncoding,
)
import random
from typing import Optional
from architecture.init import initialize


class Seq2Seq(Skeleton):
    def __init__(self):
        super().__init__()
        self.initialize_skeleton(locals())

    def forward_auto(self, x: torch.Tensor, trg_len: int):
        # implements autoregressive decoding
        raise NotImplementedError

    def forward_labeled(self, x: torch.Tensor, y: torch.Tensor):
        # implements teacher-forced decoding
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        trg_len: int,
        y: Optional[torch.Tensor] = None,
        teacher_forcing: float = -1,
    ):
        batch_size, length_x, input_size = x.size()
        p = random.uniform(0, 1)
        if p > teacher_forcing and y is not None:
            return self.forward_labeled(x, y)
        else:
            return self.forward_auto(x, trg_len)


class LSTMSeq2Seq(Seq2Seq):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
    ):
        super().__init__()
        self.initialize_skeleton(locals())
        self.D = 2 if bidirectional else 1
        self.encoder = LSTMEncoder(
            input_size, hidden_size, num_layers, bidirectional, dropout, layernorm
        )
        self.decoder = LSTMDecoder(
            output_size,
            self.D * hidden_size,
            self.D * hidden_size,
            num_layers,
            dropout,
            layernorm,
        )
        self.linear = MLP(self.D * hidden_size, output_size, self.D * hidden_size)
        initialize(self)

    def encode(self, x: torch.Tensor):
        enc_h_all, enc_c_all = self.encoder.forward(x)

        # merge directional dim of the last timestep states
        if self.bidirectional:
            dec_ht = enc_h_all[-1][: self.num_layers]
            dec_ht_ = enc_h_all[-1][self.num_layers :]
            dec_ct = enc_c_all[-1][: self.num_layers]
            dec_ct_ = enc_c_all[-1][self.num_layers :]
            dec_ht = torch.cat([dec_ht, dec_ht_], dim=-1)
            dec_ct = torch.cat([dec_ct, dec_ct_], dim=-1)
            enc_h_all_ = enc_h_all[:, self.num_layers - 1, ...]
            enc_h_all__ = enc_h_all[:, self.D * self.num_layers - 1, ...]
            enc_h_all = torch.cat([enc_h_all_, enc_h_all__], dim=-1).permute(1, 0, 2)

        else:
            dec_ht = enc_h_all[-1]
            dec_ct = enc_c_all[-1]
            enc_h_all = enc_h_all[:, self.num_layers - 1, ...].permute(1, 0, 2)

        return enc_h_all, enc_c_all, dec_ht, dec_ct

    def forward_auto(self, x, trg_len):
        batch_size, seq_len, input_size = x.size()
        enc_h_all, enc_c_all, dec_ht, dec_ct = self.encode(x)

        y0 = torch.zeros(batch_size, 1, self.output_size).to(x.device)
        out = []
        dec_input = y0
        for t in range(trg_len):
            dec_ht, dec_ct = self.decoder(dec_input, (dec_ht, dec_ct))
            dec_ht = dec_ht.squeeze(0)
            dec_ct = dec_ct.squeeze(0)
            yt = self.linear(dec_ht[-1])  # project last decoder layer
            out += [yt]  # yt should be (B,D*H)
            dec_input = yt.unsqueeze(1)
        out = torch.stack(out, dim=1)
        return out

    def forward_labeled(self, x, y):
        batch_size, seq_len, input_size = x.size()
        enc_h_all, enc_c_all, dec_ht, dec_ct = self.encode(x)

        y0 = y[:, 0:1, :]
        out = []
        dec_input = y0
        for t in range(y.shape[1]):
            dec_ht, dec_ct = self.decoder(dec_input, (dec_ht, dec_ct))
            dec_ht = dec_ht.squeeze(0)
            dec_ct = dec_ct.squeeze(0)
            yt = self.linear(dec_ht[-1])  # project last decoder layer
            out += [yt]  # yt should be (B,D*H)
            if t < y.shape[1] - 1:
                dec_input = y[:, t + 1 : t + 2, :]
        out = torch.stack(out, dim=1)
        return out


class AttentionLSTMSeq2Seq(LSTMSeq2Seq):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
    ):
        super().__init__(input_size, output_size, hidden_size)
        self.initialize_skeleton(locals())
        self.D = 2 if bidirectional else 1
        self.encoder = LSTMEncoder(
            input_size, hidden_size, num_layers, bidirectional, dropout, layernorm
        )
        self.decoder = LSTMDecoder(
            self.D * hidden_size + output_size,
            self.D * hidden_size,
            self.D * hidden_size,
            num_layers,
            dropout,
            layernorm,
        )
        self.attention = BahdanauAttention(self.D * hidden_size)
        self.linear = MLP(self.D * hidden_size, output_size, self.D * hidden_size)
        initialize(self)

    def forward_auto(self, x, trg_len):
        batch_size, seq_len, input_size = x.size()
        enc_h_all, enc_c_all, dec_ht, dec_ct = self.encode(x)

        y0 = torch.zeros(batch_size, 1, self.output_size).to(x.device)
        h0 = torch.zeros(batch_size, 1, self.D * self.hidden_size).to(x.device)
        out = [y0]
        dec_input = torch.cat([h0, y0], dim=-1)
        for t in range(trg_len):
            dec_ht, dec_ct = self.decoder(dec_input, (dec_ht, dec_ct))
            dec_ht = dec_ht.squeeze(0)
            dec_ct = dec_ct.squeeze(0)
            dec_q = dec_ht[-1].unsqueeze(1)
            attn_v, attn_scores = self.attention.forward(
                q=dec_q, k=enc_h_all, v=enc_h_all
            )
            yt = self.linear(attn_v).squeeze(1)  # project last decoder layer
            out += [yt]  # yt should be (B,D*H)
            dec_input = torch.cat([dec_q.squeeze(1), yt], dim=1).unsqueeze(1)
        out = torch.stack(out[1:], dim=1)
        return out

    def forward_labeled(self, x, y):
        batch_size, seq_len, input_size = x.size()
        enc_h_all, enc_c_all, dec_ht, dec_ct = self.encode(x)

        y0 = y[:, 0:1, :]
        h0 = torch.zeros(batch_size, 1, self.D * self.hidden_size).to(x.device)
        out = [y0]
        dec_input = torch.cat([h0, y0], dim=-1)
        for t in range(y.shape[1]):
            dec_ht, dec_ct = self.decoder(dec_input, (dec_ht, dec_ct))
            dec_ht = dec_ht.squeeze(0)
            dec_ct = dec_ct.squeeze(0)
            dec_q = dec_ht[-1].unsqueeze(1)
            attn_v, attn_scores = self.attention.forward(
                q=dec_q, k=enc_h_all, v=enc_h_all
            )
            yt = self.linear(attn_v).squeeze(1)  # project last decoder layer
            out += [yt]  # yt should be (B,D*H)
            if t < y.shape[1] - 1:
                dec_input = torch.cat(
                    [
                        dec_q.squeeze(1),
                        y[:, t + 1, :],
                    ],
                    dim=1,
                ).unsqueeze(1)
        out = torch.stack(out[1:], dim=1)
        return out


class TransformerSeq2Seq(Seq2Seq):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_layers: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super(TransformerSeq2Seq, self).__init__()
        self.initialize_skeleton(locals())

        self.embed_x = nn.Linear(input_size, d_model)
        self.embed_y = nn.Linear(output_size, d_model)
        self.linear = nn.Linear(d_model, output_size)
        self.pe_x = PositionalEncoding(d_model)
        self.pe_y = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in range(num_layers):
            self.encoders.append(TransformerEncoder(d_model, n_heads, dropout, d_ff))
            self.decoders.append(TransformerDecoder(d_model, n_heads, dropout, d_ff))
        initialize(self)

    def embedding_x(self, x):
        return self.dropout(self.embed_x(x) + self.pe_x(x))

    def embedding_y(self, y):
        return self.dropout(self.embed_y(y) + self.pe_y(y))

    def encode(self, x):
        batch_size, length_x, input_size = x.size()
        ebd_x = self.embedding_x(x)
        for encoder in self.encoders:
            ebd_x = encoder(ebd_x)
        return ebd_x

    def decode(self, ebd_x, y):
        batch_size, length_x, d_model = ebd_x.size()
        batch_size, length_y, output_size = y.size()
        ebd_y = self.embedding_y(y)
        for decoder in self.decoders:
            ebd_y = decoder.forward(ebd_x, ebd_y)
        out = self.linear(ebd_y)
        return out

    def forward_auto(
        self,
        x: torch.Tensor,
        trg_len: int = 1,
    ):
        batch_size, length_x, input_size = x.size()
        ebd_x = self.encode(x)
        sos = torch.zeros(batch_size, 1, self.output_size, device=x.device)
        dec_input = [sos]
        for t in range(trg_len):
            out = self.decode(ebd_x, torch.cat(dec_input, dim=1))
            dec_input += [out[:, -1, :].unsqueeze(1)]
        return out  # (batch_size, trg_len, output_size)

    def forward_labeled(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ):
        batch_size, length_x, input_size = x.size()

        ebd_x = self.encode(x)

        mask = create_square_mask(y.size(1), device=x.device)
        ebd_y = self.embedding_y(y)
        for decoder in self.decoders:
            ebd_y = decoder(ebd_x, ebd_y, mask=mask)
        out = self.linear(ebd_y)
        return out