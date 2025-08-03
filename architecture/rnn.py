import torch
import torch.nn as nn
from torch.nn import Parameter
from copy import deepcopy
import os, sys
from typing import List, Tuple
from torch import Tensor

sys.path.append(os.getcwd())
from architecture.skeleton import Skeleton
from architecture.rnn import *
from architecture.cnn import *
from architecture.mlp import *
from architecture.attention import *
from architecture.init import initialize

# Reference:
# https://github.com/pytorch/pytorch/blob/main/benchmarks/fastrnns/custom_lstms.py


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 4 = input, forget, cell, output
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.bias_ih = Parameter(torch.randn(4 * hidden_size))
        self.bias_hh = Parameter(torch.randn(4 * hidden_size))

    def forward(
        self, x: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h0, c0 = state  # BH,BH
        gates = (
            torch.mm(x, self.weight_ih.t())
            + self.bias_ih
            + torch.mm(h0, self.weight_hh.t())
            + self.bias_hh
        )  # B,4H
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)  # 4 chunks in dim 1

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c1 = (forgetgate * c0) + (ingate * cellgate)  # BH
        h1 = outgate * torch.tanh(c1)  # BH

        return h1, (h1, c1)


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = Parameter(torch.randn(4 * hidden_size, hidden_size))
        self.register_parameter("weight_ih", self.weight_ih)
        self.register_parameter("weight_hh", self.weight_hh)
        self.layernorm_i = nn.LayerNorm(4 * hidden_size)
        self.layernorm_h = nn.LayerNorm(4 * hidden_size)
        self.layernorm_c = nn.LayerNorm(hidden_size)

    def forward(
        self, x: Tensor, state: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        h0, c0 = state
        igates = self.layernorm_i(torch.mm(x, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(h0, self.weight_hh.t()))
        gates = igates + hgates  # B,4H
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        c1 = self.layernorm_c((forgetgate * c0) + (ingate * cellgate))  # BH
        h1 = outgate * torch.tanh(c1)  # BH

        return h1, (h1, c1)


class StackedLSTMCell(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        dropout: float = 0,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.__dict__.update(locals())
        cell = LayerNormLSTMCell if layernorm else LSTMCell
        self.stacks = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.stacks.append(cell(input_size, hidden_size))
            else:
                self.stacks.append(cell(hidden_size, hidden_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, state: Tuple[Tensor]) -> Tuple[Tensor]:
        h0_all, c0_all = state
        if len(h0_all.size()) == 2:
            h0_all = h0_all.unsqueeze(0)
            c0_all = c0_all.unsqueeze(0)
        num_layers, batch_size, hidden_size = h0_all.size()
        h1_all, c1_all = [], []
        out = x
        for i, rnn in enumerate(self.stacks):
            out, (h1, c1) = rnn(out, (h0_all[i], c0_all[i]))
            h1, c1 = self.dropout(h1), self.dropout(c1)
            h1_all += [h1]
            c1_all += [c1]
        h1_all = torch.stack(h1_all)
        c1_all = torch.stack(c1_all)
        return h1_all, c1_all


class LSTM(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        dropout: float = 0,
        layernorm: bool = False,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.D = 2 if bidirectional else 1
        self.__dict__.update(locals())
        self.cell_1 = StackedLSTMCell(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
        )
        if bidirectional:
            self.cell_2 = StackedLSTMCell(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                layernorm=layernorm,
            )

    def forward_unidirectional(self, x, state):
        batch_size, length, input_size = x.size()
        device = x.device
        h0, c0 = state
        if len(h0.size()) == 2:
            h0 = h0.unsqueeze(0)
            c0 = c0.unsqueeze(0)
        num_layers, batch_size, hidden_size = h0.size()
        x = x.unbind(1)
        ht, ct = h0, c0
        ht, ct = h0[: self.num_layers], c0[: self.num_layers]
        ht_, ct_ = (
            h0[self.num_layers :],
            c0[self.num_layers :],
        )  # in case of bidirectional encoder state input
        h_all, c_all = [], []
        for t in range(length):
            if t == 0:  # merge directional dim at initial step
                if ht_.numel() == 0:
                    ht_cat = ht
                    ct_cat = ct
                else:
                    ht_cat = torch.cat([ht, ht_], dim=-1)
                    ct_cat = torch.cat([ct, ct_], dim=-1)
                ht, ct = self.cell_1(x[t], (ht_cat, ct_cat))
            else:
                ht, ct = self.cell_1(x[t], (ht, ct))
            h_all += [ht]
            c_all += [ct]
        h_all = torch.stack(h_all)
        c_all = torch.stack(c_all)
        return h_all, c_all

    def forward_bidirectional(self, x, state):
        batch_size, length, input_size = x.size()
        h0, c0 = state
        if len(h0.size()) == 2:
            h0 = h0.unsqueeze(0)
            c0 = c0.unsqueeze(0)
        D_num_layers, batch_size, hidden_size = h0.size()
        num_layers = D_num_layers // self.D
        x = x.unbind(1)
        x_ = x[::-1]
        ht, ct = h0[: self.num_layers], c0[: self.num_layers]
        ht_, ct_ = h0[self.num_layers :], c0[self.num_layers :]
        h_all, c_all = [], []
        for t in range(length):
            ht, ct = self.cell_1(x[t], (ht, ct))  # forward direction
            ht_, ct_ = self.cell_2(x_[t], (ht_, ct_))  # backward direction
            h_all += [
                torch.cat([ht, ht_])
            ]  # (forward layer0, forward layer1... backward layer0, backward layer1...)
            c_all += [
                torch.cat([ct, ct_])
            ]  # (forward layer0, forward layer1... backward layer0, backward layer1...)
        h_all = torch.stack(h_all)
        c_all = torch.stack(c_all)
        return h_all, c_all

    def forward(self, x, state):
        if self.bidirectional:
            h_all, c_all = self.forward_bidirectional(x, state)
        else:
            h_all, c_all = self.forward_unidirectional(x, state)
        return h_all, c_all


class LSTMEncoder(Skeleton):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0,
        layernorm: bool = False,
    ) -> None:
        super(LSTMEncoder, self).__init__()
        self.initialize_skeleton(locals())
        self.D = 2 if bidirectional else 1
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
            bidirectional=bidirectional,
        )
        # Learnable initial states
        self.h0 = nn.Parameter(
            torch.empty(self.D * num_layers, 1, hidden_size).normal_(mean=0, std=1e-2)
        )
        self.c0 = nn.Parameter(
            torch.empty(self.D * num_layers, 1, hidden_size).normal_(mean=0, std=1e-2)
        )
        self.register_parameter("h0", self.h0)
        self.register_parameter("c0", self.c0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, input_size = x.size()
        state = (self.h0.expand(-1, batch_size, -1), self.c0.expand(-1, batch_size, -1))
        h_all, c_all = self.lstm.forward(x, state)
        return h_all, c_all


class LSTMDecoder(Skeleton):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_layers: int = 1,
        dropout: float = 0,
        layernorm: bool = False,
    ) -> None:
        super().__init__()
        self.initialize_skeleton(locals())
        self.D = 1
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=self.D * hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            layernorm=layernorm,
            bidirectional=False,
        )

    def forward(
        self, y: torch.Tensor, enc_last: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        batch_size, length, output_size = y.size()
        h_enc, c_enc = enc_last
        D_num_layers, batch_size, hidden_size = h_enc.size()
        h_all, c_all = self.lstm.forward(y, enc_last)
        return h_all, c_all


if __name__ == "__main__":
    x = torch.rand(32, 100, 27)
    y = torch.rand(32, 20, 6)
    model = LSTMSeq2Seq(input_size=27, output_size=6, hidden_size=256, num_layers=3, bidirectional=True)
    h_all, c_all = model.forward(x, 30)

    pass