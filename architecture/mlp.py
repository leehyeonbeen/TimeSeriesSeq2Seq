import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import platform
from copy import deepcopy

import os, sys

sys.path.append(os.getcwd())
from architecture.skeleton import Skeleton


class MLP(Skeleton):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size,
        num_hl: int = 0,
    ) -> None:
        self.initialize_skeleton(locals())

        self.num_hidden_layers = num_hl
        self.mlp = []
        self.mlp.append(nn.Linear(input_size, hidden_size))
        self.mlp.append(nn.GELU())
        for i in range(num_hl):
            self.mlp.append(nn.Linear(hidden_size, hidden_size))
            self.mlp.append(nn.GELU())
        self.mlp.append(nn.Linear(hidden_size, output_size))
        self.mlp = nn.Sequential(*self.mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.mlp(x)
        return out


if __name__ == "__main__":
    x = torch.rand(64, 10)
    encoder = MLP(10, 6, 512, num_layers=3)
    y = encoder(x)
