from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class Time2Vec(nn.Module):
    r"""Time2Vec, as described by Kazemi et al. in "Time2Vec: Learning a
    Vector Representation of Time" implemented in PyTorch.

    A piecewise function which relies upon the sinusoidal function
    to represent periodic time intervals, as inspired by Viswani et al.
    in their paper "Attention is All You Need" wherein they used the sin
    and cos functions to encode positional embeddings. The output
    embedding is designed to incorporate both a linear component (linear
    layer without periodic activation) and a periodic component (linear
    layer with sin/cos/etc. activation).

    Args:
        in_features: the length of the input sequence.
        embed_size: the output embedding size.
        activation: the activation function for the periodic layer. One of
            sin or cos.
        dropout: the dropout value.

    Resources:
        Time2Vec: Learning a Vector Representation of Time, Kazemi et al.
            https://arxiv.org/pdf/1907.05321.pdf
        Transformer and Time Embeddings
            https://rb.gy/lh5pco
        Date2Vec PyTorch
            https://github.com/ojus1/Date2Vec
    """

    def __init__(
            self,
            in_features: int,
            embed_size: int,
            activation: str = "sin",
            dropout: float = 0.1,
    ):
        super(Time2Vec, self).__init__()
        assert embed_size % 2 == 0, "Embedding size must be a multiple of 2."

        self.linear_time_proj = nn.Linear(in_features, embed_size // 2)
        self.periodic_time_proj = nn.Linear(in_features, embed_size // 2)
        self.proj = nn.Linear(embed_size, embed_size)
        self.activation = Time2Vec.get_activation(activation)

        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)
        self.dropout3 = nn.Dropout(p=dropout)

        self.init_weights()

    @staticmethod
    def get_activation(activation: str) -> Callable:
        if activation == "sin":
            return torch.sin
        elif activation == "cos":
            return torch.cos
        else:
            raise ValueError(
                f"Invalid activation function for Time2Vec "
                f"{activation} must be of type sin or cos."
            )

    def init_weights(self):
        r"""Initialize weights over a uniform distribution."""
        nn.init.uniform_(self.linear_time_proj.weight)
        nn.init.uniform_(self.periodic_time_proj.weight)

    def forward(self, src: Tensor) -> Tensor:
        r"""Forward propagate data.
        Args:
            src: tensor containing time features.

        Shapes:
            src: (*, N, F)
            output: (*, N, E)
        """
        linear = self.dropout1(self.linear_time_proj(src))
        periodic = self.dropout2(self.activation(self.periodic_time_proj(src)))
        out = F.hardswish(torch.cat([linear, periodic], dim=-1))
        out = self.dropout3(self.proj(out))
        return out


def debug():
    window_len = 32
    batch_size = 8
    in_features = 6
    embed_size = 128

    t2v = Time2Vec(in_features, embed_size)

    # 13:23:30 2021-3-30
    src = torch.tensor([[13, 23, 30, 2021, 3, 30]], dtype=torch.float)
    out = t2v(src)
    print(f"In shape: {src.shape}")
    print(f"Single example: {out.shape}\n{out}")

    src = torch.randn(window_len, batch_size, in_features)
    out = t2v(src)
    print(f"\nIn shape: {src.shape}")
    print(f"4th example: {out[:, 3, :].shape}")
    print(f"Total output: {out.shape}")


if __name__ == "__main__":
    debug()
