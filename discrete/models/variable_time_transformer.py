import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from discrete.hyperparameters import ModelHyperparameters
from discrete.models.time2vec import Time2Vec

torch.manual_seed(0)


class VariableTimeTransformer(nn.Module):
    r"""An encoder decoder transformer that can predict price movements n
    days into the future.

    Given a time series containing N data points x_{t−N+1}, ..., x_{t−1},
    x_t, for M step ahead prediction, the input X of the supervised ML model
    is x_{t−N+1}, ..., x_{t−M}, and the output Y is x_{t−M+1}, x_{t−M+2},
    ..., x_t.

    For consideration: should we use an embedding or linear layer as the
    interface for the decoder? It really depends on the expected output
    representation. Embedding could be used for a rather binary "is it gonna
    do this?", while a linear layer should be used if the expected output is
    continuous.

    Extending this model may make sense. The paper "Enhancing the Locality
    and Breaking the Memory Bottleneck of Transformer on Time Series
    Forecasting" uses causal convolutional self-attention rather than piecewise.
    This shows greater ability to detect anomalies (short squeezes, earnings,
    etc.) and performs better.

    Args:
        src_window: the number of previous timesteps.
        tgt_window: the number of timesteps to predict (n days)
        n_head: number of heads in the multi-head attention models.
        n_encoder_layers: the number of encoder sublayers.
        n_decoder_layers: the number of decoder sublayers.
        n_time_features: the number of features used as input into the
            Time2Vec model.
        n_linear_features: the number of linear, non-time features.
        n_out_features: the number of output features.
        d_time_embed: the number of features output by the Time2Vec
            model.
        d_linear_embed: the number of features for the linear projection
            layer. d_model = d_time_embed + d_linear_embed_embed
        dropout: dropout value
        device: device to send tensors to (CPU, CUDA, etc.)

    """

    def __init__(
            self,
            src_window: int,
            tgt_window: int,
            n_head: int,
            n_encoder_layers: int,
            n_decoder_layers: int,
            n_time_features: int,
            n_linear_features: int,
            n_out_features: int,
            d_time_embed: int,
            d_linear_embed: int,
            dropout: float,
            device: torch.device,
    ):
        super(VariableTimeTransformer, self).__init__()

        assert n_time_features > 0, "There must be at least one time feature."
        assert n_linear_features > 0, "There must be at least one linear " \
                                      "feature."

        self.d_model = d_time_embed + d_linear_embed
        self.n_in_features = n_time_features + n_linear_features

        self.time_embedding = Time2Vec(n_time_features, d_time_embed)
        self.linear_embedding = nn.Linear(
            n_linear_features, d_linear_embed
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=self.d_model, nhead=n_head, dropout=dropout
            ),
            num_layers=n_encoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        self.tgt_embedding = nn.Linear(n_out_features, self.d_model)
        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.d_model, nhead=n_head, dropout=dropout
            ),
            num_layers=n_decoder_layers,
            norm=nn.LayerNorm(self.d_model),
        )

        self.projection = nn.Sequential(
            nn.Hardswish(),
            nn.Linear(self.d_model, n_out_features)
        )

        self.src_window = src_window
        self.tgt_window = tgt_window
        self.n_linear_features = n_linear_features
        self.n_time_features = n_time_features
        self.n_out_features = n_out_features
        self.d_time_embed = d_time_embed
        self.d_linear_embed = d_linear_embed
        self.device = device

    def future_token_square_mask(self, sz: int) -> Tensor:
        r"""Generate a square mask for the sequence. The masked positions are
        filled with 1. Unmasked positions are filled with 0. This outputs
        a ByteTensor which, according to PyTorch docs, will mask tokens
        with non-zero values.

        Masking future tokens is only applicable to the decoder.
        https://www.reddit.com/r/MachineLearning/comments/bjgpt2
        /d_confused_about_using_masking_in_transformer/

        torch.triu(..., diagonal=1) is required to avoid masking the
        current token.
        """
        mask = torch.triu(torch.ones(sz, sz), diagonal=1).bool()
        return mask.to(self.device)

    def forward(self, src: Tensor, tgt: Tensor) -> Tensor:
        r"""Forward propagate data.

        Args:
            src: Input to create the hidden context vector.
            tgt: Expected output.

        Shapes:
            src: (S, N, E)
            tgt: (T, N, E)
        """
        T, N, E = tgt.shape
        assert T == self.tgt_window, (
            f"The output sequence length must be the same length the target "
            f"window. {T} != {self.tgt_window}"
        )

        tgt_future_mask = self.future_token_square_mask(T)

        assert src.shape[-1] == self.n_in_features, (
            f"The shape must be of size time_features + linear_features."
        )

        time_features = src[:, :, :self.n_time_features]
        linear_features = src[:, :, -self.n_linear_features:]

        assert time_features.shape[-1] > 0, (
            "There should at least be one time feature used."
        )
        assert linear_features.shape[-1] > 0, (
            "There should at least be one linear feature used."
        )

        time_embeddings = self.time_embedding(time_features) * math.sqrt(
            self.d_time_embed
        )
        linear_embeddings = F.hardswish(self.linear_embedding(linear_features))

        src_embeddings = F.hardswish(
            torch.cat([time_embeddings, linear_embeddings], dim=-1)
        )

        encoded = self.encoder(src_embeddings)

        tgt_embeddings = self.tgt_embedding(tgt)
        decoder = self.decoder(
            tgt_embeddings, encoded, tgt_mask=tgt_future_mask
        )

        out = self.projection(decoder)
        return out

    @classmethod
    def model_from_mp(cls, params: ModelHyperparameters, device: torch.device):
        return cls(
            src_window=params.src_window_len,
            tgt_window=params.tgt_window_len,
            n_head=params.n_head,
            n_encoder_layers=params.n_encoder_layers,
            n_decoder_layers=params.n_decoder_layers,
            n_time_features=params.n_time_features,
            n_linear_features=params.n_linear_features,
            n_out_features=params.n_out_features,
            d_time_embed=params.d_time_embed,
            d_linear_embed=params.d_linear_embed,
            dropout=params.dropout,
            device=device,
        )
