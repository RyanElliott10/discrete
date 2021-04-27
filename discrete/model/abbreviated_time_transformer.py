import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from discrete.model.time2vec import Time2Vec
from torch import Tensor

from discrete.hyperparameters import ModelHyperparameters

torch.manual_seed(0)


class AbbreviatedTimeTransformer(nn.Module):
    r"""A transformer that utilizes time embeddings via a Time2Vec layer.
    Strictly uses the encoder, no decoding occurs within this model.

    This model does not use masking of any type; all input sequences are of the
    same length and future token masking is not required when not using a
    decoder. This means that this model is only capable of performing a
    statically-sized seq2seq modelling. If the input window is n (128) in
    length, the
    output window will also be n (128) in length. If we would like to perform a
    different task, perhaps one where the input window is n (128) and the
    output window
    is m (8), we would need to utilize a full encoder-decoder transformer.
    This model
    architecture can be found as VariableTimeTransformer, where the N stands
    for the model's
    output size variability.

    Args:
        n_time_features: the number of features used as input into the
            Time2Vec model.
        n_linear_features: the number of linear, non-time features.
        n_out_features: the number of expected outputs.
        d_time_embed: the number of features output by the Time2Vec
            model.
        d_linear_embed: the number of features for the linear projection
            layer. d_model = d_time_embed + d_linear_embed
        n_head: number of heads in the multi-head attention models.
        n_encoder_layers: the number of sub encoder layers.
        dropout: dropout value
        device: device to send tensors to (CPU, CUDA, etc.)
    """

    def __init__(
            self,
            n_time_features: int,
            n_linear_features: int,
            n_out_features: int,
            d_time_embed: int,
            d_linear_embed: int,
            n_head: int,
            n_encoder_layers: int,
            dropout: float,
            device: torch.device,
            use_pos_enc: bool = False,
    ):
        super(AbbreviatedTimeTransformer, self).__init__()

        assert n_time_features > 0, "There must be at least one time feature."
        assert n_linear_features > 0, "There must be at least one linear " \
                                      "feature."

        self.n_in_features = n_time_features + n_linear_features
        self.d_model = d_time_embed + d_linear_embed

        self.time_embedding = Time2Vec(n_time_features, d_time_embed)

        self.positional_encoding = None
        if use_pos_enc:
            self.positional_encoding = PositionalEncoding(d_time_embed, dropout)
        self.linear_src = nn.Linear(n_linear_features, d_linear_embed)

        encoder_sublayers = nn.TransformerEncoderLayer(
            d_model=self.d_model, nhead=n_head, dropout=dropout
        )
        encoder_norm = nn.LayerNorm(self.d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_sublayers, num_layers=n_encoder_layers, norm=encoder_norm
        )

        self.projection = nn.Sequential(
            nn.Hardswish(),
            nn.Linear(self.d_model, n_out_features),
            nn.Dropout(p=dropout)
        )

        self.dropout1 = nn.Dropout(p=dropout)

        self.n_linear_features = n_linear_features
        self.n_time_features = n_time_features
        self.d_time_embed = d_time_embed
        self.d_linear_embed = d_linear_embed
        self.device = device

    def forward(self, src: Tensor) -> Tensor:
        r"""Forward propagate data. It expects the src tensor to be of shape
        (S, N, F) where the first time_features E are used in the Time2Vec
        model. The model will consume the first time_features from the src
        tensor and create time embeddings via a Time2Vec model. It will then
        pass the remaining linear_features into a standard linear layer to be
        concatenated with the time embeddings.

        Args:
            src: features

        Shapes:
            src: (S, N, F)
            out: (S, N, P)
        """
        assert (
                src.shape[-1] == self.n_in_features
        ), "The shape must be of size time_features + linear_features."

        time_features = src[:, :, : self.n_time_features]
        linear_features = src[:, :, self.n_time_features:]

        assert (
                time_features.shape[-1] > 0
        ), "There should at least be one time feature used."
        assert (
                linear_features.shape[-1] > 0
        ), "There should at least be one linear feature used."

        time_embeddings = self.time_embedding(time_features) * math.sqrt(
            self.d_time_embed
        )
        if self.positional_encoding is not None:
            time_embeddings = self.positional_encoding(time_embeddings)
        linear_proj = self.dropout1(self.linear_src(linear_features))

        # Concatenate the time embeddings and linear features that were
        # previously separated.
        x = F.hardswish(torch.cat([time_embeddings, linear_proj], dim=-1))

        assert x.shape[-1] == self.d_time_embed + self.d_linear_embed, (
            "The dimensionality of the concatenated time embeddings and "
            "linear hidden dims must be equal to d_time_embed + d_linear_embed."
        )

        encoded = F.hardswish(self.encoder(x))
        out = self.projection(encoded)

        return out

    @classmethod
    def model_from_mp(
            cls,
            params: ModelHyperparameters,
            device: torch.device
    ) -> 'AbbreviatedTimeTransformer':
        return cls(
            n_time_features=params.n_time_features,
            n_linear_features=params.n_linear_features,
            n_out_features=params.n_out_features,
            d_time_embed=params.d_time_embed,
            d_linear_embed=params.d_linear_embed,
            n_head=params.n_head,
            n_encoder_layers=params.n_encoder_layers,
            dropout=params.dropout,
            device=device,
            use_pos_enc=params.use_pos_enc,
        ).to(device)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor):
        x = x + self.pe[: x.shape[0], :]
        return self.dropout(x)
