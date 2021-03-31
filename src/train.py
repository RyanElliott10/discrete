import argparse
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch import Tensor

from hyperparameters import ModelHyperparameters, TrainingHyperparameters
from time_transformer import TimeTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_rand_data(
        mp: ModelHyperparameters,
        tp: TrainingHyperparameters
) -> Tuple[Tensor, Tensor]:
    src = torch.randn(
        mp.seq_len, tp.batch_size, mp.n_time_features + mp.n_linear_features
    ).to(device)

    tgt = torch.randint(
        low=0, high=mp.n_out_features, size=(mp.seq_len, tp.batch_size)
    ).to(device)

    return src, tgt


def print_progress(epoch: int, n_epochs: int, loss: float):
    print(
        f"\r[Overfit Epoch {epoch + 1} / {n_epochs}] Loss: {loss}",
        end='', flush=True
    )


def debug(cfg: dict):
    mp = ModelHyperparameters(cfg["model"])
    tp = TrainingHyperparameters(cfg["training"])

    src, tgt = get_rand_data(mp, tp)

    model = TimeTransformer.model_from_mp(mp, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=tp.learning_rate)

    start_loss, end_loss = 0, 0

    for epoch in range(tp.n_epochs):
        out = model(src)
        out = out.reshape(-1, mp.n_out_features)

        loss = criterion(out, tgt.reshape(-1))

        if epoch == 0:
            start_loss = loss.item()
        else:
            end_loss = loss.item()

        print_progress(epoch, tp.n_epochs, loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print(f"\nStart Loss: {start_loss} | End Loss: {end_loss}")


def main(cfg: dict):
    mp = ModelHyperparameters(cfg["model"])
    tp = TrainingHyperparameters(cfg["training"])

    src, tgt = get_rand_data(mp, tp)

    model = TimeTransformer.model_from_mp(mp, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=tp.learning_rate)

    start_loss, end_loss = 0, 0

    for epoch in range(tp.n_epochs):
        out = model(src)
        out = out.reshape(-1, mp.n_out_features)

        loss = criterion(out, tgt.reshape(-1))

        if epoch == 0:
            start_loss = loss.item()
        else:
            end_loss = loss.item()

        print_progress(epoch, tp.n_epochs, loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print(f"\nStart Loss: {start_loss} | End Loss: {end_loss}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-y', '--yaml', required=True, type=str,
        help="Path to .yaml config file"
    )
    parser.add_argument(
        '-d', '--debug', default=False, type=bool,
        help="Flag to run in debug mode."
    )

    args = parser.parse_args()

    with open(args.yaml) as f:
        config = yaml.safe_load(f)

    if args.debug:
        debug(config)
    else:
        main(config)
