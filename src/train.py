import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from attrdict import AttrDict

from time_transformer import TimeTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def debug(cfg: AttrDict):
    mp = cfg.model
    tp = cfg.training

    model = TimeTransformer(
        n_time_features=mp.n_time_features,
        n_linear_features=mp.n_linear_features,
        n_out_features=mp.n_out_features, d_time_embed=mp.d_time_embed,
        d_linear=mp.d_linear, n_head=mp.n_head,
        num_encoder_layers=mp.num_encoder_layers,
        dropout=mp.dropout, device=device
    ).to(device)

    src = torch.randn(
        mp.seq_len, tp.batch_size, mp.n_time_features + mp.n_linear_features
    ).to(device)

    tgt = torch.randint(
        low=0, high=mp.n_out_features, size=(mp.seq_len, tp.batch_size)
    ).to(device)

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

        print(
            f"\r[Overfit Epoch {epoch + 1} / {tp.n_epochs}] Loss: "
            f"{loss.item()}",
            end='', flush=True
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
        optimizer.step()

    print(f"\nStart Loss: {start_loss} | End Loss: {end_loss}")


def main(cfg: AttrDict):
    mp = cfg.model
    tp = cfg.training

    model = TimeTransformer(
        n_time_features=mp.n_time_features,
        n_linear_features=mp.n_linear_features,
        n_out_features=mp.n_out_features, d_time_embed=mp.d_time_embed,
        d_linear=mp.d_linear, n_head=mp.n_head,
        num_encoder_layers=mp.num_encoder_layers,
        dropout=mp.dropout, device=device
    ).to(device)

    src = torch.randn(
        mp.seq_len, tp.batch_size, mp.n_time_features + mp.n_linear_features
    ).to(device)

    tgt = torch.randint(
        low=0, high=mp.n_out_features, size=(mp.seq_len, tp.batch_size)
    ).to(device)

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

        print(
            f"\r[Epoch {epoch + 1} / {tp.n_epochs}] Loss: {loss.item()}",
            end='', flush=True
        )

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
        cfg_attr = AttrDict(config)

    if args.debug:
        debug(cfg_attr)
    else:
        main(cfg_attr)
