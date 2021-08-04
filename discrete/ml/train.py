import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from discrete.ml.hyperparameters import ModelHyperparameters, \
    TrainingHyperparameters
from discrete.ml.model.abbreviated_time_transformer import \
    AbbreviatedTimeTransformer
from discrete.ml.model.variable_time_transformer import VariableTimeTransformer
from discrete.ml.toy_data import ToyTimeSeriesDataset
from discrete.ml.trainer import AbbreviatedModelTrainer, VariableModelTrainer, \
    ModelTrainer
from discrete.ml.price_dataset import PriceDataset
from discrete.bot.config import price_history_sql_path
from discrete.bot.stock_sql import StockSQL

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model(mp: ModelHyperparameters):
    if mp.type == "abbreviated":
        return AbbreviatedTimeTransformer.model_from_mp(mp, device)
    elif mp.type == "variable":
        return VariableTimeTransformer.model_from_mp(mp, device)
    else:
        raise ValueError("Model type must be one of abbreviated or variable.")


def get_trainer(
        mp: ModelHyperparameters, tp: TrainingHyperparameters, model: nn.Module
) -> ModelTrainer.__class__:
    if mp.type == "abbreviated":
        return AbbreviatedModelTrainer(mp, tp, model, device)
    elif mp.type == "variable":
        return VariableModelTrainer(mp, tp, model, device)
    else:
        raise ValueError("Model type must be one of abbreviated or variable.")


def main(cfg: dict):
    mp = ModelHyperparameters(cfg["model"])
    tp = TrainingHyperparameters(cfg["training"])

    model = get_model(mp)
    trainer = get_trainer(mp, tp, model)

    pdataset = PriceDataset(
        src_window=mp.src_window_len, tgt_window=mp.tgt_window_len,
        db_name=price_history_sql_path, table_name=StockSQL.PRICE_HISTORY_TABLE,
        securities={"aapl", "gme", "spy"}
    )
    dataset = ToyTimeSeriesDataset(
        src_window=mp.src_window_len, tgt_window=mp.tgt_window_len,
        n_features=mp.n_time_features + mp.n_linear_features,
        n_out_features=mp.n_out_features, n_data=1
    )
    data_loader = DataLoader(dataset, batch_size=tp.batch_size, shuffle=True)
    data_loader = DataLoader(pdataset, batch_size=tp.batch_size, shuffle=True)

    trainer.train(data_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, type=str,
        help="Path to .yaml config file"
    )
    parser.add_argument(
        "-d", "--debug", default=False, type=bool,
        help="Flag to run in debug mode."
    )

    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    main(config)
