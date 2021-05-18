import time
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from discrete.ml.hyperparameters import ModelHyperparameters, \
    TrainingHyperparameters


def print_progress(epoch: int, n_epochs: int, loss: float, debug: bool = False):
    p_of = "Overfit " if debug else ""
    print(
        f"\r[{p_of}Epoch {epoch + 1} / {n_epochs}] Loss: {loss}", end="",
        flush=True
    )


class ModelTrainer(ABC):
    def __init__(
            self,
            params: ModelHyperparameters,
            train_params: TrainingHyperparameters,
            model: nn.Module,
            device: torch.device,
    ):
        self.mp = params
        self.tp = train_params
        self.model = model
        self.device = device

    @abstractmethod
    def generate_rand_data(self) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def train_debug(self):
        raise NotImplementedError

    @abstractmethod
    def train(self, data_loader: DataLoader = None):
        raise NotImplementedError


class AbbreviatedModelTrainer(ModelTrainer):
    def __init__(
            self,
            params: ModelHyperparameters,
            train_params: TrainingHyperparameters,
            model: nn.Module,
            device: torch.device,
    ):
        super(AbbreviatedModelTrainer, self).__init__(
            params, train_params, model, device
        )

    def generate_rand_data(self) -> Tuple[Tensor, Tensor]:
        r"""Returns random data for testing the model.

        Shapes
            src: (S, N, F)
            tgt: (S, N)
        """
        src = torch.randn(
            self.mp.src_window_len,
            self.tp.batch_size,
            self.mp.n_time_features + self.mp.n_linear_features,
        ).to(self.device)

        tgt = torch.randint(
            low=0,
            high=self.mp.n_out_features,
            size=(self.mp.src_window_len, self.tp.batch_size),
        ).to(self.device)

        return src, tgt

    def train_debug(self):
        pass

    def train(self, data_loader: DataLoader = None):
        r"""Note that not passing a data_loader is not supported at this
        moment.
        """
        if data_loader is None:
            src, tgt = self.generate_rand_data()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.tp.learning_rate
        )

        start_loss, end_loss = 0, 0
        writer = SummaryWriter(f"runs/debug")
        step = 0

        for epoch in range(self.tp.n_epochs):
            out = self.model(src)
            out = out.reshape(-1, self.mp.n_out_features)

            loss = criterion(out, tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            optimizer.step()

            num_correct = (out.argmax(dim=-1) == tgt.reshape(-1)).sum()
            running_train_acc = float(num_correct) / float(tgt.shape[0])

            writer.add_scalar("Training Loss", loss, global_step=step)
            writer.add_scalar(
                "Training Accuracy", running_train_acc, global_step=step
            )
            step += 1

            if epoch == 0:
                start_loss = loss.item()
            else:
                end_loss = loss.item()

            print_progress(epoch, self.tp.n_epochs, loss.item(), debug=True)

        print(f"\nStart Loss: {start_loss} | End Loss: {end_loss}")


class VariableModelTrainer(ModelTrainer):
    def __init__(
            self,
            params: ModelHyperparameters,
            train_params: TrainingHyperparameters,
            model: nn.Module,
            device: torch.device,
    ):
        super(VariableModelTrainer, self).__init__(
            params, train_params, model, device
        )

    def generate_rand_data(self) -> Tuple[Tensor, Tensor]:
        r"""Returns random data for testing the model. Should favor a proper
        PyTorch Dataset/DataLoader combination.

        Shapes
            src: (S, N, F)
            tgt: (T, N)
        """
        src = torch.randn(
            self.mp.src_window_len,
            self.tp.batch_size,
            self.mp.n_time_features + self.mp.n_linear_features,
        ).to(self.device)

        tgt = torch.randint(
            low=0,
            high=self.mp.n_out_features,
            size=(self.mp.tgt_window_len, self.tp.batch_size),
        ).to(self.device)

        return src, tgt

    def train_debug(self):
        pass

    def train(self, data_loader: DataLoader = None):
        r"""Note that not passing a data_loader is not supported at this
        moment. This assumes the training of a non-CrossEntropy model. i.e.,
        the output will not be undergo an argmax.

        If we would like to create a CrossEntropy model we will have to
        remove the linear layer leading into the decoder and replace it with
        an embedding layer. We would also have to reshape the out data and
        tgt data to reduce the dimensionality. See transformer.py for more
        information and examples on this transformation.
        """
        if data_loader is None:
            src, tgt = self.generate_rand_data()

        criterion = nn.MSELoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.tp.learning_rate
        )

        start_loss, end_loss = 0, 0
        step = 0
        m_str = self.mp.create_meta_string()
        t_str = self.tp.create_meta_string()
        mt_str = f"variable_gold_{m_str}_{t_str}"
        writer = SummaryWriter(f"runs/gold/adam_{mt_str}_{time.time()}")

        for epoch in range(self.tp.n_epochs):
            for batch in data_loader:
                # data_loader outputs (N, S, E), we need (S, N, E)
                src = batch['src'].transpose(0, 1)
                # data_loader outputs (N, T), we need (T, N, E)
                tgt = batch['tgt'].transpose(0, 1)

                print(src.shape, tgt.shape)
                exit(0)

                out = self.model(src, tgt)

                loss = criterion(out, tgt)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1
                )
                optimizer.step()

                num_correct = (out == tgt).sum()
                running_train_acc = float(num_correct) / float(tgt.shape[0])

                writer.add_scalar("Training Loss", loss, global_step=step)
                writer.add_scalar(
                    "Training Accuracy", running_train_acc, global_step=step
                )
                step += 1

                if epoch == 0:
                    start_loss = loss.item()
                else:
                    end_loss = loss.item()

            print_progress(epoch, self.tp.n_epochs, loss.item())

        print(f"\nStart Loss: {start_loss} | End Loss: {end_loss}")
