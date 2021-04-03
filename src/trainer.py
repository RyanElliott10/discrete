from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from hyperparameters import ModelHyperparameters, TrainingHyperparameters


def print_progress(epoch: int, n_epochs: int, loss: float):
    print(
        f"\r[Overfit Epoch {epoch + 1} / {n_epochs}] Loss: {loss}", end="",
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
    def train(self):
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

    def train(self):
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

            print_progress(epoch, self.tp.n_epochs, loss.item())

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
        r"""Returns random data for testing the model.

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

    def train(self):
        src, tgt = self.generate_rand_data()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.tp.learning_rate
        )

        start_loss, end_loss = 0, 0

        for epoch in range(self.tp.n_epochs):
            out = self.model(src, tgt)
            out = out.reshape(-1, self.mp.n_out_features)

            loss = criterion(out, tgt.reshape(-1))

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            optimizer.step()

            if epoch == 0:
                start_loss = loss.item()
            else:
                end_loss = loss.item()

            print_progress(epoch, self.tp.n_epochs, loss.item())

        print(f"\nStart Loss: {start_loss} | End Loss: {end_loss}")
