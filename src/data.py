from collections import namedtuple
from typing import Dict, List

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

TrainingSample = namedtuple('TrainingSample', 'src tgt')

np.random.seed(0)


class ToyTimeSeriesDataset(Dataset):
    r"""A local dataset meant only for testing the network's architecture
    functionality. This is intended for use with a standard encoder-decoder
    transformer (VariableTimeTransformer) since the source timesteps do not
    match the target timesteps.

    Args:
        src_window: the number of timesteps for the input.
        tgt_window: the number of timesteps for the output.
        n_features: the number of input features. This ignores the
            time/linear split that's used in the actual training data.
        n_data: the number of data points.
        random_walk: a flag to use a random walk. Does not currently work.
    """

    def __init__(
            self,
            src_window: int,
            tgt_window: int,
            n_features: int,
            n_out_features: int,
            n_data: int,
            random_walk: bool = False
    ):
        self.size = n_data * (src_window + tgt_window)
        self.n_data = n_data
        self.src_window = src_window
        self.tgt_window = tgt_window
        self.use_rw = random_walk

        if random_walk:
            rw_positions = self.random_walk()
            self.ts_data = self.random_walk_splits(rw_positions)
        else:
            self.src_data = torch.randn(n_data, src_window, n_features)
            self.tgt_data = torch.randn(n_data, tgt_window, n_out_features)

    def random_walk(self) -> List[int]:
        prob = [0.05, 0.95]
        start = 2
        positions = [start]

        rr = np.random.random(self.size)
        downp = rr < prob[0]
        upp = rr > prob[1]

        for idownp, iupp in zip(downp, upp):
            down = idownp and positions[-1] > 1
            up = iupp and positions[-1] < 400
            positions.append(positions[-1] - down + up)

        return positions

    def random_walk_splits(self, positions: List[int]):
        raw = list(
            zip(*[iter(positions)] * (self.src_window + self.tgt_window))
        )
        return [(sub[:-self.tgt_window], sub[-self.tgt_window:]) for sub in raw]

    def __len__(self) -> int:
        if self.use_rw:
            return len(self.ts_data)
        return self.src_data.shape[0]

    def __getitem__(self, item: int) -> Dict[str, Tensor]:
        r"""The simple item getter method.

        Shapes:
            src: (S, N, E)
            tgt: (T, N, E)
        """
        if self.use_rw:
            data = self.ts_data[item]
            return {
                'src': torch.tensor(data[0]),
                'tgt': torch.tensor(data[1])
            }
        return {
            'src': self.src_data[item],
            'tgt': self.tgt_data[item]
        }


def main():
    dataset = ToyTimeSeriesDataset(
        src_window=128, tgt_window=8, n_features=10, n_data=50
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    for el in dataloader:
        print(el['src'].shape)
        print(el['tgt'].shape)


if __name__ == "__main__":
    main()
