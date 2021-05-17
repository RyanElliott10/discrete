from abc import ABC, abstractmethod

import pandas as pd


class Strategy(ABC):
    @abstractmethod
    def label(self, data: pd.DataFrame):
        raise NotImplementedError
