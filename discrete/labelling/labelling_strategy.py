from abc import ABC, abstractmethod

import pandas as pd


class LabellingStrategy(ABC):
    @abstractmethod
    def label(self, data: pd.DataFrame):
        raise NotImplementedError
