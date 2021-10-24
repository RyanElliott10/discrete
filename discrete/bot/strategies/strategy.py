from abc import ABC, abstractmethod

from discrete.bot.candle import TDACandle


class Strategy(ABC):
    @abstractmethod
    def consume(self, candle: TDACandle):
        raise NotImplementedError

    @abstractmethod
    def act(self):
        raise NotImplementedError
