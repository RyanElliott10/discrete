from typing import List

import pandas as pd
import pandas_ta as ta

from discrete.bot.candle import TDACandle
from discrete.bot.candle_source_type import CandleSourceType
from discrete.bot.strategies.strategy import Strategy


# If performance becomes a concern, we can just store the values we're
# interested in rather than entire TDACandle objects
class MovingAverageCrossoverStrategy(Strategy):
    def __init__(
            self,
            lengths: List[int],
            source: CandleSourceType,
            starting_candles=None
    ):
        if starting_candles is None:
            starting_candles = []
        self.lengths = lengths
        self.source = source
        self.candles = starting_candles

    def _getvals(self) -> pd.DataFrame:
        return pd.DataFrame(
            [getattr(candle, str(self.source)) for candle in self.candles]
        )

    def consume(self, candle: TDACandle):
        self.candles.append(candle)
        vals = self._getvals()[0]
        averages = list(filter(
            lambda a: a is not None, [ta.sma(vals, length=length) for length
                                      in self.lengths]
        ))
        return averages

    def act(self):
        pass
