import math
from typing import Callable

import pandas as pd
import pandas_ta as ta

from discrete.bot.candle import TDACandle
from discrete.bot.candle_source_type import CandleSourceType
from discrete.bot.strategies.strategy import Strategy

PositionFunc = Callable[[float], float]


class GoldenCrossStrategy(Strategy):
    r"""A simple golden cross strategy. Using simple moving averages,
    when the lower MA crosses above the lower MA we begin to enter positions.
    When the thresholds are met, we begin to exit positions.

    Args:
        source: what point of each candle to utilize.
        positive_threshold: the number of timesteps until we begin entering
            positions after the SMAs have crossed.
        negative_threshold: the number of timesteps where we begin to exit
            positions
        exit_threshold: the maximum number of timesteps where we exit all
            positions
        enter_func: a function that dictates how many positions to obtain
            after we have surpassed the positive_threshold.
        exit_func: a function that we use to sell positions. Output should be
            between 0 and 1 as a percentage of original positions we should
            maintain
    """

    LOWER_LENGTH = 50
    UPPER_LENGTH = 200

    def __init__(
            self,
            positive_threshold: int,
            negative_threshold: int,
            exit_threshold: int = None,
            enter_func: PositionFunc = None,
            exit_func: PositionFunc = None,
            source: CandleSourceType = CandleSourceType.Close
    ):
        self.positive_threshold = positive_threshold
        self.negative_threshold = negative_threshold
        self.exit_threshold = math.floor(negative_threshold * 1.5) if \
            exit_threshold is None else exit_threshold
        self.enter_func = enter_func
        self.exit_func = exit_func
        self.source = source
        self.candles = []
        self.positions = []
        self.orig_position_count = len(self.positions)
        self.lowersma = pd.DataFrame
        self.uppersma = pd.DataFrame
        self.negative_flag = False
        self.negative_count = 0

    def _getvals(self) -> pd.DataFrame:
        return pd.DataFrame(
            [getattr(candle, str(self.source)) for candle in self.candles]
        )

    def consume(self, candle: TDACandle):
        self.candles.append(candle)

    def _lower_above_upper(self):
        return self.lowersma.iloc[-1] > self.uppersma.iloc[-1]

    def _handle_positive(self):
        pass

    def _handle_negative(self):
        if not self.negative_flag:
            self._handle_negative_incr()
            return
        self._set_negative_flag()

    def _handle_negative_incr(self):
        self.negative_count += 1
        if self.negative_count >= self.negative_threshold:
            ratio = self.negative_count / self.negative_threshold
            if self.exit_func is not None:
                pass
            else:
                delta = self.negative_count - self.negative_threshold
                if delta >= self.exit_threshold and len(self.positions) > 0:
                    self._exit_all_positions()
                else:
                    remaining_pos_count = math.floor(
                        ratio * self.orig_position_count
                    )
                    self._exit_positions_num(
                        self.orig_position_count - remaining_pos_count
                    )

    def _exit_positions_num(self, num_positions: int):
        pass

    def _exit_all_positions(self):
        pass

    def _reset_negative_flag(self):
        self.negative_flag = False
        self.negative_count = 0

    def _set_negative_flag(self):
        self.negative_flag = True
        self.negative_count = 1
        self.orig_position_count = len(self.positions)

    def act(self):
        vals = self._getvals()
        self.lowersma = ta.sma(vals, length=GoldenCross.LOWER_LENGTH)
        self.uppersma = ta.sma(vals, length=GoldenCross.UPPER_LENGTH)
        if not self._lower_above_upper():
            self._handle_negative()
            return
        if self.negative_flag:
            self._reset_negative_flag()
        self._handle_positive()
