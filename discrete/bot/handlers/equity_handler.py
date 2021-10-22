from itertools import repeat
from typing import List

from discrete.bot.candle import TDACandle
from discrete.bot.handlers.handler import TDAMessageHandler, TDARespMsg
from discrete.bot.strategies.strategy import Strategy


class EquityHandler(TDAMessageHandler):
    def __init__(self, strategies: List[Strategy]):
        self.candles: List[TDACandle] = []
        self.strategies = strategies

    # This calls the same strategy for different tickers. Strategies should
    # only have to support a single security. The division and creation of
    # multiple strategies should probably be done here on the fly based on
    # what securities we have seen
    def _invoke_strategies(self, candle: TDACandle):
        for strategy in self.strategies:
            strategy.consume(candle)

    def handler(self, msg: TDARespMsg):
        print("Handler")
        raw_candles = msg['content']
        timestamp = msg['timestamp']
        for raw_candle in raw_candles:
            candle = TDACandle.from_tda_content(raw_candle, timestamp)
            self._invoke_strategies(candle)
            self.candles.append(candle)
        print(len(self.candles))

    def responds_to(self, event: any) -> bool:
        return False
