from typing import List

from discrete.bot.candle import TDACandle
from discrete.bot.handlers.handler import TDAMessageHandler, TDARespMsg


class EquityHandler(TDAMessageHandler):
    def __init__(self):
        self.candles: List[TDACandle] = []

    def handler(self, msg: TDARespMsg):
        raw_candles = msg['content']
        timestamp = msg['timestamp']
        for raw_candle in raw_candles:
            candle = TDACandle.from_tda_content(raw_candle, timestamp)
            self.candles.append(candle)
        print(len(self.candles))

    def responds_to(self, event: any) -> bool:
        return False
