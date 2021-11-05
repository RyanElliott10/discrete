from datetime import datetime, timedelta
from typing import List

from pathos.multiprocessing import ThreadPool

from discrete.bot.candle import TDACandle
from discrete.bot.strategies.news_sentiment_strategy import \
    NewsSentimentStrategy
from discrete.bot.strategies.strategy import Strategy


class TickerStrategy(object):
    def __init__(self, ticker: str, strategy: Strategy):
        self.ticker = ticker
        self.strategy = strategy

    def accept(self, data: List[TDACandle]):
        # TODO: Shouldn't have to find data manually
        data = list(filter(lambda d: d.security == self.ticker, data))
        if len(data) == 0:
            return
        data = data[0]
        self.strategy.consume(data)
        self.strategy.act()


class Portfolio(object):
    r"""Controls the strategies of the entire portfolio."""

    def __init__(self, strategies: List[TickerStrategy]):
        self.strategies = strategies

    def accept(self, data: List[TDACandle]):
        def _accept(strat: TickerStrategy):
            strat.accept(data)

        with ThreadPool(5) as p:
            p.map(_accept, self.strategies)


def main():
    strategies = [
        TickerStrategy("AAPL", NewsSentimentStrategy(None)),
        TickerStrategy("SPY", NewsSentimentStrategy(None)),
        TickerStrategy("SQ", NewsSentimentStrategy(None)),
    ]

    timestamp = int((datetime.today() - timedelta(days=365)).timestamp())
    candles = [
        TDACandle("AAPL", 2, 3, 1.9, 2.5, 20, timestamp),
        TDACandle("SPY", 2, 3, 1.9, 2.5, 20, timestamp),
        TDACandle("SQ", 2, 3, 1.9, 2.5, 20, timestamp)
    ]
    portfolio = Portfolio(strategies)

    data = [candles for _ in range(10)]
    for i, dp in enumerate(data):
        timestamp = int((datetime.today() - timedelta(
            days=365 - i)).timestamp())
        for candle in dp:
            candle.open += 1
            candle.timestamp = timestamp
        portfolio.accept(dp)


if __name__ == "__main__":
    main()
