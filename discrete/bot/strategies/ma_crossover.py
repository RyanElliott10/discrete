from datetime import datetime

import backtrader as bt

from discrete.bot.strategies.data_fetcher import BacktraderDataFetcher


class MovingAverageCrossover(bt.Strategy):
    def __init__(self):
        ma_fast = bt.ind.MovingAverageSimple(period=10)
        ma_slow = bt.ind.MovingAverageSimple(period=50)

        self.crossover = bt.ind.CrossOver(ma_fast, ma_slow)

    def next(self):
        if not self.position:
            if self.crossover > 0:
                self.buy()
        elif self.crossover < 0:
            self.close()


def debug_main():
    cerebro = bt.Cerebro()
    data_fetcher = BacktraderDataFetcher(
        datetime(2017, 7, 6), datetime(2021, 7, 1)
    )
    data_feeds = data_fetcher.fetch(["AAPL"])
    cerebro.adddata(data_feeds)

    cerebro.addstrategy(MovingAverageCrossover)
    cerebro.broker.setcash(1000.0)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=10)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="Sharpe")
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="Transactions")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="Trades")

    backtest = cerebro.run()
    cerebro.plot()
    print(cerebro.broker.getvalue())


if __name__ == "__main__":
    debug_main()
