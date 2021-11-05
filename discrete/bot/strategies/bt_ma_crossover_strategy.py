from datetime import datetime

import backtrader as bt

from discrete.bot.strategies.data_fetcher import BacktraderDataFetcher


class BTMovingAverageCrossoverStrategy(bt.Strategy):
    def __init__(self):
        ma_fast = bt.ind.MovingAverageSimple(period=10)
        ma_slow = bt.ind.MovingAverageSimple(period=50)

        self.crossover = bt.ind.CrossOver(ma_fast, ma_slow)
        self.order = None

    def log(self, txt, dt=None):
        r"""Logging function for this strategy"""
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
            elif order.issell():
                self.log('SELL EXECUTED, %.2f' % order.executed.price)

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

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

    cerebro.addstrategy(BTMovingAverageCrossoverStrategy)
    cerebro.broker.setcash(1000.0)
    cerebro.addsizer(bt.sizers.PercentSizer, percents=98)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="Sharpe")
    cerebro.addanalyzer(bt.analyzers.Transactions, _name="Transactions")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="Trades")

    cerebro.run()
    cerebro.plot()
    print(cerebro.broker.getvalue())


if __name__ == "__main__":
    debug_main()
