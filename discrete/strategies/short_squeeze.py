from datetime import datetime

import pandas as pd

from discrete.strategies.strategy import Strategy
from discrete.utils import graph_ohlc, overrides


class ShortSqueezeLabeller(Strategy):
    r"""A class that labels data based on the short squeeze strategy. The
    labels are based upon the parameters: epsilon, tau, and pi. Using a
    rolling average of percent closes.

    Args:
        epsilon: the threshold ratio of percent change : days to be
        considered a short
            squeeze.
        tau: the maximum number of days to consider. This is an upper bound
        on the
            number of days to consider.
        pi: the minimum number of days that epsilon must be maintained. A
        single day
            jump shouldn't be considered a short squeeze.
    """

    def __init__(self, epsilon: float, tau: int, pi: int):
        self.epsilon = epsilon
        self.tau = tau
        self.pi = pi

    @overrides(Strategy)
    def label(self, data: pd.DataFrame):
        graph_ohlc(data)
        prev = None
        perc_closes = []
        for curr in data.itertuples():
            if prev is None:
                prev = curr
                continue
            delta_close = curr.close - prev.close
            perc_close = delta_close / prev.close

            # Calculate rolling percent change
            perc_closes.append(perc_close)
            rpc = sum(perc_closes[-self.tau:]) / len(
                perc_closes[-self.tau:]
            ) * 100
            if rpc >= 5.0:
                print(rpc, datetime.fromtimestamp(curr.datetime / 1e3).strftime(
                    "%d/%m/%Y"))

            prev = curr


def main():
    labeller = ShortSqueezeLabeller(10.0, tau=5, pi=3)
    sql = StockSQL(price_history_sql_path)
    all_data = sql.fetch_securities("sq", meta="OHLCV", table="price_history")

    for data in all_data:
        labeller.label(data)


if __name__ == "__main__":
    r"""Run as module: python -m discrete.strategies.short_squeeze"""
    from discrete.stock_sql import StockSQL
    from discrete.config import price_history_sql_path

    main()
