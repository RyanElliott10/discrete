import pandas as pd
import pandas_ta as ta

from discrete.labelling import LabellingStrategy


class BLSHLabeller(LabellingStrategy):
    r"""Buy Low Sell High is a very basic stratgey that simply identifies
    opportune times to buy into a security, and opportune times to exit the
    position.
    """

    def __init__(self):
        self.tastrat = ta.LabellingStrategy(
            name="foo", ta=[
                {"kind": "sma", "length": 7},
                {"kind": "sma", "length": 12},
                {"kind": "sma", "length": 23},
                {"kind": "macd"}
            ])

    def label(self, data: pd.DataFrame):
        # data.ta.sma(length=20, append=True)
        # data.ta.macd(append=True)
        # data.ta.percent_return(cumulative=True, append=True)
        data.ta.strategy(self.tastrat, append=True)
        print(data.tail(50))


def main():
    labeller = BLSHLabeller()
    sql = StockSQL(price_history_sql_path)
    all_data = sql.fetch_securities("sq", meta="OHLCV", table="price_history")

    for data in all_data:
        labeller.label(data)


if __name__ == "__main__":
    r"""Run as module: python -m discrete.strategies.short_squeeze"""
    from discrete.stock_sql import StockSQL
    from discrete.config import price_history_sql_path

    main()
