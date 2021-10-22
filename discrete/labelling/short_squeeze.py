from typing import List

import pandas as pd

from discrete.bot.config import price_history_sql_path
from discrete.bot.stock_sql import StockSQL
from discrete.labelling import LabellingStrategy

pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)


class ShortSqueezeLabeller(LabellingStrategy):
    r"""A class that labels data based on the short squeeze strategy. This uses
    a basic sliding standard deviation window. If any point within that window
    exceeds the threshold, this is labelled as part of a short squeeze.\

    Args:
        window: the number of days to consider when calculating standard
            deviation.
    """

    DF_LABEL = "short_squeeze"

    def __init__(
            self,
            window: int,
            stddev_threshold: float = None,
            alt_threshold: float = None
    ):
        self.window = window
        self.stddev_threshold = stddev_threshold
        self.alt_threshold = alt_threshold

    def label(self, data: pd.DataFrame):
        label_inter = self._df_label_inter()
        label_final = self.df_label()
        data[label_inter] = 0.0
        data[label_final] = 0
        for curr in data.itertuples():
            lrange = data[curr.Index:curr.Index + self.window]
            stddev = lrange.close.std()
            mean = lrange.close.mean()
            labels = lrange.close.apply(
                lambda d: ShortSqueezeLabeller._stddev_anomaly(d, stddev, mean)
            )
            prev_ssvals = data.iloc[labels.index.values][label_inter]
            labels = labels[labels > prev_ssvals]
            data.update(labels.rename(label_inter))

        data.loc[data[label_inter] >= self.stddev_threshold, label_final] = 1

    def label_alt(self, data: pd.DataFrame):
        spike = False
        window = [0] * self.window
        wsum = 0.0
        windex = 0
        prev_idx = None
        start_idx = None
        label_final = self.df_label()

        data[label_final] = 0
        for curr in data.itertuples():
            if curr.Index == 0:
                prev_idx = curr.Index
                continue
            prev_close = data.iloc[prev_idx].close
            delta = (curr.close - prev_close) / prev_close
            wsum -= window[windex]
            window[windex] = delta
            windex += 1
            windex %= self.window
            wsum += delta

            if not spike and wsum >= self.alt_threshold:
                spike = True
                start_idx = curr.Index - self.window // 2
                if start_idx < 0:
                    start_idx = 0
            elif spike and wsum < self.alt_threshold:
                spike = False
                srange = range(start_idx, curr.Index - self.window // 2)
                data.loc[srange, label_final] = 1
            prev_idx = curr.Index

    def _label_partial(self, days: List[pd.Series]):
        print(type(days), type(days[0]), days)
        label = self._df_label_inter()
        [d[label].add(True) for d in days]

    def _df_label_inter(self) -> str:
        r"""The intermediate label used for storing the multiples of the
        standard deviation.
        """
        return f"{ShortSqueezeLabeller.DF_LABEL}_{self.window}"

    @staticmethod
    def df_label() -> str:
        return f"{ShortSqueezeLabeller.DF_LABEL}"

    @staticmethod
    def _stddev_anomaly(
            curr: float,
            stddev: float,
            mean: float
    ) -> float:
        return (curr - mean) / stddev


def update_short_squeeze_db(r: pd.DataFrame, sec: str, sql: StockSQL):
    sql.execute(
        f"""UPDATE price_history SET short_squeeze={r["short_squeeze"]} """
        f"""WHERE security="{sec}" AND datetime={r["datetime"]}"""
    )


def main():
    labeller = ShortSqueezeLabeller(
        window=25, stddev_threshold=2.5, alt_threshold=0.75
    )
    sql = StockSQL(price_history_sql_path)
    securities = sql.fetch_securities_tickers("price_history")

    for data in sql.fetch_securities(
            securities, meta="OHLCV", table="price_history"
    ):
        labeller.label_alt(data)
        sec = data["security"].iloc[0]
        data.apply(lambda r: update_short_squeeze_db(r, sec, sql), axis=1)
        print(
            f"Security: {data['security'].iloc[0]}:\t"
            f"{data[data[labeller.df_label()] == 1].shape[0]} squeeze days"
        )


if __name__ == "__main__":
    r"""Run as module: python -m discrete.strategies.short_squeeze"""
    main()
