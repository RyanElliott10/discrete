from typing import List
from datetime import datetime

import backtrader as bt
import yfinance as yf


class BacktraderDataFetcher(object):
    def __init__(self, start: datetime, end: datetime, source: str = "yahoo"):
        self.start = start
        self.end = end
        if source == "yahoo":
            self.source = yf

    def fetch(self, tickers: List[str]) -> bt.feeds.PandasData:
        df = yf.download(tickers, self.start, self.end, auto_adjust=True)
        return bt.feeds.PandasData(dataname=df)
