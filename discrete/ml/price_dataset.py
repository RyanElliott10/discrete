from typing import Dict, Set

import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from discrete.bot.stock_sql import StockSQL


class PriceDataset(Dataset):

    def __init__(
            self,
            src_window: int,
            tgt_window: int,
            db_name: str,
            table_name: str,
            securities: Set[str] = None
    ):
        self.src_window = src_window
        self.tgt_window = tgt_window
        self.securities = {sec.upper() for sec in securities}
        self.src_data = None
        self.tgt_data = None
        self._fetch_data(db_name, table_name)

    def _fetch_data(
            self,
            db_name: str,
            table_name: str,
    ):
        sql = StockSQL(db_file=db_name)
        securities = sql.fetch_securities_tickers(table_name)
        if self.securities is not None:
            securities = securities.intersection(self.securities)

        for ph in sql.fetch_securities(
                securities, meta="OHLCVS", table=StockSQL.PRICE_HISTORY_TABLE
        ):
            # Split into chunks according to src_window
            for _, subdf in ph.groupby(np.arange(len(ph)) // self.src_window):
                src = subdf[["security", "datetime", "open", "high", "low",
                             "close", "volume"]]
                tgt = subdf[["short_squeeze"]]
                print(src, tgt)
                # print(df.shape)

    def __getitem__(self, item) -> Dict[str, Tensor]:
        return None

    def __len__(self) -> int:
        return self.src_data.shape[0]
