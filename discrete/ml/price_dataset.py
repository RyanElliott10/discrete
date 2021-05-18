from typing import Set, Union

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
        self.src_data = None
        self.tgt_data = None
        self._fetch_data(db_name, table_name, securities)

    def _fetch_data(
            self,
            db_name: str,
            table_name: str,
            desired_securities: Union[None, Set[str]]
    ):
        sql = StockSQL(db_file=db_name)
        securities = sql.fetch_securities_tickers(table_name)
        if desired_securities is not None:
            securities = securities.intersection(desired_securities)
        print(securities)
        self.src_data = None

    def __getitem__(self, item):
        return None
