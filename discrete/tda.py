import re
import time
from typing import List, Set, Union

import pandas as pd
import ratelimit
import requests
import tda
from selenium import webdriver
from tda.client import Client

from discrete.config import (
    token_path,
    api_key,
    redirect_uri,
    all_securities_url,
    price_history_sql_path,
)
from discrete.stock_sql import StockSQL

_securities_html_pattern = (
    r"<li><a href='https:\/\/stockanalysis.com\/"
    r"stocks\/(\D+?)\/'>(\D+?) - (\D+?)<\/a><\/li>"
)


class TdaAPI(object):
    TOO_MANY_REQUESTS_CODE = 429
    SUCCESS_CODE = 200

    def __init__(self):
        self.client = tda.auth.easy_client(
            api_key, redirect_uri, token_path,
            webdriver_func=lambda: webdriver.Chrome()
        )

    @ratelimit.sleep_and_retry
    @ratelimit.limits(calls=120, period=60)
    def price_history(
            self,
            security: str,
            period_type: Client.PriceHistory.PeriodType,
            period: Client.PriceHistory.Period,
            frequency_type: Client.PriceHistory.FrequencyType,
            frequency: Client.PriceHistory.Frequency,
    ) -> pd.DataFrame:
        resp = self.client.get_price_history(
            symbol=security.upper(),
            period_type=period_type,
            period=period,
            frequency_type=frequency_type,
            frequency=frequency,
        )
        if resp.status_code == TdaAPI.SUCCESS_CODE:
            return pd.json_normalize(
                resp.json(), record_path=["candles"], meta=["symbol"]
            )
        elif resp.status_code == TdaAPI.TOO_MANY_REQUESTS_CODE:
            time.sleep(60)
            return self.price_history(
                security, period_type, period, frequency_type, frequency
            )
        resp.raise_for_status()

    def parallel_price_history(
            self,
            securities: Union[Set[str], List[str]],
            period_type: Client.PriceHistory.PeriodType,
            period: Client.PriceHistory.Period,
            frequency_type: Client.PriceHistory.FrequencyType,
            frequency: Client.PriceHistory.Frequency,
    ):
        for security in securities:
            print(security)
            yield self.price_history(
                security, period_type, period, frequency_type, frequency
            )


def fetch_securities_list(
        intermediary: bool = False, sql: StockSQL = None) -> Set[str]:
    r = requests.get(all_securities_url)
    if r.status_code != 200:
        raise Exception("Invalid response fetching securities")

    content = r.text
    securities = set()
    for match in re.finditer(_securities_html_pattern, content):
        securities.add(match.group(1).upper())
    if not intermediary:
        return securities

    utd_secs = set(sql.select_cols("price_history", ["security"]))
    return securities.difference({el[0] for el in utd_secs})


def update_sql(data: pd.DataFrame, sql: StockSQL):
    insert_sql = StockSQL.generate_insert_sql_string(
        "price_history", value_count=StockSQL.NUM_PRICE_HISTORY_VALUES
    )
    cols = list(data.columns)
    cols = cols[-1:] + cols[:-1]
    data = data[cols]
    print(data.head(5))
    sql.insert_pd(insert_sql, data)


def fetch_all_and_update_sql(api: TdaAPI):
    sql = StockSQL(price_history_sql_path)
    securities = fetch_securities_list(intermediary=True, sql=sql)
    for price_history in api.parallel_price_history(
            securities,
            Client.PriceHistory.PeriodType.YEAR,
            Client.PriceHistory.Period.TWENTY_YEARS,
            Client.PriceHistory.FrequencyType.DAILY,
            Client.PriceHistory.Frequency.DAILY,
    ):
        update_sql(price_history, sql)


def _list_generator(l, rate: int) -> List:
    idx = rate
    prev_idx = 0
    while idx < len(l):
        yield l[prev_idx:idx]
        prev_idx = idx
        idx += rate


def main():
    api = TdaAPI()
    fetch_all_and_update_sql(api)


if __name__ == "__main__":
    main()
