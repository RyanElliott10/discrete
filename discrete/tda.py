import multiprocessing
import re
import threading
import time
from typing import List

import backoff
import pandas as pd
import ratelimit
import requests
import tda
from tda import auth

from tda.client import Client

from discrete.config import (
    token_path,
    api_key,
    chrome_driver_path,
    redirect_uri,
    all_securities_url,
    tda_requests_per_limit,
)

securities_html_pattern = (
    r"<li><a href='https:\/\/stockanalysis.com\/"
    r"stocks\/(\D+?)\/'>(\D+?) - (\D+?)<\/a><\/li>"
)


pool = threading.BoundedSemaphore(value=multiprocessing.cpu_count())


class TDAAPI(object):
    def __init__(self):
        try:
            self.client = auth.client_from_token_file(token_path, api_key)
        except FileNotFoundError:
            from selenium import webdriver

            with webdriver.Chrome(executable_path=chrome_driver_path) as driver:
                self.client = auth.client_from_login_flow(
                    driver, api_key, redirect_uri, token_path
                )

    @backoff.on_exception(backoff.expo, ratelimit.RateLimitException, max_time=60)
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
        assert resp.status_code == 200, resp.raise_for_status()
        return pd.json_normalize(resp.json(), record_path=["candles"], meta=["symbol"])

    def parallel_price_history(
        self,
        securities: List[str],
        period_type: Client.PriceHistory.PeriodType,
        period: Client.PriceHistory.Period,
        frequency_type: Client.PriceHistory.FrequencyType,
        frequency: Client.PriceHistory.Frequency,
    ):
        for security in securities:
            print(security)
            data = self.price_history(
                security, period_type, period, frequency_type, frequency
            )
            data.to_csv(f"data/{security.upper()}.csv")
            print(data)


def fetch_securities_list() -> List[str]:
    r = requests.get(all_securities_url)
    if r.status_code != 200:
        raise Exception("Invalid response fetching securities")

    content = r.text
    securities = []
    for match in re.finditer(securities_html_pattern, content):
        securities.append(match.group(1))
    return securities


def update_sql(data: pd.DataFrame):
    pass


def fetch_all_and_update_sql(api: TDAAPI):
    securities = fetch_securities_list()
    price_histories = api.parallel_price_history(
        securities,
        Client.PriceHistory.PeriodType.YEAR,
        Client.PriceHistory.Period.TWENTY_YEARS,
        Client.PriceHistory.FrequencyType.DAILY,
        Client.PriceHistory.Frequency.DAILY,
    )
    [update_sql(price_history) for price_history in price_histories]


def _list_generator(l, rate: int) -> List:
    idx = rate
    prev_idx = 0
    while idx < len(l):
        yield l[prev_idx:idx]
        prev_idx = idx
        idx += rate


def main():
    api = TDAAPI()
    fetch_all_and_update_sql(api)


if __name__ == "__main__":
    main()
