import datetime
from abc import ABC, abstractmethod
from typing import List

import requests

from discrete.news.article import Article
from discrete.news.network_exception import NetworkException
from discrete.news.news_response import NewsResp
from discrete.news.private import news_api_key


NEWS_API_MAX_ITEMS = 50


class Fetcher(ABC):
    @abstractmethod
    def get_base_url(self) -> str:
        raise NotImplementedError


class NewsFetcher(Fetcher):
    base_url = "https://stocknewsapi.com/api/v1"

    def __init__(
            self,
            start: datetime.datetime,
            end: datetime.datetime = datetime.datetime.today()
    ):
        self.api_key = news_api_key
        self.start = start
        self.end = end

    def get_base_url(self) -> str:
        return self.base_url

    @staticmethod
    def _format_date(date: datetime.datetime):
        return datetime.date.strftime(date, "%m%d%Y")

    def _build_date_range(self) -> str:
        return f"date={self._format_date(self.start)}-" \
               f"{self._format_date(self.end)}"

    def _join_tickers(self, tickers: List[str]) -> str:
        return ",".join(list(map(str.upper, tickers)))

    def _build_tickers_str(self, tickers: List[str]) -> str:
        joined_tickers = self._join_tickers(tickers)
        return f"tickers={joined_tickers}"

    def _build_items_str(self, num_items: int) -> str:
        return f"items={num_items}"

    def _build_api_key_str(self) -> str:
        return f"token={self.api_key}"

    def _build_url_str_from_components(self, comps: List[str]) -> str:
        builder = f"{self.base_url}?"
        builder += "&".join(comps)
        return builder

    def _build_tickers_url(
            self,
            tickers: List[str],
            num_items: int
    ) -> str:
        r"""Structured as follows:
        stocknewsapi.com/api/v1?tickers={tickers}\
            &items={num_items}&token={token}
        """
        comps = []
        comps.append(self._build_tickers_str(tickers))
        comps.append(self._build_items_str(num_items))
        comps.append(self._build_api_key_str())
        if self.start is not None:
            comps.append(self._build_date_range())
        return self._build_url_str_from_components(comps)

    def fetch_ticker_articles(
            self,
            ticker: str,
            num_items: int = NEWS_API_MAX_ITEMS
    ) -> List[Article]:
        return self.fetch_tickers_articles([ticker], num_items=num_items)

    def fetch_tickers_articles(
            self,
            tickers: List[str],
            num_items: int = NEWS_API_MAX_ITEMS
    ) -> List[Article]:
        return self.fetch_news_response(tickers, num_items=num_items).articles

    def fetch_news_response(
            self,
            tickers: List[str],
            num_items: int = NEWS_API_MAX_ITEMS
    ) -> NewsResp:
        if num_items > NEWS_API_MAX_ITEMS:
            print("Warning: requesting more than max items according to "
                  f"NewsAPI, downgrading to {NEWS_API_MAX_ITEMS}")

        url = self._build_tickers_url(tickers, num_items)
        resp = requests.get(url)
        if resp.status_code != 200:
            raise NetworkException(resp.text, resp.status_code)

        news_response = NewsResp.from_json_dict(resp.json())
        return news_response


def debug_main():
    fetcher = NewsFetcher(
        start=datetime.datetime(2019, 1, 1), end=datetime.datetime.now()
    )
    articles = fetcher.fetch_tickers_articles(
        ["aapl", "fb", "amzn"]
    )
    Article.print_article_list(articles)


if __name__ == "__main__":
    debug_main()