import datetime
from typing import List, Tuple, Union

from discrete.news.article import Article

NEWS_API_MAX_ITEMS = 50
NEWS_API_START_DATE = datetime.datetime(2019, 2, 1)

UnixDateTimestamp = int
UnixDateRange = Tuple[UnixDateTimestamp, UnixDateTimestamp]
DateRange = Tuple[datetime.datetime, datetime.datetime]


class NewsCache(object):
    r"""
    Largely used by the NewsFetcher. The user should have no reason to
    directly touch this class.

    Maintains a state of NewsAPI articles that have already been fetched
    based on tickers and date ranges. This reduces the number of network calls,
    thereby reducing network strain and reducing API costs.
    """

    class _NewsCacheTicker(object):
        def __init__(self, ticker: str):
            self.ticker = ticker
            self.ranges: List[UnixDateTimestamp] = []
            self.articles: List[Article] = []

        @staticmethod
        def _get_timestamp_range(
                inp_range: DateRange
        ) -> List[UnixDateTimestamp]:
            r"""Converts a pseudo-range of datetime.datetime to an expanded
            range of UnixDateTimestamp, or ints.
            """
            start_stamp = UnixDateTimestamp(inp_range[0].timestamp())
            end_stamp = UnixDateTimestamp(inp_range[1].timestamp())
            return list(range(start_stamp, end_stamp))

        def adjust_range_from_cache(self, date_range: DateRange) -> DateRange:
            r"""Accepts a range that we would like to query for articles.
            Finds any overlap between the input range and cached ranges,
            returning the new DateRange that has not been cached.
            """
            timestamp_range = self._get_timestamp_range(date_range)
            # delta = sorted(list(set(self.ranges) - set(timestamp_range)))
            delta = set(self.ranges).intersection(set(timestamp_range))
            delta = sorted(list(delta))
            if len(delta) == 0:
                return date_range
            return (datetime.datetime.utcfromtimestamp(delta[0]),
                    datetime.datetime.utcfromtimestamp(delta[-1]))

        def fetch_articles_in_range(
                self,
                date_range: DateRange
        ) -> List[Article]:
            hit = False
            articles = []

            for article in self.articles:
                if date_range[0] < article.date < date_range[1]:
                    hit = True
                    articles.append(article)
                elif hit:
                    break

            return articles

        def query_cache(self, date_range: DateRange) -> List[Article]:
            date_range = self.adjust_range_from_cache(date_range)
            return self.fetch_articles_in_range(date_range)

        def _remove_range_duplicates(self):
            self.ranges = list(dict.fromkeys(self.ranges))

        def cache_range(self, date_range: DateRange, articles: List[Article]):
            self.ranges += self._get_timestamp_range(date_range)
            self._remove_range_duplicates()
            self.articles.extend(articles)
            self.articles.sort(key=lambda a: a.date)

        def __repr__(self) -> str:
            return f"{self.ticker}, {len(self.articles)} articles"

    def __init__(self):
        self._cached_tickers: List[NewsCache._NewsCacheTicker] = []

    def _fetch_cached_ticker(
            self,
            ticker: str
    ) -> Union[_NewsCacheTicker, None]:
        cached_ticker = list(filter(
            lambda t: t.ticker.casefold() == ticker.casefold(),
            self._cached_tickers
        ))
        if len(cached_ticker) == 0:
            return None
        return cached_ticker[0]

    def query_cache(
            self,
            ticker: str,
            date_range: DateRange
    ) -> Union[List[Article], None]:
        r"""Queries the cached tickers. Returns None if the ticker has no
        cache for the given range, a list of Articles within that range,
        otherwise. An empty list means that call has been cached but no
        Articles were returned.
        """
        cached_ticker = self._fetch_cached_ticker(ticker)
        if cached_ticker is None:
            return None
        return cached_ticker.query_cache(date_range)

    def cache_range(
            self,
            ticker: str,
            date_range: DateRange,
            articles: List[Article]
    ):
        cached_ticker = self._fetch_cached_ticker(ticker)
        if cached_ticker is None:
            cached_ticker = self._NewsCacheTicker(ticker)

        cached_ticker.cache_range(date_range, articles)
        self._cached_tickers.append(cached_ticker)


def _generate_article_within_date_range(
        ticker: str,
        date_range: DateRange
) -> Article:
    import random

    rand_date = datetime.datetime.fromtimestamp(random.choice(range(
        UnixDateTimestamp(date_range[0].timestamp()),
        UnixDateTimestamp(date_range[1].timestamp())
    )))
    return Article(
        "foo.url", "foo.image.url", "Foo Title", "Foo text",
        "Foo News Network",
        rand_date.strftime("%a, %d %b %Y %H:%M:%S %z") + "-0400",
        ["Foo topic", "Bar topic"],
        "Neutral",
        "Foo type", [ticker]
    )


def debug_main():
    cache = NewsCache()
    cache.cache_range(
        "aapl", (datetime.datetime(2019, 7, 8), datetime.datetime(2019, 8, 31)),
        [
            _generate_article_within_date_range("aapl",
                (datetime.datetime(2019, 7, 8), datetime.datetime(2019, 8, 31))
            ), _generate_article_within_date_range("aapl", (datetime.datetime(
            2019, 7, 8), datetime.datetime(2019, 8, 31)))
        ]
    )

    print(cache.query_cache(
        "aapl", (datetime.datetime(2019, 7, 8), datetime.datetime(2019, 8, 31)))
    )


if __name__ == "__main__":
    debug_main()
