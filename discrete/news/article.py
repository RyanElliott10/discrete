from datetime import datetime
from typing import Iterable, Set


class Article(object):
    def __init__(
            self,
            news_url: str,
            image_url: str,
            title: str,
            text: str,
            source_name: str,
            date: str,
            topics: Iterable[str],
            sentiment: str,
            type: str,
            tickers: Iterable[str]
    ):
        self.news_url = news_url
        self.title = title
        self.text = text
        self.source = source_name
        self.date = datetime.strptime(date, "%a, %d %b %Y %H:%M:%S %z")
        self.topics = topics
        self.sentiment = sentiment
        self.type = type
        self.tickers = [t.casefold() for t in tickers]

    @staticmethod
    def print_article_list(articles: Iterable["Article"]):
        print(f"{len(articles)} articles")
        print("\n\n".join(map(str, articles)))

    def __str__(self):
        return f"title: {self.title}\nmentioned_tickers: " \
               f"{self.tickers}\nsentiment: " \
               f"{self.sentiment}\ntopics: {self.topics}\ndate: {self.date}"

    def __repr__(self):
        return self.__str__()
