from typing import List

class Article(object):
    def __init__(
            self,
            news_url: str,
            image_url: str,
            title: str,
            text: str,
            source_name: str,
            date: str,
            topics: List[str],
            sentiment: str,
            type: str,
            tickers: str
    ):
        self.news_url = news_url
        self.title = title
        self.text = text
        self.source = source_name
        self.date = date
        self.topics = topics
        self.sentiment = sentiment
        self.type = type
        self.mentioned_tickers = tickers


    def __str__(self):
        return f"title: {self.title}\nmentioned_tickers: " \
               f"{self.mentioned_tickers}\nsentiment: " \
               f"{self.sentiment}\ntopics: {self.topics}"


    def __repr__(self):
        return self.__str__()
