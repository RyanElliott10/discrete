import backtrader as bt

from discrete.news.news_fetcher import NewsFetcher
from discrete.news.analyzers.news_sentiment_analyzer import NewsSentimentAnalyzer


class NewsSentimentStrategy(bt.Strategy):
    def __init__(self):
        self.news_fetcher = NewsFetcher()
        self.news_sentiment_analyzer = NewsSentimentAnalyzer()

    def next(self):
        r"""Use self.news_fetcher to grab the news from the previous week,
        run sentiment on those articles, make decision. Be wary about API
        usage. Should probably add a cache to the NewsFetcher class.
        """
        pass
