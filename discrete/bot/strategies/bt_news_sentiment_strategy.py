import backtrader as bt

from discrete.ml.model.variable_time_transformer import VariableTimeTransformer
from discrete.news.analyzers.news_sentiment_analyzer import \
    NewsSentimentAnalyzer
from discrete.news.news_fetcher import NewsFetcher


class BTNewsSentimentStrategy(bt.Strategy):
    def __init__(self, model: VariableTimeTransformer):
        self.news_fetcher = NewsFetcher()
        self.news_sentiment_analyzer = NewsSentimentAnalyzer(model)

    def next(self):
        r"""Use self.news_fetcher to grab the news from the previous week,
        run sentiment on those articles, make decision. Be wary about API
        usage. Should probably add a cache to the NewsFetcher class.
        """
        pass
