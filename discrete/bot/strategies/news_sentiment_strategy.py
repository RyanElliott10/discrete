from datetime import datetime

from discrete.bot.candle import TDACandle
from discrete.bot.strategies.strategy import Strategy
from discrete.ml.model.variable_time_transformer import VariableTimeTransformer
from discrete.news.analyzers.news_sentiment_analyzer import \
    NewsSentimentAnalyzer
from discrete.news.news_fetcher import NewsFetcher


class NewsSentimentStrategy(Strategy):
    def __init__(self, model: VariableTimeTransformer):
        self.news_sentiment_analyzer = NewsSentimentAnalyzer(model)
        self.candles = []
        self.sent_score = 0

    @staticmethod
    def naive_sentiment_score(sentiment: str) -> int:
        if sentiment == "Positive":
            return 1
        elif sentiment == "Negative":
            return -1
        return 0

    def consume(self, candle: TDACandle):
        start_date = end_date = datetime.fromtimestamp(candle.timestamp)
        news_fetcher = NewsFetcher(start_date, end_date)
        articles = news_fetcher.fetch_news_response({candle.security}).articles
        sentiments = [self.news_sentiment_analyzer.sentiment_for(article) for \
                      article in articles]
        self.sent_score = sum(map(self.naive_sentiment_score, sentiments))
        self.candles.append(candle)

    def _buy(self, candle: TDACandle):
        date = datetime.fromtimestamp(candle.timestamp).strftime("%Y-%m-%d")
        print(f"{date}, BUY EXECUTED, {candle.close}")

    def _sell(self, candle: TDACandle):
        date = datetime.fromtimestamp(candle.timestamp).strftime("%Y-%m-%d")
        print(f"{date}, SELL EXECUTED, {candle.close}")

    def act(self):
        candle = self.candles[-1]
        if self.sent_score == 1:
            self._buy(candle)
        elif self.sent_score == -1:
            self._sell(candle)
