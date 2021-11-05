from discrete.ml.model.variable_time_transformer import VariableTimeTransformer

from discrete.news.article import Article


class NewsSentimentAnalyzer(object):
    def __init__(self, model: VariableTimeTransformer):
        self.model = model

    def sentiment_for(self, article: Article):
        content = article.text
        title = article.title
        return article.sentiment
