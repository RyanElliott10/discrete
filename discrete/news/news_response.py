import json
from typing import Set

from discrete.news.article import Article


class NewsResp(object):
    def __init__(self, articles: Set[Article]):
        self.articles = articles

    @staticmethod
    def from_json_str(jsonstr: str) -> "NewsResp":
        articles = set()
        articles_dict = json.loads(jsonstr)
        for article in articles_dict["data"]:
            articles.add(Article(**article))
        return NewsResp(articles)

    @staticmethod
    def from_json_dict(json_dict) -> "NewsResp":
        articles = set()
        for article in json_dict["data"]:
            articles.add(Article(**article))
        return NewsResp(articles)


def debug_main():
    with open("data/news/news_response.json", "r") as f:
        jsonstr = f.read()

    news = NewsResp.from_json_str(jsonstr)
    Article.print_article_list(news.articles)


if __name__ == "__main__":
    debug_main()
