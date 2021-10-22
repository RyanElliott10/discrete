import json
from typing import List

from discrete.news.article import Article


class NewsResp(object):
    def __init__(self, articles: List[Article]):
        self.articles = articles

    @staticmethod
    def from_json(jsonstr: str):
        articles = []
        articles_dict = json.loads(jsonstr)
        for article in articles_dict["data"]:
            articles.append(Article(**article))
        return NewsResp(articles)


def debug_main():
    with open("data/news/news_response.json", "r") as f:
        jsonstr = f.read()

    news = NewsResp.from_json(jsonstr)
    print(news.articles)



if __name__ == "__main__":
    debug_main()