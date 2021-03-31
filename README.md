# Discrete

## Commitments

### News

* Perform two things on each article:
    1. Determine which securities are mentioned/talked about
        * This will vary based on how the news API works
    2. Extract the pertinent parts about each ticker and perform sentiment
       analysis on them
        * Custom transfer learning model
        * Train model on next word prediction, then remove the final layer(s)
          and perform sentiment analysis
        * Or simply use word2vec and avoid transfer learning
* Rather than analyze the news, analyze financial statements and news particles
  that were released around the same time
* Or we could just boil news down to a sentiment for a given stock and use that
  as a small indicator paired with financial data

* [Reddit Thread](https://www.reddit.com/r/algotrading/comments/9dpxhm/looking_for_good_json_stock_news_api_feeds_free/)
* [Stock News API](https://stocknewsapi.com)

### Financial Data

* [Transformer with Time2Vector](https://towardsdatascience.com/stock-predictions-with-state-of-the-art-transformer-and-time-embeddings-3a4485237de6)

## Model
![Model Architecture](https://github.com/RyanElliott10/Discrete/blob/main/docs/img/architecture.svg)
