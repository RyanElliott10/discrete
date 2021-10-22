# discrete

## Commitments
* NLP
* Buying/selling
    * Importantly, we may be able to get by just writing a basic backtester. 
      Aside from data, there is no real need to interact with an API
* Multithreading

## Crude Plan
Gotta get this done ASAP so we will *not* be connecting to TDA or any other 
API to make trades. Instead, we will scrape historical data from these 
services and writing our own backtester, or more likely using a library. We 
can ask for news on a given set of securities that we'll support, turn the 
processing of these articles through the NLP model into something that 
supports multi-threading, and then run this through the algorithm. The 
StockNewsAPI already labels the articles for us so we can use that as 
training data. May also end up just using the title rather than having to 
make another request to read and parse the article.

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
![Model Architecture](https://github.com/RyanElliott10/discrete/blob/main/docs/img/architecture.svg)

## Todo
* [ ] Research viable output representations
	* Just do the price difference/percent from the last day
	* It uses the last close as the principal for each prediction; $y_0$ is independent of $y_1$
	* Granted, this (almost) defeats the purpose of using a transformer
		* With $x_N$ representing a close price of $25
		* $y_0$ would predict +1%, $25.25
		* $y_1$ would be predict +2.3%, $25.575
* [ ] Figure out how we would go about predicting short squeeze events
	* Would we have a simple indicator saying "yes, it will close higher and it'll be a short squeeze"?

## Usage
### Model Training
`python3 -m discrete.train -c path/to/config.yaml [-d]`