import asyncio
import json
from typing import Any, Dict
from abc import ABC, abstractmethod

import tda
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager

from discrete.bot.config import token_path, api_key, redirect_uri, primary_account_id

TDARespMsg = Dict[str, Any]


class TDACandle(object):
    def __init__(
        self,
        security: str,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: int,
        timestamp: int,
    ):
        self.security = security
        self.open = open
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.timestamp = timestamp

    @staticmethod
    def tda_names_to_candle() -> Dict[str, str]:
        return {
            "key": "security",
            "OPEN_PRICE": "open",
            "HIGH_PRICE": "high",
            "LOW_PRICE": "low",
            "CLOSE_PRICE": "close",
            "VOLUME": "volume",
        }

    @classmethod
    def from_tda_content(cls, content: Dict[str, any], timestamp: int) -> "TDACandle":
        init_dict = {"timestamp": timestamp}
        for key, value in cls.tda_names_to_candle().items():
            init_dict[value] = content.pop(key, 0)
        return cls(**init_dict)


class TDAMessageHandler(ABC):
    @abstractmethod
    def handler(self, msg: TDARespMsg):
        raise NotImplementedError


class EquityHandler(TDAMessageHandler):
    def __init__(self):
        self.messages = []

    def handler(self, msg: TDARespMsg):
        raw_candles = msg["content"]
        timestamp = msg["timestamp"]
        for raw_candle in raw_candles:
            candle = TDACandle.from_tda_content(raw_candle, timestamp)
        self.messages.append(msg)
        print(len(self.messages))


def options_handler(msg: TDARespMsg):
    print(json.dumps(msg, indent=4))


async def read_stream(stream_client: tda.streaming.StreamClient):
    r"""https://tda-api.readthedocs.io/en/stable/streaming.html"""
    await stream_client.login()
    await stream_client.quality_of_service(stream_client.QOSLevel.EXPRESS)

    equity_handler = EquityHandler()

    await stream_client.chart_equity_subs(["AAPL", "SPY", "SQ"])
    stream_client.add_chart_equity_handler(equity_handler.handler)

    while True:
        await stream_client.handle_message()


def main():
    client = tda.auth.easy_client(
        api_key=api_key,
        redirect_uri=redirect_uri,
        token_path=token_path,
        webdriver_func=lambda: webdriver.Firefox(
            executable_path=GeckoDriverManager().install()
        ),
    )
    stream_client = tda.streaming.StreamClient(client, account_id=primary_account_id)

    asyncio.run(read_stream(stream_client))


if __name__ == "__main__":
    main()
