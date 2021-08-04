import asyncio

import tda
from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager

from discrete.bot.config import token_path, api_key, redirect_uri, \
    primary_account_id
from discrete.bot.handlers.equity_handler import EquityHandler
from discrete.bot.handlers.stream_handler import StreamHandler


def main():
    client = tda.auth.easy_client(
        api_key=api_key,
        redirect_uri=redirect_uri,
        token_path=token_path,
        webdriver_func=lambda: webdriver.Firefox(
            executable_path=GeckoDriverManager().install()
        ),
    )
    stream_client = tda.streaming.StreamClient(client,
        account_id=primary_account_id)

    stream_handler = StreamHandler(EquityHandler())
    asyncio.run(stream_handler.read_stream(stream_client))


if __name__ == "__main__":
    main()
