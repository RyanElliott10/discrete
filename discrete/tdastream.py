import asyncio
import json
from typing import Any, Dict

import tda
from selenium import webdriver

from discrete.config import (
    token_path,
    api_key,
    redirect_uri,
    primary_account_id
)

TDARespMsg = Dict[str, Any]

# https://tda-api.readthedocs.io/en/stable/streaming.html

client = tda.auth.easy_client(
    api_key, redirect_uri, token_path, webdriver_func=lambda: webdriver.Chrome()
)
stream_client = tda.streaming.StreamClient(
    client, account_id=primary_account_id
)


def ohlcv_handler(msg: TDARespMsg):
    print(json.dumps(msg, indent=4))


def options_handler(msg: TDARespMsg):
    print(json.dumps(msg, indent=4))


async def read_stream():
    await stream_client.login()
    await stream_client.quality_of_service(stream_client.QOSLevel.EXPRESS)

    await stream_client.chart_equity_subs(["AAPL", "SPY", "SQ"])
    stream_client.add_chart_equity_handler(ohlcv_handler)

    await stream_client.level_one_option_subs(["AAPL"])
    stream_client.add_level_one_option_handler(options_handler)

    while True:
        await stream_client.handle_message()


asyncio.get_event_loop().run_until_complete(read_stream())
