import tda

from discrete.bot.handlers.equity_handler import EquityHandler


class StreamHandler(object):
    # TODO: This may make more sense to accept a List of handlers that can
    #  opt into listening to certain Enum events, determined by some instance
    #  method on each Handler (ABC)
    def __init__(self, equity_handler: EquityHandler):
        self.equity_handler = equity_handler

    async def read_stream(self, stream_client: tda.streaming.StreamClient):
        r"""https://tda-api.readthedocs.io/en/stable/streaming.html"""
        await stream_client.login()
        await stream_client.quality_of_service(stream_client.QOSLevel.EXPRESS)

        await stream_client.chart_equity_subs(["AAPL", "SPY", "SQ"])
        stream_client.add_chart_equity_handler(self.equity_handler.handler)

        while True:
            await stream_client.handle_message()
