import pandas as pd
import plotly.graph_objects as go


def graph_ohlc(data: pd.DataFrame):
    fig = go.Figure(data=[go.Candlestick(
        # x=data["datetime"],
        x=data.index,
        open=data["open"],
        high=data["high"],
        low=data["low"],
        close=data["close"]
    )])
    fig.show()


def overrides(interface_class):
    def overrider(method):
        assert (method.__name__ in dir(interface_class))
        return method

    return overrider
