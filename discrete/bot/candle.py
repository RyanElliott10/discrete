from typing import Dict


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
            "key"        : "security",
            "OPEN_PRICE" : "open",
            "HIGH_PRICE" : "high",
            "LOW_PRICE"  : "low",
            "CLOSE_PRICE": "close",
            "VOLUME"     : "volume",
        }

    @classmethod
    def from_tda_content(cls, content: Dict[str, any],
            timestamp: int) -> "TDACandle":
        init_dict = {"timestamp": timestamp}
        for key, value in cls.tda_names_to_candle().items():
            init_dict[value] = content.pop(key, 0)
        return cls(**init_dict)
