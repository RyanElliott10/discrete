from enum import Enum


class CandleSourceType(Enum):
    Open = 'open'
    High = 'high'
    Low = 'low'
    Close = 'close'
    Volume = 'volume'

    def __str__(self) -> str:
        return self.value
