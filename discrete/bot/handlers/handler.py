from abc import ABC, abstractmethod

from typing import Any, Dict
TDARespMsg = Dict[str, Any]


class TDAMessageHandler(ABC):
    @abstractmethod
    def handler(self, msg: TDARespMsg):
        raise NotImplementedError

    @abstractmethod
    def responds_to(self, event: any) -> bool:
        raise NotImplementedError
