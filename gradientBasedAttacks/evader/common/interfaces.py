from abc import ABC, abstractmethod
from typing import List

class BaseEvader(ABC):
    @abstractmethod
    def evade(self, texts: List[str], **kwargs) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def load(self, model_dir: str) -> None:
        raise NotImplementedError
