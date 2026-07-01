from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class LLMResponse:
    text: str
    model: str
    provider: str
    tokens_used: Optional[int] = None


class BaseLLMAdapter(ABC):

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> "LLMResponse":
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...
