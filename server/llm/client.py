"""LLM provider abstraction. Business logic imports only from here and factory.py."""
from abc import ABC, abstractmethod

from pydantic import BaseModel


class LLMResponse(BaseModel):
    content: str
    usage_input_tokens: int
    usage_output_tokens: int


class LLMClient(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse: ...
