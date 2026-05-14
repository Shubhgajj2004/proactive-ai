"""OpenRouter backend — the ONLY file that imports from openai."""
import json

from openai import AsyncOpenAI
from pydantic import BaseModel

from server.llm.client import LLMClient, LLMResponse


class OpenRouterClient(LLMClient):
    def __init__(self, model: str, api_key: str, base_url: str = "https://openrouter.ai/api/v1"):
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    async def complete(
        self,
        messages: list[dict],
        response_format: type[BaseModel] | None = None,
        temperature: float = 0.3,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
        }

        if response_format is not None:
            # Request JSON output and parse into the Pydantic model
            schema = response_format.model_json_schema()
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema.get("title", "Response"), "schema": schema},
            }

        resp = await self._client.chat.completions.create(**kwargs)
        content = resp.choices[0].message.content or ""

        # If structured output was requested, validate the JSON
        if response_format is not None:
            response_format.model_validate(json.loads(content))  # raises if invalid

        return LLMResponse(
            content=content,
            usage_input_tokens=resp.usage.prompt_tokens if resp.usage else 0,
            usage_output_tokens=resp.usage.completion_tokens if resp.usage else 0,
        )
