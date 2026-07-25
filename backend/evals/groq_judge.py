"""A Groq-backed judge model for DeepEval.

DeepEval defaults to OpenAI. To judge with a Groq-hosted model we subclass
DeepEvalBaseLLM and route generation through the Groq SDK. DeepEval 4.x calls the
model two ways:

- generate(prompt)               -> str            (free-text)
- generate(prompt, schema=Model) -> Model instance (structured verdict)

For the structured path we use Groq JSON mode (response_format=json_object) and
validate the returned JSON into the requested pydantic schema. temperature=0.0 so the
judge is as deterministic as the provider allows — a grader should not sample.
"""

import json
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from groq import AsyncGroq, Groq, RateLimitError
from pydantic import BaseModel
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config.settings import settings

# The free tier caps tokens-per-minute; the judge fans out several calls per case, so
# a full run brushes the limit. Back off and retry on 429 rather than failing the test.
_retry_on_rate_limit = retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(multiplier=2, min=2, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)


class GroqJudge(DeepEvalBaseLLM):
    """DeepEval judge that generates via a Groq chat model (e.g. gpt-oss-20b)."""

    def __init__(self, model: str) -> None:
        self._model = model
        self._sync = Groq(api_key=settings.groq_api_key)
        self._async = AsyncGroq(api_key=settings.groq_api_key)

    def load_model(self) -> "GroqJudge":
        return self

    def get_model_name(self) -> str:
        return self._model

    @staticmethod
    def _prompt_for(prompt: str, schema: type[BaseModel] | None) -> str:
        """Append a JSON-schema instruction when a structured verdict is required.

        Groq JSON mode requires the word 'json' to appear in the prompt.
        """
        if schema is None:
            return prompt
        return (
            f"{prompt}\n\nReturn a JSON object only, matching this schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )

    def _kwargs(self, prompt: str, schema: type[BaseModel] | None) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": self._prompt_for(prompt, schema)}],
            "temperature": 0.0,
            # gpt-oss is a reasoning model: reasoning tokens count against the completion
            # budget. Without headroom, long verdict JSON is truncated mid-document and
            # Groq raises json_validate_failed. 4096 fits multi-claim verdicts + reasoning.
            "max_tokens": 4096,
        }
        if schema is not None:
            kwargs["response_format"] = {"type": "json_object"}
        return kwargs

    @staticmethod
    def _parse(content: str, schema: type[BaseModel] | None) -> Any:
        return schema.model_validate_json(content) if schema is not None else content

    @_retry_on_rate_limit
    def generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        resp = self._sync.chat.completions.create(**self._kwargs(prompt, schema))
        return self._parse(resp.choices[0].message.content, schema)

    @_retry_on_rate_limit
    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None) -> Any:
        resp = await self._async.chat.completions.create(**self._kwargs(prompt, schema))
        return self._parse(resp.choices[0].message.content, schema)
