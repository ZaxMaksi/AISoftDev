"""Справжній провайдер: OpenAI-сумісний API (Google Gemini через OpenAI-ендпоінт).

Транслює наш `Request` у виклик OpenAI SDK, повертає `Chunk` у streaming
з `usage` та зводить усі помилки до єдиного `ProviderError`.
"""

import time
from typing import Iterator

import openai

from .base import Chunk, ProviderError, Request, Result


class OpenAICompatProvider:
    """Провайдер поверх OpenAI-сумісного API."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = openai.OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
            timeout=self.timeout,
            max_retries=0,  # Повтори контролюються нашим шлюзом
        )

    def _convert_error(self, exc: Exception) -> ProviderError:
        """Перетворити виняток SDK на ProviderError."""
        if isinstance(exc, openai.APITimeoutError):
            return ProviderError("timeout", "Перевищено час очікування відповіді провайдера", retryable=True, detail=str(exc))
        if isinstance(exc, openai.RateLimitError):
            retry_after = 5.0
            if hasattr(exc, "response") and exc.response and "retry-after" in exc.response.headers:
                try:
                    retry_after = float(exc.response.headers["retry-after"])
                except ValueError:
                    pass
            return ProviderError("rate_limit", "Перевищено ліміт запитів провайдера", retryable=True, retry_after=retry_after, detail=str(exc))
        if isinstance(exc, openai.AuthenticationError):
            return ProviderError("auth", "Помилка автентифікації API ключа", retryable=False, detail="401 Unauthorized")
        if isinstance(exc, openai.BadRequestError):
            return ProviderError("bad_request", "Некоректний запит до моделі", retryable=False, detail=str(exc))
        if isinstance(exc, openai.APIConnectionError):
            return ProviderError("unavailable", "Втрачено зʼєднання з сервером моделі", retryable=True, detail=str(exc))
        if isinstance(exc, openai.InternalServerError):
            return ProviderError("unavailable", "Внутрішня помилка сервера моделі (5xx)", retryable=True, detail=str(exc))
        if isinstance(exc, openai.APIError):
            return ProviderError("unknown", f"Помилка API провайдера: {exc.message}", retryable=False, detail=str(exc))
        return ProviderError("unknown", f"Непередбачена помилка: {exc}", retryable=False, detail=str(exc))

    def _extract_usage(self, raw_usage) -> dict | None:
        if raw_usage is None:
            return None
        if hasattr(raw_usage, "model_dump"):
            return raw_usage.model_dump(exclude_none=True)
        return dict(raw_usage)

    def complete(self, request: Request) -> Result:
        """Отримати відповідь цілком."""
        started = time.perf_counter()
        model_name = request.model or self.model
        kwargs = {
            "model": model_name,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        try:
            resp = self.client.chat.completions.create(**kwargs)
            choice = resp.choices[0] if resp.choices else None
            text = choice.message.content if choice and choice.message else ""
            finish_reason = choice.finish_reason if choice else None
            usage = self._extract_usage(getattr(resp, "usage", None))
            actual_model = getattr(resp, "model", model_name)
            return Result(
                text=text or "",
                provider=self.name,
                model=actual_model,
                usage=usage,
                finish_reason=finish_reason,
                elapsed=time.perf_counter() - started,
            )
        except Exception as exc:
            raise self._convert_error(exc) from exc

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати фрагменти відповіді в міру надходження."""
        model_name = request.model or self.model
        kwargs = {
            "model": model_name,
            "messages": request.messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        def _do_stream(opts: dict):
            return self.client.chat.completions.create(**opts)

        stream_resp = None
        try:
            stream_resp = _do_stream(kwargs)
        except openai.BadRequestError as exc:
            # Деякі провайдери не підтримують stream_options
            if "stream_options" in str(exc) or "unknown" in str(exc).lower():
                kwargs.pop("stream_options", None)
                try:
                    stream_resp = _do_stream(kwargs)
                except Exception as inner_exc:
                    raise self._convert_error(inner_exc) from inner_exc
            else:
                raise self._convert_error(exc) from exc
        except Exception as exc:
            raise self._convert_error(exc) from exc

        try:
            for chunk in stream_resp:
                text_delta = ""
                finish_reason = None
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        text_delta = delta.content
                    finish_reason = chunk.choices[0].finish_reason

                usage = self._extract_usage(getattr(chunk, "usage", None))
                if text_delta or usage or finish_reason:
                    yield Chunk(text=text_delta, usage=usage, finish_reason=finish_reason)
        except Exception as exc:
            raise self._convert_error(exc) from exc
