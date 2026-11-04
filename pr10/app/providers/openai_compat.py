"""Справжній провайдер: OpenAI-сумісний API — той самий, що в ПР3–ПР9.

Єдиний модуль застосунку, який імпортує SDK `openai`. Його справа —
перетворити наш `Request` на виклик API, а відповідь і збої API — на
`Result`, `Chunk` і `ProviderError` з `base.py`. Що робити зі збоєм —
повторити, зачекати, перейти на запасний варіант — тут не вирішується:
це справа шлюзу.

Клас нижче — заготовка. Рішення з розділу 2 практичної роботи:

* який виняток SDK у який `kind` перетворюється і який із них
  `retryable`: `APITimeoutError`, `RateLimitError` (429),
  `APIConnectionError`, `InternalServerError` (5xx),
  `AuthenticationError` (401), `BadRequestError` (400). Де взяти
  `Retry-After`, якщо провайдер його надіслав;
* таймаут: SDK приймає його при створенні клієнта і на окремий виклик
  (`with_options`). Що передавати — `LLM_TIMEOUT` на спробу чи щось
  інше — і хто рахує таймаут бездіяльності в streaming: SDK такого не
  вміє, це робиться у споживача фрагментів;
* `max_retries` SDK за замовчуванням не нульовий — SDK сам мовчки
  повторює деякі запити. Залишити це йому чи вимкнути й повторювати у
  шлюзі, де видно спроби, бюджет і запасний варіант;
* streaming: `stream=True` віддає обʼєкт, по якому йдуть фрагментами
  (`choices[0].delta.content`). Як закрити його, якщо споживач
  перестав читати (скасування, таймаут бездіяльності), і чи приходить
  `usage` в останньому фрагменті (`stream_options`), а якщо ні —
  що записати;
* що з відповіді потрапляє в `detail` для журналу, а що — у `message`
  для людини. Ключ і повний текст запиту — ні туди, ні туди.

Обробку збоїв із ПР3–ПР9 перенесіть сюди — але тепер вона не підіймає
власний `LLMError`, а зводить усе до `ProviderError`.
"""

from typing import Iterator

from .base import Chunk, ProviderError, Request, Result


class OpenAICompatProvider:
    """Провайдер поверх OpenAI-сумісного API."""

    name = "openai"

    def __init__(self, base_url: str, api_key: str, model: str, timeout: float) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def complete(self, request: Request) -> Result:
        """Отримати відповідь цілком."""
        raise NotImplementedError("OpenAICompatProvider.complete ще не реалізовано")

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати фрагменти відповіді в міру надходження."""
        raise NotImplementedError("OpenAICompatProvider.stream ще не реалізовано")
