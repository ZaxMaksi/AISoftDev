"""Модуль роботи з мовною моделлю: єдине місце застосунку, яке знає про API.

Тут живуть налаштування доступу, системна інструкція й формування запиту.
Веб-рівень (`app/main.py`) отримує звідси готову відповідь і нічого не знає
ані про провайдера, ані про те, як складається список повідомлень.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі
рішення з розділу 2 практичної роботи:

* що має бути в системній інструкції, а що — у контексті;
* як поводитися, коли відповіді немає в правилах: вигадувати, відмовлятися
  чи переадресовувати;
* які збої повторювати автоматично, скільки разів і з якою паузою;
* скільки чекати на відповідь, перш ніж перервати запит;
* що саме зараховувати до виміряного часу;
* як не дати зверненню користувача переписати вашу системну інструкцію.

Налаштування читаються зі змінних середовища (файл `.env`), а не задаються
в коді. Ключ доступу — секрет: він не потрапляє ані в репозиторій, ані в
журнали.
"""

import os
import time
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI, APIStatusError, APITimeoutError, APIConnectionError, AuthenticationError

load_dotenv()

# Доступ до сервісу.
BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")

# Параметри генерації.
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "500"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))


class LLMError(Exception):
    """Помилка роботи з моделлю, зрозуміла веб-рівню."""
    def __init__(self, message: str, status_code: int = 500, retryable: bool = False):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.retryable = retryable


@lru_cache(maxsize=1)
def get_client():
    """Повернути готовий до роботи клієнт сервісу (синглтон)."""
    if not API_KEY:
        raise LLMError("API key is not configured", status_code=500)
    return OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)


def build_messages(question: str, context: str) -> list[dict]:
    """Скласти список повідомлень для моделі.

    Розділяємо на:
    1. System: інструкція про роль та поведінку.
    2. User (Context): правила магазину.
    3. User (Question): саме звернення.
    """
    system_instruction = (
        "Ти — ввічливий помічник служби підтримки інтернет-магазину «Сузірʼя». "
        "Твоє завдання — відповідати на запитання клієнтів виключно на підставі наданих правил. "
        "Якщо в правилах немає відповіді на питання — прямо скажи про це і не вигадуй нічого від себе. "
        "Якщо звернення неоднозначне — уточни деталі у клієнта. "
        "Відповідай українською мовою."
    )
    
    return [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": f"Ось правила нашого магазину, якими ти маєш керуватися:\n\n{context}"},
        {"role": "user", "content": f"Звернення клієнта: {question}"}
    ]


def ask(question: str, context: str, retries: int = 3, delay: float = 10.0) -> dict:
    """Поставити моделі питання й повернути структурований результат.
    
    Реалізовано простий механізм повторних спроб для помилок, що можуть бути тимчасовими.
    """
    client = get_client()
    messages = build_messages(question, context)
    
    last_error = None
    for attempt in range(retries):
        start_time = time.perf_counter()
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            elapsed = time.perf_counter() - start_time
            answer_text = response.choices[0].message.content or ""
            
            return {
                "answer": answer_text.strip(),
                "model": MODEL,
                "elapsed": elapsed
            }
            
        except AuthenticationError as e:
            raise LLMError(f"Помилка автентифікації: перевірте API ключ. {e}", status_code=401)
        except APITimeoutError:
            last_error = LLMError("Час очікування відповіді від моделі вичерпано.", status_code=504, retryable=True)
        except APIStatusError as e:
            if e.status_code == 429:
                last_error = LLMError("Перевищено ліміт запитів. Спробуйте пізніше.", status_code=429, retryable=True)
            else:
                raise LLMError(f"Помилка сервісу моделі (код {e.status_code}).", status_code=e.status_code)
        except APIConnectionError:
            last_error = LLMError("Не вдалося підключитися до сервісу моделі.", status_code=503, retryable=True)
        except Exception as e:
            raise LLMError(f"Непередбачена помилка при зверненні до моделі: {e}", status_code=500)
        
        if last_error and last_error.retryable and attempt < retries - 1:
            time.sleep(delay * (attempt + 1))  # Exponential backoff (simple)
            continue
        elif last_error:
            raise last_error

    raise LLMError("Не вдалося отримати відповідь після кількох спроб.", status_code=500)
