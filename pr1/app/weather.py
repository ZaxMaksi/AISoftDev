"""Модуль інтеграції із зовнішнім API погоди.

Це єдине місце застосунку, яке знає про HTTP: адреси сервісів, параметри
запиту, коди відповіді й формат JSON. Веб-рівень (`app/main.py`) отримує
звідси готовий результат або зрозумілу помилку і нічого не знає про
`requests`.
"""

import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
DEFAULT_TIMEOUT = 5.0


class WeatherError(Exception):
    """Базовий виняток для всіх помилок отримання погоди."""


class CityNotFoundError(WeatherError):
    """Виняток, коли вказане місто не знайдено."""


class InvalidAPIRequestError(WeatherError):
    """Виняток для некоректних параметрів або помилок клієнта (4xx)."""


class APIServiceUnavailableError(WeatherError):
    """Виняток при недоступності сервісу або внутрішній помилці (5xx, таймаут)."""


class WeatherParsingError(WeatherError):
    """Виняток, коли формат відповіді API не збігається з очікуваним."""


def find_city(name: str) -> dict:
    """Знайти координати міста за його назвою.

    Повертає словник з назвою, країною, широтою та довготою або
    викидає відповідну підгрупу WeatherError.
    """
    clean_name = name.strip()
    if not clean_name:
        raise InvalidAPIRequestError("Назва міста не може бути порожньою")

    params = {
        "name": clean_name,
        "count": 1,
        "language": "uk"  # Запитуємо українською для кращого досвіду
    }

    try:
        response = requests.get(GEOCODING_URL, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.Timeout as e:
        raise APIServiceUnavailableError("Перевищено час очікування відповіді від сервісу геокодування") from e
    except requests.exceptions.RequestException as e:
        raise APIServiceUnavailableError("Сервіс геокодування недоступний або сталася помилка мережі") from e

    if response.status_code >= 500:
        raise APIServiceUnavailableError(f"Помилка сервера геокодування (код {response.status_code})")
    elif response.status_code >= 400:
        try:
            err_msg = response.json().get("reason", response.text)
        except Exception:
            err_msg = response.text
        raise InvalidAPIRequestError(f"Некоректний запит до сервісу геокодування: {err_msg}")

    try:
        data = response.json()
    except Exception as e:
        raise WeatherParsingError("Не вдалося розібрати відповідь сервісу геокодування") from e

    # Якщо міста не знайдено, ключ 'results' відсутній у відповіді
    if not data or "results" not in data or not data["results"]:
        raise CityNotFoundError(f"Місто '{clean_name}' не знайдено")

    result = data["results"][0]
    return {
        "name": result.get("name", clean_name),
        "country": result.get("country", ""),
        "latitude": result.get("latitude"),
        "longitude": result.get("longitude")
    }


def get_current_weather(city: str) -> dict:
    """Повернути поточну погоду в місті: температуру й швидкість вітру.

    Поєднує геокодування і запит прогнозу, повертає структуровані дані.
    """
    city_info = find_city(city)

    lat = city_info["latitude"]
    lon = city_info["longitude"]

    if lat is None or lon is None:
        raise WeatherParsingError("У результатах геокодування відсутні координати міста")

    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,wind_speed_10m"
    }

    try:
        response = requests.get(FORECAST_URL, params=params, timeout=DEFAULT_TIMEOUT)
    except requests.exceptions.Timeout as e:
        raise APIServiceUnavailableError("Перевищено час очікування відповіді від сервісу прогнозу погоди") from e
    except requests.exceptions.RequestException as e:
        raise APIServiceUnavailableError("Сервіс прогнозу погоди недоступний або сталася помилка мережі") from e

    if response.status_code >= 500:
        raise APIServiceUnavailableError(f"Помилка сервера прогнозу погоди (код {response.status_code})")
    elif response.status_code >= 400:
        try:
            err_msg = response.json().get("reason", response.text)
        except Exception:
            err_msg = response.text
        raise InvalidAPIRequestError(f"Некоректний запит до сервісу прогнозу погоди: {err_msg}")

    try:
        data = response.json()
    except Exception as e:
        raise WeatherParsingError("Не вдалося розібрати відповідь сервісу прогнозу погоди") from e

    current = data.get("current")
    current_units = data.get("current_units")

    if not current or not current_units:
        raise WeatherParsingError("У відповіді сервісу погоди відсутні поточні дані")

    temp = current.get("temperature_2m")
    wind_speed = current.get("wind_speed_10m")
    temp_unit = current_units.get("temperature_2m", "°C")
    wind_speed_unit = current_units.get("wind_speed_10m", "km/h")

    if temp is None or wind_speed is None:
        raise WeatherParsingError("У відповіді сервісу погоди відсутня температура чи швидкість вітру")

    return {
        "city": city_info["name"],
        "country": city_info["country"],
        "latitude": lat,
        "longitude": lon,
        "temperature": temp,
        "temperature_unit": temp_unit,
        "wind_speed": wind_speed,
        "wind_speed_unit": wind_speed_unit
    }

