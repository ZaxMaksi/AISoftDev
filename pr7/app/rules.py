"""Програмні правила: перевірка змісту вилучених даних.

Схема (`app/schema.py`) каже, що IBAN — рядок, а сума — число. Чи це
справжній IBAN і чи сходиться сума з позиціями, схема не знає. Це знає
документ: у рахунку багато надлишковості — контрольні розряди в кодах,
підсумки, що мають дорівнювати сумі рядків, ПДВ, що має дорівнювати
п'ятій частині. Правила тут не знають ні про модель, ні про веб-рівень.
"""

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"

# Допуск на округлення в арифметичних перевірках (копійки)
ARITHMETIC_TOLERANCE = Decimal("0.02")

# Ставка ПДВ
VAT_RATE = Decimal("0.20")


@dataclass
class Issue:
    """Проблема, знайдена правилом.

    `field` — шлях до поля, наприклад `supplier.iban` чи `items.1.amount`
    (другий рядок: індекс — з нуля); сторінка підсвічує поле з таким
    шляхом. `rule` — коротка назва правила. `message` — пояснення для
    людини, яка перевірятиме документ. `severity` — `error` або `warning`.
    """

    field: str
    rule: str
    message: str
    severity: str = "error"


def load_reference() -> tuple[dict, list[dict]]:
    """Прочитати реквізити своєї компанії й довідник постачальників."""
    company = json.loads((REFERENCE_DIR / "company.json").read_text(encoding="utf-8"))
    suppliers = json.loads((REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8"))
    return company, suppliers


# ---------------------------------------------------------------------------
# Допоміжні функції нормалізації
# ---------------------------------------------------------------------------

def _normalize_str(s: str | None) -> str:
    """Нормалізувати рядок: trim, нормалізація unicode, апострофи."""
    if s is None:
        return ""
    # Unicode-нормалізація NFC
    s = unicodedata.normalize("NFC", s)
    # Різні варіанти апострофа → стандартний '
    s = re.sub(r"[\u2018\u2019\u02BC\u0027\u0060]", "\u2019", s)
    return s.strip()


def _normalize_iban(iban: str | None) -> str:
    """Видалити пробіли, привести до верхнього регістру."""
    if iban is None:
        return ""
    return re.sub(r"\s+", "", iban).upper()


def _normalize_code(code: str | None) -> str:
    """Залишити лише цифри."""
    if code is None:
        return ""
    return re.sub(r"\D", "", code)


def _to_decimal(value) -> Decimal | None:
    """Перетворити рядок або число на Decimal. None якщо не вдалося."""
    if value is None:
        return None
    try:
        # Нормалізуємо: замінюємо кому на крапку, видаляємо пробіли
        s = str(value).replace(",", ".").replace(" ", "").replace("\xa0", "")
        return Decimal(s)
    except InvalidOperation:
        return None


# ---------------------------------------------------------------------------
# Перевірка IBAN (ISO 13616)
# ---------------------------------------------------------------------------

def _check_iban_format_and_checksum(iban: str) -> tuple[bool, str]:
    """Перевірити формат і контрольну суму IBAN.

    Повертає (ok, message). Для UA: 29 символів.
    """
    iban = _normalize_iban(iban)
    if not iban:
        return False, "IBAN відсутній"

    # Лише цифри та великі літери
    if not re.fullmatch(r"[A-Z]{2}[0-9]{2}[A-Z0-9]+", iban):
        return False, f"IBAN має некоректний формат: «{iban[:4]}…»"

    # Перевірка довжини для UA
    if iban.startswith("UA") and len(iban) != 29:
        return (
            False,
            f"Для України IBAN має містити 29 символів, знайдено {len(iban)}: «{iban}»"
        )

    # Алгоритм ISO 13616: перемістити 4 перших символи в кінець
    rearranged = iban[4:] + iban[:4]
    # Замінити літери на числа (A=10, B=11, ..., Z=35)
    numeric = ""
    for ch in rearranged:
        if ch.isdigit():
            numeric += ch
        elif ch.isalpha():
            numeric += str(ord(ch) - ord("A") + 10)
        else:
            return False, f"Недопустимий символ у IBAN: «{ch}»"

    remainder = int(numeric) % 97
    if remainder != 1:
        return (
            False,
            f"IBAN не пройшов перевірку контрольної суми (залишок {remainder}, очікується 1)"
        )

    return True, "OK"


# ---------------------------------------------------------------------------
# Перевірка ЄДРПОУ
# ---------------------------------------------------------------------------

def _check_edrpou(code: str) -> tuple[bool, str]:
    """Перевірити контрольний розряд ЄДРПОУ (8 цифр)."""
    digits = _normalize_code(code)
    if len(digits) != 8:
        return False, f"ЄДРПОУ має містити 8 цифр, знайдено {len(digits)}"

    d = [int(c) for c in digits]
    n = int(digits[:7])

    if n < 30000000 or n > 60000000:
        # Перший набір ваг
        weights = [1, 2, 3, 4, 5, 6, 7]
    else:
        # Другий набір ваг
        weights = [7, 1, 2, 3, 4, 5, 6]

    check = sum(d[i] * weights[i] for i in range(7)) % 11
    if check == 10:
        # Повторний розрахунок з іншими вагами
        if n < 30000000 or n > 60000000:
            weights2 = [3, 4, 5, 6, 7, 8, 9]
        else:
            weights2 = [9, 3, 4, 5, 6, 7, 8]
        check = sum(d[i] * weights2[i] for i in range(7)) % 11
        if check == 10:
            check = 0

    if check != d[7]:
        return (
            False,
            f"ЄДРПОУ: контрольний розряд не збігається "
            f"(очікується {check}, знайдено {d[7]})"
        )
    return True, "OK"


# ---------------------------------------------------------------------------
# Перевірка РНОКПП
# ---------------------------------------------------------------------------

def _check_rnokpp(code: str) -> tuple[bool, str]:
    """Перевірити контрольний розряд РНОКПП (10 цифр)."""
    digits = _normalize_code(code)
    if len(digits) != 10:
        return False, f"РНОКПП має містити 10 цифр, знайдено {len(digits)}"

    d = [int(c) for c in digits]
    weights = [-1, 5, 7, 9, 4, 6, 10, 5, 7]
    check = sum(d[i] * weights[i] for i in range(9)) % 11
    if check == 10:
        check = 0
    if check != d[9]:
        return (
            False,
            f"РНОКПП: контрольний розряд не збігається "
            f"(очікується {check}, знайдено {d[9]})"
        )
    return True, "OK"


# ---------------------------------------------------------------------------
# Головна функція перевірки
# ---------------------------------------------------------------------------

def check(document: dict) -> list[Issue]:
    """Застосувати правила до вилученого документа.

    Порожній перелік означає, що правила нічого не знайшли.
    """
    issues: list[Issue] = []
    company, suppliers = load_reference()

    # -----------------------------------------------------------------------
    # 8.8 Тип документа
    # -----------------------------------------------------------------------
    doc_type = (document.get("document_type") or "").lower().strip()
    if not doc_type:
        issues.append(Issue(
            field="document_type",
            rule="document_type",
            message="Тип документа не визначено. Переконайтеся, що завантажено рахунок на оплату.",
            severity="error"
        ))
    elif "накладна" in doc_type or "видаткова" in doc_type:
        issues.append(Issue(
            field="document_type",
            rule="document_type",
            message=f"Документ є «{doc_type}», а не рахунком на оплату. Автоматична реєстрація недопустима.",
            severity="error"
        ))
    elif "рахунок" not in doc_type and "invoice" not in doc_type:
        issues.append(Issue(
            field="document_type",
            rule="document_type",
            message=f"Невідомий тип документа: «{document.get('document_type')}». Перевірте тип вручну.",
            severity="warning"
        ))

    # -----------------------------------------------------------------------
    # 8.7 Обов'язкові поля
    # -----------------------------------------------------------------------
    required_fields = [
        ("number", "Номер рахунку"),
        ("date", "Дата рахунку"),
    ]
    for field_key, field_name in required_fields:
        if not document.get(field_key):
            issues.append(Issue(
                field=field_key,
                rule="missing_required",
                message=f"{field_name} відсутній або не вдалося прочитати.",
                severity="error"
            ))

    supplier = document.get("supplier") or {}
    buyer = document.get("buyer") or {}

    required_supplier = [
        ("name", "Назва постачальника"),
        ("code", "Код постачальника (ЄДРПОУ/РНОКПП)"),
        ("iban", "IBAN постачальника"),
    ]
    for field_key, field_name in required_supplier:
        if not supplier.get(field_key):
            issues.append(Issue(
                field=f"supplier.{field_key}",
                rule="missing_required",
                message=f"{field_name} відсутній або не вдалося прочитати.",
                severity="error"
            ))

    required_buyer = [
        ("name", "Назва покупця"),
        ("code", "Код покупця"),
    ]
    for field_key, field_name in required_buyer:
        if not buyer.get(field_key):
            issues.append(Issue(
                field=f"buyer.{field_key}",
                rule="missing_required",
                message=f"{field_name} відсутній або не вдалося прочитати.",
                severity="error"
            ))

    if not document.get("total"):
        issues.append(Issue(
            field="total",
            rule="missing_required",
            message="Загальна сума до оплати відсутня або не вдалося прочитати.",
            severity="error"
        ))

    # -----------------------------------------------------------------------
    # 8.1 Перевірка IBAN
    # -----------------------------------------------------------------------
    supplier_iban_raw = supplier.get("iban")
    if supplier_iban_raw:
        ok, msg = _check_iban_format_and_checksum(supplier_iban_raw)
        if not ok:
            issues.append(Issue(
                field="supplier.iban",
                rule="iban_checksum",
                message=f"IBAN постачальника не пройшов перевірку: {msg}.",
                severity="error"
            ))

    # -----------------------------------------------------------------------
    # 8.2 Перевірка кодів
    # -----------------------------------------------------------------------
    supplier_code_raw = supplier.get("code")
    if supplier_code_raw:
        digits = _normalize_code(supplier_code_raw)
        if len(digits) == 8:
            ok, msg = _check_edrpou(digits)
            if not ok:
                issues.append(Issue(
                    field="supplier.code",
                    rule="edrpou_checksum",
                    message=f"Код ЄДРПОУ постачальника: {msg}.",
                    severity="error"
                ))
        elif len(digits) == 10:
            ok, msg = _check_rnokpp(digits)
            if not ok:
                issues.append(Issue(
                    field="supplier.code",
                    rule="rnokpp_checksum",
                    message=f"РНОКПП постачальника: {msg}.",
                    severity="error"
                ))
        else:
            issues.append(Issue(
                field="supplier.code",
                rule="code_format",
                message=f"Код постачальника має {len(digits)} цифр, очікується 8 (ЄДРПОУ) або 10 (РНОКПП).",
                severity="error"
            ))

    buyer_code_raw = buyer.get("code")
    if buyer_code_raw:
        digits = _normalize_code(buyer_code_raw)
        if len(digits) == 8:
            ok, msg = _check_edrpou(digits)
            if not ok:
                issues.append(Issue(
                    field="buyer.code",
                    rule="edrpou_checksum",
                    message=f"Код ЄДРПОУ покупця: {msg}.",
                    severity="error"
                ))
        elif len(digits) == 10:
            ok, msg = _check_rnokpp(digits)
            if not ok:
                issues.append(Issue(
                    field="buyer.code",
                    rule="rnokpp_checksum",
                    message=f"РНОКПП покупця: {msg}.",
                    severity="error"
                ))
        # Якщо 0 цифр — вже підхоплено missing_required

    # -----------------------------------------------------------------------
    # 8.4 Перевірка дат
    # -----------------------------------------------------------------------
    invoice_date = None
    date_str = document.get("date")
    if date_str:
        try:
            invoice_date = date.fromisoformat(str(date_str))
            today = date.today()
            if invoice_date > today:
                issues.append(Issue(
                    field="date",
                    rule="date_future",
                    message=f"Дата рахунку ({date_str}) у майбутньому. Перевірте правильність дати.",
                    severity="error"
                ))
        except ValueError:
            issues.append(Issue(
                field="date",
                rule="date_format",
                message=f"Не вдалося розпізнати дату рахунку: «{date_str}». Очікується формат РРРР-ММ-ДД.",
                severity="error"
            ))

    valid_until_str = document.get("valid_until")
    if valid_until_str:
        try:
            valid_until = date.fromisoformat(str(valid_until_str))
            if invoice_date and valid_until < invoice_date:
                issues.append(Issue(
                    field="valid_until",
                    rule="date_valid_until",
                    message=f"Строк дії ({valid_until_str}) раніше дати рахунку ({date_str}).",
                    severity="error"
                ))
        except ValueError:
            issues.append(Issue(
                field="valid_until",
                rule="date_format",
                message=f"Не вдалося розпізнати строк дії: «{valid_until_str}».",
                severity="warning"
            ))

    # -----------------------------------------------------------------------
    # 8.3 Арифметичні перевірки
    # -----------------------------------------------------------------------
    items = document.get("items") or []

    line_sum = Decimal("0")
    for idx, item in enumerate(items):
        qty = _to_decimal(item.get("quantity"))
        price = _to_decimal(item.get("price"))
        amount = _to_decimal(item.get("amount"))

        if qty is not None and price is not None and amount is not None:
            expected_amount = (qty * price).quantize(Decimal("0.01"))
            diff = abs(expected_amount - amount)
            if diff > ARITHMETIC_TOLERANCE:
                issues.append(Issue(
                    field=f"items.{idx}.amount",
                    rule="arithmetic_line",
                    message=(
                        f"Рядок {idx + 1}: {qty} × {price} = {expected_amount}, "
                        f"але вказано {amount}. Різниця: {diff}."
                    ),
                    severity="error"
                ))

        if amount is not None:
            line_sum += amount

    total_without_vat = _to_decimal(document.get("total_without_vat"))
    vat = _to_decimal(document.get("vat"))
    total = _to_decimal(document.get("total"))

    # Перевірка: сума рядків = разом без ПДВ
    if items and line_sum > 0 and total_without_vat is not None:
        diff = abs(line_sum - total_without_vat)
        if diff > ARITHMETIC_TOLERANCE:
            issues.append(Issue(
                field="total_without_vat",
                rule="arithmetic_subtotal",
                message=(
                    f"Сума позицій ({line_sum}) не збігається з «Разом без ПДВ» ({total_without_vat}). "
                    f"Різниця: {diff}."
                ),
                severity="error"
            ))

    # Перевірка: разом без ПДВ + ПДВ = до сплати
    if total_without_vat is not None and vat is not None and total is not None:
        expected_total = total_without_vat + vat
        diff = abs(expected_total - total)
        if diff > ARITHMETIC_TOLERANCE:
            issues.append(Issue(
                field="total",
                rule="arithmetic_total",
                message=(
                    f"Разом без ПДВ ({total_without_vat}) + ПДВ ({vat}) = {expected_total}, "
                    f"але вказано {total}. Різниця: {diff}."
                ),
                severity="error"
            ))

    # Перевірка ПДВ: 20% від суми без ПДВ (для платників ПДВ)
    # Визначаємо, чи є постачальник платником ПДВ, за наявністю ПДВ у рахунку
    if total_without_vat is not None and vat is not None:
        if vat > Decimal("0"):
            expected_vat = (total_without_vat * VAT_RATE).quantize(Decimal("0.01"))
            diff = abs(expected_vat - vat)
            if diff > ARITHMETIC_TOLERANCE:
                issues.append(Issue(
                    field="vat",
                    rule="arithmetic_vat",
                    message=(
                        f"ПДВ ({vat}) не дорівнює 20% від суми без ПДВ "
                        f"({total_without_vat} × 20% = {expected_vat}). Різниця: {diff}."
                    ),
                    severity="warning"
                ))

    # -----------------------------------------------------------------------
    # 8.5 Перевірка покупця
    # -----------------------------------------------------------------------
    buyer_name = _normalize_str(buyer.get("name"))
    buyer_code = _normalize_code(buyer.get("code") or "")
    company_name = _normalize_str(company.get("name"))
    company_code = _normalize_code(company.get("code") or "")

    def _names_match(a: str, b: str) -> bool:
        """Нечітке порівняння назв компаній."""
        a = a.lower()
        b = b.lower()
        return a == b or a in b or b in a

    buyer_name_ok = buyer_name and _names_match(buyer_name, company_name)
    buyer_code_ok = buyer_code and buyer_code == company_code

    if buyer.get("name") and not buyer_name_ok:
        issues.append(Issue(
            field="buyer.name",
            rule="buyer",
            message=(
                f"Покупець у рахунку «{buyer.get('name')}» не відповідає компанії «{company.get('name')}»."
            ),
            severity="error"
        ))

    if buyer.get("code") and not buyer_code_ok:
        issues.append(Issue(
            field="buyer.code",
            rule="buyer",
            message=(
                f"Код покупця «{buyer.get('code')}» не відповідає коду компанії «{company.get('code')}»."
            ),
            severity="error"
        ))

    # -----------------------------------------------------------------------
    # 8.6 Перевірка постачальника за довідником
    # -----------------------------------------------------------------------
    supplier_name = _normalize_str(supplier.get("name"))
    supplier_code = _normalize_code(supplier.get("code") or "")
    supplier_iban = _normalize_iban(supplier.get("iban") or "")

    matched_supplier = None
    for s in suppliers:
        s_code = _normalize_code(s.get("code") or "")
        s_name = _normalize_str(s.get("name"))
        if supplier_code and s_code == supplier_code:
            matched_supplier = s
            break
        if supplier_name and _names_match(supplier_name, s_name):
            matched_supplier = s
            break

    if not matched_supplier:
        if supplier_name or supplier_code:
            issues.append(Issue(
                field="supplier.name",
                rule="supplier_registry",
                message=(
                    f"Постачальника «{supplier.get('name')}» "
                    f"(код {supplier.get('code')}) не знайдено в довіднику."
                ),
                severity="error"
            ))
    else:
        # Перевірка відповідності IBAN довіднику
        registered_iban = _normalize_iban(matched_supplier.get("iban") or "")
        if supplier_iban and registered_iban and supplier_iban != registered_iban:
            issues.append(Issue(
                field="supplier.iban",
                rule="iban_registry",
                message=(
                    f"IBAN у рахунку відрізняється від зареєстрованого IBAN постачальника. "
                    f"Навіть якщо контрольна сума вірна, такий рахунок не може бути оплачений автоматично."
                ),
                severity="error"
            ))

        # Перевірка коду (якщо знайдено за назвою)
        s_code = _normalize_code(matched_supplier.get("code") or "")
        if supplier_code and s_code and supplier_code != s_code:
            issues.append(Issue(
                field="supplier.code",
                rule="supplier_registry",
                message=(
                    f"Код постачальника «{supplier.get('code')}» не збігається "
                    f"з довідником (очікується «{matched_supplier.get('code')}»)."
                ),
                severity="error"
            ))

    return issues
