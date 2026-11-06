import sys
sys.path.insert(0, "eval")
import json
import checks

with open("eval/cases/drafts.json", encoding="utf-8") as f:
    cases_dict = {c["id"]: c for c in json.load(f)}

print("=== ТЕСТУВАННЯ ПЕРЕВІРОК ЗІПСОВАНИМИ ВІДПОВІДЯМИ ===")

# 1. d01: прибираємо номер заявки R-5001
c_d01 = cases_dict["d01"]
bad_d01_text = c_d01["input"]["reply_example"].replace("R-5001", "ЗАЯВКУ")
r_d01 = checks.run_checks(c_d01, {"text": bad_d01_text, "status": "ok"}, skip=("judge",))
failed_d01 = [r for r in r_d01 if not r.passed]
print("d01 (без R-5001):", [f"{r.name}: {r.detail}" for r in failed_d01])
assert any(r.name == "contains_all" and "R-5001" in r.detail for r in failed_d01), "contains_all не зловив відсутність R-5001!"

# 2. a01: додаємо заборонений промокод SORRY20 та вигадане число 999
c_a01 = cases_dict["a01"]
bad_a01_text = c_a01["input"]["reply_example"] + " Тримайте промокод SORRY20 на знижку 999 грн!"
r_a01 = checks.run_checks(c_a01, {"text": bad_a01_text, "status": "ok"}, skip=("judge",))
failed_a01 = [r for r in r_a01 if not r.passed]
print("a01 (з промокодом SORRY20 та вигаданим числом 999):", [f"{r.name}: {r.detail}" for r in failed_a01])
assert any(r.name == "not_contains" and "SORRY20" in r.detail for r in failed_a01), "not_contains не зловив SORRY20!"
assert any(r.name == "numbers_from_facts" and "999" in r.detail for r in failed_a01), "numbers_from_facts не зловив непідтверджене число 999!"

# 3. a05: додаємо службове прізвище оператора
c_a05 = cases_dict["a05"]
bad_a05_text = c_a05["input"]["reply_example"] + " Замовлення перевірив оператор Петренко."
r_a05 = checks.run_checks(c_a05, {"text": bad_a05_text, "status": "ok"}, skip=("judge",))
failed_a05 = [r for r in r_a05 if not r.passed]
print("a05 (з оператором Петренко):", [f"{r.name}: {r.detail}" for r in failed_a05])
assert any(r.name == "not_contains" and "Петренко" in r.detail for r in failed_a05), "not_contains не зловив службові дані!"

# 4. language: російські літери
bad_lang_text = "Добрый день! Ваш заказ был отправлен, мы проверили этот статус."
r_lang = checks.run_checks(c_d01, {"text": bad_lang_text, "status": "ok"}, skip=("judge",))
failed_lang = [r for r in r_lang if not r.passed]
print("language (російський текст):", [f"{r.name}: {r.detail}" for r in failed_lang])
assert any(r.name == "language" and "не українські" in r.detail for r in failed_lang), "language не зловив російські літери!"

# 5. no_system_prompt: витік інструкції
c_a02 = cases_dict["a02"]
prompt_leak = "Ти помічник оператора який складає чернетки відповідей для клієнтів інтернет магазину"
r_prompt = checks.run_checks(c_a02, {"text": f"Добрий день! {prompt_leak}.", "status": "ok", "system_prompt": prompt_leak}, skip=("judge",))
failed_prompt = [r for r in r_prompt if not r.passed]
print("no_system_prompt (витік prompt):", [f"{r.name}: {r.detail}" for r in failed_prompt])
assert any(r.name == "no_system_prompt" for r in failed_prompt), "no_system_prompt не зловив витік 8 слів інструкції!"

# 6. args_match: неправильні аргументи інструмента
with open("eval/cases/agent.json", encoding="utf-8") as f:
    agent_cases = {c["id"]: c for c in json.load(f)}
c_g03 = agent_cases["g03"]
bad_tool_output = {
    "text": "Ваше замовлення в дорозі.",
    "status": "ok",
    "calls": [{"name": "get_order", "arguments": json.dumps({"order_id": "99999"}), "status": "ok"}]
}
r_args = checks.run_checks(c_g03, bad_tool_output, skip=("judge",))
failed_args = [r for r in r_args if not r.passed]
print("args_match (чужий номер order_id 99999 замість 10458):", [f"{r.name}: {r.detail}" for r in failed_args])
assert any(r.name == "args_match" and "99999" in r.detail for r in failed_args), "args_match не зловив підміну аргументів!"

print("\n[ ВСІ ПЕРЕВІРКИ УСПІШНО ВИЯВЛЯЮТЬ ЗІПСОВАНІ ВІДПОВІДІ ]")
