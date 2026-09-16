import sys
from pathlib import Path

# Додаємо поточну папку до шляху, щоб імпортувати app
sys.path.append(str(Path(__file__).parent))

from app import llm

def test_ask(question: str):
    print(f"Питання: {question}")
    context = Path("context.md").read_text(encoding="utf-8")
    try:
        result = llm.ask(question, context)
        print(f"Відповідь: {result['answer']}")
        print(f"Модель: {result['model']}")
        print(f"Час: {result['elapsed']:.2f} с")
    except Exception as e:
        print(f"Помилка: {e}")
    print("-" * 20)

if __name__ == "__main__":
    test_ask("Чи можу я повернути товар через 10 днів?")
    test_ask("Яка вартість доставки замовлення на 3000 грн?")
    test_ask("Який ваш графік роботи?")
    test_ask("Чи продаєте ви автомобілі?")  # Питання поза контекстом
