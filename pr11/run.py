"""Скрипт запуску застосунку ПР11.
Використання:
    python run.py
"""

import os
import sys
import uvicorn

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "127.0.0.1")
    print(f"Запуск застосунку на http://{host}:{port} ...")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
