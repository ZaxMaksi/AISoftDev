"""Побудова індексу з колекції документів.

Запуск із папки pr5:
    python ingest.py
або
    python scripts/build_index.py
"""

import sys
from scripts.build_index import main

if __name__ == "__main__":
    sys.exit(main())
