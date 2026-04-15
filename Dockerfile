FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY run.py config.yaml ./
COPY src ./src

RUN python -c "import sqlite3; assert sqlite3.sqlite_version; import torch; import catboost; print('imports OK:', sqlite3.sqlite_version)" \
    && python run.py --help

CMD ["python", "run.py", "--help"]
