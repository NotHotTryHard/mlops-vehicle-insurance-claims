FROM python:3.12-slim-bookworm

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

COPY requirements.txt .
# phik (ydata-profiling) may build from source in slim images; needs a C++ compiler.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && pip install --upgrade pip && pip install -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY run.py config.yaml ./
COPY src ./src

RUN python -c "import sqlite3; assert sqlite3.sqlite_version; import torch; import catboost; print('imports OK:', sqlite3.sqlite_version)" \
    && python run.py --help

CMD ["python", "run.py", "--help"]
