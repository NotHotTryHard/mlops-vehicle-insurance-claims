FROM ghcr.io/astral-sh/uv:python3.10-bookworm-slim

WORKDIR /app

ENV PYTHONPATH=/app \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && apt-mark manual libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

COPY run.py config.yaml ./
COPY src ./src

RUN uv sync --locked --no-dev --no-editable \
    && python -c "import sqlite3; assert sqlite3.sqlite_version; import catboost; import sklearn; print('imports OK:', sqlite3.sqlite_version)" \
    && python run.py --help

CMD ["python", "run.py", "--help"]
