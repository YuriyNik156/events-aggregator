FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /usr/local/bin/uv

RUN addgroup --system --gid 1000 appuser && \
    adduser --system --uid 1000 --ingroup appuser appuser

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev

COPY --chown=appuser:appuser main.py alembic.ini ./
COPY --chown=appuser:appuser app ./app
COPY --chown=appuser:appuser alembic ./alembic

USER appuser

# Migrate first: a failed migration exits non-zero, so the app never serves
# requests without its tables. exec keeps uvicorn as PID 1 for clean shutdown.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn main:app --host 0.0.0.0 --port 8000"]
