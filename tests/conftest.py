import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session

from app.models import Base

ROOT = Path(__file__).resolve().parent.parent
DB_ENV_VARS = (
    "POSTGRES_CONNECTION_STRING",
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DATABASE_NAME",
    "POSTGRES_USERNAME",
    "POSTGRES_PASSWORD",
)


@pytest.fixture
def clean_db_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in DB_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.fixture
def alembic_config() -> Config:
    return Config(str(ROOT / "alembic.ini"))


@pytest.fixture
def session() -> Iterator[Session]:
    """In-memory SQLite session. Good for logic; not proof of PostgreSQL behaviour."""
    engine = create_engine("sqlite://")

    @event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    engine.dispose()


@pytest.fixture
def postgres_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL", "")
    if not url:
        pytest.skip("TEST_DATABASE_URL is not set")
    return url


def make_event(event_id: str = "e1", **overrides) -> dict:
    data = {
        "id": event_id,
        "name": "Concert",
        "place_id": "p1",
        "place_name": "Hall",
        "place_city": "Moscow",
        "place_address": "Main st. 1",
        "place_seats_pattern": "A1-10",
        "event_time": datetime(2026, 1, 11, 14, 0, tzinfo=UTC),
        "registration_deadline": datetime(2026, 1, 10, 14, 0, tzinfo=UTC),
        "status": "published",
        "number_of_visitors": 0,
        "changed_at": datetime(2026, 1, 1, tzinfo=UTC),
        "created_at": datetime(2025, 12, 1, tzinfo=UTC),
    }
    data.update(overrides)
    return data


def make_ticket(ticket_id: str = "t1", event_id: str = "e1", **overrides) -> dict:
    data = {
        "ticket_id": ticket_id,
        "event_id": event_id,
        "first_name": "Ivan",
        "last_name": "Ivanov",
        "email": "ivan@example.com",
        "seat": "A15",
    }
    data.update(overrides)
    return data
