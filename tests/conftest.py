import asyncio
import os
import sys
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from alembic.config import Config
from sqlalchemy import create_engine, event, select
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import Session
from sqlalchemy.pool import NullPool

import main  # noqa: F401  (imported so the app is loaded)
from app import events_provider
from app.models import Base, Event, Ticket

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


@pytest.fixture(autouse=True)
def background_sync_off(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never start the background worker (and so never reach a provider)."""
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "0")


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
def anyio_backend():
    """Async tests (``@pytest.mark.anyio``) run on asyncio only; trio is not used."""
    if sys.platform == "win32":
        # psycopg's async mode cannot use the default Proactor loop of Windows.
        return "asyncio", {"loop_factory": asyncio.SelectorEventLoop}
    return "asyncio"


@pytest.fixture
async def async_session() -> AsyncIterator[AsyncSession]:
    """In-memory SQLite AsyncSession (aiosqlite): logic only, not PostgreSQL."""
    engine = create_async_engine("sqlite+aiosqlite://")

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    async with AsyncSession(engine, expire_on_commit=False) as s:
        yield s
    await engine.dispose()


@pytest.fixture
async def async_engine(tmp_path) -> AsyncIterator[AsyncEngine]:
    """SQLite file behind a real connection pool, to watch connections come and go.

    Several sessions can use it at once, as in the application; the in-memory
    ``async_session`` has a single shared connection and cannot show that.
    """
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'test.db'}")

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
def session_factory(async_engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(async_engine, expire_on_commit=False)


@pytest.fixture
async def async_db(session_factory) -> AsyncIterator[AsyncSession]:
    """The session a service under test is given (own session per request)."""
    async with session_factory() as s:
        yield s


@pytest.fixture(autouse=True)
def provider_client_is_not_shared_between_tests(monkeypatch: pytest.MonkeyPatch):
    """``main`` keeps one provider client per process; every test starts without it."""
    monkeypatch.setattr(events_provider, "_shared_client", None)


@pytest.fixture
def db(tmp_path) -> Iterator[Session]:
    """A sync Session on a SQLite file, for seeding and checking in route tests.

    The application reaches the same file through ``session_override`` (aiosqlite).
    Every ORM query here refreshes what it loads, so rows the application changed
    meanwhile are never read from this session's cache.
    """
    path = tmp_path / "routes.db"
    engine = create_engine(
        f"sqlite:///{path}", connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        s.info["path"] = path

        @event.listens_for(s, "do_orm_execute")
        def _fresh(state):
            if state.is_select:
                state.update_execution_options(populate_existing=True)

        yield s
    engine.dispose()


def session_override(db: Session):
    """A replacement for ``main.get_session``: an AsyncSession on the file ``db`` uses.

    No pool: TestClient serves each request in a loop of its own, and pooled
    connections must not travel between loops.
    """
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db.info['path']}", poolclass=NullPool
    )

    @event.listens_for(engine.sync_engine, "connect")
    def _enable_fk(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

    async def get_session():
        async with AsyncSession(engine, expire_on_commit=False) as session:
            yield session

    return get_session


LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1")


def checked_postgres_url() -> str:
    """``TEST_DATABASE_URL``, or a skip / failure; never a database that is not ours.

    The PostgreSQL tests drop and truncate tables, so the URL must name a local
    database whose name ends in ``_test`` and which is not the one the application
    itself is configured with. ``REQUIRE_POSTGRES_TESTS`` (set in CI) turns a missing
    URL into a failure, so these tests cannot be skipped there unnoticed.
    """
    url = os.environ.get("TEST_DATABASE_URL", "")
    if not url:
        if os.environ.get("REQUIRE_POSTGRES_TESTS"):
            pytest.fail("TEST_DATABASE_URL is required here but not set")
        pytest.skip("TEST_DATABASE_URL is not set")
    target = make_url(url)
    if not (target.database or "").endswith("_test"):
        pytest.fail("TEST_DATABASE_URL: the database name must end with '_test'")
    if target.host not in LOCAL_HOSTS:
        pytest.fail("TEST_DATABASE_URL: the database must be on this machine")
    app_url = os.environ.get("POSTGRES_CONNECTION_STRING", "")
    if app_url:
        app = make_url(app_url)
        if (app.host, app.port or 5432, app.database) == (
            target.host,
            target.port or 5432,
            target.database,
        ):
            pytest.fail("TEST_DATABASE_URL is the application database")
    return url


@pytest.fixture
def postgres_url() -> str:
    return checked_postgres_url()


def store_event(db: Session, data: dict) -> None:
    """Insert or update an event the way a sync does (test side, sync session)."""
    db.merge(Event(**data))
    db.flush()


def store_ticket(db: Session, **fields) -> Ticket:
    ticket = Ticket(**fields)
    db.add(ticket)
    db.flush()
    return ticket


def active_ticket(db: Session, ticket_id: str) -> Ticket | None:
    return db.scalars(
        select(Ticket).where(
            Ticket.ticket_id == ticket_id, Ticket.cancelled_at.is_(None)
        )
    ).first()


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


INITIAL = date(2000, 1, 1)


def provider_event_json(event_id: str = "e1", **overrides) -> dict:
    """An event in the Events Provider response format."""
    data = {
        "id": event_id,
        "name": "Concert",
        "place": {
            "id": "p1",
            "name": "Hall",
            "city": "Moscow",
            "address": "Main st. 1",
            "seats_pattern": "A1-10",
        },
        "event_time": "2026-01-11T17:00:00+03:00",
        "registration_deadline": "2026-01-10T17:00:00+03:00",
        "status": "published",
        "number_of_visitors": 5,
        "changed_at": "2026-01-05T07:00:00+00:00",
        "created_at": "2025-12-01T10:00:00+03:00",
    }
    data.update(overrides)
    return data


def provider_page(results: list[dict], next_url: str | None = None) -> dict:
    return {"next": next_url, "previous": None, "results": results}
