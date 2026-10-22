"""The async database setup: URL, engine factory, session fixtures."""

import pytest
from sqlalchemy import select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from app import database
from app.config import get_async_database_url, get_database_url
from app.models import Event, Ticket
from tests.conftest import make_event, make_ticket

PG_URL = "postgresql://user:p%40ss@db.example:5433/appdb?sslmode=require"


def test_async_url_differs_from_sync_url_only_by_driver(clean_db_env):
    clean_db_env.setenv("POSTGRES_CONNECTION_STRING", PG_URL)

    sync_url, async_url = get_database_url(), get_async_database_url()

    assert sync_url.drivername == "postgresql+psycopg"  # Alembic keeps this one
    assert async_url.drivername == "postgresql+psycopg_async"
    assert async_url.set(drivername=sync_url.drivername) == sync_url
    assert async_url.password == "p@ss"
    assert async_url.query == {"sslmode": "require"}


def test_async_url_from_separate_variables(clean_db_env):
    for name, value in {
        "POSTGRES_HOST": "db",
        "POSTGRES_PORT": "5432",
        "POSTGRES_DATABASE_NAME": "app",
        "POSTGRES_USERNAME": "u",
        "POSTGRES_PASSWORD": "secret",
    }.items():
        clean_db_env.setenv(name, value)

    url = get_async_database_url()

    assert url.drivername == "postgresql+psycopg_async"
    assert (url.host, url.port, url.database) == ("db", 5432, "app")
    assert "secret" not in repr(url)


def test_async_url_without_configuration_fails_clearly(clean_db_env):
    from app.config import ConfigError

    with pytest.raises(ConfigError, match="POSTGRES_CONNECTION_STRING"):
        get_async_database_url()


def test_psycopg_async_dialect_loads_without_connecting():
    engine = create_async_engine(
        make_url(PG_URL).set(drivername="postgresql+psycopg_async")
    )

    assert isinstance(engine, AsyncEngine)
    assert engine.url.drivername == "postgresql+psycopg_async"
    assert engine.dialect.is_async


@pytest.mark.anyio
async def test_async_engine_is_lazy_shared_and_disposable(clean_db_env):
    clean_db_env.setenv("POSTGRES_CONNECTION_STRING", PG_URL)

    engine = database.get_async_engine()  # no connection is opened here
    assert database.get_async_engine() is engine
    factory = database.get_async_session_factory()
    assert database.get_async_session_factory() is factory
    assert factory.kw["expire_on_commit"] is False

    await database.dispose_async_engine()

    assert database._async_engine is None
    assert database._async_session_factory is None
    await database.dispose_async_engine()  # a second call is harmless


@pytest.mark.anyio
async def test_async_session_fixture_roundtrip(async_session):
    async_session.add(Event(**make_event()))
    await async_session.commit()

    event = (await async_session.scalars(select(Event))).one()

    assert event.id == "e1"
    assert event.name == "Concert"  # not expired by commit


@pytest.mark.anyio
async def test_async_session_enforces_foreign_keys(async_session):
    async_session.add(Ticket(**make_ticket(event_id="missing")))

    with pytest.raises(IntegrityError):
        await async_session.flush()


@pytest.mark.postgres
@pytest.mark.anyio
async def test_psycopg_async_works_against_postgres(postgres_url):
    url = make_url(postgres_url).set(drivername="postgresql+psycopg_async")
    engine = create_async_engine(url)
    try:
        async with engine.connect() as connection:
            one = await connection.scalar(text("SELECT 1"))
            now = await connection.scalar(text("SELECT now()"))
    finally:
        await engine.dispose()

    assert one == 1
    assert now.tzinfo is not None
