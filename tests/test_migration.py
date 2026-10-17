"""Migration checks.

Offline tests render SQL and need no database. Tests marked ``postgres`` run
against a *disposable* PostgreSQL given by ``TEST_DATABASE_URL`` and are
skipped otherwise. They run ``downgrade``, so the database name must contain
"test"; never point them at a real database.
"""

from io import StringIO

import pytest
from alembic import command
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

from app.models import Base


def test_offline_upgrade_renders_postgres_ddl(alembic_config, clean_db_env):
    # Offline mode must not require any database configuration.
    alembic_config.output_buffer = buffer = StringIO()

    command.upgrade(alembic_config, "head", sql=True)

    sql = buffer.getvalue()
    for table in ("events", "tickets", "sync_metadata"):
        assert f"CREATE TABLE {table}" in sql
    assert "TIMESTAMP WITH TIME ZONE" in sql
    assert (
        "CREATE UNIQUE INDEX uq_tickets_ticket_id_active ON tickets (ticket_id) "
        "WHERE cancelled_at IS NULL" in sql
    )
    assert "FOREIGN KEY(event_id) REFERENCES events (id)" in sql
    assert "CREATE INDEX ix_events_event_time ON events (event_time)" in sql
    assert "CREATE INDEX ix_events_changed_at ON events (changed_at)" in sql


def test_online_migration_without_configuration_fails_clearly(
    alembic_config, clean_db_env
):
    from app.config import ConfigError

    with pytest.raises(ConfigError, match="POSTGRES_CONNECTION_STRING"):
        command.upgrade(alembic_config, "head")


@pytest.fixture
def pg_engine(postgres_url, alembic_config, monkeypatch):
    name = make_url(postgres_url).database or ""
    if "test" not in name:
        pytest.fail("TEST_DATABASE_URL database name must contain 'test'")
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", postgres_url)
    engine = create_engine(make_url(postgres_url).set(drivername="postgresql+psycopg"))
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS tickets, events, sync_metadata"))
        conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
    yield engine
    command.downgrade(alembic_config, "base")
    engine.dispose()


@pytest.mark.postgres
def test_upgrade_matches_models_and_downgrades(pg_engine, alembic_config):
    command.upgrade(alembic_config, "head")

    assert {"events", "tickets", "sync_metadata"} <= set(
        inspect(pg_engine).get_table_names()
    )
    with pg_engine.connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == []

    command.downgrade(alembic_config, "base")
    assert "events" not in inspect(pg_engine).get_table_names()
    command.upgrade(alembic_config, "head")  # re-appliable


@pytest.mark.postgres
def test_partial_unique_index_and_fk_in_postgres(pg_engine, alembic_config):
    command.upgrade(alembic_config, "head")
    insert_event = text(
        "INSERT INTO events (id, name, place_id, place_name, place_city,"
        " place_address, place_seats_pattern, event_time, status, changed_at,"
        " created_at) VALUES ('e1','n','p','pn','c','a','A1-1', now(), 'published',"
        " now(), now())"
    )
    insert_ticket = text(
        "INSERT INTO tickets (ticket_id, event_id, first_name, last_name, email,"
        " seat, cancelled_at) VALUES (:t, :e, 'a', 'b', 'c@d.e', 's', :c)"
    )
    with pg_engine.begin() as conn:
        conn.execute(insert_event)
        conn.execute(insert_ticket, {"t": "t1", "e": "e1", "c": None})

    with pytest.raises(IntegrityError), pg_engine.begin() as conn:
        conn.execute(insert_ticket, {"t": "t1", "e": "e1", "c": None})  # duplicate

    with pytest.raises(IntegrityError), pg_engine.begin() as conn:
        conn.execute(insert_ticket, {"t": "t2", "e": "missing", "c": None})  # FK

    with pg_engine.begin() as conn:
        conn.execute(text("UPDATE tickets SET cancelled_at = now()"))
        conn.execute(insert_ticket, {"t": "t1", "e": "e1", "c": None})  # reuse OK
