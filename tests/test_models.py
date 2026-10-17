from datetime import UTC, datetime

import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import IntegrityError
from sqlalchemy.schema import CreateIndex

from app.models import Event, SyncMetadata, Ticket
from tests.conftest import make_event, make_ticket


def test_tables_and_columns():
    assert Event.__table__.c.id.primary_key
    assert str(Event.__table__.c.id.type) == "TEXT"
    ticket = Ticket.__table__
    assert ticket.c.id.primary_key
    assert not ticket.c.ticket_id.primary_key
    assert ticket.c.cancelled_at.nullable
    assert not ticket.c.event_id.nullable
    assert {fk.target_fullname for fk in ticket.foreign_keys} == {"events.id"}
    assert SyncMetadata.__table__.c.last_changed_at.nullable


def test_datetimes_are_timezone_aware():
    for column in ("event_time", "registration_deadline", "changed_at", "created_at"):
        assert Event.__table__.c[column].type.timezone
    for column in ("created_at", "cancelled_at"):
        assert Ticket.__table__.c[column].type.timezone


def test_event_indexes_exist():
    names = {i.name for i in Event.__table__.indexes}
    assert {"ix_events_event_time", "ix_events_changed_at"} <= names


def test_partial_unique_index_on_active_tickets_in_postgres_ddl():
    index = next(
        i for i in Ticket.__table__.indexes if i.name == "uq_tickets_ticket_id_active"
    )
    ddl = str(CreateIndex(index).compile(dialect=postgresql.dialect()))

    assert "CREATE UNIQUE INDEX uq_tickets_ticket_id_active" in ddl
    assert "(ticket_id)" in ddl
    assert "WHERE cancelled_at IS NULL" in ddl


def test_event_requires_name(session):
    session.add(Event(**make_event(name=None)))
    with pytest.raises(IntegrityError):
        session.flush()


def test_nullable_registration_deadline(session):
    session.add(Event(**make_event(registration_deadline=None)))
    session.flush()


def test_ticket_requires_existing_event(session):
    session.add(Ticket(**make_ticket(event_id="missing")))
    with pytest.raises(IntegrityError):
        session.flush()


def test_active_ticket_id_must_be_unique(session):
    session.add(Event(**make_event()))
    session.add(Ticket(**make_ticket("t1")))
    session.flush()
    session.add(Ticket(**make_ticket("t1", seat="A16")))
    with pytest.raises(IntegrityError):
        session.flush()


def test_ticket_id_can_be_reused_after_cancellation(session):
    session.add(Event(**make_event()))
    session.add(Ticket(**make_ticket("t1", cancelled_at=datetime.now(UTC))))
    session.add(Ticket(**make_ticket("t1")))
    session.flush()
