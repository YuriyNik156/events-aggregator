from datetime import UTC, datetime

from sqlalchemy import func, select

from app.models import Event
from app.repositories import (
    EventRepository,
    SyncMetadataRepository,
    TicketRepository,
)
from tests.conftest import make_event, make_ticket


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


def test_event_get_missing_returns_none(session):
    assert EventRepository(session).get("nope") is None


def test_event_upsert_inserts_then_updates_without_duplicates(session):
    repo = EventRepository(session)
    repo.upsert(make_event("e1", name="Old"))
    repo.upsert(make_event("e1", name="New", status="finished"))

    assert session.scalar(select(func.count()).select_from(Event)) == 1
    event = repo.get("e1")
    assert (event.name, event.status) == ("New", "finished")


def test_event_list_order_pagination_and_count(session):
    repo = EventRepository(session)
    repo.upsert(make_event("b", event_time=utc(2026, 1, 2)))
    repo.upsert(make_event("a", event_time=utc(2026, 1, 2)))
    repo.upsert(make_event("c", event_time=utc(2026, 1, 1)))

    page1, total = repo.list(page=1, page_size=2)
    page2, _ = repo.list(page=2, page_size=2)
    page3, _ = repo.list(page=3, page_size=2)

    assert total == 3
    assert [e.id for e in page1] == ["c", "a"]  # event_time, then id
    assert [e.id for e in page2] == ["b"]
    assert page3 == []


def test_event_list_date_from_is_inclusive(session):
    repo = EventRepository(session)
    repo.upsert(make_event("old", event_time=utc(2026, 1, 1)))
    repo.upsert(make_event("edge", event_time=utc(2026, 1, 2)))
    repo.upsert(make_event("new", event_time=utc(2026, 1, 3)))

    events, total = repo.list(date_from=utc(2026, 1, 2))

    assert total == 2
    assert [e.id for e in events] == ["edge", "new"]


def test_event_list_empty_database(session):
    assert EventRepository(session).list() == ([], 0)


def test_ticket_add_and_find_active(session):
    EventRepository(session).upsert(make_event())
    repo = TicketRepository(session)
    added = repo.add(**make_ticket("t1"))

    assert added.id is not None
    assert added.cancelled_at is None
    assert repo.get_active("t1") is added
    assert repo.get_active("other") is None


def test_cancel_hides_ticket_and_allows_reuse_of_external_id(session):
    EventRepository(session).upsert(make_event())
    repo = TicketRepository(session)
    first = repo.add(**make_ticket("t1"))

    repo.cancel(first, utc(2026, 1, 5))
    assert repo.get_active("t1") is None
    assert first.cancelled_at is not None

    second = repo.add(**make_ticket("t1", seat="B1"))
    assert repo.get_active("t1") is second
    assert second.id != first.id


def test_sync_metadata_starts_empty(session):
    assert SyncMetadataRepository(session).get() is None


def test_sync_metadata_cursor_is_not_changed_unless_given(session):
    repo = SyncMetadataRepository(session)
    repo.save(
        sync_status="success",
        last_sync_time=utc(2026, 1, 1),
        last_changed_at=utc(2026, 1, 1),
    )
    cursor = repo.get().last_changed_at

    repo.save(sync_status="error")

    meta = repo.get()
    assert meta.sync_status == "error"
    assert meta.last_changed_at == cursor
    assert session.scalar(select(func.count()).select_from(type(meta))) == 1
