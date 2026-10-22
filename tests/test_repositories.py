"""Async repositories on an in-memory SQLite ``AsyncSession`` (aiosqlite).

SQLite is enough for the logic. The partial unique index, timestamptz handling and
real transaction isolation are PostgreSQL matters (integration stage).
"""

import gc
import inspect
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError, MissingGreenlet, StatementError

from app.models import Event, SyncMetadata, Ticket
from app.repositories import (
    EventRepository,
    SyncMetadataRepository,
    TicketRepository,
)
from tests.conftest import make_event, make_ticket

pytestmark = pytest.mark.anyio


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=UTC)


async def count(session, model) -> int:
    return await session.scalar(select(func.count()).select_from(model))


# ---- events ----


async def test_event_get_missing_returns_none(async_session):
    assert await EventRepository(async_session).get("nope") is None


async def test_event_upsert_creates_and_get_returns_it(async_session):
    repo = EventRepository(async_session)

    created = await repo.upsert(make_event("e1", name="Concert"))

    assert isinstance(created, Event)
    assert await repo.get("e1") is created
    assert created.place_seats_pattern == "A1-10"
    assert created.number_of_visitors == 0


async def test_event_upsert_inserts_then_updates_without_duplicates(async_session):
    repo = EventRepository(async_session)
    first = await repo.upsert(make_event("e1", name="Old"))
    second = await repo.upsert(make_event("e1", name="New", status="finished"))

    assert await count(async_session, Event) == 1
    assert second is first  # merge returns the instance the session already holds
    event = await repo.get("e1")
    assert (event.name, event.status) == ("New", "finished")


async def test_event_upsert_of_an_event_the_session_does_not_hold(async_session):
    repo = EventRepository(async_session)
    await repo.upsert(make_event("e1", name="Old"))
    await async_session.commit()
    async_session.expunge_all()  # as in a new session: merge has to load the row

    updated = await repo.upsert(make_event("e1", name="New"))
    await async_session.commit()

    assert await count(async_session, Event) == 1
    assert updated.name == "New"
    assert updated.created_at is not None


async def test_event_upsert_is_idempotent(async_session):
    repo = EventRepository(async_session)
    for _ in range(3):
        await repo.upsert(make_event("e1"))
        await repo.upsert(make_event("e2"))
    await async_session.commit()

    assert await count(async_session, Event) == 2


async def test_event_list_order_pagination_and_count(async_session):
    repo = EventRepository(async_session)
    await repo.upsert(make_event("b", event_time=utc(2026, 1, 2)))
    await repo.upsert(make_event("a", event_time=utc(2026, 1, 2)))
    await repo.upsert(make_event("c", event_time=utc(2026, 1, 1)))

    page1, total = await repo.list(page=1, page_size=2)
    page2, _ = await repo.list(page=2, page_size=2)
    page3, _ = await repo.list(page=3, page_size=2)

    assert total == 3
    assert [e.id for e in page1] == ["c", "a"]  # event_time, then id
    assert [e.id for e in page2] == ["b"]
    assert page3 == []


async def test_event_list_date_from_is_inclusive(async_session):
    repo = EventRepository(async_session)
    await repo.upsert(make_event("old", event_time=utc(2026, 1, 1)))
    await repo.upsert(make_event("edge", event_time=utc(2026, 1, 2)))
    await repo.upsert(make_event("new", event_time=utc(2026, 1, 3)))

    events, total = await repo.list(date_from=utc(2026, 1, 2))

    assert total == 2  # the count honours the filter too
    assert [e.id for e in events] == ["edge", "new"]


async def test_event_list_count_covers_all_pages_not_only_the_page(async_session):
    repo = EventRepository(async_session)
    for i in range(5):
        await repo.upsert(make_event(f"e{i}", event_time=utc(2026, 1, 1 + i)))

    events, total = await repo.list(page=2, page_size=2)

    assert total == 5
    assert [e.id for e in events] == ["e2", "e3"]


async def test_event_list_empty_database(async_session):
    assert await EventRepository(async_session).list() == ([], 0)


# ---- tickets ----


async def test_ticket_add_and_find_active(async_session):
    await EventRepository(async_session).upsert(make_event())
    repo = TicketRepository(async_session)

    added = await repo.add(**make_ticket("t1"))

    assert added.id is not None  # flush assigned the primary key
    assert added.cancelled_at is None
    assert await repo.get_active("t1") is added
    assert await repo.get_active("other") is None


async def test_cancel_hides_ticket_and_allows_reuse_of_external_id(async_session):
    await EventRepository(async_session).upsert(make_event())
    repo = TicketRepository(async_session)
    first = await repo.add(**make_ticket("t1"))

    await repo.cancel(first, utc(2026, 1, 5))
    assert await repo.get_active("t1") is None
    assert first.cancelled_at is not None

    second = await repo.add(**make_ticket("t1", seat="B1"))
    assert await repo.get_active("t1") is second
    assert second.id != first.id
    assert await count(async_session, Ticket) == 2  # the history is kept


async def test_active_ticket_id_must_be_unique(async_session):
    await EventRepository(async_session).upsert(make_event())
    repo = TicketRepository(async_session)
    await repo.add(**make_ticket("t1"))

    with pytest.raises(IntegrityError):
        await repo.add(**make_ticket("t1", seat="B1"))


async def test_ticket_needs_an_existing_event(async_session):
    with pytest.raises(IntegrityError):
        await TicketRepository(async_session).add(**make_ticket("t1"))


async def test_cancel_only_touches_the_given_ticket(async_session):
    await EventRepository(async_session).upsert(make_event())
    repo = TicketRepository(async_session)
    one = await repo.add(**make_ticket("t1", seat="A1"))
    two = await repo.add(**make_ticket("t2", seat="A2"))

    await repo.cancel(one, utc(2026, 1, 5))

    assert await repo.get_active("t2") is two
    assert two.cancelled_at is None


# ---- sync metadata ----


async def test_sync_metadata_starts_empty(async_session):
    assert await SyncMetadataRepository(async_session).get() is None


async def test_sync_metadata_is_created_on_first_save(async_session):
    repo = SyncMetadataRepository(async_session)

    meta = await repo.save(
        sync_status="success",
        last_sync_time=utc(2026, 1, 2),
        last_changed_at=utc(2026, 1, 1),
    )

    assert meta.id == 1
    assert (meta.sync_status, meta.last_sync_time) == ("success", utc(2026, 1, 2))
    assert await repo.get() is meta


async def test_sync_metadata_cursor_is_not_changed_unless_given(async_session):
    repo = SyncMetadataRepository(async_session)
    await repo.save(
        sync_status="success",
        last_sync_time=utc(2026, 1, 1),
        last_changed_at=utc(2026, 1, 1),
    )
    cursor = (await repo.get()).last_changed_at

    await repo.save(sync_status="error")

    meta = await repo.get()
    assert meta.sync_status == "error"
    assert meta.last_changed_at == cursor
    assert await count(async_session, SyncMetadata) == 1


async def test_sync_metadata_cursor_moves_when_given(async_session):
    repo = SyncMetadataRepository(async_session)
    await repo.save(sync_status="success", last_changed_at=utc(2026, 1, 1))

    await repo.save(
        sync_status="success",
        last_sync_time=utc(2026, 2, 2),
        last_changed_at=utc(2026, 2, 1),
    )

    meta = await repo.get()
    # SQLite hands datetimes back without a time zone; compare the wall clock.
    assert meta.last_changed_at.replace(tzinfo=None) == datetime(2026, 2, 1)
    assert meta.last_sync_time.replace(tzinfo=None) == datetime(2026, 2, 2)
    assert await count(async_session, SyncMetadata) == 1


# ---- flush / commit / rollback ----


async def test_repositories_flush_but_do_not_commit(async_session):
    # Whatever the repositories wrote is visible in the transaction, yet it is gone
    # after a rollback: only the caller (the service) commits.
    await EventRepository(async_session).upsert(make_event("e1"))
    await TicketRepository(async_session).add(**make_ticket("t1"))
    await SyncMetadataRepository(async_session).save(sync_status="success")
    assert async_session.in_transaction()
    assert await count(async_session, Event) == 1
    assert await count(async_session, Ticket) == 1
    assert await count(async_session, SyncMetadata) == 1

    await async_session.rollback()

    assert await count(async_session, Event) == 0
    assert await count(async_session, Ticket) == 0
    assert await count(async_session, SyncMetadata) == 0


async def test_commit_keeps_the_data(async_session):
    await EventRepository(async_session).upsert(make_event("e1"))
    await async_session.commit()

    await async_session.rollback()  # nothing is left to undo

    assert await count(async_session, Event) == 1


async def test_committed_objects_stay_usable_after_commit(async_session):
    event = await EventRepository(async_session).upsert(make_event("e1"))
    await async_session.commit()

    assert event.name == "Concert"  # expire_on_commit=False: no lazy reload needed


async def test_failed_flush_rolls_back_to_the_previous_state(async_session):
    events = EventRepository(async_session)
    tickets = TicketRepository(async_session)
    await events.upsert(make_event("e1"))
    await tickets.add(**make_ticket("t1"))
    await async_session.commit()

    await events.upsert(make_event("e1", name="Changed"))
    with pytest.raises(IntegrityError):
        await tickets.add(**make_ticket("t1", seat="B1"))  # active duplicate
    await async_session.rollback()

    assert (await events.get("e1")).name == "Concert"  # the update is undone too
    assert await count(async_session, Ticket) == 1
    # The session is usable again after the rollback.
    assert await tickets.add(**make_ticket("t2", seat="B2")) is not None


async def test_failed_cancel_flush_can_be_rolled_back(async_session):
    await EventRepository(async_session).upsert(make_event())
    repo = TicketRepository(async_session)
    ticket = await repo.add(**make_ticket("t1"))
    await async_session.commit()

    await repo.cancel(ticket, utc(2026, 1, 5))
    await async_session.rollback()

    assert (await repo.get_active("t1")).id == ticket.id  # still active


# ---- no blocking SQL ----


@pytest.mark.parametrize(
    "repository",
    [EventRepository, TicketRepository, SyncMetadataRepository],
)
def test_every_public_method_is_a_coroutine_function(repository):
    methods = [
        member
        for name, member in inspect.getmembers(repository, inspect.isfunction)
        if not name.startswith("_")
    ]

    assert methods
    assert all(inspect.iscoroutinefunction(m) for m in methods)


# The refused call leaves an un-awaited driver coroutine; collect it here so its
# warning is raised (and ignored) in this test, not in a later one.
@pytest.mark.filterwarnings("ignore::pytest.PytestUnraisableExceptionWarning")
@pytest.mark.filterwarnings("ignore:coroutine 'Connection.cursor' was never awaited")
async def test_sync_style_sql_cannot_run_on_the_async_session(async_session):
    # The proof that no repository can quietly do blocking I/O: the sync API of
    # the same session refuses to run outside an await.
    # SQLAlchemy 2.1 wraps MissingGreenlet in a StatementError.
    with pytest.raises((MissingGreenlet, StatementError), match="greenlet_spawn"):
        async_session.sync_session.get(Event, "e1")
    gc.collect()


async def test_queries_do_not_block_the_event_loop(async_session):
    import asyncio

    repo = EventRepository(async_session)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        for i in range(20):
            await repo.upsert(make_event(f"e{i}"))
    finally:
        task.cancel()

    # Every query hands control back to the loop at least once.
    assert ticks >= 20
