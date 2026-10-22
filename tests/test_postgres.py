"""The application on a real PostgreSQL (``postgresql+psycopg_async``).

Needs a disposable database in ``TEST_DATABASE_URL`` (see ``checked_postgres_url``:
local, name ends in ``_test``, never the application's own). The schema comes from
Alembic, the engine and sessions from ``app.database`` exactly as the application
builds them; every test starts with empty tables.

What SQLite cannot tell: isolation, ``timestamptz``, the partial unique index,
real row locks, and which connections the server sees as "idle in transaction".
Concurrent scenarios use independent sessions.
"""

import asyncio
import logging
from datetime import UTC, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import IntegrityError

import main
from app import database, events_provider, services, sync_worker
from app.events_provider import (
    EventsPage,
    EventsProviderClient,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.models import Event, SyncMetadata
from app.repositories import EventRepository, SyncMetadataRepository, TicketRepository
from app.services import (
    CancellationInProgressError,
    SeatsCache,
    SyncAlreadyRunningError,
    TicketNotFoundError,
    TicketNotSavedError,
    cancel_ticket,
    get_available_seats,
    register_ticket,
    sync_events,
)
from tests.conftest import (
    ROOT,
    checked_postgres_url,
    make_event,
    provider_event_json,
    provider_page,
)
from tests.support import (
    ARGS,
    FUTURE,
    SECRET,
    Provider,
    Ticker,
    add_ticket,
    all_tickets,
    seed_event,
)

pytestmark = [pytest.mark.postgres, pytest.mark.anyio]

IDLE_IN_TRANSACTION = (
    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
    "AND pid <> pg_backend_pid() AND state LIKE 'idle in transaction%'"
)
WAITING_FOR_A_LOCK = (
    "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
    "AND wait_event_type = 'Lock'"
)


# ---- the database: Alembic schema, application engine ------------------------


@pytest.fixture(scope="module")
def schema():
    """The schema as Alembic builds it (the synchronous path stays in use)."""
    url = checked_postgres_url()
    with pytest.MonkeyPatch.context() as env:
        env.setenv("POSTGRES_CONNECTION_STRING", url)
        sync_engine = create_engine(make_url(url).set(drivername="postgresql+psycopg"))
        with sync_engine.begin() as connection:
            connection.execute(
                text("DROP TABLE IF EXISTS tickets, events, sync_metadata")
            )
            connection.execute(text("DROP TABLE IF EXISTS alembic_version"))
        sync_engine.dispose()
        command.upgrade(Config(str(ROOT / "alembic.ini")), "head")
        yield url


@pytest.fixture
async def pg(schema, monkeypatch):
    """Empty tables, and the application's own engine and session factory."""
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", schema)
    await database.dispose_async_engine()
    engine = database.get_async_engine()
    factory = database.get_async_session_factory()
    async with factory() as session:
        await session.execute(
            text("TRUNCATE tickets, events, sync_metadata RESTART IDENTITY CASCADE")
        )
        await session.commit()
    yield SimpleNamespace(url=schema, engine=engine, factory=factory)
    leaked = engine.pool.checkedout()
    await database.dispose_async_engine()
    assert leaked == 0, "a connection was not given back to the pool"


async def server_count(pg, query: str) -> int:
    async with pg.factory() as session:
        return await session.scalar(text(query))


async def idle_in_transaction(pg) -> int:
    return await server_count(pg, IDLE_IN_TRANSACTION)


async def assert_nothing_is_held(pg) -> None:
    """Neither the pool nor the server sees a connection kept by the application."""
    assert pg.engine.pool.checkedout() == 0
    assert await idle_in_transaction(pg) == 0


def gate_client() -> tuple[Provider, EventsProviderClient]:
    gate = Provider()
    client = EventsProviderClient(
        "http://provider:8000",
        SECRET,
        http_client=httpx.AsyncClient(
            base_url="http://provider:8000", transport=httpx.MockTransport(gate)
        ),
    )
    return gate, client


def page(*events: dict, next_url: str | None = None) -> EventsPage:
    return EventsPage.model_validate(provider_page(list(events), next_url=next_url))


async def count(pg, model) -> int:
    async with pg.factory() as session:
        return await session.scalar(select(func.count()).select_from(model))


def constraint_of(error: IntegrityError) -> str:
    return error.orig.diag.constraint_name


# ---- the server and the driver ---------------------------------------------


async def test_the_application_connects_through_psycopg_async(pg):
    async with pg.factory() as session:
        assert await session.scalar(text("SELECT 1")) == 1
        version = await session.scalar(text("SHOW server_version_num"))
    assert pg.engine.url.drivername == "postgresql+psycopg_async"
    assert int(version) >= 120000


async def test_default_isolation_is_read_committed(pg):
    # The "read, release, call, write" design relies on a read transaction holding
    # no snapshot; that is true for READ COMMITTED, the PostgreSQL default.
    async with pg.factory() as session:
        assert (
            await session.scalar(text("SHOW transaction_isolation")) == "read committed"
        )


async def test_flush_commit_and_rollback_across_independent_sessions(pg):
    async with pg.factory() as writer, pg.factory() as reader:
        await EventRepository(writer).upsert(make_event("e1"))  # flushed only
        assert await EventRepository(reader).get("e1") is None  # not visible yet
        await reader.commit()
        await writer.commit()
        assert (await EventRepository(reader).get("e1")).id == "e1"  # now it is
        await reader.commit()

        await TicketRepository(writer).add(
            ticket_id="t", event_id="e1", first_name="a", last_name="b",
            email="a@b.cc", seat="A1",
        )  # fmt: skip
        await writer.rollback()
        assert await TicketRepository(reader).get_active("t") is None
        await reader.commit()


async def test_a_failed_flush_rolls_the_whole_transaction_back(pg):
    await seed_event(pg.factory)
    await add_ticket(pg.factory, "t-1", seat="A4")
    async with pg.factory() as session:
        await EventRepository(session).upsert(make_event("e1", name="Changed"))
        with pytest.raises(IntegrityError):
            await TicketRepository(session).add(
                ticket_id="t-1", event_id="e1", first_name="x", last_name="y",
                email="x@y.zz", seat="A5",
            )  # fmt: skip
        await session.rollback()
        assert (await EventRepository(session).get("e1")).name == "Concert"
        await session.commit()


# ---- events, time zones ----------------------------------------------------


async def test_event_upsert_inserts_updates_and_is_idempotent(pg):
    async with pg.factory() as session:
        repo = EventRepository(session)
        await repo.upsert(make_event("e1", name="Old"))
        await repo.upsert(make_event("e1", name="New", status="finished"))
        await repo.upsert(make_event("e2"))
        await session.commit()
    async with pg.factory() as session:
        await EventRepository(session).upsert(
            make_event("e1", name="New", status="finished")
        )  # the same data again
        await session.commit()

    assert await count(pg, Event) == 2
    async with pg.factory() as session:
        event = await EventRepository(session).get("e1")
    assert (event.name, event.status) == ("New", "finished")


async def test_timestamptz_values_come_back_timezone_aware_and_equal(pg):
    moscow = timezone(timedelta(hours=3))
    original = datetime(2026, 1, 11, 17, 0, tzinfo=moscow)
    async with pg.factory() as session:
        await EventRepository(session).upsert(
            make_event("e1", event_time=original, registration_deadline=None)
        )
        await session.commit()

    async with pg.factory() as session:
        event = await EventRepository(session).get("e1")
        column = await session.scalar(
            text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_name = 'events' AND column_name = 'event_time'"
            )
        )
    assert column == "timestamp with time zone"
    assert event.event_time.tzinfo is not None
    assert event.event_time == original  # the same instant
    assert event.event_time.astimezone(UTC).hour == 14
    assert event.registration_deadline is None


async def test_provider_times_are_stored_as_instants_and_the_cursor_is_aware(pg):
    gate, client = gate_client()
    raw = provider_event_json(
        "e1",
        event_time="2026-01-11T17:00:00+03:00",
        changed_at="2026-01-05T07:00:00+00:00",
    )
    raw["registration_deadline"] = "2026-01-10T14:00:00"  # no offset: taken as UTC
    clients = AsyncMock()
    clients.get_events_page.side_effect = [page(raw), page()]

    async with pg.factory() as session:
        result = await sync_events(session, clients)

    assert result.changed_at_cursor.isoformat() == "2026-01-05"
    async with pg.factory() as session:
        event = await EventRepository(session).get("e1")
        meta = await SyncMetadataRepository(session).get()
    assert event.event_time == datetime(2026, 1, 11, 14, 0, tzinfo=UTC)
    assert event.registration_deadline == datetime(2026, 1, 10, 14, 0, tzinfo=UTC)
    assert meta.last_changed_at == datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
    assert meta.last_changed_at.tzinfo is not None and meta.last_sync_time.tzinfo

    async with pg.factory() as session:  # the next sync starts from the stored cursor
        await sync_events(session, clients)
    clients.get_events_page.assert_awaited_with(changed_at=datetime(2026, 1, 5).date())
    await client.aclose()


# ---- tickets: the partial unique index, history ------------------------------


async def test_partial_unique_index_refuses_a_second_active_ticket_id(pg):
    await seed_event(pg.factory)
    await add_ticket(pg.factory, "t-1", seat="A4")

    async with pg.factory() as session:
        with pytest.raises(IntegrityError) as info:
            await TicketRepository(session).add(
                ticket_id="t-1", event_id="e1", first_name="x", last_name="y",
                email="x@y.zz", seat="A5",
            )  # fmt: skip
        await session.rollback()
    assert constraint_of(info.value) == "uq_tickets_ticket_id_active"


async def test_a_ticket_needs_an_existing_event(pg):
    async with pg.factory() as session:
        with pytest.raises(IntegrityError) as info:
            await TicketRepository(session).add(
                ticket_id="t", event_id="missing", first_name="x", last_name="y",
                email="x@y.zz", seat="A5",
            )  # fmt: skip
        await session.rollback()
    assert constraint_of(info.value) == "fk_tickets_event_id_events"


async def test_cancellation_keeps_history_and_frees_the_ticket_id(pg):
    await seed_event(pg.factory)
    gate, client = gate_client()
    cache = SeatsCache()

    async with pg.factory() as session:
        assert await register_ticket(session, client, cache, **ARGS) == "t-1"
    async with pg.factory() as session:
        await cancel_ticket(session, client, cache, "t-1")
    other = {"first_name": "Пётр", "last_name": "Петров", "email": "petr@example.com"}
    async with pg.factory() as session:  # the provider hands out the same id again
        assert (
            await register_ticket(session, client, cache, **{**ARGS, **other}) == "t-1"
        )

    rows = await all_tickets(pg.factory)
    assert len(rows) == 2  # the old row is history, the new one is active
    assert rows[0].cancelled_at is not None and rows[0].email == "ivan@example.com"
    assert rows[1].cancelled_at is None and rows[1].email == "petr@example.com"
    async with pg.factory() as session:
        assert (await TicketRepository(session).get_active("t-1")).id == rows[1].id
        await cancel_ticket(session, client, cache, "t-1")
        with pytest.raises(TicketNotFoundError):
            await cancel_ticket(session, client, cache, "t-1")
    assert all(t.cancelled_at is not None for t in await all_tickets(pg.factory))
    await client.aclose()


# ---- concurrent registrations and cancellations ----------------------------


async def test_two_parallel_registrations_of_one_seat_save_one_ticket(pg):
    await seed_event(pg.factory)
    taken: set[str] = set()
    posts = 0

    async def provider_register(event_id, *, seat, **kwargs):
        nonlocal posts
        posts += 1
        number = posts
        await asyncio.sleep(0.05)  # both are inside the call together
        if seat in taken:
            raise ProviderRejectedError(400, "")
        taken.add(seat)
        return f"t-{number}"

    provider = AsyncMock()
    provider.get_event_seats.return_value = ["A5"]
    provider.register.side_effect = provider_register
    cache = SeatsCache()

    async def one():
        async with pg.factory() as session:
            return await register_ticket(session, provider, cache, **ARGS)

    results = await asyncio.gather(one(), one(), return_exceptions=True)

    assert sorted(type(r).__name__ for r in results) == [
        "RegistrationRejectedError",
        "str",
    ]
    assert provider.register.await_count == 2  # each POST was sent once
    [ticket] = await all_tickets(pg.factory)
    assert ticket.seat == "A5" and ticket.cancelled_at is None
    await assert_nothing_is_held(pg)


async def test_a_ticket_id_the_provider_repeats_is_refused_by_the_index(pg):
    # Both registrations pass the provider (which is wrong to repeat an id) and race
    # to save; the partial unique index lets exactly one in.
    await seed_event(pg.factory)
    provider = AsyncMock()
    provider.get_event_seats.return_value = ["A4", "A5"]
    provider.register.return_value = "same-id"
    cache = SeatsCache()

    async def one(seat):
        async with pg.factory() as session:
            return await register_ticket(
                session, provider, cache, **{**ARGS, "seat": seat}
            )

    results = await asyncio.gather(one("A4"), one("A5"), return_exceptions=True)

    assert sorted(type(r).__name__ for r in results) == ["TicketNotSavedError", "str"]
    assert len(await all_tickets(pg.factory)) == 1
    await assert_nothing_is_held(pg)


async def test_parallel_cancellations_of_one_ticket_send_one_delete(pg):
    await seed_event(pg.factory)
    await add_ticket(pg.factory, "t")

    async def slow_unregister(event_id, ticket_id):
        await asyncio.sleep(0.1)

    provider = AsyncMock()
    provider.unregister.side_effect = slow_unregister
    cache = SeatsCache()

    async def one():
        async with pg.factory() as session:
            await cancel_ticket(session, provider, cache, "t")

    results = await asyncio.gather(*(one() for _ in range(8)), return_exceptions=True)

    assert provider.unregister.await_count == 1
    assert sum(r is None for r in results) == 1
    assert all(
        isinstance(r, CancellationInProgressError | TicketNotFoundError)
        for r in results
        if r is not None
    )
    [ticket] = await all_tickets(pg.factory)
    assert ticket.cancelled_at is not None
    assert services._cancelling == set()
    await assert_nothing_is_held(pg)


# ---- manual and background synchronization ---------------------------------


@pytest.fixture
def worker_stack(pg, monkeypatch):
    """The real worker path on PostgreSQL: its own session factory comes from
    ``app.database`` (no patching), the provider is the gate."""
    gate, client = gate_client()
    monkeypatch.setattr(events_provider, "_shared_client", client)
    return SimpleNamespace(gate=gate, client=client)


async def test_manual_sync_is_refused_while_the_worker_syncs_on_postgres(
    pg, worker_stack
):
    gate = worker_stack.gate
    gate.hold(1)
    done = asyncio.Event()

    async def run_once(stop):
        try:
            return await sync_worker.run_sync_once(stop)
        finally:
            done.set()

    async def first_delay(interval):
        return 0.0

    worker = sync_worker.SyncWorker(86400, run_once=run_once, first_delay=first_delay)
    worker.start()
    await asyncio.wait_for(gate.all_entered.wait(), 5)

    async with pg.factory() as manual:
        with pytest.raises(SyncAlreadyRunningError):
            await sync_events(manual, AsyncMock())
    assert await idle_in_transaction(pg) == 1  # the background sync's own transaction

    gate.release.set()
    await asyncio.wait_for(done.wait(), 5)
    assert await worker.stop() is True
    assert await count(pg, Event) == 1
    async with pg.factory() as session:
        assert (await SyncMetadataRepository(session).get()).sync_status == "success"
    assert services._sync_running is False
    await assert_nothing_is_held(pg)


async def test_manual_trigger_is_409_over_http_while_the_worker_syncs(pg, worker_stack):
    gate = worker_stack.gate
    gate.hold(1)
    done = asyncio.Event()

    async def run_once(stop):
        try:
            return await sync_worker.run_sync_once(stop)
        finally:
            done.set()

    async def first_delay(interval):
        return 0.0

    worker = sync_worker.SyncWorker(86400, run_once=run_once, first_delay=first_delay)
    worker.start()
    await asyncio.wait_for(gate.all_entered.wait(), 5)

    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
        answer = await http.post("/api/sync/trigger")  # the real get_session

    assert answer.status_code == 409
    gate.release.set()
    await asyncio.wait_for(done.wait(), 5)
    assert await worker.stop() is True
    assert await count(pg, Event) == 1
    await assert_nothing_is_held(pg)


async def test_two_syncs_without_the_guard_never_corrupt_the_data(pg):
    # What a second replica would do: the in-process guard cannot see it. The
    # primary key and the transaction keep the data whole: one writer wins, the
    # other fails cleanly. This is why only one replica is supported.
    one_event = AsyncMock()
    one_event.get_events_page.return_value = page(provider_event_json("e1"))

    async with pg.factory() as first, pg.factory() as second:
        await EventRepository(first).upsert(make_event("e1"))  # holds the key
        racing = asyncio.create_task(services._sync(second, one_event, None))
        for _ in range(500):  # until the second one waits for the first one's row
            if await server_count(pg, WAITING_FOR_A_LOCK) >= 1:
                break
            await asyncio.sleep(0.01)
        assert await server_count(pg, WAITING_FOR_A_LOCK) >= 1
        await first.commit()
        with pytest.raises(IntegrityError):
            await racing

    assert await count(pg, Event) == 1  # no duplicate
    async with pg.factory() as session:
        assert (await SyncMetadataRepository(session).get()).sync_status == "error"
    await assert_nothing_is_held(pg)


# ---- atomicity of events and cursor ----------------------------------------


async def test_events_and_cursor_are_committed_together(pg):
    seen = []

    async def pages(**kwargs):
        if kwargs.get("url") is None:
            return page(provider_event_json("e1"), next_url="http://x/?page=2")
        async with pg.factory() as other:  # a different session, mid-sync
            seen.append(
                (
                    await other.scalar(select(func.count()).select_from(Event)),
                    await other.scalar(select(func.count()).select_from(SyncMetadata)),
                )
            )
        return page(provider_event_json("e2"))

    client = AsyncMock()
    client.get_events_page.side_effect = pages

    async with pg.factory() as session:
        await sync_events(session, client)

    assert seen == [(0, 0)]  # nothing is visible before the commit
    assert await count(pg, Event) == 2
    async with pg.factory() as session:
        meta = await SyncMetadataRepository(session).get()
    assert meta.last_changed_at is not None  # ...and then both are there at once


async def test_a_failure_on_the_second_page_leaves_events_and_cursor_untouched(pg):
    first = AsyncMock()
    first.get_events_page.return_value = page(provider_event_json("e1"))
    async with pg.factory() as session:
        await sync_events(session, first)
    async with pg.factory() as session:
        before = (await SyncMetadataRepository(session).get()).last_changed_at

    failing = AsyncMock()
    failing.get_events_page.side_effect = [
        page(
            provider_event_json("e2", changed_at="2026-03-01T00:00:00+00:00"),
            next_url="http://x/?page=2",
        ),
        ProviderUnavailableError("boom"),
    ]
    async with pg.factory() as session:
        with pytest.raises(ProviderUnavailableError):
            await sync_events(session, failing)

    assert await count(pg, Event) == 1  # e2 was rolled back
    async with pg.factory() as session:
        meta = await SyncMetadataRepository(session).get()
    assert meta.last_changed_at == before and meta.sync_status == "error"
    await assert_nothing_is_held(pg)


async def test_stop_between_events_rolls_back_on_postgres(pg, monkeypatch):
    stop = asyncio.Event()
    upsert = EventRepository.upsert

    async def upsert_then_stop(self, data):
        result = await upsert(self, data)
        stop.set()
        return result

    monkeypatch.setattr(EventRepository, "upsert", upsert_then_stop)
    client = AsyncMock()
    client.get_events_page.return_value = page(
        provider_event_json("e1"), provider_event_json("e2")
    )

    async with pg.factory() as session:
        with pytest.raises(services.SyncStoppedError):
            await sync_events(session, client, stop)

    assert await count(pg, Event) == 0 and await count(pg, SyncMetadata) == 0
    await assert_nothing_is_held(pg)


async def test_cancelled_sync_rolls_back_and_frees_its_connection(pg):
    in_second_page = asyncio.Event()

    async def pages(**kwargs):
        if kwargs.get("url") is None:
            return page(provider_event_json("e1"), next_url="http://x/?page=2")
        in_second_page.set()
        await asyncio.Event().wait()

    client = AsyncMock()
    client.get_events_page.side_effect = pages
    async with pg.factory() as session:
        task = asyncio.create_task(sync_events(session, client))
        await asyncio.wait_for(in_second_page.wait(), 5)
        assert await idle_in_transaction(pg) == 1  # e1 is flushed, not committed
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        # The service itself rolled back: the session is still open, yet it holds
        # no transaction and the server sees none.
        assert not session.in_transaction()
        assert await idle_in_transaction(pg) == 0
        assert await count(pg, Event) == 0

    assert await count(pg, SyncMetadata) == 0
    assert services._sync_running is False
    await assert_nothing_is_held(pg)


async def test_cancel_during_the_commit_leaves_a_consistent_database(pg, caplog):
    # The COMMIT is in flight when the task is cancelled; whichever way it ends,
    # the events and the cursor are either both there or both absent.
    in_commit = asyncio.Event()
    client = AsyncMock()
    client.get_events_page.return_value = page(provider_event_json("e1"))
    async with pg.factory() as session:
        real_commit = session.commit

        async def slow_commit():
            in_commit.set()
            await asyncio.sleep(0.05)
            await real_commit()

        session.commit = slow_commit
        with caplog.at_level(logging.INFO):
            task = asyncio.create_task(sync_events(session, client))
            await asyncio.wait_for(in_commit.wait(), 5)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    events, metadata = await count(pg, Event), await count(pg, SyncMetadata)
    assert (events, metadata) in {(0, 0), (1, 1)}
    assert "not known whether the events and the cursor were saved" in caplog.text
    await assert_nothing_is_held(pg)


# ---- the connection is not held during a provider call -------------------------


async def test_the_probe_sees_a_connection_that_is_held(pg):
    # Control for the checks below: an open read transaction does show up.
    await seed_event(pg.factory)
    async with pg.factory() as session:
        await EventRepository(session).get("e1")
        assert pg.engine.pool.checkedout() == 1
        assert await idle_in_transaction(pg) == 1
        await session.commit()
    await assert_nothing_is_held(pg)


async def test_no_connection_or_transaction_is_held_during_provider_calls(pg):
    await seed_event(pg.factory)
    await add_ticket(pg.factory, "t-9", seat="A1")
    samples = []

    async def sample(label):
        samples.append(
            (label, pg.engine.pool.checkedout(), await idle_in_transaction(pg))
        )

    async def seats(event_id):
        await sample("seats")
        return ["A5"]

    async def register(event_id, **kwargs):
        await sample("register")
        return "t-1"

    async def unregister(event_id, ticket_id):
        await sample("unregister")

    provider = AsyncMock()
    provider.get_event_seats.side_effect = seats
    provider.register.side_effect = register
    provider.unregister.side_effect = unregister
    cache = SeatsCache()

    async with pg.factory() as session:
        await get_available_seats(session, provider, cache, "e1")
        cache.invalidate("e1")
        await register_ticket(session, provider, cache, **ARGS)
        await cancel_ticket(session, provider, cache, "t-9")

    assert samples == [
        ("seats", 0, 0),
        ("seats", 0, 0),
        ("register", 0, 0),
        ("unregister", 0, 0),
    ]
    assert len(await all_tickets(pg.factory)) == 2


async def test_forty_requests_wait_for_the_provider_on_a_pool_of_fifteen(
    pg, monkeypatch
):
    assert pg.engine.pool.size() + pg.engine.pool._max_overflow == 15
    count_ = 40
    async with pg.factory() as session:
        repo = EventRepository(session)
        for i in range(count_):
            await repo.upsert(
                make_event(f"e{i}", registration_deadline=FUTURE, event_time=FUTURE)
            )
        await session.commit()
    gate, client = gate_client()
    monkeypatch.setattr(events_provider, "_shared_client", client)
    fresh = SeatsCache()
    main.app.dependency_overrides[main.get_seats_cache] = lambda: fresh
    gate.hold(count_)
    try:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            waiting = [
                asyncio.create_task(http.get(f"/api/events/e{i}/seats"))
                for i in range(count_)
            ]
            await asyncio.wait_for(gate.all_entered.wait(), 15)

            assert gate.waiting == count_  # all 40 are at the provider at once
            await assert_nothing_is_held(pg)
            assert (await http.get("/api/health")).status_code == 200
            assert (await http.get("/api/events")).json()["count"] == count_

            gate.release.set()
            answers = await asyncio.wait_for(asyncio.gather(*waiting), 15)
    finally:
        main.app.dependency_overrides.pop(main.get_seats_cache, None)

    assert [a.status_code for a in answers] == [200] * count_
    await assert_nothing_is_held(pg)


async def test_slow_provider_does_not_block_the_loop_on_postgres(pg):
    await seed_event(pg.factory)

    async def slow_seats(event_id):
        await asyncio.sleep(0.3)
        return ["A5"]

    provider = AsyncMock()
    provider.get_event_seats.side_effect = slow_seats

    async with pg.factory() as session, Ticker() as ticker:
        await get_available_seats(session, provider, SeatsCache(), "e1")

    assert ticker.count >= 10


# ---- nothing is left behind after errors and cancellation ----------------------


def _failing(operation, error):
    async def run(pg):
        provider = AsyncMock()
        provider.get_event_seats.return_value = ["A5"]
        getattr(provider, operation).side_effect = error
        async with pg.factory() as session:
            with pytest.raises(type(error)):
                if operation == "get_event_seats":
                    await get_available_seats(session, provider, SeatsCache(), "e1")
                elif operation == "register":
                    await register_ticket(session, provider, SeatsCache(), **ARGS)
                else:
                    await cancel_ticket(session, provider, SeatsCache(), "t-9")

    return run


@pytest.mark.parametrize(
    "scenario",
    [
        _failing("get_event_seats", ProviderUnavailableError("down")),
        _failing("get_event_seats", ProviderRejectedError(500, "")),
        _failing("register", ProviderUnavailableError("timeout")),
        _failing("register", ProviderRejectedError(503, "")),
        _failing("unregister", ProviderUnavailableError("timeout")),
        _failing("unregister", ProviderRejectedError(502, "")),
    ],
    ids=[
        "seats-unavailable",
        "seats-rejected",
        "register-unavailable",
        "register-rejected",
        "unregister-unavailable",
        "unregister-rejected",
    ],
)
async def test_provider_errors_leave_no_connection_behind(pg, scenario):
    await seed_event(pg.factory)
    await add_ticket(pg.factory, "t-9", seat="A1")

    await scenario(pg)

    await assert_nothing_is_held(pg)
    assert len(await all_tickets(pg.factory)) == 1  # nothing was changed


async def test_failed_local_saves_leave_no_connection_behind(pg):
    await seed_event(pg.factory)
    await add_ticket(pg.factory, "t-1", seat="A4")
    provider = AsyncMock()
    provider.get_event_seats.return_value = ["A5"]
    provider.register.return_value = "t-1"  # still active: the index refuses it

    async with pg.factory() as session:
        with pytest.raises(TicketNotSavedError):
            await register_ticket(session, provider, SeatsCache(), **ARGS)

    await assert_nothing_is_held(pg)


@pytest.mark.parametrize("operation", ["seats", "register", "unregister"])
async def test_cancelled_requests_leave_no_connection_behind(pg, operation):
    await seed_event(pg.factory)
    await add_ticket(pg.factory, "t-9", seat="A1")
    waiting = asyncio.Event()

    async def never_answers(*args, **kwargs):
        waiting.set()
        await asyncio.Event().wait()

    provider = AsyncMock()
    provider.get_event_seats.return_value = ["A5"]
    getattr(
        provider,
        {
            "seats": "get_event_seats",
            "register": "register",
            "unregister": "unregister",
        }[operation],
    ).side_effect = never_answers
    cache = SeatsCache()

    async with pg.factory() as session:
        if operation == "seats":
            work = get_available_seats(session, provider, cache, "e1")
        elif operation == "register":
            work = register_ticket(session, provider, cache, **ARGS)
        else:
            work = cancel_ticket(session, provider, cache, "t-9")
        task = asyncio.create_task(work)
        await asyncio.wait_for(waiting.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    await assert_nothing_is_held(pg)
    assert services._cancelling == set()
    assert len(await all_tickets(pg.factory)) == 1


# ---- the application over HTTP on PostgreSQL -----------------------------------


@pytest.fixture
async def http(pg, monkeypatch):
    """The real app, the real ``get_session``, a provider the test controls."""
    gate, client = gate_client()
    monkeypatch.setattr(events_provider, "_shared_client", client)
    fresh = SeatsCache()
    main.app.dependency_overrides[main.get_seats_cache] = lambda: fresh
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        c.gate = gate
        yield c
    main.app.dependency_overrides.pop(main.get_seats_cache, None)


async def test_events_list_detail_and_pagination_on_postgres(http, pg, monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://events.example.org")
    async with pg.factory() as session:
        repo = EventRepository(session)
        for i, day in enumerate((3, 1, 2, 2)):
            await repo.upsert(
                make_event(f"e{i}", event_time=datetime(2026, 1, day, tzinfo=UTC))
            )
        await session.commit()

    first = (await http.get("/api/events", params={"page_size": 2})).json()
    filtered = (
        await http.get("/api/events", params={"date_from": "2026-01-02"})
    ).json()
    detail = await http.get("/api/events/e0")
    missing = await http.get("/api/events/nope")

    assert first["count"] == 4
    assert [e["id"] for e in first["results"]] == ["e1", "e2"]  # by time, then id
    assert first["next"] == "https://events.example.org/api/events?page_size=2&page=2"
    assert [e["id"] for e in filtered["results"]] == ["e2", "e3", "e0"]
    assert detail.status_code == 200 and detail.json()["place"]["seats_pattern"]
    assert detail.json()["event_time"].startswith("2026-01-03T00:00:00")
    assert missing.status_code == 404


async def test_register_and_cancel_over_http_on_postgres(http, pg):
    await seed_event(pg.factory)
    body = {
        "event_id": "e1",
        "first_name": "Иван",
        "last_name": "Иванов",
        "email": "ivan@example.com",
        "seat": "A5",
    }

    created = await http.post("/api/tickets", json=body)
    again = await http.post("/api/tickets", json={**body, "email": "bad"})
    deleted = await http.delete("/api/tickets/t-1")
    gone = await http.delete("/api/tickets/t-1")

    assert created.status_code == 201 and created.json() == {"ticket_id": "t-1"}
    assert again.status_code == 400
    assert deleted.status_code == 200 and deleted.json() == {"success": True}
    assert gone.status_code == 404
    [ticket] = await all_tickets(pg.factory)
    assert ticket.cancelled_at is not None
    sent = [(r.method, r.url.path) for r in http.gate.requests]
    assert sent == [
        ("GET", "/api/events/e1/seats/"),
        ("POST", "/api/events/e1/register/"),
        ("DELETE", "/api/events/e1/unregister/"),
    ]  # each mutation exactly once, even for the refused requests
    assert all(r.headers["x-api-key"] == SECRET for r in http.gate.requests)
    await assert_nothing_is_held(pg)


async def test_manual_sync_over_http_on_postgres_is_idempotent(http, pg):
    http.gate.respond = lambda request: httpx.Response(
        200, json=provider_page([provider_event_json("p1"), provider_event_json("p2")])
    )

    first = await http.post("/api/sync/trigger")
    second = await http.post("/api/sync/trigger")

    assert first.status_code == second.status_code == 200
    assert await count(pg, Event) == 2  # the same events twice: still two rows
    assert (await http.get("/api/events")).json()["count"] == 2
    second_request = http.gate.requests[1].url.params["changed_at"]
    assert second_request == "2026-01-05"  # incremental: from the stored cursor
    await assert_nothing_is_held(pg)
