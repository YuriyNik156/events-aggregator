"""``sync_events``: cursor, atomicity, one sync at a time, stop and cancellation.

The service runs on ``async_db``; the state is always read back through a fresh
session, so only what was really committed counts.
"""

import asyncio
import logging
import time
from datetime import UTC, date, datetime
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import func, select

from app import services
from app.events_provider import (
    EventsPage,
    EventsProviderClient,
    ProviderUnavailableError,
)
from app.models import Event
from app.repositories import EventRepository, SyncMetadataRepository
from app.services import SyncAlreadyRunningError, SyncStoppedError, sync_events
from tests.conftest import INITIAL, provider_event_json, provider_page

pytestmark = pytest.mark.anyio


def page(*events: dict, next_url: str | None = None) -> EventsPage:
    return EventsPage.model_validate(provider_page(list(events), next_url=next_url))


def client_with(*pages) -> AsyncMock:
    client = AsyncMock()
    client.get_events_page.side_effect = list(pages)
    return client


async def event_count(factory) -> int:
    async with factory() as s:
        return await s.scalar(select(func.count()).select_from(Event))


async def metadata(factory):
    async with factory() as s:
        return await SyncMetadataRepository(s).get()


async def stored_cursor(factory) -> datetime | None:
    value = (await metadata(factory)).last_changed_at
    return value.replace(tzinfo=UTC) if value and value.tzinfo is None else value


@pytest.fixture(autouse=True)
def sync_flag_is_free():
    assert services._sync_running is False
    yield
    assert services._sync_running is False  # no test leaves it set


# ---- the same scenarios as the sync service ----


async def test_first_sync_uses_initial_cursor_and_saves_metadata(
    async_db, session_factory
):
    client = client_with(page(provider_event_json("e1"), provider_event_json("e2")))

    result = await sync_events(async_db, client)

    client.get_events_page.assert_awaited_once_with(changed_at=INITIAL)
    assert result.processed == 2
    assert await event_count(session_factory) == 2
    meta = await metadata(session_factory)
    assert meta.sync_status == "success"
    assert meta.last_sync_time is not None
    assert await stored_cursor(session_factory) == datetime(2026, 1, 5, 7, tzinfo=UTC)
    assert result.changed_at_cursor == date(2026, 1, 5)


async def test_all_pages_are_saved(async_db, session_factory):
    client = client_with(
        page(provider_event_json("e1"), next_url="http://x/?page=2"),
        page(provider_event_json("e2")),
    )

    assert (await sync_events(async_db, client)).processed == 2
    assert await event_count(session_factory) == 2


async def test_second_sync_uses_stored_cursor_and_updates_without_duplicates(
    async_db, session_factory
):
    await sync_events(
        async_db, client_with(page(provider_event_json("e1", name="Old")))
    )
    changed = provider_event_json(
        "e1", name="New", changed_at="2026-02-01T00:00:00+00:00"
    )
    client = client_with(page(changed))

    await sync_events(async_db, client)

    client.get_events_page.assert_awaited_once_with(changed_at=date(2026, 1, 5))
    assert await event_count(session_factory) == 1
    async with session_factory() as s:
        assert (await EventRepository(s).get("e1")).name == "New"
    assert (await stored_cursor(session_factory)).date() == date(2026, 2, 1)


async def test_sync_without_events_keeps_cursor(async_db, session_factory):
    await sync_events(async_db, client_with(page(provider_event_json("e1"))))
    before = await stored_cursor(session_factory)

    result = await sync_events(async_db, client_with(page()))

    assert result.processed == 0
    assert await stored_cursor(session_factory) == before
    assert (await metadata(session_factory)).sync_status == "success"


async def test_cursor_never_moves_back(async_db, session_factory):
    await sync_events(async_db, client_with(page(provider_event_json("e1"))))
    before = await stored_cursor(session_factory)
    older = provider_event_json("e2", changed_at="2025-01-01T00:00:00+00:00")

    await sync_events(async_db, client_with(page(older)))

    assert await stored_cursor(session_factory) == before


async def test_pagination_error_keeps_cursor_and_events(async_db, session_factory):
    await sync_events(async_db, client_with(page(provider_event_json("e1"))))
    before = await stored_cursor(session_factory)
    client = client_with(
        page(
            provider_event_json("e2", changed_at="2026-03-01T00:00:00+00:00"),
            next_url="http://x/?page=2",
        ),
        ProviderUnavailableError("boom"),
    )

    with pytest.raises(ProviderUnavailableError):
        await sync_events(async_db, client)

    assert await event_count(session_factory) == 1  # page 1 of the failed run: undone
    assert await stored_cursor(session_factory) == before
    assert (await metadata(session_factory)).sync_status == "error"


async def test_failed_first_sync_leaves_no_cursor(async_db, session_factory):
    with pytest.raises(ProviderUnavailableError):
        await sync_events(async_db, client_with(ProviderUnavailableError("boom")))

    meta = await metadata(session_factory)
    assert meta.sync_status == "error"
    assert meta.last_changed_at is None
    assert meta.last_sync_time is None


async def test_save_error_keeps_cursor(async_db, session_factory, monkeypatch):
    await sync_events(async_db, client_with(page(provider_event_json("e1"))))
    before = await stored_cursor(session_factory)

    async def broken_upsert(self, data):
        raise RuntimeError("db down")

    monkeypatch.setattr(EventRepository, "upsert", broken_upsert)
    with pytest.raises(RuntimeError):
        await sync_events(async_db, client_with(page(provider_event_json("e2"))))

    assert await stored_cursor(session_factory) == before
    assert not async_db.in_transaction()


async def test_stop_before_start_changes_nothing(async_db, session_factory):
    await sync_events(async_db, client_with(page(provider_event_json("e1"))))
    before = await stored_cursor(session_factory)
    stop = asyncio.Event()
    stop.set()
    newer = provider_event_json("e2", changed_at="2026-03-01T00:00:00+00:00")

    with pytest.raises(SyncStoppedError):
        await sync_events(async_db, client_with(page(newer)), stop)

    assert await event_count(session_factory) == 1
    assert await stored_cursor(session_factory) == before
    assert (await metadata(session_factory)).sync_status == "success"  # not error


async def test_stop_between_events_rolls_back(async_db, session_factory, monkeypatch):
    stop = asyncio.Event()
    upsert = EventRepository.upsert

    async def upsert_then_stop(self, data):
        result = await upsert(self, data)
        stop.set()  # as if shutdown began while the first event was saved
        return result

    monkeypatch.setattr(EventRepository, "upsert", upsert_then_stop)
    client = client_with(page(provider_event_json("e1"), provider_event_json("e2")))

    with pytest.raises(SyncStoppedError):
        await sync_events(async_db, client, stop)

    assert await event_count(session_factory) == 0
    assert await metadata(session_factory) is None  # no error row either
    assert not async_db.in_transaction()
    assert services._sync_running is False


async def test_unset_stop_does_not_change_the_sync(async_db):
    result = await sync_events(
        async_db, client_with(page(provider_event_json("e1"))), asyncio.Event()
    )

    assert result.processed == 1


# ---- one sync at a time ----


async def test_parallel_sync_is_refused(async_db):
    services._sync_running = True  # as if another sync were running
    client = client_with(page())
    try:
        with pytest.raises(SyncAlreadyRunningError):
            await sync_events(async_db, client)

        client.get_events_page.assert_not_called()
        assert services._sync_running is True  # the refusal did not clear it
    finally:
        services._sync_running = False


async def test_second_sync_started_during_the_first_is_refused_without_waiting(
    session_factory,
):
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_page(**kwargs):
        entered.set()
        await release.wait()
        return page(provider_event_json("e1"))

    slow = AsyncMock()
    slow.get_events_page.side_effect = slow_page
    other = client_with(page())

    async with session_factory() as first_session, session_factory() as second_session:
        first = asyncio.create_task(sync_events(first_session, slow))
        await asyncio.wait_for(entered.wait(), 5)

        started = time.monotonic()
        with pytest.raises(SyncAlreadyRunningError):
            await sync_events(second_session, other)
        assert time.monotonic() - started < 0.5  # refused, not queued

        release.set()
        assert (await first).processed == 1

    other.get_events_page.assert_not_called()
    assert services._sync_running is False
    assert await event_count(session_factory) == 1


async def test_flag_is_released_after_every_outcome(async_db):
    with pytest.raises(ProviderUnavailableError):
        await sync_events(async_db, client_with(ProviderUnavailableError("boom")))
    assert services._sync_running is False

    stop = asyncio.Event()
    stop.set()
    with pytest.raises(SyncStoppedError):
        await sync_events(async_db, client_with(page(provider_event_json("e1"))), stop)
    assert services._sync_running is False

    # ...so the next sync can run
    assert (await sync_events(async_db, client_with(page()))).processed == 0


# ---- one transaction for events and cursor ----


async def test_events_and_cursor_are_committed_together(async_db, session_factory):
    seen_during_second_page = []

    async def pages(**kwargs):
        if kwargs.get("url") is None:
            return page(provider_event_json("e1"), next_url="http://x/?page=2")
        # The first page is flushed, but nobody else may see it yet.
        seen_during_second_page.append(
            (async_db.in_transaction(), await event_count(session_factory))
        )
        return page(provider_event_json("e2"))

    client = AsyncMock()
    client.get_events_page.side_effect = pages

    await sync_events(async_db, client)

    assert seen_during_second_page == [(True, 0)]
    assert await event_count(session_factory) == 2
    assert not async_db.in_transaction()


# ---- cancellation ----


async def test_cancelled_sync_rolls_back_and_is_not_a_failure(
    async_db, session_factory
):
    await sync_events(async_db, client_with(page(provider_event_json("e1"))))
    before = await stored_cursor(session_factory)
    in_second_page = asyncio.Event()

    async def pages(**kwargs):
        if kwargs.get("url") is None:
            return page(
                provider_event_json("e2", changed_at="2026-03-01T00:00:00+00:00"),
                next_url="http://x/?page=2",
            )
        in_second_page.set()
        await asyncio.Event().wait()  # the provider never answers

    client = AsyncMock()
    client.get_events_page.side_effect = pages

    task = asyncio.create_task(sync_events(async_db, client))
    await asyncio.wait_for(in_second_page.wait(), 5)
    assert async_db.in_transaction()  # e2 is flushed, not committed
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not async_db.in_transaction()  # rolled back, nothing left open
    assert await event_count(session_factory) == 1
    assert await stored_cursor(session_factory) == before
    assert (await metadata(session_factory)).sync_status == "success"  # not "error"
    assert services._sync_running is False


async def test_cancel_during_a_database_write_leaves_no_transaction(
    async_db, session_factory, monkeypatch
):
    started = asyncio.Event()
    upsert = EventRepository.upsert

    async def slow_upsert(self, data):
        await upsert(self, data)
        started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(EventRepository, "upsert", slow_upsert)
    task = asyncio.create_task(
        sync_events(async_db, client_with(page(provider_event_json("e1"))))
    )
    await asyncio.wait_for(started.wait(), 5)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not async_db.in_transaction()
    assert await event_count(session_factory) == 0
    assert await metadata(session_factory) is None
    assert services._sync_running is False


# ---- the event loop stays free ----


async def test_slow_provider_does_not_block_the_event_loop(async_db):
    async def slow_page(**kwargs):
        await asyncio.sleep(0.3)
        return page(provider_event_json("e1"))

    client = AsyncMock()
    client.get_events_page.side_effect = slow_page
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        assert (await sync_events(async_db, client)).processed == 1
    finally:
        task.cancel()

    assert ticks >= 10  # a blocked loop would have counted about one


# ---- with the real async client (MockTransport) ----


async def test_sync_through_the_real_client_follows_next_on_the_provider_host(
    async_db, session_factory
):
    seen: list[httpx.Request] = []
    next_url = "http://public.example.org/api/events/?changed_at=2000-01-01&cursor=2"

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if request.url.params.get("cursor") is None:
            body = provider_page([provider_event_json("e1")], next_url=next_url)
        else:
            body = provider_page([provider_event_json("e2")])
        return httpx.Response(200, json=body)

    http = httpx.AsyncClient(
        base_url="http://provider:8000", transport=httpx.MockTransport(handler)
    )
    client = EventsProviderClient("http://provider:8000", "key", http_client=http)
    try:
        result = await sync_events(async_db, client)
    finally:
        await client.aclose()

    assert result.processed == 2
    assert await event_count(session_factory) == 2
    assert {r.url.host for r in seen} == {"provider"}
    assert all(r.headers["x-api-key"] == "key" for r in seen)


# ---- cancelled during the commit: the outcome is not known ----


async def cancelled_in_commit(async_db, session_factory, caplog, *, reaches_database):
    """Cancel a sync while its COMMIT is in flight; the COMMIT may or may not land."""
    real_commit = async_db.commit
    in_commit = asyncio.Event()

    async def hanging_commit():
        if reaches_database:
            await real_commit()  # the database has it, the caller does not know yet
        in_commit.set()
        await asyncio.Event().wait()

    async_db.commit = hanging_commit
    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(
            sync_events(async_db, client_with(page(provider_event_json("e1"))))
        )
        await asyncio.wait_for(in_commit.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):  # not hidden
            await task
    return await event_count(session_factory)


@pytest.mark.parametrize("reaches_database", [False, True])
async def test_cancel_during_the_commit_does_not_claim_a_result(
    async_db, session_factory, caplog, reaches_database
):
    stored = await cancelled_in_commit(
        async_db, session_factory, caplog, reaches_database=reaches_database
    )

    assert stored == (1 if reaches_database else 0)  # either is possible
    assert "not known whether the events and the cursor were saved" in caplog.text
    assert "rolled back" not in caplog.text  # no claim in either direction
    assert "Sync finished" not in caplog.text and "Sync failed" not in caplog.text
    assert not async_db.in_transaction()
    assert services._sync_running is False


async def test_next_sync_repairs_whatever_a_cancelled_commit_left(
    async_db, session_factory, caplog
):
    # Not saved: the next run starts from the old cursor and takes the event again.
    await cancelled_in_commit(async_db, session_factory, caplog, reaches_database=False)
    del async_db.commit  # back to the real commit
    result = await sync_events(async_db, client_with(page(provider_event_json("e1"))))

    assert result.processed == 1
    assert await event_count(session_factory) == 1  # once, whatever happened before


async def test_cancel_outside_the_commit_says_it_was_rolled_back(
    async_db, session_factory, caplog
):
    in_page = asyncio.Event()

    async def never(**kwargs):
        in_page.set()
        await asyncio.Event().wait()

    client = AsyncMock()
    client.get_events_page.side_effect = never
    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(sync_events(async_db, client))
        await asyncio.wait_for(in_page.wait(), 5)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert "cancelled" in caplog.text and "rolled back" in caplog.text
    assert "not known" not in caplog.text
