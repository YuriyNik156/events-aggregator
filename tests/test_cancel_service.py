"""``cancel_ticket`` (the routes are in ``test_cancel.py``): rules, provider
outcomes, partial failures, connection release, concurrent cancellations,
cancellation of the task.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app import services
from app.events_provider import ProviderRejectedError, ProviderUnavailableError
from app.repositories import TicketRepository
from app.services import (
    CancellationInProgressError,
    CancellationNotSavedError,
    CancellationRejectedError,
    EventStartedError,
    SeatsCache,
    TicketNotFoundError,
    cancel_ticket,
    register_ticket,
)
from tests.support import (
    ARGS,
    PAST,
    SECRET,
    Ticker,
    add_ticket,
    all_tickets,
    db_state,
    real_client,
    seed_event,
)

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def nothing_is_being_cancelled():
    assert services._cancelling == set()
    yield
    leaked = set(services._cancelling)
    services._cancelling.clear()  # never let one failure spread to the next test
    assert leaked == set()  # released on every outcome


@pytest.fixture
def provider() -> AsyncMock:
    mock = AsyncMock()
    mock.get_event_seats.return_value = ["A4", "A5"]
    mock.register.return_value = "t-1"
    return mock


@pytest.fixture
def cache() -> SeatsCache:
    return SeatsCache(ttl=30)


async def active(factory, ticket_id: str = "t-1"):
    async with factory() as s:
        return await TicketRepository(s).get_active(ticket_id)


# ---- success ----


async def test_cancel_keeps_the_row_and_marks_it_cancelled(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)

    await cancel_ticket(async_db, provider, cache, "t-1")

    provider.unregister.assert_awaited_once_with("e1", "t-1")
    [ticket] = await all_tickets(session_factory)  # not physically deleted
    assert ticket.ticket_id == "t-1"
    assert ticket.cancelled_at is not None
    assert await active(session_factory) is None
    assert not async_db.in_transaction()


async def test_cancel_invalidates_seats_cache(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    cache.set("e1", ["A1"])

    await cancel_ticket(async_db, provider, cache, "t-1")

    assert cache.get("e1") is None


# ---- not found / not allowed ----


async def test_unknown_ticket_without_provider_call(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)

    with pytest.raises(TicketNotFoundError):
        await cancel_ticket(async_db, provider, cache, "nope")

    provider.unregister.assert_not_called()
    assert db_state(async_db) == (False, 0)


async def test_repeated_cancel_does_not_send_a_second_delete(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    await cancel_ticket(async_db, provider, cache, "t-1")

    with pytest.raises(TicketNotFoundError):
        await cancel_ticket(async_db, provider, cache, "t-1")

    assert provider.unregister.await_count == 1


async def test_started_event_is_refused_without_provider_call(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory, event_time=PAST)
    await add_ticket(session_factory)

    with pytest.raises(EventStartedError):
        await cancel_ticket(async_db, provider, cache, "t-1")

    provider.unregister.assert_not_called()
    assert (await all_tickets(session_factory))[0].cancelled_at is None


async def test_event_time_boundary_is_compared_in_utc(
    async_db, session_factory, provider, cache
):
    await seed_event(
        session_factory, event_time=datetime(2026, 1, 10, 7, 0, tzinfo=UTC)
    )
    await add_ticket(session_factory, "t")

    with pytest.raises(EventStartedError):
        await cancel_ticket(
            async_db, provider, cache, "t", now=datetime(2026, 1, 10, 7, 0, tzinfo=UTC)
        )
    await cancel_ticket(
        async_db, provider, cache, "t", now=datetime(2026, 1, 10, 6, 59, tzinfo=UTC)
    )

    assert (await all_tickets(session_factory))[0].cancelled_at is not None


# ---- provider outcomes: never a false success ----


async def test_provider_400_is_rejected_and_ticket_stays_active(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    cache.set("e1", ["A1"])
    provider.unregister.side_effect = ProviderRejectedError(400, "")

    with pytest.raises(CancellationRejectedError):
        await cancel_ticket(async_db, provider, cache, "t-1")

    assert (await all_tickets(session_factory))[0].cancelled_at is None
    assert cache.get("e1") is not None  # unsuccessful: cache untouched


async def test_provider_404_is_not_found_and_ticket_stays_active(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    provider.unregister.side_effect = ProviderRejectedError(404, "")

    with pytest.raises(TicketNotFoundError):
        await cancel_ticket(async_db, provider, cache, "t-1")

    assert (await all_tickets(session_factory))[0].cancelled_at is None


@pytest.mark.parametrize("status", [401, 429, 500, 503])
async def test_other_provider_rejections_propagate(
    async_db, session_factory, provider, cache, status
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    provider.unregister.side_effect = ProviderRejectedError(status, "<html>x</html>")

    with pytest.raises(ProviderRejectedError) as info:
        await cancel_ticket(async_db, provider, cache, "t-1")

    assert info.value.status_code == status
    assert (await all_tickets(session_factory))[0].cancelled_at is None


async def test_provider_unavailable_is_not_retried(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    provider.unregister.side_effect = ProviderUnavailableError("timeout")

    with pytest.raises(ProviderUnavailableError):
        await cancel_ticket(async_db, provider, cache, "t-1")

    assert provider.unregister.await_count == 1
    assert (await all_tickets(session_factory))[0].cancelled_at is None


@pytest.mark.parametrize(
    "error",
    [
        ProviderRejectedError(400, ""),
        ProviderRejectedError(404, ""),
        ProviderRejectedError(500, ""),
        ProviderUnavailableError("timeout"),
        RuntimeError("unexpected"),
    ],
)
async def test_guard_is_released_after_provider_failure(
    async_db, session_factory, provider, cache, error
):
    await seed_event(session_factory)
    await add_ticket(session_factory, "t")
    provider.unregister.side_effect = error

    with pytest.raises(Exception):  # noqa: B017 - every failure kind ends here
        await cancel_ticket(async_db, provider, cache, "t")

    assert services._cancelling == set()  # also checked by the autouse fixture


# ---- partial failure: provider confirmed, local save failed ----


async def test_local_save_failure_reports_ticket_id_and_logs_no_personal_data(
    async_db, session_factory, provider, cache, caplog, monkeypatch
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    cache.set("e1", ["A1"])

    async def broken_cancel(self, ticket, cancelled_at):
        raise RuntimeError(f"db down for {ticket.email}")

    monkeypatch.setattr(TicketRepository, "cancel", broken_cancel)

    with (
        caplog.at_level(logging.INFO),
        pytest.raises(CancellationNotSavedError) as info,
    ):
        await cancel_ticket(async_db, provider, cache, "t-1")

    assert info.value.ticket_id == "t-1"
    assert "ivan@example.com" not in str(info.value)
    assert provider.unregister.await_count == 1  # no automatic repeat
    assert cache.get("e1") is None  # the seat is free at the provider
    assert (await all_tickets(session_factory))[0].cancelled_at is None
    assert not async_db.in_transaction()  # rolled back
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "ivan@example.com" not in logged
    assert "Иван" not in logged


async def test_local_save_failure_on_commit_is_rolled_back(
    async_db, session_factory, provider, cache, monkeypatch
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    real_commit = async_db.commit
    calls = 0

    async def commit_fails_the_second_time():
        nonlocal calls
        calls += 1
        if calls == 2:  # the first one releases the read transaction
            raise RuntimeError("commit failed")
        await real_commit()

    monkeypatch.setattr(async_db, "commit", commit_fails_the_second_time)

    with pytest.raises(CancellationNotSavedError):
        await cancel_ticket(async_db, provider, cache, "t-1")

    assert (await all_tickets(session_factory))[0].cancelled_at is None
    assert not async_db.in_transaction()


# ---- the database connection is not held during the provider call ----


async def test_connection_is_released_during_the_provider_call(
    async_db, session_factory, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    states = []

    async def unregister(event_id, ticket_id):
        states.append(db_state(async_db))

    provider = AsyncMock()
    provider.unregister.side_effect = unregister

    await cancel_ticket(async_db, provider, cache, "t-1")

    assert states == [(False, 0)]
    assert (await all_tickets(session_factory))[0].cancelled_at is not None


async def test_works_when_the_session_expires_objects_on_commit(
    async_engine, session_factory, provider, cache
):
    # The ticket was loaded in the first transaction and is updated in a new one;
    # the service must not rely on expire_on_commit=False.
    await seed_event(session_factory)
    await add_ticket(session_factory)

    async with AsyncSession(async_engine, expire_on_commit=True) as session:
        await cancel_ticket(session, provider, cache, "t-1")

    assert (await all_tickets(session_factory))[0].cancelled_at is not None


async def test_slow_provider_does_not_block_the_event_loop(
    async_db, session_factory, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)

    async def slow_unregister(event_id, ticket_id):
        await asyncio.sleep(0.3)

    provider = AsyncMock()
    provider.unregister.side_effect = slow_unregister

    async with Ticker() as ticker:
        await cancel_ticket(async_db, provider, cache, "t-1")

    assert ticker.count >= 10


# ---- concurrency ----


async def test_concurrent_cancel_of_one_ticket_sends_one_delete(
    async_db, session_factory, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory, "t")
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_unregister(event_id, ticket_id):
        entered.set()
        await release.wait()

    provider = AsyncMock()
    provider.unregister.side_effect = slow_unregister

    async with session_factory() as other_session:
        first = asyncio.create_task(cancel_ticket(async_db, provider, cache, "t"))
        await asyncio.wait_for(entered.wait(), 5)

        with pytest.raises(CancellationInProgressError):
            await cancel_ticket(other_session, provider, cache, "t")
        assert db_state(other_session) == (False, 0)  # the refusal read nothing

        release.set()
        await first

    assert provider.unregister.await_count == 1
    assert (await all_tickets(session_factory))[0].cancelled_at is not None
    assert services._cancelling == set()


async def test_many_concurrent_cancels_of_one_ticket_send_one_delete(
    session_factory, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory, "t")

    async def slow_unregister(event_id, ticket_id):
        await asyncio.sleep(0.1)

    provider = AsyncMock()
    provider.unregister.side_effect = slow_unregister

    async def one():
        async with session_factory() as session:
            await cancel_ticket(session, provider, cache, "t")

    results = await asyncio.gather(*(one() for _ in range(10)), return_exceptions=True)

    assert provider.unregister.await_count == 1
    assert sum(r is None for r in results) == 1
    # Each of the others lost the race: refused while it ran, or not found after.
    assert all(
        isinstance(r, CancellationInProgressError | TicketNotFoundError)
        for r in results
        if r is not None
    )


async def test_in_progress_guard_refuses_without_provider_call(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    services._cancelling.add("t-1")
    try:
        with pytest.raises(CancellationInProgressError):
            await cancel_ticket(async_db, provider, cache, "t-1")

        provider.unregister.assert_not_called()
        # The refused call must not release somebody else's guard.
        assert services._cancelling == {"t-1"}
    finally:
        services._cancelling.discard("t-1")


async def test_cancels_of_different_tickets_run_together(session_factory, cache):
    await seed_event(session_factory)
    await add_ticket(session_factory, "a", seat="A1")
    await add_ticket(session_factory, "b", seat="A2")
    active_calls = peak = 0

    async def slow_unregister(event_id, ticket_id):
        nonlocal active_calls, peak
        active_calls += 1
        peak = max(peak, active_calls)
        await asyncio.sleep(0.1)
        active_calls -= 1

    provider = AsyncMock()
    provider.unregister.side_effect = slow_unregister

    async def one(ticket_id):
        async with session_factory() as session:
            await cancel_ticket(session, provider, cache, ticket_id)

    await asyncio.gather(one("a"), one("b"))

    assert peak == 2
    assert all(t.cancelled_at is not None for t in await all_tickets(session_factory))


async def test_waiter_after_a_finished_cancellation_gets_not_found_without_delete(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory, "t")
    await cancel_ticket(async_db, provider, cache, "t")

    with pytest.raises(TicketNotFoundError):
        await cancel_ticket(async_db, provider, cache, "t")

    assert provider.unregister.await_count == 1


# ---- cancellation of the task ----


async def test_cancelled_while_the_provider_is_asked_leaves_the_ticket_active(
    async_db, session_factory, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    cache.set("e1", ["A1"])
    waiting = asyncio.Event()

    async def never_answers(event_id, ticket_id):
        waiting.set()
        await asyncio.Event().wait()

    provider = AsyncMock()
    provider.unregister.side_effect = never_answers

    task = asyncio.create_task(cancel_ticket(async_db, provider, cache, "t-1"))
    await asyncio.wait_for(waiting.wait(), 5)
    assert services._cancelling == {"t-1"}
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert services._cancelling == set()  # the guard is let go
    assert db_state(async_db) == (False, 0)
    assert (await all_tickets(session_factory))[0].cancelled_at is None
    assert cache.get("e1") is not None  # the outcome is unknown: cache untouched
    assert provider.unregister.await_count == 1


async def test_cancelled_during_the_local_save_rolls_back_and_logs_the_ticket(
    async_db, session_factory, provider, cache, caplog, monkeypatch
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    saving = asyncio.Event()
    cancel = TicketRepository.cancel

    async def hanging_cancel(self, ticket, cancelled_at):
        await cancel(self, ticket, cancelled_at)  # flushed, not committed
        saving.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(TicketRepository, "cancel", hanging_cancel)

    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(cancel_ticket(async_db, provider, cache, "t-1"))
        await asyncio.wait_for(saving.wait(), 5)
        assert async_db.in_transaction()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert db_state(async_db) == (False, 0)
    assert (await all_tickets(session_factory))[0].cancelled_at is None
    assert cache.get("e1") is None  # the seat is free at the provider
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "t-1" in logged
    assert "ivan@example.com" not in logged


# ---- ticket_id reuse after cancellation (end to end) ----


async def test_ticket_id_reuse_after_cancellation(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await register_ticket(async_db, provider, cache, **ARGS)  # user 1 on A5 -> t-1
    await cancel_ticket(async_db, provider, cache, "t-1")
    [old] = await all_tickets(session_factory)
    assert old.cancelled_at is not None

    provider.register.return_value = "t-1"  # the provider hands out the same id
    other = {"first_name": "Пётр", "last_name": "Петров", "email": "petr@example.com"}
    await register_ticket(async_db, provider, cache, **{**ARGS, **other})  # user 2

    rows = await all_tickets(session_factory)
    assert len(rows) == 2  # no uniqueness conflict, history kept
    new = await active(session_factory)
    assert new is not None and new.id != old.id
    assert new.email == "petr@example.com"
    assert rows[0].cancelled_at is not None and rows[0].email == "ivan@example.com"

    await cancel_ticket(async_db, provider, cache, "t-1")
    assert all(t.cancelled_at is not None for t in await all_tickets(session_factory))
    assert provider.unregister.await_count == 2
    with pytest.raises(TicketNotFoundError):
        await cancel_ticket(async_db, provider, cache, "t-1")


# ---- the real async client (MockTransport) ----


async def test_real_client_sends_the_exact_contract_once(
    async_db, session_factory, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory, "1fed0122")
    client, seen = real_client(lambda r: httpx.Response(200, json={"success": True}))

    await cancel_ticket(async_db, client, cache, "1fed0122")

    [request] = seen
    assert request.method == "DELETE"
    assert request.url.path == "/api/events/e1/unregister/"
    assert request.headers["x-api-key"] == SECRET
    assert json.loads(request.content) == {"ticket_id": "1fed0122"}


async def test_real_client_timeout_is_not_retried_and_leaks_nothing(
    async_db, session_factory, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)

    def boom(request):
        raise httpx.ReadTimeout("timeout", request=request)

    client, seen = real_client(boom)

    with pytest.raises(ProviderUnavailableError) as info:
        await cancel_ticket(async_db, client, cache, "t-1")

    assert len(seen) == 1
    assert SECRET not in str(info.value)
    assert (await all_tickets(session_factory))[0].cancelled_at is None


async def test_real_client_unconfirmed_answer_is_not_a_cancellation(
    async_db, session_factory, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    client, _ = real_client(lambda r: httpx.Response(200, json={"success": False}))

    with pytest.raises(ProviderUnavailableError):
        await cancel_ticket(async_db, client, cache, "t-1")

    assert (await all_tickets(session_factory))[0].cancelled_at is None


async def test_real_client_redirect_is_not_followed(async_db, session_factory, cache):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    client, seen = real_client(
        lambda r: httpx.Response(302, headers={"location": "http://evil.example/x"})
    )

    with pytest.raises(ProviderRejectedError):
        await cancel_ticket(async_db, client, cache, "t-1")

    assert [r.url.host for r in seen] == ["provider"]
    assert (await all_tickets(session_factory))[0].cancelled_at is None
