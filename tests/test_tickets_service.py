"""``register_ticket`` (the routes are in ``test_tickets.py``): rules, provider
outcomes, partial failures, connection release, parallel registrations, cancellation.
"""

import asyncio
import json
import logging
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.events_provider import ProviderRejectedError, ProviderUnavailableError
from app.repositories import TicketRepository
from app.services import (
    EventNotFoundError,
    EventNotPublishedError,
    InvalidSeatError,
    RegistrationClosedError,
    RegistrationRejectedError,
    SeatsCache,
    SeatUnavailableError,
    TicketNotSavedError,
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


@pytest.fixture
def provider() -> AsyncMock:
    mock = AsyncMock()
    mock.get_event_seats.return_value = ["A4", "A5"]
    mock.register.return_value = "t-1"
    return mock


@pytest.fixture
def cache() -> SeatsCache:
    return SeatsCache(ttl=30)


async def register(session, provider, cache, **overrides):
    return await register_ticket(session, provider, cache, **{**ARGS, **overrides})


# ---- success ----


async def test_register_returns_ticket_id_and_saves_it(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)

    ticket_id = await register(async_db, provider, cache)

    assert ticket_id == "t-1"
    [ticket] = await all_tickets(session_factory)  # committed, seen by a new session
    assert (ticket.ticket_id, ticket.event_id, ticket.seat) == ("t-1", "e1", "A5")
    assert (ticket.first_name, ticket.last_name) == ("Иван", "Иванов")
    assert ticket.email == "ivan@example.com"
    assert ticket.cancelled_at is None
    provider.register.assert_awaited_once_with(
        "e1", first_name="Иван", last_name="Иванов", seat="A5", email="ivan@example.com"
    )
    assert not async_db.in_transaction()


async def test_success_invalidates_seats_cache(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    cache.set("e1", ["A4", "A5"])

    await register(async_db, provider, cache)

    assert cache.get("e1") is None


async def test_seats_come_from_the_cache_when_it_is_fresh(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    cache.set("e1", ["A5"])

    await register(async_db, provider, cache)

    provider.get_event_seats.assert_not_called()


async def test_no_deadline_means_open_registration(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory, registration_deadline=None)

    assert await register(async_db, provider, cache) == "t-1"


# ---- local business checks ----


async def test_unknown_event(async_db, provider, cache):
    with pytest.raises(EventNotFoundError):
        await register(async_db, provider, cache)

    assert provider.mock_calls == []


async def test_unpublished_event(async_db, session_factory, provider, cache):
    await seed_event(session_factory, status="draft")

    with pytest.raises(EventNotPublishedError):
        await register(async_db, provider, cache)

    provider.register.assert_not_called()


async def test_expired_deadline(async_db, session_factory, provider, cache):
    await seed_event(session_factory, registration_deadline=PAST)

    with pytest.raises(RegistrationClosedError):
        await register(async_db, provider, cache)

    provider.register.assert_not_called()
    assert await all_tickets(session_factory) == []


async def test_deadline_boundary_is_compared_in_utc(
    async_db, session_factory, provider, cache
):
    # SQLite returns a naive value, which the service reads as UTC (as PostgreSQL's
    # timestamptz gives UTC-equivalent instants).
    await seed_event(
        session_factory, registration_deadline=datetime(2026, 1, 10, 7, 0, tzinfo=UTC)
    )

    with pytest.raises(RegistrationClosedError):
        await register(
            async_db, provider, cache, now=datetime(2026, 1, 10, 7, 30, tzinfo=UTC)
        )
    assert (
        await register(
            async_db, provider, cache, now=datetime(2026, 1, 10, 6, 30, tzinfo=UTC)
        )
        == "t-1"
    )


async def test_seat_outside_pattern(async_db, session_factory, provider, cache):
    await seed_event(session_factory, place_seats_pattern="A1-10")

    with pytest.raises(InvalidSeatError):
        await register(async_db, provider, cache, seat="A11")

    provider.register.assert_not_called()


async def test_seat_not_free_is_refused_without_registration(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    provider.get_event_seats.return_value = ["A4"]

    with pytest.raises(SeatUnavailableError):
        await register(async_db, provider, cache)

    provider.register.assert_not_called()
    assert await all_tickets(session_factory) == []


# ---- provider outcomes ----


async def test_provider_400_is_rejected_and_not_saved(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    cache.set("e1", ["A5"])
    provider.register.side_effect = ProviderRejectedError(400, "")

    with pytest.raises(RegistrationRejectedError):
        await register(async_db, provider, cache)

    assert await all_tickets(session_factory) == []
    assert cache.get("e1") is not None  # unsuccessful: cache untouched


async def test_provider_404_is_event_not_found(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    provider.register.side_effect = ProviderRejectedError(404, "")

    with pytest.raises(EventNotFoundError):
        await register(async_db, provider, cache)

    assert await all_tickets(session_factory) == []


@pytest.mark.parametrize("status", [401, 429, 500, 503])
async def test_other_provider_rejections_propagate(
    async_db, session_factory, provider, cache, status
):
    await seed_event(session_factory)
    cache.set("e1", ["A5"])
    provider.register.side_effect = ProviderRejectedError(status, "")

    with pytest.raises(ProviderRejectedError) as info:
        await register(async_db, provider, cache)

    assert info.value.status_code == status
    assert await all_tickets(session_factory) == []
    assert cache.get("e1") is not None


async def test_provider_unavailable_is_not_retried(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    provider.register.side_effect = ProviderUnavailableError("timeout")

    with pytest.raises(ProviderUnavailableError):
        await register(async_db, provider, cache)

    assert provider.register.await_count == 1
    assert await all_tickets(session_factory) == []


async def test_seats_check_failure_means_no_registration(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    provider.get_event_seats.side_effect = ProviderUnavailableError("down")

    with pytest.raises(ProviderUnavailableError):
        await register(async_db, provider, cache)

    provider.register.assert_not_called()


# ---- partial failure: provider confirmed, local save failed ----


async def test_local_save_failure_reports_ticket_id_and_logs_no_personal_data(
    async_db, session_factory, provider, cache, caplog, monkeypatch
):
    await seed_event(session_factory)
    cache.set("e1", ["A5"])

    async def broken_add(self, **fields):
        raise RuntimeError(f"db down for {fields['email']}")

    monkeypatch.setattr(TicketRepository, "add", broken_add)

    with caplog.at_level(logging.INFO), pytest.raises(TicketNotSavedError) as info:
        await register(async_db, provider, cache)

    assert info.value.ticket_id == "t-1"
    assert "ivan@example.com" not in str(info.value)
    assert provider.register.await_count == 1  # no automatic repeat
    assert cache.get("e1") is None  # the seat is taken at the provider
    assert await all_tickets(session_factory) == []
    assert not async_db.in_transaction()  # rolled back
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "t-1" in logged
    assert "ivan@example.com" not in logged
    assert "Иван" not in logged


async def test_save_failure_on_the_unique_index_keeps_the_existing_ticket(
    async_db, session_factory, provider, cache
):
    # The provider hands out an id that is still active here: the partial unique
    # index refuses the second row. A real database error, not a stub.
    await seed_event(session_factory)
    await add_ticket(session_factory, "t-1", seat="A4", first_name="Пётр")

    with pytest.raises(TicketNotSavedError):
        await register(async_db, provider, cache)

    [existing] = await all_tickets(session_factory)
    assert (existing.first_name, existing.seat) == ("Пётр", "A4")
    assert not async_db.in_transaction()


async def test_session_is_usable_after_a_failed_save(
    async_db, session_factory, provider, cache
):
    await seed_event(session_factory)
    await add_ticket(session_factory, "t-1", seat="A4")
    with pytest.raises(TicketNotSavedError):
        await register(async_db, provider, cache)

    provider.register.return_value = "t-2"
    assert await register(async_db, provider, cache, seat="A5") == "t-2"

    assert {t.ticket_id for t in await all_tickets(session_factory)} == {"t-1", "t-2"}


# ---- the database connection is not held during provider calls ----


async def test_connection_is_released_during_both_provider_calls(
    async_db, session_factory, cache
):
    await seed_event(session_factory)
    states = []

    async def seats(event_id):
        states.append(("seats", *db_state(async_db)))
        return ["A5"]

    async def reg(event_id, **kwargs):
        states.append(("register", *db_state(async_db)))
        return "t-1"

    provider = AsyncMock()
    provider.get_event_seats.side_effect = seats
    provider.register.side_effect = reg

    await register(async_db, provider, cache)

    assert states == [("seats", False, 0), ("register", False, 0)]


async def test_connection_is_released_on_every_early_exit(
    async_db, session_factory, provider, cache
):
    with pytest.raises(EventNotFoundError):
        await register(async_db, provider, cache)
    assert db_state(async_db) == (False, 0)

    await seed_event(session_factory, status="draft")
    with pytest.raises(EventNotPublishedError):
        await register(async_db, provider, cache)
    assert db_state(async_db) == (False, 0)


async def test_works_when_the_session_expires_objects_on_commit(
    async_engine, session_factory, provider, cache
):
    # The service reads what it needs before the release; it does not rely on
    # expire_on_commit=False.
    await seed_event(session_factory)

    async with AsyncSession(async_engine, expire_on_commit=True) as session:
        assert await register(session, provider, cache) == "t-1"

    assert len(await all_tickets(session_factory)) == 1


# ---- concurrency ----


async def test_two_registrations_for_one_seat_save_exactly_one_ticket(
    session_factory, cache
):
    # Both pass the local pre-filter (the seat looks free to both); the provider
    # decides: the first POST wins, the second one is refused with 400.
    await seed_event(session_factory)
    taken: set[str] = set()
    posts = 0

    async def provider_register(event_id, *, seat, **kwargs):
        nonlocal posts
        posts += 1
        await asyncio.sleep(0.05)  # both requests are inside the call together
        if seat in taken:
            raise ProviderRejectedError(400, "")
        taken.add(seat)
        return f"t-{posts}"

    provider = AsyncMock()
    provider.get_event_seats.return_value = ["A5"]
    provider.register.side_effect = provider_register

    async def one():
        async with session_factory() as session:
            return await register(session, provider, cache)

    results = await asyncio.gather(one(), one(), return_exceptions=True)

    assert sorted(type(r).__name__ for r in results) == [
        "RegistrationRejectedError",
        "str",
    ]
    assert provider.register.await_count == 2  # each POST sent once
    tickets = await all_tickets(session_factory)
    assert len(tickets) == 1
    assert tickets[0].seat == "A5"


async def test_registrations_for_different_seats_are_all_saved(session_factory, cache):
    await seed_event(session_factory, place_seats_pattern="A1-100")

    async def provider_register(event_id, *, seat, **kwargs):
        await asyncio.sleep(0.05)
        return f"t-{seat}"

    provider = AsyncMock()
    provider.get_event_seats.return_value = [f"A{i}" for i in range(1, 101)]
    provider.register.side_effect = provider_register

    async def one(seat):
        async with session_factory() as session:
            return await register(session, provider, cache, seat=seat)

    results = await asyncio.gather(*(one(f"A{i}") for i in range(1, 6)))

    assert sorted(results) == [f"t-A{i}" for i in range(1, 6)]
    assert len(await all_tickets(session_factory)) == 5


async def test_more_slow_registrations_than_pool_connections_all_finish(
    async_engine, session_factory, cache
):
    pool = async_engine.pool
    assert pool.size() + pool._max_overflow == 15
    count = 24
    await seed_event(session_factory, place_seats_pattern="A1-100")
    active = peak = 0

    async def provider_register(event_id, *, seat, **kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.3)
        active -= 1
        return f"t-{seat}"

    provider = AsyncMock()
    provider.get_event_seats.return_value = [f"A{i}" for i in range(1, 101)]
    provider.register.side_effect = provider_register

    async def one(i):
        async with session_factory() as session:
            return await register(session, provider, cache, seat=f"A{i}")

    results = await asyncio.wait_for(
        asyncio.gather(*(one(i) for i in range(1, count + 1))), 20
    )

    assert len(results) == count
    assert peak == count  # all POSTs were waiting together, none parked behind the pool
    assert len(await all_tickets(session_factory)) == count
    assert pool.checkedout() == 0


async def test_slow_provider_does_not_block_the_event_loop(
    async_db, session_factory, cache
):
    await seed_event(session_factory)

    async def slow_register(event_id, **kwargs):
        await asyncio.sleep(0.3)
        return "t-1"

    provider = AsyncMock()
    provider.get_event_seats.return_value = ["A5"]
    provider.register.side_effect = slow_register

    async with Ticker() as ticker:
        await register(async_db, provider, cache)

    assert ticker.count >= 10


# ---- cancellation of the task ----


async def test_cancelled_while_the_provider_is_asked_saves_nothing(
    async_db, session_factory, cache
):
    await seed_event(session_factory)
    cache.set("e1", ["A5"])
    waiting = asyncio.Event()

    async def never_answers(event_id, **kwargs):
        waiting.set()
        await asyncio.Event().wait()

    provider = AsyncMock()
    provider.get_event_seats.return_value = ["A5"]
    provider.register.side_effect = never_answers

    task = asyncio.create_task(register(async_db, provider, cache))
    await asyncio.wait_for(waiting.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert db_state(async_db) == (False, 0)
    assert await all_tickets(session_factory) == []
    assert cache.get("e1") is not None  # the outcome is unknown: cache untouched


async def test_cancelled_during_the_local_save_rolls_back_and_logs_the_ticket(
    async_db, session_factory, provider, cache, caplog, monkeypatch
):
    await seed_event(session_factory)
    saving = asyncio.Event()
    add = TicketRepository.add

    async def hanging_add(self, **fields):
        await add(self, **fields)  # flushed, not committed
        saving.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(TicketRepository, "add", hanging_add)

    with caplog.at_level(logging.INFO):
        task = asyncio.create_task(register(async_db, provider, cache))
        await asyncio.wait_for(saving.wait(), 5)
        assert async_db.in_transaction()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert db_state(async_db) == (False, 0)  # nothing half saved, nothing open
    assert await all_tickets(session_factory) == []
    assert cache.get("e1") is None  # the seat is taken at the provider
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "t-1" in logged  # the ticket can be found and repaired by hand
    assert "ivan@example.com" not in logged


# ---- the real async client (MockTransport) ----


async def test_real_client_sends_the_exact_contract_once(
    async_db, session_factory, cache
):
    await seed_event(session_factory)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"seats": ["A5"]})
        return httpx.Response(201, json={"ticket_id": "1fed0122"})

    client, seen = real_client(handler)

    assert await register(async_db, client, cache) == "1fed0122"

    post = [r for r in seen if r.method == "POST"]
    assert len(post) == 1
    assert post[0].url.path == "/api/events/e1/register/"
    assert post[0].headers["x-api-key"] == SECRET
    assert json.loads(post[0].content) == {
        "first_name": "Иван",
        "last_name": "Иванов",
        "seat": "A5",
        "email": "ivan@example.com",
    }
    assert {r.url.host for r in seen} == {"provider"}


async def test_real_client_timeout_is_not_retried_and_leaks_nothing(
    async_db, session_factory, cache
):
    await seed_event(session_factory)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"seats": ["A5"]})
        raise httpx.ReadTimeout("timeout", request=request)

    client, seen = real_client(handler)

    with pytest.raises(ProviderUnavailableError) as info:
        await register(async_db, client, cache)

    assert [r.method for r in seen].count("POST") == 1
    for secret in (SECRET, "ivan@example.com", "Иван"):
        assert secret not in str(info.value)
    assert await all_tickets(session_factory) == []


async def test_real_client_error_body_with_personal_data_is_not_kept(
    async_db, session_factory, cache
):
    await seed_event(session_factory)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"seats": ["A5"]})
        return httpx.Response(500, text="<html>ivan@example.com</html>")

    client, _ = real_client(handler)

    with pytest.raises(ProviderRejectedError) as info:
        await register(async_db, client, cache)

    leaked = str(info.value) + info.value.body
    assert "ivan@example.com" not in leaked
    assert SECRET not in leaked


async def test_real_client_redirect_is_not_followed(async_db, session_factory, cache):
    await seed_event(session_factory)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            return httpx.Response(200, json={"seats": ["A5"]})
        return httpx.Response(302, headers={"location": "http://evil.example/x"})

    client, seen = real_client(handler)

    with pytest.raises(ProviderRejectedError):
        await register(async_db, client, cache)

    assert {r.url.host for r in seen} == {"provider"}
