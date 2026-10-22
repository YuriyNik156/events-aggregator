"""``get_available_seats`` (the routes are in ``test_seats.py``), and the checks
that the database connection is not held while the provider answers.
"""

import asyncio
from unittest.mock import AsyncMock

import httpx
import pytest

from app.events_provider import (
    EventsProviderClient,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.repositories import EventRepository
from app.services import (
    EventNotFoundError,
    EventNotPublishedError,
    SeatsCache,
    get_available_seats,
)
from tests.conftest import make_event

pytestmark = pytest.mark.anyio

SECRET = "super-secret-key"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def provider() -> AsyncMock:
    return AsyncMock()


@pytest.fixture
def cache(clock) -> SeatsCache:
    return SeatsCache(ttl=30, clock=clock)


async def seed(factory, *events: dict) -> None:
    async with factory() as s:
        repo = EventRepository(s)
        for e in events:
            await repo.upsert(e)
        await s.commit()


# ---- service ----


async def test_seats_success(async_db, session_factory, provider, cache):
    await seed(session_factory, make_event("e1"))
    provider.get_event_seats.return_value = ["A1", "A3", "B1"]

    seats = await get_available_seats(async_db, provider, cache, "e1")

    assert seats == ["A1", "A3", "B1"]
    provider.get_event_seats.assert_awaited_once_with("e1")


async def test_unknown_event_without_provider_call(async_db, provider, cache):
    with pytest.raises(EventNotFoundError):
        await get_available_seats(async_db, provider, cache, "missing")

    assert provider.mock_calls == []


async def test_unpublished_event_without_provider_call(
    async_db, session_factory, provider, cache
):
    await seed(session_factory, make_event("e1", status="draft"))

    with pytest.raises(EventNotPublishedError):
        await get_available_seats(async_db, provider, cache, "e1")

    assert provider.mock_calls == []


async def test_empty_seats_returned_and_cached(
    async_db, session_factory, provider, cache
):
    await seed(session_factory, make_event("e1"))
    provider.get_event_seats.return_value = []

    first = await get_available_seats(async_db, provider, cache, "e1")
    second = await get_available_seats(async_db, provider, cache, "e1")

    assert first == second == []
    assert provider.get_event_seats.await_count == 1


async def test_cache_hit_within_ttl_and_refresh_after(
    async_db, session_factory, provider, clock, cache
):
    await seed(session_factory, make_event("e1"))
    provider.get_event_seats.side_effect = [["A1"], ["A2"]]

    assert await get_available_seats(async_db, provider, cache, "e1") == ["A1"]
    clock.now += 29.9
    assert await get_available_seats(async_db, provider, cache, "e1") == ["A1"]
    assert provider.get_event_seats.await_count == 1
    clock.now += 0.2
    assert await get_available_seats(async_db, provider, cache, "e1") == ["A2"]
    assert provider.get_event_seats.await_count == 2


async def test_cache_is_per_event(async_db, session_factory, provider, cache):
    await seed(session_factory, make_event("e1"), make_event("e2"))
    provider.get_event_seats.side_effect = lambda eid: [f"{eid}-A"]

    assert await get_available_seats(async_db, provider, cache, "e1") == ["e1-A"]
    assert await get_available_seats(async_db, provider, cache, "e2") == ["e2-A"]
    assert provider.get_event_seats.await_count == 2


async def test_provider_error_is_not_cached_or_turned_into_empty(
    async_db, session_factory, provider, cache
):
    await seed(session_factory, make_event("e1"))
    provider.get_event_seats.side_effect = [ProviderUnavailableError("down"), ["A1"]]

    with pytest.raises(ProviderUnavailableError):
        await get_available_seats(async_db, provider, cache, "e1")
    assert cache.get("e1") is None

    assert await get_available_seats(async_db, provider, cache, "e1") == ["A1"]


@pytest.mark.parametrize("status", [401, 429, 500, 503])
async def test_provider_rejection_propagates(
    async_db, session_factory, provider, cache, status
):
    await seed(session_factory, make_event("e1"))
    provider.get_event_seats.side_effect = ProviderRejectedError(status, "<html>")

    with pytest.raises(ProviderRejectedError) as info:
        await get_available_seats(async_db, provider, cache, "e1")

    assert info.value.status_code == status
    assert cache.get("e1") is None


async def test_provider_404_is_event_not_found_and_drops_cache(
    async_db, session_factory, provider, cache
):
    await seed(session_factory, make_event("e1"))
    cache.set("e1", ["A1"])
    cache._items["e1"] = (0.0, ["A1"])  # expired, so the provider is asked again
    provider.get_event_seats.side_effect = ProviderRejectedError(404, "nope")

    with pytest.raises(EventNotFoundError):
        await get_available_seats(async_db, provider, cache, "e1")

    assert cache.get("e1") is None


# ---- the database connection is not held while the provider answers ----


async def test_pool_counter_sees_an_open_transaction(async_db, session_factory):
    # Control for the tests below: the counter does show a held connection.
    await seed(session_factory, make_event("e1"))
    engine = async_db.bind

    await EventRepository(async_db).get("e1")  # a read starts a transaction

    assert async_db.in_transaction()
    assert engine.pool.checkedout() == 1
    await async_db.commit()
    assert engine.pool.checkedout() == 0


async def test_connection_is_released_before_the_provider_is_asked(
    async_db, session_factory, cache
):
    await seed(session_factory, make_event("e1"))
    engine = async_db.bind
    during_request = []

    async def seats(event_id):
        during_request.append((async_db.in_transaction(), engine.pool.checkedout()))
        return ["A1"]

    provider = AsyncMock()
    provider.get_event_seats.side_effect = seats

    assert await get_available_seats(async_db, provider, cache, "e1") == ["A1"]

    assert during_request == [(False, 0)]


async def test_connection_is_released_on_every_early_exit(
    async_db, session_factory, provider, cache
):
    await seed(session_factory, make_event("d", status="draft"))
    engine = async_db.bind

    with pytest.raises(EventNotFoundError):
        await get_available_seats(async_db, provider, cache, "missing")
    assert engine.pool.checkedout() == 0
    with pytest.raises(EventNotPublishedError):
        await get_available_seats(async_db, provider, cache, "d")
    assert engine.pool.checkedout() == 0
    assert not async_db.in_transaction()


async def test_works_when_the_session_expires_objects_on_commit(
    async_engine, session_factory, provider, cache
):
    # The service must not rely on expire_on_commit=False: it reads what it needs
    # before it releases the connection.
    from sqlalchemy.ext.asyncio import AsyncSession

    await seed(session_factory, make_event("e1"))
    provider.get_event_seats.return_value = ["A1"]

    async with AsyncSession(async_engine, expire_on_commit=True) as session:
        assert await get_available_seats(session, provider, cache, "e1") == ["A1"]


async def test_more_slow_requests_than_pool_connections_all_finish(
    async_engine, session_factory, cache
):
    # The default pool has 5 + 10 connections. If a request kept its connection
    # while waiting for the provider, request 16 onwards would wait for the pool.
    pool = async_engine.pool
    assert pool.size() + pool._max_overflow == 15
    count = 40
    await seed(session_factory, *[make_event(f"e{i}") for i in range(count)])

    active = peak = 0

    async def slow_seats(event_id):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.3)
        active -= 1
        return [event_id]

    provider = AsyncMock()
    provider.get_event_seats.side_effect = slow_seats

    async def one(i: int):
        async with session_factory() as session:
            return await get_available_seats(session, provider, cache, f"e{i}")

    results = await asyncio.wait_for(
        asyncio.gather(*(one(i) for i in range(count))), 10
    )

    assert results == [[f"e{i}"] for i in range(count)]
    # All 40 provider calls were waiting at the same time: no request was parked
    # behind the 15 pooled connections while another waited for the provider.
    assert peak == count
    assert pool.checkedout() == 0


async def test_slow_provider_does_not_block_the_event_loop(
    async_db, session_factory, cache
):
    await seed(session_factory, make_event("e1"))

    async def slow_seats(event_id):
        await asyncio.sleep(0.3)
        return ["A1"]

    provider = AsyncMock()
    provider.get_event_seats.side_effect = slow_seats
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        await get_available_seats(async_db, provider, cache, "e1")
    finally:
        task.cancel()

    assert ticks >= 10


async def test_cancelled_request_leaves_no_transaction_and_no_cache_entry(
    async_db, session_factory, cache
):
    await seed(session_factory, make_event("e1"))
    waiting = asyncio.Event()

    async def never_answers(event_id):
        waiting.set()
        await asyncio.Event().wait()

    provider = AsyncMock()
    provider.get_event_seats.side_effect = never_answers

    task = asyncio.create_task(get_available_seats(async_db, provider, cache, "e1"))
    await asyncio.wait_for(waiting.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert not async_db.in_transaction()
    assert async_db.bind.pool.checkedout() == 0
    assert cache.get("e1") is None


# ---- through the real async client (MockTransport) ----


def real_client(handler) -> tuple[EventsProviderClient, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    http = httpx.AsyncClient(
        base_url="http://provider:8000", transport=httpx.MockTransport(record)
    )
    return EventsProviderClient("http://provider:8000", SECRET, http_client=http), seen


async def test_real_client_sends_key_only_to_the_provider(
    async_db, session_factory, cache
):
    await seed(session_factory, make_event("e1"))
    client, seen = real_client(lambda r: httpx.Response(200, json={"seats": ["A1"]}))

    assert await get_available_seats(async_db, client, cache, "e1") == ["A1"]

    assert seen[0].url.path == "/api/events/e1/seats/"
    assert seen[0].headers["x-api-key"] == SECRET
    assert seen[0].url.host == "provider"


async def test_real_client_redirect_is_an_error_and_the_key_stays_home(
    async_db, session_factory, cache
):
    await seed(session_factory, make_event("e1"))
    client, seen = real_client(
        lambda r: httpx.Response(302, headers={"location": "http://evil.example/x"})
    )

    with pytest.raises(ProviderRejectedError):
        await get_available_seats(async_db, client, cache, "e1")

    assert [r.url.host for r in seen] == ["provider"]
    assert cache.get("e1") is None


async def test_real_client_errors_do_not_contain_the_key(
    async_db, session_factory, cache
):
    await seed(session_factory, make_event("e1"))

    def boom(request):
        raise httpx.ReadTimeout("timeout", request=request)

    client, _ = real_client(boom)

    with pytest.raises(ProviderUnavailableError) as info:
        await get_available_seats(async_db, client, cache, "e1")

    assert SECRET not in str(info.value)
