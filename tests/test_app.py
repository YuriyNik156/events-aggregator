"""The application as a whole: routes, dependencies, lifespan.

The requests go through the real ASGI stack (``httpx.ASGITransport``, same event
loop as the test), the database is the pooled SQLite file of ``async_engine`` and the
provider is the real ``EventsProviderClient`` on a ``MockTransport`` whose
answers the test releases with events. No test here depends on elapsed time.
"""

import asyncio
import inspect
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

import main
from app import database, events_provider, repositories, services, sync_worker
from app.events_provider import EventsProviderClient
from app.services import SeatsCache
from tests.conftest import make_event
from tests.support import (
    SECRET,
    Provider,
    add_ticket,
    all_tickets,
    seed_event,
)

pytestmark = pytest.mark.anyio

FUTURE = datetime.now(UTC) + timedelta(days=30)
TICKET = {
    "event_id": "e1",
    "first_name": "Иван",
    "last_name": "Иванов",
    "email": "ivan@example.com",
    "seat": "A5",
}


@pytest.fixture
def provider(monkeypatch) -> Provider:
    """The provider behind the process-wide client that the routes really use."""
    gate = Provider()
    client = EventsProviderClient(
        "http://provider:8000",
        SECRET,
        http_client=httpx.AsyncClient(
            base_url="http://provider:8000", transport=httpx.MockTransport(gate)
        ),
    )
    monkeypatch.setattr(events_provider, "_shared_client", client)
    return gate


@pytest.fixture
def cache():
    fresh = SeatsCache(ttl=30)
    main.app.dependency_overrides[main.get_seats_cache] = lambda: fresh
    yield fresh
    main.app.dependency_overrides.pop(main.get_seats_cache, None)


@pytest.fixture
def sessions(session_factory):
    """``main.get_session`` on the pooled engine; every session handed out is kept."""
    handed_out = []

    async def get_session():
        async with session_factory() as session:
            handed_out.append(session)
            yield session

    main.app.dependency_overrides[main.get_session] = get_session
    yield handed_out
    main.app.dependency_overrides.pop(main.get_session, None)


@pytest.fixture
async def http(provider, cache, sessions):
    transport = httpx.ASGITransport(app=main.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    assert services._cancelling == set()
    assert services._sync_running is False


async def seed_events(factory, count: int) -> None:
    async with factory() as s:
        repo = services.EventRepository(s)
        for i in range(count):
            await repo.upsert(
                make_event(
                    f"e{i}",
                    registration_deadline=FUTURE,
                    event_time=FUTURE + timedelta(days=1),
                )
            )
        await s.commit()


# ---- nothing synchronous is left on the request path ----


def _calls(dependant):
    yield dependant.call
    for dependency in dependant.dependencies:
        yield from _calls(dependency)


def test_every_route_and_dependency_is_async():
    routes = [r for r in main.app.routes if isinstance(r, APIRoute)]
    assert len(routes) == 7
    for route in routes:
        for call in _calls(route.dependant):
            assert inspect.iscoroutinefunction(call) or inspect.isasyncgenfunction(
                call
            ), f"{route.path}: {call.__name__} would run in the thread pool"


def test_the_sync_database_and_provider_code_is_gone():
    # The blocking variants were deleted, not just left unused.
    for name in ("get_session_factory", "get_engine", "dispose_engine"):
        assert not hasattr(database, name), name
    for name in ("Session", "sessionmaker", "create_engine"):
        assert not hasattr(main, name), name
        assert not hasattr(services, name), name
    assert not hasattr(events_provider, "AsyncEventsProviderClient")
    assert "to_thread" not in inspect.getsource(sync_worker)


def _public_methods(cls):
    return [
        member
        for name, member in inspect.getmembers(cls, inspect.isfunction)
        if not name.startswith("_")
    ]


def test_everything_that_waits_is_a_coroutine():
    # Repositories, the provider client and the services: no blocking twin is left.
    for cls in (
        repositories.EventRepository,
        repositories.TicketRepository,
        repositories.SyncMetadataRepository,
        events_provider.EventsProviderClient,
    ):
        for method in _public_methods(cls):
            assert inspect.iscoroutinefunction(method), f"{cls.__name__}.{method}"
    for name in (
        "sync_events",
        "get_available_seats",
        "register_ticket",
        "cancel_ticket",
    ):
        function = getattr(main, name)
        assert function is getattr(services, name)
        assert function.__module__ == "app.services"
        assert inspect.iscoroutinefunction(function)
    assert inspect.iscoroutinefunction(sync_worker.run_sync_once)
    assert inspect.iscoroutinefunction(sync_worker.seconds_until_due)
    assert inspect.isasyncgenfunction(events_provider.EventsPaginator.__aiter__)


def test_routes_raise_the_domain_exceptions_of_the_shared_module():
    # One class per error, so a handler written for one is reached by the other.
    for name in (
        "EventNotFoundError",
        "TicketNotSavedError",
        "CancellationInProgressError",
        "SyncAlreadyRunningError",
    ):
        assert getattr(main, name) is getattr(services, name)


# ---- a slow provider does not hold the application up ----


async def test_slow_provider_does_not_block_health(http, provider, session_factory):
    await seed_events(session_factory, 1)
    provider.hold(1)

    slow = asyncio.create_task(http.get("/api/events/e0/seats"))
    await asyncio.wait_for(provider.all_entered.wait(), 5)

    health = await asyncio.wait_for(http.get("/api/health"), 5)

    assert health.status_code == 200 and health.json() == {"status": "ok"}
    assert not slow.done()  # the provider has not answered yet
    provider.release.set()
    answer = await asyncio.wait_for(slow, 5)
    assert answer.status_code == 200
    assert answer.json() == {"event_id": "e0", "available_seats": ["A4", "A5"]}


async def test_health_is_served_when_more_requests_than_threads_wait_for_the_provider(
    http, provider, session_factory, async_engine
):
    # 60 requests wait for the provider at once: more than the 40 worker threads
    # of a sync application, and more than the 15 connections of the pool.
    count = 60
    assert async_engine.pool.size() + async_engine.pool._max_overflow == 15
    await seed_events(session_factory, count)
    provider.hold(count)

    waiting = [
        asyncio.create_task(http.get(f"/api/events/e{i}/seats")) for i in range(count)
    ]
    await asyncio.wait_for(provider.all_entered.wait(), 10)

    assert provider.waiting == count
    health = await asyncio.wait_for(http.get("/api/health"), 5)
    assert health.status_code == 200
    assert async_engine.pool.checkedout() == 0  # no request holds a DB connection
    other = await asyncio.wait_for(http.get("/api/events"), 5)  # the DB works too
    assert other.status_code == 200 and other.json()["count"] == count

    provider.release.set()
    answers = await asyncio.wait_for(asyncio.gather(*waiting), 10)
    assert [a.status_code for a in answers] == [200] * count


async def test_registration_waiting_for_the_provider_holds_no_db_connection(
    http, provider, session_factory, async_engine
):
    await seed_event(session_factory)
    provider.hold(1)  # the seats check waits first

    registering = asyncio.create_task(http.post("/api/tickets", json=TICKET))
    await asyncio.wait_for(provider.all_entered.wait(), 5)
    assert async_engine.pool.checkedout() == 0
    provider.release.set()

    answer = await asyncio.wait_for(registering, 5)
    assert answer.status_code == 201 and answer.json() == {"ticket_id": "t-1"}
    assert async_engine.pool.checkedout() == 0


async def test_second_manual_sync_is_409_while_the_first_waits_for_the_provider(
    http, provider, session_factory
):
    provider.hold(1)

    first = asyncio.create_task(http.post("/api/sync/trigger"))
    await asyncio.wait_for(provider.all_entered.wait(), 5)

    second = await asyncio.wait_for(http.post("/api/sync/trigger"), 5)

    assert second.status_code == 409
    assert second.json() == {"detail": "Synchronization is already running"}
    provider.release.set()
    done = await asyncio.wait_for(first, 5)
    assert done.status_code == 200 and done.json() == {"success": True}
    assert len(provider.requests) == 1  # the refused trigger never reached the provider
    assert (await http.get("/api/events")).json()["count"] == 1


# ---- sessions: one per request, always closed ----


async def test_concurrent_requests_get_separate_sessions_that_end_clean(
    http, provider, session_factory, sessions, async_engine
):
    await seed_events(session_factory, 3)
    provider.hold(3)

    waiting = [
        asyncio.create_task(http.get(f"/api/events/e{i}/seats")) for i in range(3)
    ]
    await asyncio.wait_for(provider.all_entered.wait(), 5)

    assert len(sessions) == 3
    assert len({id(s) for s in sessions}) == 3  # nobody shares a session
    provider.release.set()
    await asyncio.wait_for(asyncio.gather(*waiting), 5)

    assert all(not s.in_transaction() for s in sessions)
    assert async_engine.pool.checkedout() == 0


async def test_session_is_closed_when_the_route_fails(
    http, provider, session_factory, sessions, async_engine
):
    provider.respond = lambda request: httpx.Response(500, text="boom")
    await seed_events(session_factory, 1)

    answer = await http.get("/api/events/e0/seats")

    assert answer.status_code == 502
    assert len(sessions) == 1 and not sessions[0].in_transaction()
    assert async_engine.pool.checkedout() == 0


# ---- the shared provider client ----


async def test_provider_client_is_created_once_and_closed_on_request(monkeypatch):
    monkeypatch.setenv("EVENTS_PROVIDER_API_KEY", SECRET)

    first = await main.get_provider_client()
    second = await main.get_provider_client()

    assert first is second
    assert not first.http_client.is_closed
    assert first.http_client.follow_redirects is False
    await events_provider.close_shared_client()
    assert first.http_client.is_closed
    assert events_provider._shared_client is None
    await events_provider.close_shared_client()  # nothing left: no error


async def test_requests_do_not_close_the_shared_client(http, provider, session_factory):
    await seed_events(session_factory, 1)
    client = events_provider._shared_client

    for _ in range(3):
        await http.post("/api/sync/trigger")
        assert not client.http_client.is_closed
        assert events_provider._shared_client is client

    assert len(provider.requests) == 3  # all three went through the same client


async def test_missing_api_key_is_a_500_and_recovers_when_it_appears(monkeypatch):
    monkeypatch.setattr(events_provider, "_shared_client", None)
    monkeypatch.delenv("EVENTS_PROVIDER_API_KEY", raising=False)

    with pytest.raises(main.HTTPException) as info:
        await main.get_provider_client()
    assert info.value.status_code == 500
    assert info.value.detail == "Events Provider is not configured"
    assert events_provider._shared_client is None  # nothing half-built is kept

    monkeypatch.setenv("EVENTS_PROVIDER_API_KEY", SECRET)
    assert await main.get_provider_client() is events_provider._shared_client
    await events_provider.close_shared_client()


# ---- cancellation of a request ----


async def test_cancelled_request_leaves_nothing_behind(
    http, provider, session_factory, sessions, async_engine
):
    await seed_events(session_factory, 1)
    provider.hold(1)
    request = asyncio.create_task(http.get("/api/events/e0/seats"))
    await asyncio.wait_for(provider.all_entered.wait(), 5)

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert not sessions[0].in_transaction()
    assert async_engine.pool.checkedout() == 0
    provider.release.set()  # let the (cancelled) provider answer go
    assert (await http.get("/api/health")).status_code == 200  # the app carries on


async def test_cancelled_ticket_cancellation_frees_the_guard_and_keeps_the_ticket(
    http, provider, session_factory, sessions
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    provider.hold(1)
    request = asyncio.create_task(http.delete("/api/tickets/t-1"))
    await asyncio.wait_for(provider.all_entered.wait(), 5)
    assert services._cancelling == {"t-1"}

    request.cancel()
    with pytest.raises(asyncio.CancelledError):
        await request

    assert services._cancelling == set()
    assert (await all_tickets(session_factory))[0].cancelled_at is None
    assert not sessions[0].in_transaction()


# ---- the contract end to end, with the real client and database ----


async def test_register_and_cancel_end_to_end(http, provider, session_factory):
    await seed_event(session_factory)

    created = await http.post("/api/tickets", json=TICKET)
    assert created.status_code == 201 and created.json() == {"ticket_id": "t-1"}
    [ticket] = await all_tickets(session_factory)
    assert (ticket.ticket_id, ticket.seat, ticket.cancelled_at) == ("t-1", "A5", None)

    deleted = await http.delete("/api/tickets/t-1")
    assert deleted.status_code == 200 and deleted.json() == {"success": True}
    assert (await all_tickets(session_factory))[0].cancelled_at is not None
    assert (await http.delete("/api/tickets/t-1")).status_code == 404

    sent = [(r.method, r.url.path) for r in provider.requests]
    assert sent == [
        ("GET", "/api/events/e1/seats/"),
        ("POST", "/api/events/e1/register/"),
        ("DELETE", "/api/events/e1/unregister/"),
    ]  # each mutation exactly once
    assert all(r.headers["x-api-key"] == SECRET for r in provider.requests)
    assert {r.url.host for r in provider.requests} == {"provider"}


async def test_error_answers_do_not_contain_the_key_or_personal_data(
    http, provider, session_factory
):
    await seed_event(session_factory)
    provider.respond = lambda request: (
        httpx.Response(200, json={"seats": ["A5"]})
        if request.method == "GET"
        else httpx.Response(500, text="<html>ivan@example.com</html>")
    )

    answer = await http.post("/api/tickets", json=TICKET)

    assert answer.status_code == 502
    assert answer.json() == {"detail": "Events Provider request failed"}
    for secret in (SECRET, "ivan@example.com", "Иван", "<html>"):
        assert secret not in answer.text


@pytest.mark.parametrize(
    ("method", "url", "body"),
    [
        ("GET", "/api/events/e1/seats", None),
        ("POST", "/api/tickets", TICKET),
        ("DELETE", "/api/tickets/t-1", None),
        ("POST", "/api/sync/trigger", None),
    ],
)
async def test_redirect_from_the_provider_is_a_502_and_the_key_stays_home(
    http, provider, session_factory, method, url, body
):
    await seed_event(session_factory)
    await add_ticket(session_factory)
    provider.respond = lambda request: (
        httpx.Response(200, json={"seats": ["A5"]})
        if request.method == "GET" and "tickets" in url
        else httpx.Response(302, headers={"location": "http://evil.example/x"})
    )

    answer = await http.request(method, url, json=body)

    assert answer.status_code == 502
    assert {r.url.host for r in provider.requests} == {"provider"}  # never evil.example
    assert SECRET not in answer.text


# ---- lifespan: startup and shutdown ----


class StoppingWorker:
    instances: list["StoppingWorker"] = []
    stopped_cleanly = True

    def __init__(self, interval):
        self.interval = interval
        StoppingWorker.instances.append(self)

    def start(self):
        pass

    async def stop(self):
        return StoppingWorker.stopped_cleanly


@pytest.fixture
def workers(monkeypatch):
    StoppingWorker.instances = []
    StoppingWorker.stopped_cleanly = True
    monkeypatch.setattr(main, "SyncWorker", StoppingWorker)
    return StoppingWorker


def shared_client_to_close() -> EventsProviderClient:
    return EventsProviderClient(
        "http://provider:8000",
        SECRET,
        http_client=httpx.AsyncClient(
            base_url="http://provider:8000",
            transport=httpx.MockTransport(lambda r: httpx.Response(200)),
        ),
    )


def test_app_starts_and_stops_with_the_background_sync_off(monkeypatch):
    # SYNC_INTERVAL_SECONDS=0 comes from conftest; no engine or client exists yet.
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", "postgresql://u:p@db.invalid/x")
    engine = None

    with TestClient(main.app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        engine = database.get_async_engine()  # created lazily; it does not connect
        assert database._async_engine is engine

    assert database._async_engine is None  # disposed at shutdown
    assert events_provider._shared_client is None


def test_shutdown_closes_the_client_and_then_the_engine(monkeypatch):
    client = shared_client_to_close()
    monkeypatch.setattr(events_provider, "_shared_client", client)
    order = []
    real_close = events_provider.close_shared_client

    async def closing():
        await real_close()
        order.append("client")

    async def disposing():
        order.append("engine")

    monkeypatch.setattr(main, "close_shared_client", closing)
    monkeypatch.setattr(main, "dispose_async_engine", disposing)

    with TestClient(main.app):
        assert not client.http_client.is_closed  # open while the app runs
        assert order == []

    assert client.http_client.is_closed
    assert events_provider._shared_client is None
    assert order == ["client", "engine"]


def test_engine_is_disposed_even_if_closing_the_client_fails(monkeypatch):
    disposed = AsyncMock()

    async def failing():
        raise RuntimeError("close failed")

    monkeypatch.setattr(main, "close_shared_client", failing)
    monkeypatch.setattr(main, "dispose_async_engine", disposed)

    with pytest.raises(RuntimeError, match="close failed"), TestClient(main.app):
        pass

    disposed.assert_awaited_once()


def test_nothing_is_closed_under_a_worker_that_is_still_stopping(monkeypatch, workers):
    # Active tasks must not lose their client or engine.
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "3600")
    workers.stopped_cleanly = False
    client = shared_client_to_close()
    monkeypatch.setattr(events_provider, "_shared_client", client)
    disposed = AsyncMock()
    monkeypatch.setattr(main, "dispose_async_engine", disposed)

    with TestClient(main.app):
        pass

    assert not client.http_client.is_closed
    disposed.assert_not_called()


def test_invalid_interval_still_fails_at_startup(
    monkeypatch,
):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "10")

    with pytest.raises(main.ConfigError), TestClient(main.app):
        pass
