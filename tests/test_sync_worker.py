"""The background sync worker (async): schedule, backoff, shutdown, the one-sync-at-a-
time guard it shares with the manual trigger, and its place in the app lifespan."""

import asyncio
import time
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

import main
from app import events_provider, services, sync_worker
from app.config import ConfigError
from app.events_provider import (
    EventsPage,
    EventsProviderClient,
    ProviderUnavailableError,
)
from app.models import Event, SyncMetadata
from app.repositories import SyncMetadataRepository
from app.services import (
    SyncAlreadyRunningError,
    SyncResult,
    SyncStoppedError,
    sync_events,
)
from app.sync_worker import (
    STARTUP_DELAY,
    SyncWorker,
    run_sync_once,
    seconds_until_due,
)
from tests.conftest import (
    provider_event_json,
    provider_page,
    session_override,
)
from tests.support import SECRET, Provider

pytestmark = pytest.mark.anyio

DAY = 86400
RESULT = SyncResult(processed=0, changed_at_cursor=datetime(2000, 1, 1).date())


def scripted(*outcomes):
    """A ``run_once`` returning or raising the given outcomes in order."""
    calls = []

    async def run_once(stop):
        calls.append(time.monotonic())
        outcome = outcomes[len(calls) - 1] if len(calls) <= len(outcomes) else RESULT
        if isinstance(outcome, Exception):
            raise outcome
        return RESULT

    run_once.calls = calls
    return run_once


def record_sleeps(worker: SyncWorker, stop_after: int) -> list[float]:
    """Replace the wait: record the delays, no real time passes."""
    delays: list[float] = []

    async def fake_sleep(seconds: float) -> bool:
        delays.append(seconds)
        if len(delays) > stop_after:
            worker.request_stop()
            return True
        return False

    worker._sleep = fake_sleep
    return delays


async def drive(worker: SyncWorker) -> None:
    worker.start()
    await asyncio.wait_for(worker._task, 5)


def delay_of(*values: float):
    """A ``first_delay`` answering the given values in turn (the last one repeats)."""
    answers = list(values)

    async def first_delay(interval: float) -> float:
        return answers.pop(0) if len(answers) > 1 else answers[0]

    return first_delay


def make_worker(interval=DAY, *, run_once, first_delay=None, **options):
    return SyncWorker(
        interval, run_once=run_once, first_delay=first_delay or delay_of(0.0), **options
    )


# --- schedule ---------------------------------------------------------------


async def test_runs_repeatedly_and_stops_on_request():
    run_once = scripted()
    worker = make_worker(0.05, run_once=run_once, first_delay=delay_of(0.01))

    worker.start()
    await asyncio.sleep(0.3)
    assert await worker.stop() is True

    assert len(run_once.calls) >= 3


async def test_first_wait_comes_from_the_database_and_nothing_runs_before_it():
    run_once = scripted()
    worker = make_worker(run_once=run_once, first_delay=delay_of(1234.5))
    delays = record_sleeps(worker, stop_after=0)

    await drive(worker)

    assert delays == [1234.5]
    assert run_once.calls == []


async def test_next_run_is_counted_from_the_start_of_the_previous_one():
    async def slow_run(stop):
        await asyncio.sleep(0.2)
        return RESULT

    worker = make_worker(100.0, run_once=slow_run)
    delays = record_sleeps(worker, stop_after=1)

    await drive(worker)

    assert delays[0] == 0.0
    assert 99.0 < delays[1] < 99.85  # the run's own 0.2 s is not added on top


async def test_event_loop_stays_responsive_while_a_sync_runs():
    started = asyncio.Event()

    async def long_run(stop):
        started.set()
        await asyncio.sleep(0.4)
        return RESULT

    worker = make_worker(run_once=long_run)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    tick_task = asyncio.create_task(ticker())
    worker.start()
    await asyncio.wait_for(started.wait(), 5)
    before = ticks
    await asyncio.sleep(0.3)  # the sync is in progress all this time
    during = ticks - before
    await worker.stop()
    tick_task.cancel()

    assert during >= 15


# --- errors and backoff -----------------------------------------------------


async def test_provider_errors_back_off_then_reset_after_success():
    boom = ProviderUnavailableError("down")
    run_once = scripted(boom, boom, boom, boom, boom, boom, RESULT, boom)
    worker = make_worker(run_once=run_once)
    delays = record_sleeps(worker, stop_after=8)

    await drive(worker)

    assert delays[:7] == [0.0, 300, 600, 1200, 2400, 3600, 3600]
    assert DAY - 5 < delays[7] <= DAY  # back on the schedule after the success
    assert delays[8] == 300  # the backoff started over
    assert len(run_once.calls) == 8


async def test_backoff_never_exceeds_a_shorter_interval():
    boom = ProviderUnavailableError("down")
    worker = make_worker(100.0, run_once=scripted(boom, boom))
    delays = record_sleeps(worker, stop_after=2)

    await drive(worker)

    assert delays == [0.0, 100.0, 100.0]


async def test_any_exception_keeps_the_loop_alive():
    run_once = scripted(RuntimeError("db down"), ConfigError("no key"), RESULT)
    worker = make_worker(run_once=run_once)
    record_sleeps(worker, stop_after=3)

    await drive(worker)

    assert len(run_once.calls) == 3


async def test_running_sync_is_not_an_error_and_is_not_retried_at_once():
    run_once = scripted(SyncAlreadyRunningError("manual"), RESULT)
    worker = make_worker(run_once=run_once, first_delay=delay_of(0.0, 31.0))
    delays = record_sleeps(worker, stop_after=1)

    await drive(worker)

    # No backoff step; the wait comes from the database again.
    assert delays == [0.0, 31.0]
    assert len(run_once.calls) == 1


async def test_stopped_sync_ends_the_worker_without_retry():
    run_once = scripted(SyncStoppedError())
    worker = make_worker(run_once=run_once)

    await drive(worker)

    assert len(run_once.calls) == 1


# --- shutdown ---------------------------------------------------------------


async def test_stop_interrupts_the_wait_at_once():
    worker = make_worker(run_once=scripted(), first_delay=delay_of(3600.0))

    worker.start()
    await asyncio.sleep(0.05)
    started = time.monotonic()
    stopped = await worker.stop()

    assert stopped is True
    assert time.monotonic() - started < 1


async def test_stop_signals_the_running_sync_and_waits_for_it():
    seen = asyncio.Event()
    finished = asyncio.Event()

    async def cooperative(stop):
        seen.set()
        await asyncio.wait_for(stop.wait(), 5)
        finished.set()
        raise SyncStoppedError

    worker = make_worker(run_once=cooperative)
    worker.start()
    await asyncio.wait_for(seen.wait(), 5)

    assert await worker.stop() is True

    assert finished.is_set()
    assert worker._task.done()
    assert not worker._task.cancelled()  # it ended by itself, nobody cancelled it


async def test_stop_cancels_a_sync_that_does_not_notice_the_request():
    cancelled = asyncio.Event()
    waiting = asyncio.Event()

    async def blind(stop):
        waiting.set()
        try:
            await asyncio.sleep(60)  # e.g. a provider that never answers
        except asyncio.CancelledError:
            cancelled.set()
            raise

    worker = make_worker(run_once=blind)
    worker.start()
    await asyncio.wait_for(waiting.wait(), 5)

    assert await worker.stop(timeout=0.05) is True

    assert cancelled.is_set()
    assert worker._task.cancelled()  # CancelledError is not hidden by the worker


async def test_stop_reports_a_sync_that_cannot_be_cancelled_in_time():
    release = asyncio.Event()
    waiting = asyncio.Event()

    async def slow_to_unwind(stop):
        waiting.set()
        try:
            await asyncio.sleep(60)
        except asyncio.CancelledError:
            await release.wait()  # cleanup that takes its time

    worker = make_worker(run_once=slow_to_unwind)
    worker.start()
    await asyncio.wait_for(waiting.wait(), 5)

    assert await worker.stop(timeout=0.05, cancel_timeout=0.05) is False
    assert not worker._task.done()  # the caller must keep the client and the engine

    release.set()
    await asyncio.wait_for(worker._task, 5)
    assert worker._task.done()


async def test_stop_without_start_is_fine():
    assert await make_worker(run_once=scripted()).stop() is True


async def test_stop_twice_is_fine():
    worker = make_worker(run_once=scripted(), first_delay=delay_of(3600.0))
    worker.start()
    await asyncio.sleep(0.01)

    assert await worker.stop() is True
    assert await worker.stop() is True


# --- schedule from sync_metadata -------------------------------------------


@pytest.fixture
def factory(session_factory, monkeypatch):
    monkeypatch.setattr(
        sync_worker, "get_async_session_factory", lambda: session_factory
    )
    return session_factory


async def save_last_sync(factory, when: datetime, status: str = "success") -> None:
    async with factory() as session:
        await SyncMetadataRepository(session).save(
            sync_status=status, last_sync_time=when
        )
        await session.commit()


async def test_empty_database_starts_after_the_short_delay(factory):
    assert await seconds_until_due(DAY) == STARTUP_DELAY


async def test_fresh_sync_schedules_the_next_one_a_full_interval_later(factory):
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    await save_last_sync(factory, now - timedelta(hours=1))

    assert await seconds_until_due(DAY, now) == pytest.approx(23 * 3600)


async def test_overdue_sync_starts_after_the_short_delay(factory):
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    await save_last_sync(factory, now - timedelta(days=3))

    assert await seconds_until_due(DAY, now) == STARTUP_DELAY


async def test_almost_due_sync_still_waits_the_short_delay(factory):
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    await save_last_sync(factory, now - timedelta(seconds=DAY - 5))

    assert await seconds_until_due(DAY, now) == STARTUP_DELAY


async def test_last_sync_in_the_future_is_capped_to_one_interval(factory):
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    await save_last_sync(factory, now + timedelta(days=30))

    assert await seconds_until_due(DAY, now) == DAY


async def test_failed_attempt_does_not_postpone_the_next_one(factory):
    # A failure writes only the status; last_sync_time still points at the last
    # success, so after a restart the overdue sync is retried soon.
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    await save_last_sync(factory, now - timedelta(days=2))
    async with factory() as session:
        await SyncMetadataRepository(session).save(sync_status="error")
        await session.commit()

    assert await seconds_until_due(DAY, now) == STARTUP_DELAY


async def test_unreadable_database_does_not_stop_the_worker(monkeypatch):
    def broken():
        raise RuntimeError("db down")

    monkeypatch.setattr(sync_worker, "get_async_session_factory", broken)

    assert await seconds_until_due(DAY) == STARTUP_DELAY


async def test_schedule_lookup_gives_the_connection_back(factory, async_engine):
    await save_last_sync(factory, datetime.now(UTC))

    await seconds_until_due(DAY)

    assert async_engine.pool.checkedout() == 0


# --- resources of one run ---------------------------------------------------


class FakeClient:
    """Stands in for the process-wide provider client."""

    closed = False

    async def aclose(self):
        self.closed = True


class FakeSession:
    def __init__(self):
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True


@pytest.fixture
def fakes(monkeypatch):
    sessions: list[FakeSession] = []
    client = FakeClient()

    def make_session():
        sessions.append(FakeSession())
        return sessions[-1]

    monkeypatch.setattr(sync_worker, "get_shared_client", lambda: client)
    monkeypatch.setattr(sync_worker, "get_async_session_factory", lambda: make_session)
    return SimpleNamespace(client=client, sessions=sessions)


async def test_run_closes_its_session_and_leaves_the_shared_client_open(
    fakes, monkeypatch
):
    stop = asyncio.Event()
    seen = {}

    async def fake_sync(session, client, stop_arg):
        seen.update(session=session, client=client, stop=stop_arg)
        return RESULT

    monkeypatch.setattr(sync_worker, "sync_events", fake_sync)

    assert await run_sync_once(stop) is RESULT

    assert seen["stop"] is stop
    assert seen["client"] is fakes.client
    assert seen["session"].closed
    assert not fakes.client.closed  # the client belongs to the process


@pytest.mark.parametrize(
    "error",
    [
        ProviderUnavailableError("x"),
        SyncStoppedError(),
        RuntimeError("x"),
        asyncio.CancelledError(),
    ],
)
async def test_run_closes_its_session_on_error_and_on_cancellation(
    fakes, monkeypatch, error
):
    async def failing(session, client, stop):
        raise error

    monkeypatch.setattr(sync_worker, "sync_events", failing)

    with pytest.raises(type(error)):
        await run_sync_once()

    assert fakes.sessions[0].closed
    assert not fakes.client.closed


async def test_every_run_gets_a_session_of_its_own_and_the_same_client(
    fakes, monkeypatch
):
    seen = []

    async def fake_sync(session, client, stop):
        seen.append((session, client))
        return RESULT

    monkeypatch.setattr(sync_worker, "sync_events", fake_sync)

    await run_sync_once()
    await run_sync_once()

    assert seen[0][0] is not seen[1][0]
    assert seen[0][1] is seen[1][1] is fakes.client
    assert all(s.closed for s in fakes.sessions)


async def test_run_without_api_key_creates_nothing(fakes, monkeypatch):
    def no_key():
        raise ConfigError("EVENTS_PROVIDER_API_KEY is not set")

    monkeypatch.setattr(sync_worker, "get_shared_client", no_key)

    with pytest.raises(ConfigError):
        await run_sync_once()

    assert fakes.sessions == []


# --- one guard for manual and background syncs -------------------------------


@pytest.fixture
def stack(factory, monkeypatch):
    """The real worker path: the pooled database and a provider the test controls."""
    gate = Provider()
    client = EventsProviderClient(
        "http://provider:8000",
        SECRET,
        http_client=httpx.AsyncClient(
            base_url="http://provider:8000", transport=httpx.MockTransport(gate)
        ),
    )
    monkeypatch.setattr(events_provider, "_shared_client", client)
    return SimpleNamespace(gate=gate, factory=factory, client=client)


async def count_events(factory) -> int:
    async with factory() as s:
        return await s.scalar(select(func.count()).select_from(Event))


def one_run_then_signal():
    """The real ``run_sync_once``, plus an event set when the run is over."""
    done = asyncio.Event()

    async def run_once(stop):
        try:
            return await run_sync_once(stop)
        finally:
            done.set()

    return run_once, done


async def test_manual_sync_is_refused_while_the_worker_syncs(stack):
    stack.gate.hold(1)
    run_once, done = one_run_then_signal()
    worker = make_worker(DAY, run_once=run_once)
    worker.start()
    await asyncio.wait_for(stack.gate.all_entered.wait(), 5)
    other = AsyncMock()

    async with stack.factory() as manual_session:  # what POST /api/sync/trigger does
        with pytest.raises(SyncAlreadyRunningError):
            await sync_events(manual_session, other)

    other.get_events_page.assert_not_called()
    stack.gate.release.set()
    await asyncio.wait_for(done.wait(), 5)
    assert await worker.stop() is True
    assert await count_events(stack.factory) == 1
    async with stack.factory() as check:
        meta = await SyncMetadataRepository(check).get()
        assert meta.sync_status == "success"
    assert len(stack.gate.requests) == 1
    assert services._sync_running is False


async def test_background_sync_does_not_start_a_second_one_during_a_manual_sync(stack):
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_page(**kwargs):
        entered.set()
        await release.wait()
        return EventsPage.model_validate(provider_page([provider_event_json("e1")]))

    manual_client = AsyncMock()
    manual_client.get_events_page.side_effect = slow_page
    skipped = asyncio.Event()

    async def run_once(stop):
        try:
            return await run_sync_once(stop)
        except SyncAlreadyRunningError:
            skipped.set()
            raise

    worker = make_worker(DAY, run_once=run_once, first_delay=delay_of(0.0, 3600.0))
    waits: list[float] = []
    real_sleep = worker._sleep

    async def spy(seconds):
        waits.append(seconds)
        return await real_sleep(seconds)

    worker._sleep = spy

    async with stack.factory() as manual_session:
        manual = asyncio.create_task(sync_events(manual_session, manual_client))
        await asyncio.wait_for(entered.wait(), 5)
        worker.start()
        await asyncio.wait_for(skipped.wait(), 5)
        while len(waits) < 2:  # the worker has looked at the database again
            await asyncio.sleep(0)
        release.set()
        result = await manual

    assert await worker.stop() is True
    assert result.processed == 1
    assert stack.gate.requests == []  # the background run never asked the provider
    assert waits == [0.0, 3600.0]  # skipped, not failed: no backoff step
    assert await count_events(stack.factory) == 1


async def test_manual_trigger_over_http_is_409_while_the_worker_syncs(stack, factory):
    async def get_session():
        async with factory() as session:
            yield session

    main.app.dependency_overrides[main.get_session] = get_session
    try:
        stack.gate.hold(1)
        run_once, done = one_run_then_signal()
        worker = make_worker(DAY, run_once=run_once)
        worker.start()
        await asyncio.wait_for(stack.gate.all_entered.wait(), 5)
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://t") as http:
            answer = await http.post("/api/sync/trigger")

        assert answer.status_code == 409
        assert answer.json() == {"detail": "Synchronization is already running"}
        stack.gate.release.set()
        await asyncio.wait_for(done.wait(), 5)
        assert await worker.stop() is True
        assert await count_events(factory) == 1
    finally:
        main.app.dependency_overrides.clear()


async def test_provider_failure_in_the_real_run_backs_off_and_keeps_the_cursor(
    stack, factory
):
    stack.gate.respond = lambda request: httpx.Response(503, text="down")
    worker = SyncWorker(DAY, first_delay=delay_of(0.0))  # the real run_sync_once
    delays = record_sleeps(worker, stop_after=2)

    await drive(worker)

    assert delays == [0.0, 300, 600]  # backoff steps, the worker kept going
    assert len(stack.gate.requests) == 2  # one request per run
    async with factory() as check:
        meta = await SyncMetadataRepository(check).get()
        assert meta.sync_status == "error"
        assert meta.last_changed_at is None  # no cursor was invented
    assert await count_events(factory) == 0
    assert services._sync_running is False


# --- lifespan ---------------------------------------------------------------


class FakeWorker:
    instances: list["FakeWorker"] = []
    stopped_cleanly = True

    def __init__(self, interval):
        self.interval = interval
        self.started = False
        self.stop_calls = 0
        FakeWorker.instances.append(self)

    def start(self):
        self.started = True

    async def stop(self):
        self.stop_calls += 1
        return FakeWorker.stopped_cleanly


@pytest.fixture
def lifespan_fakes(monkeypatch):
    FakeWorker.instances = []
    FakeWorker.stopped_cleanly = True
    disposed = AsyncMock()
    monkeypatch.setattr(main, "SyncWorker", FakeWorker)
    monkeypatch.setattr(main, "dispose_async_engine", disposed)
    return disposed


def test_worker_is_off_when_interval_is_zero(lifespan_fakes):
    with TestClient(main.app):
        pass  # SYNC_INTERVAL_SECONDS=0 comes from conftest

    assert FakeWorker.instances == []
    lifespan_fakes.assert_awaited_once()


def test_worker_starts_with_the_app_and_stops_with_it(lifespan_fakes, monkeypatch):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "3600")

    with TestClient(main.app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        (worker,) = FakeWorker.instances
        assert worker.interval == 3600 and worker.started
        lifespan_fakes.assert_not_called()

    assert worker.stop_calls == 1
    lifespan_fakes.assert_awaited_once()


def test_default_interval_is_24_hours(lifespan_fakes, monkeypatch):
    monkeypatch.delenv("SYNC_INTERVAL_SECONDS")

    with TestClient(main.app):
        pass

    assert FakeWorker.instances[0].interval == DAY


def test_engine_is_kept_while_the_worker_has_not_stopped(lifespan_fakes, monkeypatch):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "3600")
    FakeWorker.stopped_cleanly = False

    with TestClient(main.app):
        pass

    lifespan_fakes.assert_not_called()


def test_interval_above_a_day_is_allowed_with_a_warning(
    lifespan_fakes, monkeypatch, caplog
):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", str(2 * DAY))

    with TestClient(main.app):
        pass

    assert FakeWorker.instances[0].interval == 2 * DAY
    assert "24 hours" in caplog.text


@pytest.mark.parametrize("value", ["-5", "10", "daily"])
def test_invalid_interval_fails_at_startup(lifespan_fakes, monkeypatch, value):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", value)

    with pytest.raises(ConfigError), TestClient(main.app):
        pass

    assert FakeWorker.instances == []


def test_manual_trigger_still_works_with_the_worker_on(lifespan_fakes, monkeypatch, db):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "3600")
    provider = AsyncMock()
    provider.get_events_page.return_value = EventsPage.model_validate(
        provider_page([provider_event_json("e1")])
    )
    main.app.dependency_overrides[main.get_session] = session_override(db)
    main.app.dependency_overrides[main.get_provider_client] = lambda: provider
    try:
        with TestClient(main.app) as client:
            response = client.post("/api/sync/trigger")
    finally:
        main.app.dependency_overrides.clear()

    assert db.scalar(select(func.count()).select_from(Event)) == 1
    assert response.status_code == 200
    assert response.json() == {"success": True}


# --- lifespan with the real worker ------------------------------------------


class QuickStopWorker(SyncWorker):
    """The real worker with a short wait before the backup cancellation, so a test
    need not wait 20 s. ``stop_timeout`` can be raised for the cooperative case."""

    instances: list["QuickStopWorker"] = []
    stop_timeout = 0.05

    def __init__(self, interval):
        super().__init__(interval)
        QuickStopWorker.instances.append(self)

    async def stop(self, timeout=None, cancel_timeout=5.0):
        return await super().stop(timeout or self.stop_timeout, cancel_timeout)


@pytest.fixture
def real_worker_app(db, monkeypatch):
    """The app with the real worker, the file database of ``db`` and a controlled
    provider. ``opened`` collects every session the worker opened."""
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "120")
    monkeypatch.setattr(sync_worker, "STARTUP_DELAY", 0.05)  # empty database: soon
    monkeypatch.setattr(main, "SyncWorker", QuickStopWorker)
    monkeypatch.setattr(QuickStopWorker, "stop_timeout", 0.05)
    QuickStopWorker.instances = []
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{db.info['path']}", poolclass=NullPool
    )
    opened: list[AsyncSession] = []

    def make_session():
        opened.append(AsyncSession(engine, expire_on_commit=False))
        return opened[-1]

    monkeypatch.setattr(sync_worker, "get_async_session_factory", lambda: make_session)
    gate = Provider()
    client = EventsProviderClient(
        "http://provider:8000",
        SECRET,
        http_client=httpx.AsyncClient(
            base_url="http://provider:8000", transport=httpx.MockTransport(gate)
        ),
    )
    monkeypatch.setattr(events_provider, "_shared_client", client)

    # What had already happened when the client and the engine were closed: the
    # worker task must be over by then, never still running on them.
    order: list[tuple[str, bool]] = []
    real_aclose = client.aclose

    async def recording_aclose():
        order.append(("client", QuickStopWorker.instances[0]._task.done()))
        await real_aclose()

    async def recording_dispose():
        order.append(("engine", QuickStopWorker.instances[0]._task.done()))
        await engine.dispose()

    client.aclose = recording_aclose
    monkeypatch.setattr(main, "dispose_async_engine", recording_dispose)
    return SimpleNamespace(gate=gate, client=client, opened=opened, db=db, order=order)


def wait_until(condition, seconds=10.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.02)
    return False


def event_count(db) -> int:
    return db.scalar(select(func.count()).select_from(Event))


def test_worker_runs_a_sync_after_startup_and_shuts_down_cleanly(real_worker_app):
    app = real_worker_app
    app.gate.respond = lambda request: httpx.Response(
        200, json=provider_page([provider_event_json("e1")])
    )

    with TestClient(main.app) as client:
        assert wait_until(lambda: event_count(app.db) == 1)  # scheduled and run
        assert client.get("/api/health").json() == {"status": "ok"}
        (worker,) = QuickStopWorker.instances
        assert worker._interval == 120 and not worker._task.done()
        assert not app.client.http_client.is_closed

    assert worker._task.done()
    assert app.client.http_client.is_closed  # closed after the worker, not before
    assert events_provider._shared_client is None
    assert app.opened and all(not s.in_transaction() for s in app.opened)
    assert {r.url.host for r in app.gate.requests} == {"provider"}
    assert all(r.headers["x-api-key"] == SECRET for r in app.gate.requests)
    assert app.order == [("client", True), ("engine", True)]  # after the worker


def test_shutdown_cancels_a_sync_that_waits_for_the_provider(real_worker_app):
    app = real_worker_app
    app.gate.hold(1)  # the provider never answers

    with TestClient(main.app):
        assert wait_until(lambda: app.gate.waiting == 1)  # the sync is in flight
        (worker,) = QuickStopWorker.instances

    assert worker._task.cancelled()  # stopped by the backup cancellation
    assert app.gate.waiting == 0  # the pending request was cancelled with it
    assert event_count(app.db) == 0
    assert app.db.scalar(select(func.count()).select_from(SyncMetadata)) == 0
    assert app.client.http_client.is_closed
    assert events_provider._shared_client is None
    assert services._sync_running is False
    assert all(not s.in_transaction() for s in app.opened)
    assert app.order == [("client", True), ("engine", True)]


def test_shutdown_between_two_events_rolls_back_without_cancelling(
    real_worker_app, monkeypatch
):
    # The sync is between two events when the app stops. With time to notice, it
    # sees the stop request itself, rolls back and ends: nothing is saved.
    app = real_worker_app
    monkeypatch.setattr(QuickStopWorker, "stop_timeout", 5.0)
    app.gate.respond = lambda request: httpx.Response(
        200,
        json=provider_page([provider_event_json("e1"), provider_event_json("e2")]),
    )
    upserted = []
    original = services.EventRepository.upsert

    async def upsert_then_wait(self, data):
        result = await original(self, data)
        upserted.append(data["id"])
        await asyncio.sleep(0.2)  # long enough for the shutdown to arrive
        return result

    monkeypatch.setattr(services.EventRepository, "upsert", upsert_then_wait)

    with TestClient(main.app):
        assert wait_until(lambda: upserted == ["e1"])
        (worker,) = QuickStopWorker.instances

    assert worker._task.done() and not worker._task.cancelled()
    assert upserted == ["e1"]  # the second event was never taken
    assert event_count(app.db) == 0
    assert app.db.scalar(select(func.count()).select_from(SyncMetadata)) == 0
    assert app.client.http_client.is_closed
    assert all(not s.in_transaction() for s in app.opened)
    assert app.order == [("client", True), ("engine", True)]
