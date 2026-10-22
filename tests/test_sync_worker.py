import asyncio
import threading
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import main
from app import sync_worker
from app.config import ConfigError
from app.events_provider import EventsPage, ProviderUnavailableError
from app.models import Base, Event
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
from tests.conftest import provider_event_json, provider_page

DAY = 86400
RESULT = SyncResult(processed=0, changed_at_cursor=datetime(2000, 1, 1).date())


def scripted(*outcomes):
    """A ``run_once`` returning or raising the given outcomes in order."""
    calls = []

    def run_once(stop):
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


def make_worker(interval=DAY, *, run_once, first_delay=lambda _: 0.0, **options):
    return SyncWorker(interval, run_once=run_once, first_delay=first_delay, **options)


# --- schedule ---------------------------------------------------------------


def test_runs_repeatedly_and_stops_on_request():
    run_once = scripted()
    worker = make_worker(0.05, run_once=run_once, first_delay=lambda _: 0.01)

    async def scenario():
        worker.start()
        await asyncio.sleep(0.3)
        assert await worker.stop() is True

    asyncio.run(scenario())

    assert len(run_once.calls) >= 3


def test_first_wait_comes_from_the_database_and_nothing_runs_before_it():
    run_once = scripted()
    worker = make_worker(run_once=run_once, first_delay=lambda interval: 1234.5)
    delays = record_sleeps(worker, stop_after=0)

    asyncio.run(drive(worker))

    assert delays == [1234.5]
    assert run_once.calls == []


def test_next_run_is_counted_from_the_start_of_the_previous_one():
    def slow_run(stop):
        time.sleep(0.2)
        return RESULT

    worker = make_worker(100.0, run_once=slow_run)
    delays = record_sleeps(worker, stop_after=1)

    asyncio.run(drive(worker))

    assert delays[0] == 0.0
    assert 99.0 < delays[1] < 99.85  # the run's own 0.2 s is not added on top


def test_event_loop_stays_responsive_during_a_blocking_sync():
    def blocking(stop):
        time.sleep(0.4)
        return RESULT

    worker = make_worker(run_once=blocking)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    async def scenario():
        tick_task = asyncio.create_task(ticker())
        worker.start()
        await asyncio.sleep(0.3)  # the sync is running in its thread now
        count_during_sync = ticks
        await worker.stop()
        tick_task.cancel()
        return count_during_sync

    assert asyncio.run(scenario()) >= 15


# --- errors and backoff -----------------------------------------------------


def test_provider_errors_back_off_then_reset_after_success():
    boom = ProviderUnavailableError("down")
    run_once = scripted(boom, boom, boom, boom, boom, boom, RESULT, boom)
    worker = make_worker(run_once=run_once)
    delays = record_sleeps(worker, stop_after=8)

    asyncio.run(drive(worker))

    assert delays[:7] == [0.0, 300, 600, 1200, 2400, 3600, 3600]
    assert DAY - 5 < delays[7] <= DAY  # back on the schedule after the success
    assert delays[8] == 300  # the backoff started over
    assert len(run_once.calls) == 8


def test_backoff_never_exceeds_a_shorter_interval():
    boom = ProviderUnavailableError("down")
    worker = make_worker(100.0, run_once=scripted(boom, boom))
    delays = record_sleeps(worker, stop_after=2)

    asyncio.run(drive(worker))

    assert delays == [0.0, 100.0, 100.0]


def test_any_exception_keeps_the_loop_alive():
    run_once = scripted(RuntimeError("db down"), ConfigError("no key"), RESULT)
    worker = make_worker(run_once=run_once)
    record_sleeps(worker, stop_after=3)

    asyncio.run(drive(worker))

    assert len(run_once.calls) == 3


def test_running_sync_is_not_an_error_and_is_not_retried_at_once():
    answers = iter([0.0, 31.0])
    run_once = scripted(SyncAlreadyRunningError("manual"), RESULT)
    worker = make_worker(run_once=run_once, first_delay=lambda _: next(answers))
    delays = record_sleeps(worker, stop_after=1)

    asyncio.run(drive(worker))

    # No backoff step; the wait comes from the database again.
    assert delays == [0.0, 31.0]
    assert len(run_once.calls) == 1


def test_stopped_sync_ends_the_worker_without_retry():
    run_once = scripted(SyncStoppedError())
    worker = make_worker(run_once=run_once)

    asyncio.run(drive(worker))

    assert len(run_once.calls) == 1


# --- shutdown ---------------------------------------------------------------


def test_stop_interrupts_the_wait_at_once():
    worker = make_worker(run_once=scripted(), first_delay=lambda _: 3600.0)

    async def scenario():
        worker.start()
        await asyncio.sleep(0.05)
        started = time.monotonic()
        stopped = await worker.stop()
        return stopped, time.monotonic() - started

    stopped, took = asyncio.run(scenario())

    assert stopped is True
    assert took < 1


def test_stop_signals_the_running_sync_and_waits_for_it():
    seen = threading.Event()
    finished = threading.Event()

    def cooperative(stop):
        seen.set()
        assert stop.wait(5)
        finished.set()
        raise SyncStoppedError

    worker = make_worker(run_once=cooperative)
    threads_before = threading.active_count()

    async def scenario():
        worker.start()
        while not seen.is_set():
            await asyncio.sleep(0.01)
        return await worker.stop()

    assert asyncio.run(scenario()) is True

    assert finished.is_set()
    assert threading.active_count() == threads_before  # no thread left behind
    assert worker._task.done()


def test_stop_reports_a_sync_that_does_not_finish_in_time():
    release = threading.Event()

    def stubborn(stop):
        release.wait(5)
        return RESULT

    worker = make_worker(run_once=stubborn)

    async def scenario():
        worker.start()
        await asyncio.sleep(0.05)
        slow = await worker.stop(timeout=0.05)
        release.set()
        await asyncio.wait_for(worker._task, 5)
        return slow

    assert asyncio.run(scenario()) is False
    assert worker._task.done()


def test_stop_without_start_is_fine():
    assert asyncio.run(make_worker(run_once=scripted()).stop()) is True


# --- schedule from sync_metadata -------------------------------------------


@pytest.fixture
def factory(session: Session, monkeypatch):
    maker = sessionmaker(bind=session.get_bind(), expire_on_commit=False)
    monkeypatch.setattr(sync_worker, "get_session_factory", lambda: maker)
    return maker


def save_last_sync(session: Session, when: datetime) -> None:
    SyncMetadataRepository(session).save(sync_status="success", last_sync_time=when)
    session.commit()


def test_empty_database_starts_after_the_short_delay(factory):
    assert seconds_until_due(DAY) == STARTUP_DELAY


def test_fresh_sync_schedules_the_next_one_a_full_interval_later(session, factory):
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    save_last_sync(session, now - timedelta(hours=1))

    assert seconds_until_due(DAY, now) == pytest.approx(23 * 3600)


def test_overdue_sync_starts_after_the_short_delay(session, factory):
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    save_last_sync(session, now - timedelta(days=3))

    assert seconds_until_due(DAY, now) == STARTUP_DELAY


def test_almost_due_sync_still_waits_the_short_delay(session, factory):
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    save_last_sync(session, now - timedelta(seconds=DAY - 5))

    assert seconds_until_due(DAY, now) == STARTUP_DELAY


def test_last_sync_in_the_future_is_capped_to_one_interval(session, factory):
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    save_last_sync(session, now + timedelta(days=30))

    assert seconds_until_due(DAY, now) == DAY


def test_failed_attempt_does_not_postpone_the_next_one(session, factory):
    # A failure writes only the status; last_sync_time still points at the last
    # success, so after a restart the overdue sync is retried soon.
    now = datetime(2026, 10, 22, 12, tzinfo=UTC)
    save_last_sync(session, now - timedelta(days=2))
    SyncMetadataRepository(session).save(sync_status="error")
    session.commit()

    assert seconds_until_due(DAY, now) == STARTUP_DELAY


def test_unreadable_database_does_not_stop_the_worker(monkeypatch):
    def broken():
        raise RuntimeError("db down")

    monkeypatch.setattr(sync_worker, "get_session_factory", broken)

    assert seconds_until_due(DAY) == STARTUP_DELAY


# --- resources of one run ---------------------------------------------------


class FakeClient:
    instances: list["FakeClient"] = []

    def __init__(self, base_url, api_key):
        self.closed = False
        FakeClient.instances.append(self)

    def close(self):
        self.closed = True


class FakeSession:
    def __init__(self):
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True


@pytest.fixture
def fakes(monkeypatch):
    FakeClient.instances = []
    sessions: list[FakeSession] = []

    def make_session():
        sessions.append(FakeSession())
        return sessions[-1]

    monkeypatch.setattr(
        sync_worker,
        "get_provider_settings",
        lambda: Mock(base_url="http://x", api_key="k"),
    )
    monkeypatch.setattr(sync_worker, "EventsProviderClient", FakeClient)
    monkeypatch.setattr(sync_worker, "get_session_factory", lambda: make_session)
    return sessions


def test_run_closes_client_and_session_on_success(fakes, monkeypatch):
    stop = threading.Event()
    seen = {}

    def fake_sync(session, client, stop_arg):
        seen.update(session=session, client=client, stop=stop_arg)
        return RESULT

    monkeypatch.setattr(sync_worker, "sync_events", fake_sync)

    assert run_sync_once(stop) is RESULT

    assert seen["stop"] is stop
    assert seen["client"].closed and seen["session"].closed


@pytest.mark.parametrize(
    "error", [ProviderUnavailableError("x"), SyncStoppedError(), RuntimeError("x")]
)
def test_run_closes_client_and_session_on_error(fakes, monkeypatch, error):
    def failing(session, client, stop):
        raise error

    monkeypatch.setattr(sync_worker, "sync_events", failing)

    with pytest.raises(type(error)):
        run_sync_once()

    assert FakeClient.instances[0].closed
    assert fakes[0].closed


def test_run_without_api_key_creates_nothing(fakes, monkeypatch):
    def no_key():
        raise ConfigError("EVENTS_PROVIDER_API_KEY is not set")

    monkeypatch.setattr(sync_worker, "get_provider_settings", no_key)

    with pytest.raises(ConfigError):
        run_sync_once()

    assert FakeClient.instances == [] and fakes == []


# --- one lock for manual and background syncs -------------------------------


def test_manual_sync_is_refused_while_the_worker_syncs(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'worker.db'}")
    Base.metadata.create_all(engine)
    maker = sessionmaker(bind=engine, expire_on_commit=False)
    entered = threading.Event()
    release = threading.Event()
    provider = Mock()

    def slow_page(**kwargs):
        entered.set()
        release.wait(5)
        return EventsPage.model_validate(provider_page([provider_event_json("e1")]))

    provider.get_events_page.side_effect = slow_page
    provider.close = Mock()
    monkeypatch.setattr(sync_worker, "get_session_factory", lambda: maker)
    monkeypatch.setattr(
        sync_worker,
        "get_provider_settings",
        lambda: Mock(base_url="http://x", api_key="k"),
    )
    monkeypatch.setattr(sync_worker, "EventsProviderClient", lambda *a: provider)
    worker = SyncWorker(DAY, first_delay=lambda _: 0.0)
    refused = []

    async def scenario():
        worker.start()
        while not entered.is_set():
            await asyncio.sleep(0.01)
        with maker() as manual_session:  # what POST /api/sync/trigger does
            try:
                sync_events(manual_session, Mock())
            except SyncAlreadyRunningError:
                refused.append(True)
        release.set()
        await asyncio.sleep(0.2)  # let the background sync commit
        assert await worker.stop() is True

    try:
        asyncio.run(scenario())
    finally:
        release.set()

    assert refused == [True]
    with maker() as check:
        assert check.scalar(select(func.count()).select_from(Event)) == 1
        assert SyncMetadataRepository(check).get().sync_status == "success"
    provider.close.assert_called_once()
    engine.dispose()


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
    disposed = Mock()
    monkeypatch.setattr(main, "SyncWorker", FakeWorker)
    monkeypatch.setattr(main, "dispose_engine", disposed)
    return disposed


def test_worker_is_off_when_interval_is_zero(lifespan_fakes):
    with TestClient(main.app):
        pass  # SYNC_INTERVAL_SECONDS=0 comes from conftest

    assert FakeWorker.instances == []
    lifespan_fakes.assert_called_once()


def test_worker_starts_with_the_app_and_stops_with_it(lifespan_fakes, monkeypatch):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "3600")

    with TestClient(main.app) as client:
        assert client.get("/api/health").json() == {"status": "ok"}
        (worker,) = FakeWorker.instances
        assert worker.interval == 3600 and worker.started
        lifespan_fakes.assert_not_called()

    assert worker.stop_calls == 1
    lifespan_fakes.assert_called_once()


def test_default_interval_is_24_hours(lifespan_fakes, monkeypatch):
    monkeypatch.delenv("SYNC_INTERVAL_SECONDS")

    with TestClient(main.app):
        pass

    assert FakeWorker.instances[0].interval == DAY


def test_engine_is_kept_while_the_sync_thread_still_runs(lifespan_fakes, monkeypatch):
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


def test_manual_trigger_still_works_with_the_worker_on(lifespan_fakes, monkeypatch):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", "3600")
    # One shared connection: TestClient runs sync endpoints in another thread.
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    provider = Mock()
    provider.get_events_page.return_value = EventsPage.model_validate(
        provider_page([provider_event_json("e1")])
    )
    with Session(engine) as db:
        main.app.dependency_overrides[main.get_session] = lambda: db
        main.app.dependency_overrides[main.get_provider_client] = lambda: provider
        try:
            with TestClient(main.app) as client:
                response = client.post("/api/sync/trigger")
        finally:
            main.app.dependency_overrides.clear()
        assert db.scalar(select(func.count()).select_from(Event)) == 1
    engine.dispose()

    assert response.status_code == 200
    assert response.json() == {"success": True}
