import asyncio
import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from app.config import get_provider_settings
from app.database import get_session_factory
from app.events_provider import EventsProviderClient
from app.repositories import SyncMetadataRepository
from app.services import (
    SyncAlreadyRunningError,
    SyncResult,
    SyncStoppedError,
    sync_events,
)

logger = logging.getLogger(__name__)

STARTUP_DELAY = 30.0  # first run of an empty or overdue database
BACKOFF_FIRST = 5 * 60.0  # 5, 10, 20, 40 minutes, then 60 at most
BACKOFF_MAX = 60 * 60.0
# Below the usual 30 s termination grace period of a container.
SHUTDOWN_TIMEOUT = 20.0


def run_sync_once(stop: threading.Event | None = None) -> SyncResult:
    """One sync with its own provider client and session; both always closed."""
    settings = get_provider_settings()
    client = EventsProviderClient(settings.base_url, settings.api_key)
    try:
        with get_session_factory()() as session:
            return sync_events(session, client, stop)
    finally:
        client.close()


def seconds_until_due(interval: float, now: datetime | None = None) -> float:
    """Wait before the next run, from the last *successful* sync in the database.

    A fresh sync schedules the next one at ``last_sync_time + interval``. No sync
    yet, an overdue one, or an unreadable database give ``STARTUP_DELAY``, so a
    failed attempt of a previous process is retried soon after the restart.
    """
    try:
        with get_session_factory()() as session:
            meta = SyncMetadataRepository(session).get()
            last = meta.last_sync_time if meta else None
    except Exception as exc:
        logger.warning("Could not read the last sync time: %s", type(exc).__name__)
        return STARTUP_DELAY
    if last is None:
        return STARTUP_DELAY
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)  # drivers without tz
    remaining = (
        last + timedelta(seconds=interval) - (now or datetime.now(UTC))
    ).total_seconds()
    # Never longer than one interval (a last_sync_time from the future) nor shorter
    # than the startup delay.
    return max(min(remaining, interval), STARTUP_DELAY)


class SyncWorker:
    """Runs the existing sync periodically from the FastAPI lifespan.

    The blocking sync runs in a thread (``asyncio.to_thread``). A cancelled task
    cannot stop that thread, so shutdown goes through ``stop()``, which sets a flag
    the sync checks between events.
    """

    def __init__(
        self,
        interval: float,
        *,
        run_once: Callable[[threading.Event], SyncResult] = run_sync_once,
        first_delay: Callable[[float], float] = seconds_until_due,
        backoff_first: float = BACKOFF_FIRST,
        backoff_max: float = BACKOFF_MAX,
    ):
        self._interval = interval
        self._run_once = run_once
        self._first_delay = first_delay
        self._backoff_first = backoff_first
        self._backoff_max = backoff_max
        self._stop = threading.Event()  # seen by the sync thread
        self._wake = asyncio.Event()  # interrupts the wait in the event loop
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self.run(), name="sync-worker")

    def request_stop(self) -> None:
        self._stop.set()
        self._wake.set()

    async def stop(self, timeout: float = SHUTDOWN_TIMEOUT) -> bool:
        """Ask the worker to stop and wait for it; ``False`` if it is still busy.

        A sync stops after the provider request in flight (up to the HTTP timeout
        per request) and rolls back. When this returns ``False`` the thread still
        uses the database: the caller must not dispose the engine, and the
        interpreter waits for the thread on exit.
        """
        self.request_stop()
        if self._task is None or self._task.done():
            return True
        done, _ = await asyncio.wait({self._task}, timeout=timeout)
        return bool(done)

    async def _sleep(self, seconds: float) -> bool:
        """Wait; ``True`` if a stop was requested meanwhile."""
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=max(seconds, 0.0))
        except TimeoutError:
            pass
        return self._stop.is_set()

    async def run(self) -> None:
        delay = await asyncio.to_thread(self._first_delay, self._interval)
        failures = 0
        logger.info("Background sync scheduled in %.0f s", delay)
        while not await self._sleep(delay):
            started = time.monotonic()
            try:
                await asyncio.to_thread(self._run_once, self._stop)
            except SyncStoppedError:
                return
            except SyncAlreadyRunningError:
                # A manual sync is running; look at the database again afterwards.
                logger.info("Background sync skipped: a sync is already running")
                delay = await asyncio.to_thread(self._first_delay, self._interval)
            except Exception as exc:
                # Details and traceback are logged where the sync failed.
                failures += 1
                delay = min(
                    self._backoff_first * 2 ** (failures - 1),
                    self._backoff_max,
                    self._interval,
                )
                logger.warning(
                    "Background sync failed (%s), attempt %d, retry in %.0f s",
                    type(exc).__name__,
                    failures,
                    delay,
                )
            else:
                failures = 0
                # Counted from the start of this run: its duration adds no delay.
                delay = self._interval - (time.monotonic() - started)
                logger.info("Background sync done, next in %.0f s", delay)
