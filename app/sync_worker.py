import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from app.database import get_async_session_factory
from app.events_provider import get_shared_client
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
# Both together stay below the usual 30 s termination grace period of a container.
SHUTDOWN_TIMEOUT = 20.0  # for the running sync to notice the stop request
CANCEL_TIMEOUT = 5.0  # for the cancelled task to unwind (rollback)


async def run_sync_once(stop: asyncio.Event | None = None) -> SyncResult:
    """One sync with the shared provider client and a session of its own.

    The client belongs to the process and stays open; the session is always closed
    (an unfinished transaction is rolled back).
    """
    client = get_shared_client()  # ConfigError before anything else is created
    async with get_async_session_factory()() as session:
        return await sync_events(session, client, stop)


async def seconds_until_due(interval: float, now: datetime | None = None) -> float:
    """Wait before the next run, from the last *successful* sync in the database.

    A fresh sync schedules the next one at ``last_sync_time + interval``. No sync
    yet, an overdue one, or an unreadable database give ``STARTUP_DELAY``, so a
    failed attempt of a previous process is retried soon after the restart.
    """
    try:
        async with get_async_session_factory()() as session:
            meta = await SyncMetadataRepository(session).get()
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

    One asyncio task; the sync itself is ``async`` and shares the "one sync at a
    time" guard with the manual trigger. Shutdown goes through ``stop()``: the stop
    event is checked between events (rollback, nothing saved), and a sync that does
    not notice it in time is cancelled.
    """

    def __init__(
        self,
        interval: float,
        *,
        run_once: Callable[[asyncio.Event], Awaitable[SyncResult]] = run_sync_once,
        first_delay: Callable[[float], Awaitable[float]] = seconds_until_due,
        backoff_first: float = BACKOFF_FIRST,
        backoff_max: float = BACKOFF_MAX,
    ):
        self._interval = interval
        self._run_once = run_once
        self._first_delay = first_delay
        self._backoff_first = backoff_first
        self._backoff_max = backoff_max
        self._stop = asyncio.Event()  # ends the wait and is seen by the sync
        self._task: asyncio.Task | None = None

    def start(self) -> None:
        self._task = asyncio.create_task(self.run(), name="sync-worker")

    def request_stop(self) -> None:
        self._stop.set()

    async def stop(
        self, timeout: float = SHUTDOWN_TIMEOUT, cancel_timeout: float = CANCEL_TIMEOUT
    ) -> bool:
        """Ask the worker to stop and wait for it; ``False`` if it is still busy.

        A sync stops between two events and rolls back. If it has not finished after
        ``timeout`` (for example it waits for a slow provider), the task is
        cancelled and gets ``cancel_timeout`` to unwind. Only ``False`` means the
        task may still use the database: the caller must then keep the client and
        the engine open.
        """
        self.request_stop()
        if self._task is None or self._task.done():
            return True
        done, _ = await asyncio.wait({self._task}, timeout=timeout)
        if done:
            return True
        logger.warning("Background sync did not stop in %.0f s; cancelling it", timeout)
        self._task.cancel()
        done, _ = await asyncio.wait({self._task}, timeout=cancel_timeout)
        return bool(done)

    async def _sleep(self, seconds: float) -> bool:
        """Wait; ``True`` if a stop was requested meanwhile."""
        try:
            await asyncio.wait_for(self._stop.wait(), timeout=max(seconds, 0.0))
        except TimeoutError:
            pass
        return self._stop.is_set()

    async def run(self) -> None:
        delay = await self._first_delay(self._interval)
        failures = 0
        logger.info("Background sync scheduled in %.0f s", delay)
        while not await self._sleep(delay):
            started = time.monotonic()
            try:
                await self._run_once(self._stop)
            except SyncStoppedError:
                return
            except SyncAlreadyRunningError:
                # A manual sync is running; look at the database again afterwards.
                logger.info("Background sync skipped: a sync is already running")
                delay = await self._first_delay(self._interval)
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
