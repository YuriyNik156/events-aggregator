"""Business rules: event synchronization, free seats, ticket registration and
cancellation. Everything that waits (the database, the provider) is ``await``-ed.

Transactions: the services own them (repositories only flush). The rule for every
operation that talks to the provider is *read, release, call, then write*: the
connection goes back to the pool before the provider is asked, and the result is
saved in a new transaction. Only ``sync_events`` keeps one transaction for its whole
run, because the events and the cursor must be saved together.
"""

import asyncio
import logging
import re
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy.ext.asyncio import AsyncSession

from app.events_provider import (
    EventsPaginator,
    EventsProviderClient,
    ProviderRejectedError,
)
from app.repositories import EventRepository, SyncMetadataRepository, TicketRepository

logger = logging.getLogger(__name__)

INITIAL_CURSOR = date(2000, 1, 1)
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

# One process, one sync at a time (DESIGN 8.5), for the manual trigger and the
# background worker alike. Non-blocking: a second caller is told so instead of
# queueing. A plain flag is enough: there is no ``await`` between the check and the
# assignment, so in one event loop nobody can slip in between.
_sync_running = False


class SyncAlreadyRunningError(RuntimeError):
    pass


class SyncStoppedError(RuntimeError):
    """The sync was stopped on request (shutdown); nothing was saved."""


@dataclass(frozen=True)
class SyncResult:
    processed: int
    changed_at_cursor: date


async def _rollback(session: AsyncSession) -> None:
    """Roll back, and let it finish even if this task is cancelled meanwhile."""
    await asyncio.shield(session.rollback())


async def _rollback_after_cancel(session: AsyncSession) -> None:
    """Undo what a cancelled task left open; a failure here is only logged."""
    try:
        await _rollback(session)
    except Exception:
        logger.exception("Rollback after cancellation failed")


async def sync_events(
    session: AsyncSession,
    client: EventsProviderClient,
    stop: asyncio.Event | None = None,
) -> SyncResult:
    """Pull changed events from the provider and upsert them.

    Events and the new cursor are saved in one transaction, so a failure at any
    point leaves both as they were. Safe to call again later (manual trigger and
    the background worker both call it).

    ``stop`` (set by the worker on shutdown) is checked between events: the
    transaction is then rolled back and ``SyncStoppedError`` raised. If the task is
    cancelled instead, the transaction is rolled back too and the cancellation goes on.
    """
    global _sync_running
    if _sync_running:
        raise SyncAlreadyRunningError("synchronization is already running")
    _sync_running = True
    try:
        return await _sync(session, client, stop)
    finally:
        _sync_running = False


async def _sync(
    session: AsyncSession,
    client: EventsProviderClient,
    stop: asyncio.Event | None,
) -> SyncResult:
    events = EventRepository(session)
    metadata = SyncMetadataRepository(session)
    saved = await metadata.get()
    stored_cursor = saved.last_changed_at if saved else None
    if stored_cursor is not None and stored_cursor.tzinfo is None:
        stored_cursor = stored_cursor.replace(tzinfo=UTC)  # drivers without tz
    # The stored value is used as is (its UTC date), no +1 day (DESIGN 8.2).
    cursor = stored_cursor.astimezone(UTC).date() if stored_cursor else INITIAL_CURSOR

    started = time.monotonic()
    logger.info("Sync started: changed_at=%s", cursor)
    committing = False
    try:
        processed = 0
        newest: datetime | None = None
        async for item in EventsPaginator(client, changed_at=cursor):
            if stop is not None and stop.is_set():
                raise SyncStoppedError
            await events.upsert(item.to_row())
            processed += 1
            if newest is None or item.changed_at > newest:
                newest = item.changed_at
        if stop is not None and stop.is_set():
            raise SyncStoppedError
        if newest is not None and stored_cursor is not None:
            newest = max(newest, stored_cursor)
        await metadata.save(
            sync_status=STATUS_SUCCESS,
            last_sync_time=datetime.now(UTC),
            last_changed_at=newest,  # None (no events) keeps the old cursor
        )
        committing = True
        await session.commit()
        committing = False
    except SyncStoppedError:
        # Not a failure: the status and the cursor stay as they were.
        await _rollback(session)
        logger.info("Sync stopped on request after %.2fs", time.monotonic() - started)
        raise
    except asyncio.CancelledError:
        # Not a failure either. CancelledError is not an ``Exception``, so the
        # clause below does not see it; nothing may stay half saved.
        await _rollback_after_cancel(session)
        if committing:
            # The COMMIT may have reached the database before the cancellation did.
            logger.warning(
                "Sync cancelled during the commit after %.2fs: it is not known "
                "whether the events and the cursor were saved; the next sync starts "
                "from the cursor that is stored",
                time.monotonic() - started,
            )
        else:
            logger.info(
                "Sync cancelled after %.2fs and rolled back", time.monotonic() - started
            )
        raise
    except Exception:
        await _rollback(session)
        logger.exception(
            "Sync failed after %.2fs, cursor stays at %s",
            time.monotonic() - started,
            cursor,
        )
        await _record_failure(session)
        raise
    new_cursor = newest.astimezone(UTC).date() if newest else cursor
    logger.info(
        "Sync finished in %.2fs: processed=%d, next changed_at=%s",
        time.monotonic() - started,
        processed,
        new_cursor,
    )
    return SyncResult(processed=processed, changed_at_cursor=new_cursor)


async def _record_failure(session: AsyncSession) -> None:
    try:
        await SyncMetadataRepository(session).save(sync_status=STATUS_ERROR)
        await session.commit()
    except Exception:
        await session.rollback()
        logger.exception("Could not record the sync error status")


SEATS_CACHE_TTL = 30.0
PUBLISHED = "published"


class EventNotFoundError(LookupError):
    pass


class EventNotPublishedError(RuntimeError):
    pass


class SeatsCache:
    """In-memory TTL cache of free seats by event id.

    Plain dictionary operations, no ``await`` inside: nothing can interleave. The
    lock only keeps it safe if it is ever touched from a thread as well.
    """

    def __init__(self, ttl: float = SEATS_CACHE_TTL, clock=time.monotonic):
        self._ttl = ttl
        self._clock = clock
        self._lock = threading.Lock()
        self._items: dict[str, tuple[float, list[str]]] = {}

    def get(self, event_id: str) -> list[str] | None:
        with self._lock:
            item = self._items.get(event_id)
            if item is None:
                return None
            expires_at, seats = item
            if self._clock() >= expires_at:
                del self._items[event_id]
                return None
            return list(seats)

    def set(self, event_id: str, seats: list[str]) -> None:
        with self._lock:
            self._items[event_id] = (self._clock() + self._ttl, list(seats))

    def invalidate(self, event_id: str) -> None:
        with self._lock:
            self._items.pop(event_id, None)


seats_cache = SeatsCache()


async def get_available_seats(
    session: AsyncSession,
    client: EventsProviderClient,
    cache: SeatsCache,
    event_id: str,
) -> list[str]:
    """Free seats of a published event; cached successes only.

    Read the event, release the connection, then (maybe) ask the provider: no
    database connection is held while the provider answers. The cache lock is never
    held during the request either, so concurrent misses may each fetch once; that
    is harmless and keeps slow requests from blocking others.
    """
    event = await EventRepository(session).get(event_id)
    status = event.status if event is not None else None  # read before the release
    await session.commit()  # read-only: just gives the connection back
    if event is None:
        raise EventNotFoundError(event_id)
    if status != PUBLISHED:
        raise EventNotPublishedError(event_id)
    return await _free_seats(client, cache, event_id)


async def _free_seats(
    client: EventsProviderClient, cache: SeatsCache, event_id: str
) -> list[str]:
    """Cache or provider; touches no database."""
    seats = cache.get(event_id)
    if seats is not None:
        return seats
    try:
        seats = await client.get_event_seats(event_id)
    except ProviderRejectedError as exc:
        if exc.status_code == 404:
            # The provider does not know an event we have locally.
            cache.invalidate(event_id)
            raise EventNotFoundError(event_id) from None
        raise
    cache.set(event_id, seats)
    return seats


class RegistrationClosedError(RuntimeError):
    pass


class InvalidSeatError(ValueError):
    pass


class SeatUnavailableError(RuntimeError):
    pass


class RegistrationRejectedError(RuntimeError):
    """The provider refused with HTTP 400 (documented: e.g. the seat is taken)."""


class TicketNotSavedError(RuntimeError):
    """The provider registered the participant but the local save failed."""

    def __init__(self, ticket_id: str):
        super().__init__("registration confirmed by the provider, not saved locally")
        self.ticket_id = ticket_id


_PATTERN_SEGMENT = re.compile(r"\s*([A-Za-z]+)(\d+)(?:-(\d+))?\s*")
_SEAT = re.compile(r"([A-Za-z]+)(\d+)")


def seat_in_pattern(seat: str, pattern: str) -> bool:
    """Whether ``seat`` (``A15``) fits ``pattern`` (``A1-1000,B1-2000``).

    Ranges are compared, never expanded. A segment in an unknown format makes the
    answer "maybe", i.e. True: the provider decides then.
    """
    parsed = _SEAT.fullmatch(seat)
    unknown = False
    for segment in pattern.split(","):
        m = _PATTERN_SEGMENT.fullmatch(segment)
        if m is None:
            unknown = True
            continue
        row, first, last = m.group(1), int(m.group(2)), m.group(3)
        if parsed and parsed.group(1) == row:
            number = int(parsed.group(2))
            if first <= number <= (int(last) if last else first):
                return True
    return unknown


async def register_ticket(
    session: AsyncSession,
    client: EventsProviderClient,
    cache: SeatsCache,
    *,
    event_id: str,
    first_name: str,
    last_name: str,
    email: str,
    seat: str,
    now: datetime | None = None,
) -> str:
    """Register a participant at the provider, then save the ticket locally.

    Read the event, release the connection, check, ask the provider, and save the
    ticket in a new transaction. The local checks are only a pre-filter; the provider
    decides. The POST is sent once and never retried. No cache lock is held during
    the network call.
    """
    event = await EventRepository(session).get(event_id)
    if event is not None:  # everything needed is read before the release
        status = event.status
        deadline = event.registration_deadline
        seats_pattern = event.place_seats_pattern
    await session.commit()  # read-only: just gives the connection back

    if event is None:
        raise EventNotFoundError(event_id)
    if status != PUBLISHED:
        raise EventNotPublishedError(event_id)
    if deadline is not None:
        if deadline.tzinfo is None:
            deadline = deadline.replace(tzinfo=UTC)  # drivers without tz
        if (now or datetime.now(UTC)) > deadline:
            raise RegistrationClosedError(event_id)
    if not seat_in_pattern(seat, seats_pattern):
        raise InvalidSeatError(event_id)
    if seat not in await _free_seats(client, cache, event_id):
        raise SeatUnavailableError(event_id)

    try:
        ticket_id = await client.register(
            event_id,
            first_name=first_name,
            last_name=last_name,
            seat=seat,
            email=email,
        )
    except ProviderRejectedError as exc:
        if exc.status_code == 400:
            raise RegistrationRejectedError(event_id) from None
        if exc.status_code == 404:
            raise EventNotFoundError(event_id) from None
        raise

    # The seat is taken at the provider from now on, whatever happens locally.
    cache.invalidate(event_id)
    try:
        await TicketRepository(session).add(
            ticket_id=ticket_id,
            event_id=event_id,
            first_name=first_name,
            last_name=last_name,
            email=email,
            seat=seat,
        )
        await session.commit()
    except asyncio.CancelledError:
        # Nobody is left to report to; the ids make the ticket findable in the log.
        logger.error(
            "Provider confirmed ticket %s for event %s but the task was cancelled "
            "during the local save: whether the ticket was stored is not known",
            ticket_id,
            event_id,
        )
        await _rollback_after_cancel(session)
        raise
    except Exception:
        await _rollback(session)
        # Ids only: no names or emails in logs. Not compensated automatically.
        logger.exception(
            "Provider confirmed ticket %s for event %s but the local save failed",
            ticket_id,
            event_id,
        )
        raise TicketNotSavedError(ticket_id) from None
    logger.info("Ticket %s registered for event %s", ticket_id, event_id)
    return ticket_id


class TicketNotFoundError(LookupError):
    pass


class EventStartedError(RuntimeError):
    pass


class CancellationInProgressError(RuntimeError):
    pass


class CancellationRejectedError(RuntimeError):
    """The provider refused with HTTP 400 (documented: e.g. the event has started)."""


class CancellationNotSavedError(RuntimeError):
    """The provider cancelled the registration but the local save failed."""

    def __init__(self, ticket_id: str):
        super().__init__("cancellation confirmed by the provider, not saved locally")
        self.ticket_id = ticket_id


# Tickets whose DELETE is in flight in this process. A second request for the same
# ticket must not send another DELETE: the provider's idempotency is unconfirmed.
# No lock: checking and adding happen without an ``await`` in between, so in one
# event loop nobody can slip in between.
_cancelling: set[str] = set()


async def cancel_ticket(
    session: AsyncSession,
    client: EventsProviderClient,
    cache: SeatsCache,
    ticket_id: str,
    now: datetime | None = None,
) -> None:
    """Cancel a registration at the provider, then mark the local ticket cancelled.

    The DELETE is sent once and never retried. The ticket row is kept (history);
    only ``cancelled_at`` is set, and only after the provider confirmed.
    """
    # The guard comes first: the ticket is read only while holding it, so a request
    # that came after a finished cancellation sees the ticket as cancelled.
    if ticket_id in _cancelling:
        raise CancellationInProgressError(ticket_id)
    _cancelling.add(ticket_id)
    try:
        event_id = await _cancel(session, client, cache, ticket_id, now)
    finally:
        _cancelling.discard(ticket_id)
    logger.info("Ticket %s cancelled for event %s", ticket_id, event_id)


async def _cancel(
    session: AsyncSession,
    client: EventsProviderClient,
    cache: SeatsCache,
    ticket_id: str,
    now: datetime | None,
) -> str:
    tickets = TicketRepository(session)
    ticket = await tickets.get_active(ticket_id)
    if ticket is None:
        await session.commit()  # nothing more to read: give the connection back
        raise TicketNotFoundError(ticket_id)
    event_id = ticket.event_id
    event = await EventRepository(session).get(event_id)
    started_at = event.event_time if event is not None else None
    await session.commit()  # read-only: just gives the connection back

    if event is None:
        raise EventNotFoundError(event_id)
    if started_at.tzinfo is None:
        started_at = started_at.replace(tzinfo=UTC)  # drivers without tz
    if (now or datetime.now(UTC)) >= started_at:
        raise EventStartedError(event_id)

    try:
        await client.unregister(event_id, ticket_id)
    except ProviderRejectedError as exc:
        if exc.status_code == 400:
            raise CancellationRejectedError(ticket_id) from None
        if exc.status_code == 404:
            # The event or the registration is unknown to the provider.
            raise TicketNotFoundError(ticket_id) from None
        raise

    # The seat is free at the provider from now on, whatever happens locally.
    cache.invalidate(event_id)
    try:
        # ``ticket`` was loaded in the first transaction; this UPDATE opens a new one.
        await tickets.cancel(ticket, datetime.now(UTC))
        await session.commit()
    except asyncio.CancelledError:
        logger.error(
            "Provider cancelled ticket %s for event %s but the task was cancelled "
            "during the local save: whether the cancellation was stored is not known",
            ticket_id,
            event_id,
        )
        await _rollback_after_cancel(session)
        raise
    except Exception:
        await _rollback(session)
        logger.exception(
            "Provider cancelled ticket %s for event %s but the local save failed",
            ticket_id,
            event_id,
        )
        raise CancellationNotSavedError(ticket_id) from None
    return event_id
