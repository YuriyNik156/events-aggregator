import logging
import threading
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime

from sqlalchemy.orm import Session

from app.events_provider import EventsPaginator, EventsProviderClient
from app.repositories import EventRepository, SyncMetadataRepository

logger = logging.getLogger(__name__)

INITIAL_CURSOR = date(2000, 1, 1)
STATUS_SUCCESS = "success"
STATUS_ERROR = "error"

# One process, one sync at a time (DESIGN 8.5). Non-blocking: a second caller is
# told so instead of queueing. A threading lock because the sync is blocking code.
_sync_lock = threading.Lock()


class SyncAlreadyRunningError(RuntimeError):
    pass


@dataclass(frozen=True)
class SyncResult:
    processed: int
    changed_at_cursor: date


def sync_events(session: Session, client: EventsProviderClient) -> SyncResult:
    """Pull changed events from the provider and upsert them.

    Events and the new cursor are saved in one transaction, so a failure at any
    point leaves both as they were. Safe to call again later (manual trigger now,
    a background worker in the future).
    """
    if not _sync_lock.acquire(blocking=False):
        raise SyncAlreadyRunningError("synchronization is already running")
    try:
        return _sync(session, client)
    finally:
        _sync_lock.release()


def _sync(session: Session, client: EventsProviderClient) -> SyncResult:
    events = EventRepository(session)
    metadata = SyncMetadataRepository(session)
    saved = metadata.get()
    stored_cursor = saved.last_changed_at if saved else None
    if stored_cursor is not None and stored_cursor.tzinfo is None:
        stored_cursor = stored_cursor.replace(tzinfo=UTC)  # drivers without tz
    # The stored value is used as is (its UTC date), no +1 day (DESIGN 8.2).
    cursor = stored_cursor.astimezone(UTC).date() if stored_cursor else INITIAL_CURSOR

    started = time.monotonic()
    logger.info("Sync started: changed_at=%s", cursor)
    try:
        processed = 0
        newest: datetime | None = None
        for item in EventsPaginator(client, changed_at=cursor):
            events.upsert(item.to_row())
            processed += 1
            if newest is None or item.changed_at > newest:
                newest = item.changed_at
        if newest is not None and stored_cursor is not None:
            newest = max(newest, stored_cursor)
        metadata.save(
            sync_status=STATUS_SUCCESS,
            last_sync_time=datetime.now(UTC),
            last_changed_at=newest,  # None (no events) keeps the old cursor
        )
        session.commit()
    except Exception:
        session.rollback()
        logger.exception(
            "Sync failed after %.2fs, cursor stays at %s",
            time.monotonic() - started,
            cursor,
        )
        _record_failure(session)
        raise
    new_cursor = newest.astimezone(UTC).date() if newest else cursor
    logger.info(
        "Sync finished in %.2fs: processed=%d, next changed_at=%s",
        time.monotonic() - started,
        processed,
        new_cursor,
    )
    return SyncResult(processed=processed, changed_at_cursor=new_cursor)


def _record_failure(session: Session) -> None:
    try:
        SyncMetadataRepository(session).save(sync_status=STATUS_ERROR)
        session.commit()
    except Exception:
        session.rollback()
        logger.exception("Could not record the sync error status")
