from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Event, SyncMetadata, Ticket

SYNC_METADATA_ID = 1


class EventRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self, event_id: str) -> Event | None:
        return self.session.get(Event, event_id)

    def list(
        self,
        *,
        date_from: datetime | None = None,
        page: int = 1,
        page_size: int = 20,
    ) -> tuple[list[Event], int]:
        """Return (events of the page, total count) ordered by event_time, id.

        ``date_from`` is inclusive: events with ``event_time >= date_from``.
        """
        conditions = []
        if date_from is not None:
            conditions.append(Event.event_time >= date_from)
        total = self.session.scalar(
            select(func.count()).select_from(Event).where(*conditions)
        )
        rows = self.session.scalars(
            select(Event)
            .where(*conditions)
            .order_by(Event.event_time, Event.id)
            .limit(page_size)
            .offset((page - 1) * page_size)
        ).all()
        return list(rows), total or 0

    def upsert(self, data: dict[str, Any]) -> Event:
        """Insert or update an event by its external ``id`` (flushes, no commit)."""
        event = self.session.merge(Event(**data))
        self.session.flush()
        return event


class TicketRepository:
    def __init__(self, session: Session):
        self.session = session

    def add(self, **fields: Any) -> Ticket:
        ticket = Ticket(**fields)
        self.session.add(ticket)
        self.session.flush()
        return ticket

    def get_active(self, ticket_id: str) -> Ticket | None:
        return self.session.scalars(
            select(Ticket).where(
                Ticket.ticket_id == ticket_id, Ticket.cancelled_at.is_(None)
            )
        ).first()

    def cancel(self, ticket: Ticket, cancelled_at: datetime) -> None:
        ticket.cancelled_at = cancelled_at
        self.session.flush()


class SyncMetadataRepository:
    def __init__(self, session: Session):
        self.session = session

    def get(self) -> SyncMetadata | None:
        return self.session.get(SyncMetadata, SYNC_METADATA_ID)

    def save(
        self,
        *,
        sync_status: str,
        last_sync_time: datetime | None = None,
        last_changed_at: datetime | None = None,
    ) -> SyncMetadata:
        """Set the status; cursor fields change only when a value is passed."""
        meta = self.get()
        if meta is None:
            meta = SyncMetadata(id=SYNC_METADATA_ID)
            self.session.add(meta)
        meta.sync_status = sync_status
        if last_sync_time is not None:
            meta.last_sync_time = last_sync_time
        if last_changed_at is not None:
            meta.last_changed_at = last_changed_at
        self.session.flush()
        return meta
