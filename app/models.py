from datetime import datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Text,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class Event(Base):
    """Local copy of a provider event. Place fields are stored flat (DESIGN 5.1)."""

    __tablename__ = "events"

    id: Mapped[str] = mapped_column(Text, primary_key=True)
    name: Mapped[str] = mapped_column(Text)
    place_id: Mapped[str] = mapped_column(Text)
    place_name: Mapped[str] = mapped_column(Text)
    place_city: Mapped[str] = mapped_column(Text)
    place_address: Mapped[str] = mapped_column(Text)
    place_seats_pattern: Mapped[str] = mapped_column(Text)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    registration_deadline: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True)
    )
    status: Mapped[str] = mapped_column(Text)
    number_of_visitors: Mapped[int] = mapped_column(Integer, server_default=text("0"))
    changed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index("ix_events_event_time", "event_time"),
        Index("ix_events_changed_at", "changed_at"),
    )


class Ticket(Base):
    """Successful registration. ``cancelled_at IS NULL`` means active."""

    __tablename__ = "tickets"

    id: Mapped[int] = mapped_column(primary_key=True)
    ticket_id: Mapped[str] = mapped_column(Text)
    event_id: Mapped[str] = mapped_column(ForeignKey("events.id"))
    first_name: Mapped[str] = mapped_column(Text)
    last_name: Mapped[str] = mapped_column(Text)
    email: Mapped[str] = mapped_column(Text)
    seat: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    cancelled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        # Provider may reuse ticket_id after cancellation: unique among active only.
        Index(
            "uq_tickets_ticket_id_active",
            "ticket_id",
            unique=True,
            postgresql_where=text("cancelled_at IS NULL"),
            sqlite_where=text("cancelled_at IS NULL"),
        ),
        Index("ix_tickets_event_id", "event_id"),
    )


class SyncMetadata(Base):
    """Single-row synchronization state."""

    __tablename__ = "sync_metadata"

    id: Mapped[int] = mapped_column(primary_key=True)
    last_sync_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_changed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    sync_status: Mapped[str | None] = mapped_column(Text)
