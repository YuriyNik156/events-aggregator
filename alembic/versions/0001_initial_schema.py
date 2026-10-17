"""initial schema: events, tickets, sync_metadata

Revision ID: 0001
Revises:
Create Date: 2026-10-17
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "events",
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("place_id", sa.Text(), nullable=False),
        sa.Column("place_name", sa.Text(), nullable=False),
        sa.Column("place_city", sa.Text(), nullable=False),
        sa.Column("place_address", sa.Text(), nullable=False),
        sa.Column("place_seats_pattern", sa.Text(), nullable=False),
        sa.Column("event_time", sa.DateTime(timezone=True), nullable=False),
        sa.Column("registration_deadline", sa.DateTime(timezone=True), nullable=True),
        sa.Column("status", sa.Text(), nullable=False),
        sa.Column(
            "number_of_visitors",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column("changed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id", name="pk_events"),
    )
    op.create_index("ix_events_event_time", "events", ["event_time"])
    op.create_index("ix_events_changed_at", "events", ["changed_at"])

    op.create_table(
        "tickets",
        sa.Column("id", sa.Integer(), sa.Identity(), nullable=False),
        sa.Column("ticket_id", sa.Text(), nullable=False),
        sa.Column("event_id", sa.Text(), nullable=False),
        sa.Column("first_name", sa.Text(), nullable=False),
        sa.Column("last_name", sa.Text(), nullable=False),
        sa.Column("email", sa.Text(), nullable=False),
        sa.Column("seat", sa.Text(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("cancelled_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(
            ["event_id"], ["events.id"], name="fk_tickets_event_id_events"
        ),
        sa.PrimaryKeyConstraint("id", name="pk_tickets"),
    )
    op.create_index("ix_tickets_event_id", "tickets", ["event_id"])
    op.create_index(
        "uq_tickets_ticket_id_active",
        "tickets",
        ["ticket_id"],
        unique=True,
        postgresql_where=sa.text("cancelled_at IS NULL"),
    )

    op.create_table(
        "sync_metadata",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("last_sync_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_changed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sync_status", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_sync_metadata"),
    )


def downgrade() -> None:
    op.drop_table("sync_metadata")
    op.drop_index("uq_tickets_ticket_id_active", table_name="tickets")
    op.drop_index("ix_tickets_event_id", table_name="tickets")
    op.drop_table("tickets")
    op.drop_index("ix_events_changed_at", table_name="events")
    op.drop_index("ix_events_event_time", table_name="events")
    op.drop_table("events")
