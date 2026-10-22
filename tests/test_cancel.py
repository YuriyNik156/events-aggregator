import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

import main
from app import services
from app.events_provider import (
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.models import Ticket
from app.repositories import (
    TicketRepository,
)
from app.services import SeatsCache
from tests.conftest import (
    active_ticket,
    make_event,
    session_override,
    store_event,
    store_ticket,
)

SECRET = "super-secret-key"
FUTURE = datetime.now(UTC) + timedelta(days=30)
PAST = datetime.now(UTC) - timedelta(days=1)

BODY = {
    "event_id": "e1",
    "first_name": "Иван",
    "last_name": "Иванов",
    "email": "ivan@example.com",
    "seat": "A5",
}


@pytest.fixture
def provider() -> AsyncMock:
    mock = AsyncMock()
    mock.get_event_seats.return_value = ["A4", "A5"]
    mock.register.return_value = "t-1"
    return mock


@pytest.fixture
def cache() -> SeatsCache:
    return SeatsCache(ttl=30)


@pytest.fixture
def client(db, provider, cache):
    main.app.dependency_overrides[main.get_session] = session_override(db)
    main.app.dependency_overrides[main.get_provider_client] = lambda: provider
    main.app.dependency_overrides[main.get_seats_cache] = lambda: cache
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def seed(db, **overrides) -> None:
    overrides.setdefault("registration_deadline", FUTURE)
    overrides.setdefault("event_time", FUTURE + timedelta(days=1))
    store_event(db, make_event("e1", **overrides))
    db.commit()


def tickets(db) -> list[Ticket]:
    return list(db.scalars(select(Ticket)))


def register(client, **overrides):
    return client.post("/api/tickets", json={**BODY, **overrides})


# ---- success ----


def test_cancel_returns_200_and_keeps_cancelled_row(client, db, provider):
    seed(db)
    register(client)

    response = client.delete("/api/tickets/t-1")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    provider.unregister.assert_called_once_with("e1", "t-1")
    [ticket] = tickets(db)  # not physically deleted
    assert ticket.ticket_id == "t-1"
    assert ticket.cancelled_at is not None
    assert active_ticket(db, "t-1") is None


def test_cancel_invalidates_seats_cache(client, db, cache):
    seed(db)
    register(client)
    client.get("/api/events/e1/seats")
    assert cache.get("e1") is not None

    client.delete("/api/tickets/t-1")

    assert cache.get("e1") is None


# ---- not found ----


def test_unknown_ticket_404_without_provider_call(client, db, provider):
    seed(db)

    assert client.delete("/api/tickets/nope").status_code == 404
    provider.unregister.assert_not_called()


def test_repeated_cancel_404_without_second_provider_call(client, db, provider):
    seed(db)
    register(client)
    assert client.delete("/api/tickets/t-1").status_code == 200

    assert client.delete("/api/tickets/t-1").status_code == 404
    assert provider.unregister.call_count == 1


def test_started_event_400_without_provider_call(client, db, provider):
    seed(db, event_time=PAST)
    store_ticket(
        db,
        ticket_id="t-1", event_id="e1", first_name="a", last_name="b",
        email="a@b.cc", seat="A5",
    )  # fmt: skip
    db.commit()

    response = client.delete("/api/tickets/t-1")

    assert response.status_code == 400
    provider.unregister.assert_not_called()
    assert tickets(db)[0].cancelled_at is None


# ---- provider outcomes: never a false success ----


def test_provider_400_is_409_and_ticket_stays_active(client, db, provider, cache):
    seed(db)
    register(client)
    client.get("/api/events/e1/seats")
    provider.unregister.side_effect = ProviderRejectedError(400, "")

    response = client.delete("/api/tickets/t-1")

    assert response.status_code == 409
    assert tickets(db)[0].cancelled_at is None
    assert cache.get("e1") is not None  # unsuccessful: cache untouched


def test_provider_404_is_404_and_ticket_stays_active(client, db, provider):
    seed(db)
    register(client)
    provider.unregister.side_effect = ProviderRejectedError(404, "")

    assert client.delete("/api/tickets/t-1").status_code == 404
    assert tickets(db)[0].cancelled_at is None


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_other_provider_rejections_are_502(client, db, provider, status):
    seed(db)
    register(client)
    provider.unregister.side_effect = ProviderRejectedError(status, "<html>x</html>")

    response = client.delete("/api/tickets/t-1")

    assert response.status_code == 502
    assert response.json() == {"detail": "Events Provider request failed"}
    assert "<html>" not in response.text
    assert tickets(db)[0].cancelled_at is None


def test_provider_unavailable_is_502_and_not_retried(client, db, provider):
    seed(db)
    register(client)
    provider.unregister.side_effect = ProviderUnavailableError("timeout")

    assert client.delete("/api/tickets/t-1").status_code == 502
    assert provider.unregister.call_count == 1
    assert tickets(db)[0].cancelled_at is None


# ---- partial failure: provider confirmed, local save failed ----


def test_local_save_failure_reports_ticket_id_and_logs_no_personal_data(
    client, db, provider, cache, caplog, monkeypatch
):
    seed(db)
    register(client)
    client.get("/api/events/e1/seats")

    async def broken_cancel(self, ticket, cancelled_at):
        raise RuntimeError(f"db down for {ticket.email}")

    monkeypatch.setattr(TicketRepository, "cancel", broken_cancel)

    with caplog.at_level(logging.INFO):
        response = client.delete("/api/tickets/t-1")

    assert response.status_code == 500
    assert "t-1" in response.json()["detail"]
    assert "ivan@example.com" not in response.text
    assert provider.unregister.call_count == 1  # no automatic repeat
    assert cache.get("e1") is None  # the seat is free at the provider
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "ivan@example.com" not in logged
    assert "Иван" not in logged


# ---- concurrency ----


def test_in_progress_is_409(client, db, provider):
    seed(db)
    register(client)
    services._cancelling.add("t-1")
    try:
        assert client.delete("/api/tickets/t-1").status_code == 409
    finally:
        services._cancelling.discard("t-1")
    provider.unregister.assert_not_called()


# ---- ticket_id reuse after cancellation (end to end) ----


def test_ticket_id_reuse_after_cancellation(client, db, provider, cache):
    seed(db)
    assert register(client).status_code == 201  # user 1 on A5 -> t-1
    assert client.delete("/api/tickets/t-1").status_code == 200
    old = tickets(db)[0]
    assert old.cancelled_at is not None

    provider.register.return_value = "t-1"  # the provider hands out the same id
    other = {"first_name": "Пётр", "last_name": "Петров", "email": "petr@example.com"}
    assert register(client, **other).status_code == 201  # user 2 on A5

    rows = tickets(db)
    assert len(rows) == 2  # no uniqueness conflict, history kept
    new = active_ticket(db, "t-1")
    assert new is not None and new.id != old.id
    assert new.email == "petr@example.com"
    assert old.cancelled_at is not None and old.email == "ivan@example.com"

    assert client.delete("/api/tickets/t-1").status_code == 200
    assert all(t.cancelled_at is not None for t in tickets(db))
    assert provider.unregister.call_count == 2
    assert client.delete("/api/tickets/t-1").status_code == 404
