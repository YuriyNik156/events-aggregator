import json
import logging
import threading
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import main
from app import services
from app.events_provider import (
    EventsProviderClient,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.models import Base, Ticket
from app.repositories import EventRepository, TicketRepository
from app.services import SeatsCache, cancel_ticket
from tests.conftest import make_event

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
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    engine.dispose()


@pytest.fixture
def provider() -> Mock:
    mock = Mock()
    mock.get_event_seats.return_value = ["A4", "A5"]
    mock.register.return_value = "t-1"
    return mock


@pytest.fixture
def cache() -> SeatsCache:
    return SeatsCache(ttl=30)


@pytest.fixture
def client(db, provider, cache):
    main.app.dependency_overrides[main.get_session] = lambda: db
    main.app.dependency_overrides[main.get_provider_client] = lambda: provider
    main.app.dependency_overrides[main.get_seats_cache] = lambda: cache
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def seed(db, **overrides) -> None:
    overrides.setdefault("registration_deadline", FUTURE)
    overrides.setdefault("event_time", FUTURE + timedelta(days=1))
    EventRepository(db).upsert(make_event("e1", **overrides))
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
    assert TicketRepository(db).get_active("t-1") is None


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
    TicketRepository(db).add(
        ticket_id="t-1", event_id="e1", first_name="a", last_name="b",
        email="a@b.cc", seat="A5",
    )  # fmt: skip
    db.commit()

    response = client.delete("/api/tickets/t-1")

    assert response.status_code == 400
    provider.unregister.assert_not_called()
    assert tickets(db)[0].cancelled_at is None


def test_event_time_boundary_is_compared_in_utc(db):
    seed(db, event_time=datetime(2026, 1, 10, 7, 0, tzinfo=UTC))
    TicketRepository(db).add(
        ticket_id="t", event_id="e1", first_name="a", last_name="b",
        email="a@b.cc", seat="A5",
    )  # fmt: skip
    db.commit()
    provider = Mock()

    with pytest.raises(services.EventStartedError):
        cancel_ticket(
            db, provider, SeatsCache(), "t", now=datetime(2026, 1, 10, 7, 0, tzinfo=UTC)
        )
    cancel_ticket(
        db, provider, SeatsCache(), "t", now=datetime(2026, 1, 10, 6, 59, tzinfo=UTC)
    )
    assert tickets(db)[0].cancelled_at is not None


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

    def broken_cancel(self, ticket, cancelled_at):
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


def test_concurrent_cancel_sends_one_delete(db):
    seed(db)
    TicketRepository(db).add(
        ticket_id="t", event_id="e1", first_name="a", last_name="b",
        email="a@b.cc", seat="A5",
    )  # fmt: skip
    db.commit()
    entered, release = threading.Event(), threading.Event()
    provider = Mock()

    def slow_unregister(event_id, ticket_id):
        entered.set()
        assert release.wait(5)

    provider.unregister.side_effect = slow_unregister
    first = threading.Thread(
        target=cancel_ticket, args=(db, provider, SeatsCache(), "t")
    )
    first.start()
    assert entered.wait(5)

    with pytest.raises(services.CancellationInProgressError):
        cancel_ticket(db, provider, SeatsCache(), "t")

    release.set()
    first.join(5)
    assert provider.unregister.call_count == 1
    assert tickets(db)[0].cancelled_at is not None
    assert services._cancelling == set()


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
    new = TicketRepository(db).get_active("t-1")
    assert new is not None and new.id != old.id
    assert new.email == "petr@example.com"
    assert old.cancelled_at is not None and old.email == "ivan@example.com"

    assert client.delete("/api/tickets/t-1").status_code == 200
    assert all(t.cancelled_at is not None for t in tickets(db))
    assert provider.unregister.call_count == 2
    assert client.delete("/api/tickets/t-1").status_code == 404


# ---- provider client ----


def transport_client(handler):
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    http = httpx.Client(
        base_url="http://provider:8000",
        transport=httpx.MockTransport(record),
        follow_redirects=True,
    )
    return EventsProviderClient("http://provider:8000", SECRET, http_client=http), seen


def test_client_sends_exact_contract():
    client, seen = transport_client(
        lambda r: httpx.Response(200, json={"success": True})
    )

    client.unregister("e1", "1fed0122")

    [request] = seen
    assert request.method == "DELETE"
    assert request.url.path == "/api/events/e1/unregister/"
    assert request.headers["x-api-key"] == SECRET
    assert json.loads(request.read()) == {"ticket_id": "1fed0122"}


def test_client_quotes_event_id():
    client, seen = transport_client(
        lambda r: httpx.Response(200, json={"success": True})
    )

    client.unregister("a/../b", "t")

    assert seen[0].url.raw_path == b"/api/events/a%2F..%2Fb/unregister/"


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500])
def test_client_rejection_keeps_status_only(status):
    client, _ = transport_client(
        lambda r: httpx.Response(status, text="<html>t</html>")
    )

    with pytest.raises(ProviderRejectedError) as info:
        client.unregister("e1", "t")

    assert info.value.status_code == status
    assert "<html>" not in str(info.value) + info.value.body


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>oops</html>"),
        httpx.Response(200, text="{not json"),
        httpx.Response(200, json={}),
        httpx.Response(200, json={"success": False}),
        httpx.Response(200, json={"success": None}),
    ],
)
def test_client_unconfirmed_response_is_unavailable(response):
    client, _ = transport_client(lambda r: response)

    with pytest.raises(ProviderUnavailableError):
        client.unregister("e1", "t")


def test_client_timeout_is_unavailable_without_key_and_not_repeated():
    def boom(request):
        raise httpx.ReadTimeout("timeout", request=request)

    client, seen = transport_client(boom)

    with pytest.raises(ProviderUnavailableError) as info:
        client.unregister("e1", "t")

    assert SECRET not in str(info.value)
    assert len(seen) == 1


def test_client_does_not_follow_redirect_with_api_key():
    client, seen = transport_client(
        lambda r: httpx.Response(302, headers={"location": "http://evil.example/x"})
    )

    with pytest.raises(ProviderRejectedError):
        client.unregister("e1", "t")

    assert [r.url.host for r in seen] == ["provider"]


# ---- _cancelling is released on every outcome ----


@pytest.mark.parametrize(
    "error",
    [
        ProviderRejectedError(400, ""),
        ProviderRejectedError(404, ""),
        ProviderRejectedError(500, ""),
        ProviderUnavailableError("timeout"),
        RuntimeError("unexpected"),
    ],
)
def test_lock_released_after_provider_failure(db, error):
    seed(db)
    TicketRepository(db).add(
        ticket_id="t", event_id="e1", first_name="a", last_name="b",
        email="a@b.cc", seat="A5",
    )  # fmt: skip
    db.commit()
    provider = Mock()
    provider.unregister.side_effect = error

    with pytest.raises(
        type(error) if not isinstance(error, ProviderRejectedError) else Exception
    ):
        cancel_ticket(db, provider, SeatsCache(), "t")

    assert services._cancelling == set()


def test_lock_released_after_local_save_failure(db, monkeypatch):
    seed(db)
    TicketRepository(db).add(
        ticket_id="t", event_id="e1", first_name="a", last_name="b",
        email="a@b.cc", seat="A5",
    )  # fmt: skip
    db.commit()

    def broken(self, ticket, cancelled_at):
        raise RuntimeError("db down")

    monkeypatch.setattr(TicketRepository, "cancel", broken)

    with pytest.raises(services.CancellationNotSavedError):
        cancel_ticket(db, Mock(), SeatsCache(), "t")

    assert services._cancelling == set()


def test_lock_released_for_unknown_ticket(db):
    seed(db)

    with pytest.raises(services.TicketNotFoundError):
        cancel_ticket(db, Mock(), SeatsCache(), "nope")

    assert services._cancelling == set()


def test_waiter_after_finished_cancellation_gets_404_without_delete(db):
    seed(db)
    TicketRepository(db).add(
        ticket_id="t", event_id="e1", first_name="a", last_name="b",
        email="a@b.cc", seat="A5",
    )  # fmt: skip
    db.commit()
    provider = Mock()
    cancel_ticket(db, provider, SeatsCache(), "t")

    with pytest.raises(services.TicketNotFoundError):
        cancel_ticket(db, provider, SeatsCache(), "t")

    assert provider.unregister.call_count == 1
