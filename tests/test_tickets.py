import json
import logging
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import main
from app.events_provider import (
    EventsProviderClient,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.models import Base, Ticket
from app.repositories import EventRepository, TicketRepository
from app.services import (
    RegistrationClosedError,
    SeatsCache,
    register_ticket,
    seat_in_pattern,
)
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
    EventRepository(db).upsert(make_event("e1", **overrides))
    db.commit()


def tickets(db) -> list[Ticket]:
    return list(db.scalars(select(Ticket)))


# ---- success ----


def test_register_returns_201_and_saves_ticket(client, db, provider):
    seed(db)

    response = client.post("/api/tickets", json=BODY)

    assert response.status_code == 201
    assert response.json() == {"ticket_id": "t-1"}
    [ticket] = tickets(db)
    assert (ticket.ticket_id, ticket.event_id, ticket.seat) == ("t-1", "e1", "A5")
    assert (ticket.first_name, ticket.last_name) == ("Иван", "Иванов")
    assert ticket.email == "ivan@example.com"
    assert ticket.cancelled_at is None
    provider.register.assert_called_once_with(
        "e1", first_name="Иван", last_name="Иванов", seat="A5", email="ivan@example.com"
    )


def test_success_invalidates_seats_cache(client, db, cache):
    seed(db)
    client.get("/api/events/e1/seats")
    assert cache.get("e1") is not None

    client.post("/api/tickets", json=BODY)

    assert cache.get("e1") is None


def test_no_deadline_means_open_registration(client, db):
    seed(db, registration_deadline=None)

    assert client.post("/api/tickets", json=BODY).status_code == 201


# ---- invalid body: 400 on this route (LMS contract) ----


@pytest.mark.parametrize("field", list(BODY))
def test_missing_field_is_400(client, db, provider, field):
    seed(db)

    response = client.post(
        "/api/tickets", json={k: v for k, v in BODY.items() if k != field}
    )

    assert response.status_code == 400
    assert provider.mock_calls == []


@pytest.mark.parametrize("field", list(BODY))
@pytest.mark.parametrize("value", ["", "   "])
def test_blank_field_is_400(client, db, provider, field, value):
    seed(db)

    assert client.post("/api/tickets", json={**BODY, field: value}).status_code == 400
    assert provider.mock_calls == []


@pytest.mark.parametrize(
    "email", ["ivan", "ivan@", "@example.com", "ivan@example", "a b@example.com"]
)
def test_invalid_email_is_400(client, db, provider, email):
    seed(db)

    assert client.post("/api/tickets", json={**BODY, "email": email}).status_code == 400
    assert provider.mock_calls == []


# ---- local business checks ----


def test_unknown_event_404(client, provider):
    response = client.post("/api/tickets", json=BODY)

    assert response.status_code == 404
    assert provider.mock_calls == []


def test_unpublished_event_400(client, db, provider):
    seed(db, status="draft")

    assert client.post("/api/tickets", json=BODY).status_code == 400
    provider.register.assert_not_called()


def test_expired_deadline_400(client, db, provider):
    seed(db, registration_deadline=PAST)

    response = client.post("/api/tickets", json=BODY)

    assert response.status_code == 400
    assert response.json() == {"detail": "Registration is closed"}
    provider.register.assert_not_called()
    assert tickets(db) == []


def test_deadline_boundary_is_compared_in_utc(client, db):
    # SQLite returns a naive value, which the service reads as UTC (as PostgreSQL's
    # timestamptz gives UTC-equivalent instants).
    seed(db, registration_deadline=datetime(2026, 1, 10, 7, 0, tzinfo=UTC))
    args = dict(first_name="a", last_name="b", email="a@b.cc", seat="A5")
    cache = SeatsCache()
    provider = Mock()
    provider.get_event_seats.return_value = ["A5"]
    provider.register.return_value = "t"

    with pytest.raises(RegistrationClosedError):
        register_ticket(
            db, provider, cache, event_id="e1",
            now=datetime(2026, 1, 10, 7, 30, tzinfo=UTC), **args,
        )  # fmt: skip
    assert (
        register_ticket(
            db,
            provider,
            cache,
            event_id="e1",
            now=datetime(2026, 1, 10, 6, 30, tzinfo=UTC),
            **args,
        )  # fmt: skip
        == "t"
    )


def test_seat_outside_pattern_400(client, db, provider):
    seed(db, place_seats_pattern="A1-10")

    response = client.post("/api/tickets", json={**BODY, "seat": "A11"})

    assert response.status_code == 400
    assert response.json() == {"detail": "Seat does not exist"}
    provider.register.assert_not_called()


def test_seat_not_free_409_without_registration(client, db, provider):
    seed(db)
    provider.get_event_seats.return_value = ["A4"]

    response = client.post("/api/tickets", json=BODY)

    assert response.status_code == 409
    provider.register.assert_not_called()
    assert tickets(db) == []


@pytest.mark.parametrize(
    ("seat", "pattern", "expected"),
    [
        ("A1", "A1-1000,B1-2000", True),
        ("A1000", "A1-1000,B1-2000", True),
        ("B1500", "A1-1000,B1-2000", True),
        ("A1001", "A1-1000,B1-2000", False),
        ("A0", "A1-1000", False),
        ("C1", "A1-1000,B1-2000", False),
        ("a1", "A1-10", False),
        ("A", "A1-10", False),
        ("15", "A1-10", False),
        ("A5", "A5", True),
        ("A6", "A5", False),
        ("A99999999999", "A1-1000000000000", True),  # ranges are never expanded
        ("A5", "weird", True),  # unknown pattern format: the provider decides
        ("Z9", "A1-10,weird", True),
    ],
)
def test_seat_in_pattern(seat, pattern, expected):
    assert seat_in_pattern(seat, pattern) is expected


# ---- provider outcomes ----


def test_provider_400_is_409_and_not_saved(client, db, provider, cache):
    seed(db)
    client.get("/api/events/e1/seats")
    provider.register.side_effect = ProviderRejectedError(400, "")

    response = client.post("/api/tickets", json=BODY)

    assert response.status_code == 409
    assert tickets(db) == []
    assert cache.get("e1") is not None  # unsuccessful: cache untouched


def test_provider_404_is_404(client, db, provider):
    seed(db)
    provider.register.side_effect = ProviderRejectedError(404, "")

    assert client.post("/api/tickets", json=BODY).status_code == 404
    assert tickets(db) == []


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_other_provider_rejections_are_502(client, db, provider, cache, status):
    seed(db)
    client.get("/api/events/e1/seats")
    provider.register.side_effect = ProviderRejectedError(status, "")

    response = client.post("/api/tickets", json=BODY)

    assert response.status_code == 502
    assert response.json() == {"detail": "Events Provider request failed"}
    assert tickets(db) == []
    assert cache.get("e1") is not None


def test_provider_unavailable_is_502_and_not_retried(client, db, provider):
    seed(db)
    provider.register.side_effect = ProviderUnavailableError("timeout")

    assert client.post("/api/tickets", json=BODY).status_code == 502
    assert provider.register.call_count == 1
    assert tickets(db) == []


def test_seats_check_failure_is_502_without_registration(client, db, provider):
    seed(db)
    provider.get_event_seats.side_effect = ProviderUnavailableError("down")

    assert client.post("/api/tickets", json=BODY).status_code == 502
    provider.register.assert_not_called()


# ---- partial failure: provider confirmed, local save failed ----


def test_local_save_failure_reports_ticket_id_and_logs_no_personal_data(
    client, db, provider, cache, caplog, monkeypatch
):
    seed(db)
    client.get("/api/events/e1/seats")

    def broken_add(self, **fields):
        raise RuntimeError(f"db down for {fields['email']}")

    monkeypatch.setattr(TicketRepository, "add", broken_add)

    with caplog.at_level(logging.INFO):
        response = client.post("/api/tickets", json=BODY)

    assert response.status_code == 500
    assert "t-1" in response.json()["detail"]
    assert "ivan@example.com" not in response.text
    assert provider.register.call_count == 1  # no automatic repeat
    assert cache.get("e1") is None  # the seat is taken at the provider
    assert tickets(db) == []
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "t-1" in logged
    assert "ivan@example.com" not in logged
    assert "Иван" not in logged


def test_error_responses_do_not_leak_key_or_personal_data(client, db, provider):
    seed(db)
    provider.register.side_effect = ProviderRejectedError(500, "<html>ivan</html>")

    response = client.post("/api/tickets", json=BODY)

    for secret in (SECRET, "ivan@example.com", "Иван", "<html>"):
        assert secret not in response.text


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


def register(client):
    return client.register(
        "e1",
        first_name="Иван",
        last_name="Иванов",
        seat="A15",
        email="ivan@example.com",
    )


def test_client_sends_exact_contract():
    client, seen = transport_client(
        lambda r: httpx.Response(201, json={"ticket_id": "1fed0122"})
    )

    assert register(client) == "1fed0122"

    [request] = seen
    assert request.method == "POST"
    assert request.url.path == "/api/events/e1/register/"
    assert request.headers["x-api-key"] == SECRET
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.read()) == {
        "first_name": "Иван",
        "last_name": "Иванов",
        "seat": "A15",
        "email": "ivan@example.com",
    }


def test_client_quotes_event_id():
    client, seen = transport_client(
        lambda r: httpx.Response(201, json={"ticket_id": "t"})
    )

    client.register("a/../b", first_name="a", last_name="b", seat="A1", email="a@b.cc")

    assert seen[0].url.raw_path == b"/api/events/a%2F..%2Fb/register/"


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500])
def test_client_rejection_keeps_status_only(status):
    client, _ = transport_client(
        lambda r: httpx.Response(status, text="<html>ivan@example.com</html>")
    )

    with pytest.raises(ProviderRejectedError) as info:
        register(client)

    assert info.value.status_code == status
    leaked = str(info.value) + info.value.body + SECRET
    assert "ivan@example.com" not in leaked.replace(SECRET, "")


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(201, text="<html>oops</html>"),
        httpx.Response(201, text="{not json"),
        httpx.Response(201, json={}),
        httpx.Response(201, json={"ticket_id": ""}),
        httpx.Response(201, json={"ticket_id": None}),
    ],
)
def test_client_invalid_body_is_unavailable(response):
    client, _ = transport_client(lambda r: response)

    with pytest.raises(ProviderUnavailableError):
        register(client)


def test_client_timeout_is_unavailable_without_key_or_email():
    def boom(request):
        raise httpx.ReadTimeout("timeout", request=request)

    client, seen = transport_client(boom)

    with pytest.raises(ProviderUnavailableError) as info:
        register(client)

    assert SECRET not in str(info.value)
    assert "ivan" not in str(info.value)
    assert len(seen) == 1  # one POST, never repeated


def test_client_does_not_follow_redirect_with_api_key():
    client, seen = transport_client(
        lambda r: httpx.Response(302, headers={"location": "http://evil.example/x"})
    )

    with pytest.raises(ProviderRejectedError):
        register(client)

    assert [r.url.host for r in seen] == ["provider"]


# ---- storage: ticket_id reuse after cancellation ----


def test_second_registration_may_reuse_cancelled_ticket_id(db):
    seed(db)
    repo = TicketRepository(db)
    fields = dict(event_id="e1", first_name="a", last_name="b", email="a@b.cc")
    first = repo.add(ticket_id="T", seat="A15", **fields)
    repo.cancel(first, datetime.now(UTC))
    second = repo.add(ticket_id="T", seat="A15", **fields)
    db.commit()

    assert repo.get_active("T") is second
    assert {t.id for t in tickets(db)} == {first.id, second.id}  # history kept
    assert first.cancelled_at is not None


def test_active_ticket_id_cannot_be_registered_twice(db):
    seed(db)
    repo = TicketRepository(db)
    fields = dict(event_id="e1", first_name="a", last_name="b", email="a@b.cc")
    repo.add(ticket_id="T", seat="A1", **fields)

    with pytest.raises(IntegrityError):
        repo.add(ticket_id="T", seat="A2", **fields)


def test_invalid_body_error_does_not_echo_input(client, db):
    seed(db)

    response = client.post("/api/tickets", json={**BODY, "email": "secret-not-email"})

    assert response.status_code == 400
    assert "secret-not-email" not in response.text
    assert response.json()["detail"][0]["loc"] == ["body", "email"]


def test_malformed_json_is_400(client, db, provider):
    seed(db)

    response = client.post(
        "/api/tickets", content="{oops", headers={"content-type": "application/json"}
    )

    assert response.status_code == 400
    assert provider.mock_calls == []


def test_other_endpoints_keep_422(client):
    assert client.get("/api/events?page=0").status_code == 422
    assert client.get("/api/events?page=abc").status_code == 422
