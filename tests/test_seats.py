from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import main
from app.events_provider import (
    EventsProviderClient,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from app.models import Base
from app.repositories import EventRepository
from app.services import SeatsCache
from tests.conftest import make_event

SECRET = "super-secret-key"


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


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
    return Mock()


@pytest.fixture
def cache(clock) -> SeatsCache:
    return SeatsCache(ttl=30, clock=clock)


@pytest.fixture
def client(db, provider, cache):
    main.app.dependency_overrides[main.get_session] = lambda: db
    main.app.dependency_overrides[main.get_provider_client] = lambda: provider
    main.app.dependency_overrides[main.get_seats_cache] = lambda: cache
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def seed(db, *events: dict) -> None:
    repo = EventRepository(db)
    for e in events:
        repo.upsert(e)
    db.commit()


# ---- route / service ----


def test_seats_success_maps_to_available_seats(client, db, provider):
    seed(db, make_event("e1"))
    provider.get_event_seats.return_value = ["A1", "A3", "B1"]

    response = client.get("/api/events/e1/seats")

    assert response.status_code == 200
    assert response.json() == {"event_id": "e1", "available_seats": ["A1", "A3", "B1"]}
    provider.get_event_seats.assert_called_once_with("e1")


def test_unknown_event_404_without_provider_call(client, provider):
    response = client.get("/api/events/missing/seats")

    assert response.status_code == 404
    assert provider.mock_calls == []


def test_unpublished_event_400_without_provider_call(client, db, provider):
    seed(db, make_event("e1", status="draft"))

    response = client.get("/api/events/e1/seats")

    assert response.status_code == 400
    assert provider.mock_calls == []


def test_empty_seats_returned_and_cached(client, db, provider):
    seed(db, make_event("e1"))
    provider.get_event_seats.return_value = []

    first = client.get("/api/events/e1/seats")
    second = client.get("/api/events/e1/seats")

    assert first.json() == second.json() == {"event_id": "e1", "available_seats": []}
    assert provider.get_event_seats.call_count == 1


def test_cache_hit_within_ttl_and_refresh_after(client, db, provider, clock):
    seed(db, make_event("e1"))
    provider.get_event_seats.side_effect = [["A1"], ["A2"]]

    assert client.get("/api/events/e1/seats").json()["available_seats"] == ["A1"]
    clock.now += 29.9
    assert client.get("/api/events/e1/seats").json()["available_seats"] == ["A1"]
    assert provider.get_event_seats.call_count == 1
    clock.now += 0.2
    assert client.get("/api/events/e1/seats").json()["available_seats"] == ["A2"]
    assert provider.get_event_seats.call_count == 2


def test_cache_is_per_event(client, db, provider):
    seed(db, make_event("e1"), make_event("e2"))
    provider.get_event_seats.side_effect = lambda eid: [f"{eid}-A"]

    assert client.get("/api/events/e1/seats").json()["available_seats"] == ["e1-A"]
    assert client.get("/api/events/e2/seats").json()["available_seats"] == ["e2-A"]
    assert provider.get_event_seats.call_count == 2


def test_provider_error_is_not_cached_or_turned_into_empty(client, db, provider):
    seed(db, make_event("e1"))
    provider.get_event_seats.side_effect = [ProviderUnavailableError("down"), ["A1"]]

    failed = client.get("/api/events/e1/seats")
    ok = client.get("/api/events/e1/seats")

    assert failed.status_code == 502
    assert failed.json() == {"detail": "Events Provider request failed"}
    assert ok.json()["available_seats"] == ["A1"]


@pytest.mark.parametrize("status", [401, 429, 500, 503])
def test_provider_rejection_returns_502(client, db, provider, status):
    seed(db, make_event("e1"))
    provider.get_event_seats.side_effect = ProviderRejectedError(status, "<html>")

    response = client.get("/api/events/e1/seats")

    assert response.status_code == 502
    assert "html" not in response.text


def test_provider_404_returns_404_and_drops_cache(client, db, provider, cache):
    seed(db, make_event("e1"))
    provider.get_event_seats.side_effect = ProviderRejectedError(404, "nope")

    response = client.get("/api/events/e1/seats")

    assert response.status_code == 404
    assert cache.get("e1") is None


def test_invalidate_resets_entry(client, db, provider, cache):
    seed(db, make_event("e1"), make_event("e2"))
    provider.get_event_seats.return_value = ["A1"]
    client.get("/api/events/e1/seats")
    client.get("/api/events/e2/seats")

    cache.invalidate("e1")

    assert cache.get("e1") is None
    assert cache.get("e2") == ["A1"]
    cache.invalidate("unknown")  # no error


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


def test_client_requests_trailing_slash_path_with_api_key():
    client, seen = transport_client(
        lambda r: httpx.Response(200, json={"seats": ["A1"]})
    )

    assert client.get_event_seats("550e8400-e29b-41d4-a716-446655440000") == ["A1"]

    assert seen[0].url.path == "/api/events/550e8400-e29b-41d4-a716-446655440000/seats/"
    assert seen[0].headers["x-api-key"] == SECRET


def test_client_quotes_event_id():
    client, seen = transport_client(lambda r: httpx.Response(200, json={"seats": []}))

    client.get_event_seats("a/../b")

    assert seen[0].url.raw_path == b"/api/events/a%2F..%2Fb/seats/"


@pytest.mark.parametrize("status", [401, 404, 429, 500, 503])
def test_client_rejection_has_status_and_no_key(status):
    client, _ = transport_client(
        lambda r: httpx.Response(status, text="<html>x</html>")
    )

    with pytest.raises(ProviderRejectedError) as info:
        client.get_event_seats("e1")

    assert info.value.status_code == status
    assert SECRET not in str(info.value) + info.value.body


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="<html>UnexpectedEventStatus</html>"),
        httpx.Response(200, text="{not json"),
        httpx.Response(200, json={"seats": "A1"}),
        httpx.Response(200, json={"other": []}),
        httpx.Response(200, json={"seats": [1, 2]}),
    ],
)
def test_client_invalid_body_is_unavailable(response):
    client, _ = transport_client(lambda r: response)

    with pytest.raises(ProviderUnavailableError) as info:
        client.get_event_seats("e1")

    assert SECRET not in str(info.value)


def test_client_timeout_has_no_key():
    def boom(request):
        raise httpx.ReadTimeout("timeout", request=request)

    client, _ = transport_client(boom)

    with pytest.raises(ProviderUnavailableError) as info:
        client.get_event_seats("e1")

    assert SECRET not in str(info.value)


def test_client_does_not_follow_redirect_with_api_key():
    client, seen = transport_client(
        lambda r: httpx.Response(302, headers={"location": "http://evil.example/x"})
    )

    with pytest.raises(ProviderRejectedError):
        client.get_event_seats("e1")

    assert [r.url.host for r in seen] == ["provider"]
