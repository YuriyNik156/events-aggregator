from collections.abc import Iterator
from datetime import UTC, datetime
from unittest.mock import Mock
from urllib.parse import parse_qs, urlparse

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

import main
from app.config import ConfigError
from app.events_provider import EventsPage, ProviderUnavailableError
from app.models import Base
from app.repositories import EventRepository
from tests.conftest import make_event, provider_event_json, provider_page


@pytest.fixture
def db() -> Iterator[Session]:
    # One shared connection: TestClient runs sync endpoints in another thread.
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    engine.dispose()


@pytest.fixture
def provider() -> Mock:
    return Mock()


@pytest.fixture
def client(db, provider) -> Iterator[TestClient]:
    main.app.dependency_overrides[main.get_session] = lambda: db
    main.app.dependency_overrides[main.get_provider_client] = lambda: provider
    yield TestClient(main.app)
    main.app.dependency_overrides.clear()


def seed(db, *events: dict) -> None:
    repo = EventRepository(db)
    for e in events:
        repo.upsert(e)
    db.commit()


def test_empty_database_returns_all_fields(client):
    response = client.get("/api/events")

    assert response.status_code == 200
    assert response.json() == {
        "count": 0,
        "next": None,
        "previous": None,
        "results": [],
    }


def test_event_structure(client, db):
    seed(db, make_event("e1"))

    body = client.get("/api/events").json()

    assert body["count"] == 1
    event = body["results"][0]
    assert set(event) == {
        "id",
        "name",
        "place",
        "event_time",
        "registration_deadline",
        "status",
        "number_of_visitors",
    }
    assert event["place"] == {
        "id": "p1",
        "name": "Hall",
        "city": "Moscow",
        "address": "Main st. 1",
    }
    assert event["status"] == "published"


def test_pagination_links_and_count(client, db):
    seed(db, *[make_event(f"e{i}") for i in range(5)])

    first = client.get("/api/events", params={"page_size": 2}).json()
    middle = client.get("/api/events", params={"page": 2, "page_size": 2}).json()
    last = client.get("/api/events", params={"page": 3, "page_size": 2}).json()

    assert first["count"] == middle["count"] == last["count"] == 5
    assert [len(p["results"]) for p in (first, middle, last)] == [2, 2, 1]
    assert first["previous"] is None
    assert last["next"] is None
    assert urlparse(first["next"]).path == "/api/events"
    assert parse_qs(urlparse(first["next"]).query) == {
        "page_size": ["2"],
        "page": ["2"],
    }
    assert parse_qs(urlparse(middle["previous"]).query)["page"] == ["1"]
    assert parse_qs(urlparse(middle["next"]).query)["page"] == ["3"]


def test_date_from_filters_and_is_kept_in_links(client, db):
    seed(
        db,
        make_event("old", event_time=datetime(2026, 1, 1, 23, 59, tzinfo=UTC)),
        make_event("edge", event_time=datetime(2026, 1, 2, 0, 0, tzinfo=UTC)),
        make_event("new", event_time=datetime(2026, 1, 3, tzinfo=UTC)),
    )

    body = client.get(
        "/api/events", params={"date_from": "2026-01-02", "page_size": 1}
    ).json()

    assert body["count"] == 2
    assert [e["id"] for e in body["results"]] == ["edge"]
    assert parse_qs(urlparse(body["next"]).query)["date_from"] == ["2026-01-02"]


def test_page_past_the_end_is_empty_not_404(client, db):
    seed(db, make_event("e1"))

    response = client.get("/api/events", params={"page": 5})

    assert response.status_code == 200
    assert response.json()["results"] == []


@pytest.mark.parametrize(
    "params",
    [
        {"page": 0},
        {"page": "x"},
        {"page_size": 0},
        {"page_size": -1},
        {"date_from": "yesterday"},
    ],
)
def test_invalid_query_returns_422(client, params):
    assert client.get("/api/events", params=params).status_code == 422


def test_sync_trigger_fills_event_list(client, provider):
    provider.get_events_page.return_value = EventsPage.model_validate(
        provider_page([provider_event_json("e1")])
    )

    response = client.post("/api/sync/trigger")

    assert response.status_code == 200
    assert response.json() == {"success": True}
    assert client.get("/api/events").json()["count"] == 1


def test_sync_trigger_provider_failure_returns_502(client, provider):
    provider.get_events_page.side_effect = ProviderUnavailableError("down")

    response = client.post("/api/sync/trigger")

    assert response.status_code == 502
    assert client.get("/api/events").json()["count"] == 0


def test_sync_trigger_without_api_key_returns_500(db, monkeypatch):
    monkeypatch.delenv("EVENTS_PROVIDER_API_KEY", raising=False)
    main.app.dependency_overrides[main.get_session] = lambda: db
    try:
        response = TestClient(main.app).post("/api/sync/trigger")
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 500
    assert response.json() == {"detail": "Events Provider is not configured"}


def test_health_still_works(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_event_detail_returns_place_with_seats_pattern(client, db, provider):
    seed(db, make_event("e1", place_seats_pattern="A1-10,B1-20"))

    response = client.get("/api/events/e1")

    assert response.status_code == 200
    body = response.json()
    # SQLite drops the UTC offset, so the date fields are checked by prefix only.
    assert body.pop("event_time").startswith("2026-01-11T14:00:00")
    assert body.pop("registration_deadline").startswith("2026-01-10T14:00:00")
    assert body == {
        "id": "e1",
        "name": "Concert",
        "place": {
            "id": "p1",
            "name": "Hall",
            "city": "Moscow",
            "address": "Main st. 1",
            "seats_pattern": "A1-10,B1-20",
        },
        "status": "published",
        "number_of_visitors": 0,
    }
    assert provider.mock_calls == []


def test_event_detail_with_null_registration_deadline(client, db):
    seed(db, make_event("e1", registration_deadline=None))

    response = client.get("/api/events/e1")

    assert response.status_code == 200
    assert response.json()["registration_deadline"] is None


def test_event_detail_unknown_event_returns_404(client, db, provider):
    seed(db, make_event("e1"))

    response = client.get("/api/events/missing")

    assert response.status_code == 404
    assert provider.mock_calls == []


def test_list_still_hides_seats_pattern_after_detail_route(client, db):
    seed(db, make_event("e1"))

    place = client.get("/api/events").json()["results"][0]["place"]

    assert "seats_pattern" not in place


PUBLIC = "https://events.example.org"


def seed_pages(db, n=5):
    seed(db, *[make_event(f"e{i}") for i in range(n)])


def test_links_use_public_base_url(client, db, monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", PUBLIC)
    seed_pages(db)

    first = client.get("/api/events", params={"page_size": 2}).json()
    middle = client.get("/api/events", params={"page": 2, "page_size": 2}).json()

    assert first["next"] == f"{PUBLIC}/api/events?page_size=2&page=2"
    assert first["previous"] is None
    assert middle["previous"] == f"{PUBLIC}/api/events?page_size=2&page=1"
    assert middle["next"] == f"{PUBLIC}/api/events?page_size=2&page=3"


def test_public_base_url_keeps_filters_without_duplicating_page(
    client, db, monkeypatch
):
    monkeypatch.setenv("PUBLIC_BASE_URL", PUBLIC)
    seed_pages(db)

    body = client.get(
        "/api/events", params={"date_from": "2020-01-01", "page": 2, "page_size": 1}
    ).json()

    for link, page in ((body["next"], "3"), (body["previous"], "1")):
        parts = urlparse(link)
        assert f"{parts.scheme}://{parts.netloc}" == PUBLIC
        assert parts.path == "/api/events"
        assert parse_qs(parts.query) == {
            "date_from": ["2020-01-01"],
            "page_size": ["1"],
            "page": [page],
        }


def test_public_base_url_ignores_host_and_forwarded_headers(client, db, monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", PUBLIC)
    seed_pages(db)

    body = client.get(
        "/api/events",
        params={"page_size": 2},
        headers={
            "Host": "evil.example",
            "X-Forwarded-Host": "evil.example",
            "X-Forwarded-Proto": "http",
        },
    ).json()

    assert body["next"].startswith(f"{PUBLIC}/api/events?")
    assert "evil.example" not in body["next"]


def test_public_base_url_trailing_slash_and_port(client, db, monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://events.example.org:8443/")
    seed_pages(db)

    body = client.get("/api/events", params={"page_size": 2}).json()

    assert (
        body["next"] == "https://events.example.org:8443/api/events?page_size=2&page=2"
    )


def test_without_public_base_url_links_follow_the_request(client, db, monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    seed_pages(db)

    body = client.get("/api/events", params={"page_size": 2}).json()

    assert body["next"] == "http://testserver/api/events?page_size=2&page=2"


def test_http_request_is_served_without_redirect(client, db, monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", PUBLIC)
    seed_pages(db)

    response = client.get(
        "/api/events", params={"page_size": 2}, follow_redirects=False
    )

    assert response.status_code == 200
    assert "location" not in response.headers
    assert set(response.json()) == {"count", "next", "previous", "results"}


def test_invalid_public_base_url_fails_at_startup(db, monkeypatch):
    monkeypatch.setenv("PUBLIC_BASE_URL", "events.example.org")

    with pytest.raises(ConfigError), TestClient(main.app):
        pass
