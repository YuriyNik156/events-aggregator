from unittest.mock import Mock, call

import httpx
import pytest

from app.events_provider import (
    EventsPage,
    EventsPaginator,
    EventsProviderClient,
    ProviderRejectedError,
    ProviderUnavailableError,
)
from tests.conftest import INITIAL, provider_event_json, provider_page


def make_client(*responses) -> tuple[EventsProviderClient, Mock]:
    http = Mock(spec=httpx.Client)
    http.get.side_effect = list(responses)
    return EventsProviderClient("http://provider", "test-key", http_client=http), http


def ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body)


def test_first_page_sends_changed_at_and_api_key():
    client, http = make_client(ok(provider_page([provider_event_json("e1")])))

    page = client.get_events_page(changed_at=INITIAL)

    args, kwargs = http.get.call_args
    assert args == ("/api/events/",)
    assert kwargs["params"] == {"changed_at": "2000-01-01"}
    assert kwargs["headers"] == {"x-api-key": "test-key"}
    assert [e.id for e in page.results] == ["e1"]
    assert page.results[0].place.seats_pattern == "A1-10"


def test_next_url_is_requested_as_is_with_api_key():
    client, http = make_client(ok(provider_page([])))

    client.get_events_page(url="http://other-host/api/events/?cursor=abc")

    args, kwargs = http.get.call_args
    assert args == ("http://other-host/api/events/?cursor=abc",)
    assert kwargs["headers"] == {"x-api-key": "test-key"}


@pytest.mark.parametrize("status", [401, 404, 500])
def test_http_error_status_raises_rejected(status):
    client, _ = make_client(httpx.Response(status, text="denied"))

    with pytest.raises(ProviderRejectedError) as info:
        client.get_events_page(changed_at=INITIAL)

    assert info.value.status_code == status
    assert "test-key" not in str(info.value)


@pytest.mark.parametrize("error", [httpx.ReadTimeout("t"), httpx.ConnectError("c")])
def test_network_errors_raise_unavailable(error):
    client, _ = make_client(error)

    with pytest.raises(ProviderUnavailableError):
        client.get_events_page(changed_at=INITIAL)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"next": None}),  # no results
        httpx.Response(200, json=provider_page([{"id": "e1"}])),  # event incomplete
    ],
)
def test_invalid_payload_raises_unavailable(response):
    client, _ = make_client(response)

    with pytest.raises(ProviderUnavailableError):
        client.get_events_page(changed_at=INITIAL)


def test_provider_event_tolerates_nulls_and_naive_time():
    raw = provider_event_json("e1", registration_deadline=None, number_of_visitors=None)
    raw["event_time"] = "2026-01-11T14:00:00"
    client, _ = make_client(ok(provider_page([raw])))

    row = client.get_events_page(changed_at=INITIAL).results[0].to_row()

    assert row["registration_deadline"] is None
    assert row["number_of_visitors"] == 0
    assert row["event_time"].utcoffset().total_seconds() == 0


def page_of(ids, next_url=None) -> EventsPage:
    return EventsPage.model_validate(
        provider_page([provider_event_json(i) for i in ids], next_url=next_url)
    )


def test_paginator_single_page():
    client = Mock()
    client.get_events_page.return_value = page_of(["a", "b"])

    assert [e.id for e in EventsPaginator(client, INITIAL)] == ["a", "b"]
    client.get_events_page.assert_called_once_with(changed_at=INITIAL)


def test_paginator_follows_next_until_null():
    client = Mock()
    client.get_events_page.side_effect = [
        page_of(["a"], next_url="http://x/api/events/?page=2"),
        page_of(["b"], next_url="http://x/api/events/?page=3"),
        page_of(["c"]),
    ]

    assert [e.id for e in EventsPaginator(client, INITIAL)] == ["a", "b", "c"]
    assert client.get_events_page.call_args_list == [
        call(changed_at=INITIAL),
        call(url="http://x/api/events/?page=2"),
        call(url="http://x/api/events/?page=3"),
    ]


def test_paginator_empty_result():
    client = Mock()
    client.get_events_page.return_value = page_of([])

    assert list(EventsPaginator(client, INITIAL)) == []


def test_paginator_error_is_not_end_of_list():
    client = Mock()
    client.get_events_page.side_effect = [
        page_of(["a"], next_url="http://x/api/events/?page=2"),
        ProviderUnavailableError("boom"),
    ]
    seen = []

    with pytest.raises(ProviderUnavailableError):
        for event in EventsPaginator(client, INITIAL):
            seen.append(event.id)

    assert seen == ["a"]
