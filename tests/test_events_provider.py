"""Async Events Provider client and paginator, on ``httpx.MockTransport``.

Nothing here reaches a real provider.
"""

import asyncio
import inspect
import json
import time
from unittest.mock import AsyncMock, call

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

pytestmark = pytest.mark.anyio

KEY = "test-key"
BASE = "http://provider:8000"
NEXT = "http://public.example.org/api/events/?changed_at=2000-01-01&cursor=2"


def ok(body: dict) -> httpx.Response:
    return httpx.Response(200, json=body)


def make_client(
    handler, *, follow_redirects: bool = False, timeout: float = 30.0
) -> tuple[EventsProviderClient, list[httpx.Request]]:
    """Real ``httpx.AsyncClient`` (base_url, redirects) on a mock transport."""
    seen: list[httpx.Request] = []

    async def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        result = handler(request)
        return await result if inspect.isawaitable(result) else result

    http = httpx.AsyncClient(
        base_url=BASE,
        transport=httpx.MockTransport(record),
        follow_redirects=follow_redirects,
    )
    return EventsProviderClient(BASE, KEY, timeout=timeout, http_client=http), seen


def always(response: httpx.Response):
    return lambda request: response


def raises(error: Exception):
    def handler(request: httpx.Request):
        raise error

    return handler


# one call per provider operation, for the checks that apply to all of them
OPERATIONS = {
    "events": lambda c: c.get_events_page(changed_at=INITIAL),
    "next": lambda c: c.get_events_page(url=NEXT),
    "seats": lambda c: c.get_event_seats("e1"),
    "register": lambda c: c.register(
        "e1", first_name="Ivan", last_name="Ivanov", seat="A5", email="i@example.com"
    ),
    "unregister": lambda c: c.unregister("e1", "t-1"),
}
ALL = list(OPERATIONS)
WRITES = ["register", "unregister"]


# ---- events page ----


async def test_first_page_sends_changed_at_and_api_key():
    client, seen = make_client(always(ok(provider_page([provider_event_json("e1")]))))

    page = await client.get_events_page(changed_at=INITIAL)

    (request,) = seen
    assert request.method == "GET"
    assert request.url.path == "/api/events/"
    assert dict(request.url.params) == {"changed_at": "2000-01-01"}
    assert request.headers["x-api-key"] == KEY
    assert [e.id for e in page.results] == ["e1"]
    assert page.results[0].place.seats_pattern == "A1-10"


async def test_first_page_without_changed_at_has_no_query():
    client, seen = make_client(always(ok(provider_page([]))))

    await client.get_events_page()

    assert seen[0].url.query == b""


@pytest.mark.parametrize(
    "next_url",
    [
        "http://public.example.org/api/events/?changed_at=2000-01-01&cursor=a%3D%3D",
        "https://provider/api/events/?changed_at=2000-01-01&cursor=a%3D%3D",  # no port
        "//evil.example.org/api/events/?changed_at=2000-01-01&cursor=a%3D%3D",
        "/api/events/?changed_at=2000-01-01&cursor=a%3D%3D",  # relative
    ],
)
async def test_next_is_requested_on_base_url_keeping_query(next_url):
    client, seen = make_client(always(ok(provider_page([]))))

    await client.get_events_page(url=next_url)

    (request,) = seen
    assert (request.url.scheme, request.url.host, request.url.port) == (
        "http",
        "provider",
        8000,
    )
    assert request.url.path == "/api/events/"
    assert request.url.params["changed_at"] == "2000-01-01"
    assert request.url.params["cursor"] == "a=="
    assert request.headers["x-api-key"] == KEY


@pytest.mark.parametrize(
    "next_url",
    [
        "http://provider:8000/api/other/?cursor=a",
        "http://provider:8000/api/events/../admin/",
        "http://provider:8000/",
        "http://[::1/api/events/",
    ],
)
async def test_unexpected_next_is_rejected_without_request(next_url):
    client, seen = make_client(always(ok(provider_page([]))))

    with pytest.raises(ProviderUnavailableError) as info:
        await client.get_events_page(url=next_url)

    assert seen == []
    assert KEY not in str(info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, text="not json"),
        httpx.Response(200, json={"next": None}),  # no results
        httpx.Response(200, json=provider_page([{"id": "e1"}])),  # event incomplete
    ],
)
async def test_invalid_events_page_raises_unavailable(response):
    client, _ = make_client(always(response))

    with pytest.raises(ProviderUnavailableError):
        await client.get_events_page(changed_at=INITIAL)


async def test_provider_event_tolerates_nulls_and_naive_time():
    raw = provider_event_json("e1", registration_deadline=None, number_of_visitors=None)
    raw["event_time"] = "2026-01-11T14:00:00"
    client, _ = make_client(always(ok(provider_page([raw]))))

    row = (await client.get_events_page(changed_at=INITIAL)).results[0].to_row()

    assert row["registration_deadline"] is None
    assert row["number_of_visitors"] == 0
    assert row["event_time"].utcoffset().total_seconds() == 0


# ---- seats, register, unregister ----


async def test_seats_request_and_result():
    client, seen = make_client(always(ok({"seats": ["A1", "A2"]})))

    seats = await client.get_event_seats("a/b")

    (request,) = seen
    assert request.method == "GET"
    assert request.url.raw_path == b"/api/events/a%2Fb/seats/"  # id is quoted
    assert request.headers["x-api-key"] == KEY
    assert seats == ["A1", "A2"]


async def test_register_request_and_result():
    client, seen = make_client(always(ok({"ticket_id": 77})))

    ticket_id = await OPERATIONS["register"](client)

    (request,) = seen
    assert request.method == "POST"
    assert request.url.path == "/api/events/e1/register/"
    assert request.headers["x-api-key"] == KEY
    assert request.headers["content-type"] == "application/json"
    assert json.loads(request.content) == {
        "first_name": "Ivan",
        "last_name": "Ivanov",
        "seat": "A5",
        "email": "i@example.com",
    }
    assert ticket_id == "77"  # ids are text, as with the sync client


async def test_unregister_request_and_result():
    client, seen = make_client(always(ok({"success": True})))

    assert await client.unregister("e1", "t-1") is None

    (request,) = seen
    assert request.method == "DELETE"
    assert request.url.path == "/api/events/e1/unregister/"
    assert request.headers["x-api-key"] == KEY
    assert json.loads(request.content) == {"ticket_id": "t-1"}


@pytest.mark.parametrize(
    ("operation", "response"),
    [
        ("seats", httpx.Response(200, json={"nope": 1})),
        ("seats", httpx.Response(200, text="<html>")),
        ("seats", httpx.Response(200, text="<html>UnexpectedEventStatus</html>")),
        ("seats", httpx.Response(200, text="{not json")),
        ("seats", httpx.Response(200, json={"seats": "A1"})),
        ("seats", httpx.Response(200, json={"other": []})),
        ("seats", httpx.Response(200, json={"seats": [1, 2]})),
        ("register", httpx.Response(200, json={})),
        ("register", httpx.Response(200, json={"ticket_id": ""})),
        ("register", httpx.Response(200, json={"ticket_id": None})),
        ("register", httpx.Response(200, text="oops")),
        ("register", httpx.Response(201, text="{not json")),
        ("unregister", httpx.Response(200, json={"success": False})),
        ("unregister", httpx.Response(200, json={})),
        ("unregister", httpx.Response(200, json={"success": None})),
        ("unregister", httpx.Response(200, text="oops")),
        ("unregister", httpx.Response(200, text="{not json")),
    ],
)
async def test_invalid_or_unconfirmed_answer_raises_unavailable(operation, response):
    client, _ = make_client(always(response))

    with pytest.raises(ProviderUnavailableError):
        await OPERATIONS[operation](client)


# ---- HTTP errors, network errors, timeouts ----


@pytest.mark.parametrize("status", [400, 401, 404, 429, 500, 503])
@pytest.mark.parametrize("operation", ALL)
async def test_http_error_status_raises_rejected(operation, status):
    client, _ = make_client(always(httpx.Response(status, text="denied")))

    with pytest.raises(ProviderRejectedError) as info:
        await OPERATIONS[operation](client)

    assert info.value.status_code == status
    assert KEY not in str(info.value)


@pytest.mark.parametrize("operation", ["events", "next", "seats"])
async def test_reads_keep_the_error_body(operation):
    client, _ = make_client(always(httpx.Response(404, text="no such event")))

    with pytest.raises(ProviderRejectedError) as info:
        await OPERATIONS[operation](client)

    assert info.value.body == "no such event"


@pytest.mark.parametrize("operation", WRITES)
async def test_writes_do_not_keep_the_error_body(operation):
    # The provider may echo the participant's data in an error answer.
    client, _ = make_client(always(httpx.Response(400, text="i@example.com taken")))

    with pytest.raises(ProviderRejectedError) as info:
        await OPERATIONS[operation](client)

    assert info.value.body == ""
    assert "i@example.com" not in str(info.value)


@pytest.mark.parametrize(
    "error",
    [
        httpx.ReadTimeout("slow"),
        httpx.ConnectTimeout("slow"),
        httpx.ConnectError("down"),
        httpx.ReadError("reset"),
    ],
)
@pytest.mark.parametrize("operation", ALL)
async def test_network_errors_and_timeouts_raise_unavailable(operation, error):
    client, _ = make_client(raises(error))

    with pytest.raises(ProviderUnavailableError) as info:
        await OPERATIONS[operation](client)

    assert type(error).__name__ in str(info.value)
    assert KEY not in str(info.value)


async def test_timeout_is_applied_to_every_request():
    client, seen = make_client(always(ok({"seats": []})), timeout=7.5)

    await client.get_event_seats("e1")

    assert set(seen[0].extensions["timeout"].values()) == {7.5}


async def test_default_timeout_is_30_seconds():
    client, seen = make_client(always(ok({"seats": []})))

    await client.get_event_seats("e1")

    assert set(seen[0].extensions["timeout"].values()) == {30.0}


# ---- no automatic retries ----


@pytest.mark.parametrize(
    "outcome",
    [
        raises(httpx.ReadTimeout("slow")),
        raises(httpx.ConnectError("down")),
        always(httpx.Response(500)),
        always(httpx.Response(502)),
        always(httpx.Response(503)),
    ],
)
@pytest.mark.parametrize("operation", WRITES)
async def test_register_and_unregister_are_sent_exactly_once(operation, outcome):
    client, seen = make_client(outcome)

    with pytest.raises((ProviderUnavailableError, ProviderRejectedError)):
        await OPERATIONS[operation](client)

    assert len(seen) == 1


# ---- redirects and the API key ----


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
@pytest.mark.parametrize("operation", ALL)
async def test_redirect_is_not_followed_and_key_does_not_leave(operation, status):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, headers={"location": "http://evil.example.org/api/events/"}
        )

    # Even a client that was built to follow redirects must not follow them.
    client, seen = make_client(handler, follow_redirects=True)

    with pytest.raises(ProviderRejectedError) as info:
        await OPERATIONS[operation](client)

    assert info.value.status_code == status
    assert [r.url.host for r in seen] == ["provider"]  # nothing went to evil.example
    assert "evil.example.org" not in str(info.value)
    assert KEY not in str(info.value)


async def test_default_client_does_not_follow_redirects():
    client = EventsProviderClient(BASE, KEY)
    try:
        assert client.http_client.follow_redirects is False
        assert str(client.http_client.base_url) == BASE
        assert client.http_client.timeout == httpx.Timeout(30.0)
    finally:
        await client.aclose()


async def test_ordinary_pagination_works_without_redirects():
    def handler(request: httpx.Request) -> httpx.Response:
        cursor = request.url.params.get("cursor")
        pages = {
            None: ([provider_event_json("a")], NEXT),
            "2": ([provider_event_json("b")], None),
        }
        events, next_url = pages[cursor]
        return ok(provider_page(events, next_url=next_url))

    client, seen = make_client(handler)  # redirects off, as in production

    events = [e async for e in EventsPaginator(client, INITIAL)]

    assert [e.id for e in events] == ["a", "b"]
    assert len(seen) == 2
    assert {r.url.host for r in seen} == {"provider"}
    assert all(r.headers["x-api-key"] == KEY for r in seen)


# ---- closing ----


async def test_aclose_closes_the_http_client():
    client, _ = make_client(always(ok({"seats": []})))
    assert not client.http_client.is_closed

    await client.aclose()

    assert client.http_client.is_closed


async def test_aclose_closes_a_client_that_it_created():
    client = EventsProviderClient(BASE, KEY)

    await client.aclose()

    assert client.http_client.is_closed


# ---- the event loop stays free ----


async def test_slow_provider_does_not_block_the_event_loop():
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return ok({"seats": ["A1"]})

    client, _ = make_client(slow)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        assert await client.get_event_seats("e1") == ["A1"]
    finally:
        task.cancel()

    assert ticks >= 10  # a blocked loop would have counted about one


async def test_slow_requests_run_concurrently():
    async def slow(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.3)
        return ok({"seats": []})

    client, seen = make_client(slow)

    started = time.monotonic()
    await asyncio.gather(*(client.get_event_seats(f"e{i}") for i in range(5)))
    elapsed = time.monotonic() - started

    assert len(seen) == 5
    assert elapsed < 0.9  # one after another it would take 1.5 s


# ---- paginator ----


def page_of(ids, next_url=None) -> EventsPage:
    return EventsPage.model_validate(
        provider_page([provider_event_json(i) for i in ids], next_url=next_url)
    )


async def test_paginator_single_page():
    client = AsyncMock()
    client.get_events_page.return_value = page_of(["a", "b"])

    assert [e.id async for e in EventsPaginator(client, INITIAL)] == ["a", "b"]
    client.get_events_page.assert_awaited_once_with(changed_at=INITIAL)


async def test_paginator_follows_next_until_null_in_order():
    client = AsyncMock()
    client.get_events_page.side_effect = [
        page_of(["a", "b"], next_url="http://x/api/events/?page=2"),
        page_of(["c"], next_url="http://x/api/events/?page=3"),
        page_of(["d", "e"]),
    ]

    events = [e async for e in EventsPaginator(client, INITIAL)]

    assert [e.id for e in events] == ["a", "b", "c", "d", "e"]
    assert client.get_events_page.await_args_list == [
        call(changed_at=INITIAL),
        call(url="http://x/api/events/?page=2"),
        call(url="http://x/api/events/?page=3"),
    ]


async def test_paginator_empty_result():
    client = AsyncMock()
    client.get_events_page.return_value = page_of([])

    assert [e async for e in EventsPaginator(client, INITIAL)] == []


async def test_paginator_error_is_not_end_of_list():
    client = AsyncMock()
    client.get_events_page.side_effect = [
        page_of(["a"], next_url="http://x/api/events/?page=2"),
        ProviderUnavailableError("boom"),
    ]
    seen = []

    with pytest.raises(ProviderUnavailableError):
        async for event in EventsPaginator(client, INITIAL):
            seen.append(event.id)

    assert seen == ["a"]


async def test_paginator_fetches_pages_one_by_one_and_lazily():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.params.get("cursor") is None:
            return ok(provider_page([provider_event_json("a")], next_url=NEXT))
        return ok(provider_page([provider_event_json("b")]))

    client, seen = make_client(handler)

    iterator = EventsPaginator(client, INITIAL).__aiter__()
    assert (await iterator.__anext__()).id == "a"
    assert len(seen) == 1  # the second page is not requested yet
    assert (await iterator.__anext__()).id == "b"
    assert len(seen) == 2
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()


async def test_paginator_stops_cleanly_when_consumer_breaks():
    client, seen = make_client(
        always(ok(provider_page([provider_event_json("a")], next_url=NEXT)))
    )

    async for event in EventsPaginator(client, INITIAL):
        assert event.id == "a"
        break

    assert len(seen) == 1  # no further page is fetched after the break


@pytest.mark.parametrize(
    ("operation", "suffix"), [("register", "register"), ("unregister", "unregister")]
)
async def test_event_id_is_quoted_in_the_path_of_writes(operation, suffix):
    client, seen = make_client(always(ok({"ticket_id": "t", "success": True})))

    if operation == "register":
        await client.register(
            "a/../b", first_name="a", last_name="b", seat="A1", email="a@b.cc"
        )
    else:
        await client.unregister("a/../b", "t")

    assert seen[0].url.raw_path == f"/api/events/a%2F..%2Fb/{suffix}/".encode()


async def test_unregister_timeout_is_not_repeated_and_has_no_key():
    client, seen = make_client(raises(httpx.ReadTimeout("timeout")))

    with pytest.raises(ProviderUnavailableError) as info:
        await client.unregister("e1", "t")

    assert KEY not in str(info.value)
    assert len(seen) == 1
