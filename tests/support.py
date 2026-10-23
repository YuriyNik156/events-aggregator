"""Helpers shared by the service and application tests."""

import asyncio
from datetime import UTC, datetime, timedelta

import httpx
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.events_provider import EventsProviderClient
from app.models import Ticket
from app.repositories import EventRepository, TicketRepository
from tests.conftest import make_event, provider_event_json, provider_page

SECRET = "super-secret-key"
FUTURE = datetime.now(UTC) + timedelta(days=30)
PAST = datetime.now(UTC) - timedelta(days=1)

# What a registration request carries (the same person and seat as the sync tests).
ARGS = {
    "event_id": "e1",
    "first_name": "Иван",
    "last_name": "Иванов",
    "email": "ivan@example.com",
    "seat": "A5",
}


async def seed_event(factory: async_sessionmaker[AsyncSession], **overrides) -> None:
    overrides.setdefault("registration_deadline", FUTURE)
    overrides.setdefault("event_time", FUTURE + timedelta(days=1))
    async with factory() as s:
        await EventRepository(s).upsert(make_event("e1", **overrides))
        await s.commit()


async def add_ticket(
    factory: async_sessionmaker[AsyncSession],
    ticket_id: str = "t-1",
    seat: str = "A5",
    **overrides,
) -> None:
    fields = {
        "ticket_id": ticket_id,
        "event_id": "e1",
        "first_name": "Иван",
        "last_name": "Иванов",
        "email": "ivan@example.com",
        "seat": seat,
    }
    fields.update(overrides)
    async with factory() as s:
        await TicketRepository(s).add(**fields)
        await s.commit()


async def all_tickets(factory: async_sessionmaker[AsyncSession]) -> list[Ticket]:
    """Every ticket row as committed, read through a fresh session."""
    async with factory() as s:
        return list(await s.scalars(select(Ticket).order_by(Ticket.id)))


def db_state(session: AsyncSession) -> tuple[bool, int]:
    """(is the session in a transaction, how many pooled connections are in use)."""
    return session.in_transaction(), session.bind.pool.checkedout()


class Ticker:
    """Counts loop turns while a block runs: a blocked loop counts almost none."""

    def __init__(self) -> None:
        self.count = 0
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(0.01)
            self.count += 1

    async def __aenter__(self) -> "Ticker":
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc) -> None:
        self._task.cancel()


def real_client(handler) -> tuple[EventsProviderClient, list[httpx.Request]]:
    """The real async client on a mock transport; the requests it made are listed."""
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    http = httpx.AsyncClient(
        base_url="http://provider:8000", transport=httpx.MockTransport(record)
    )
    return EventsProviderClient("http://provider:8000", SECRET, http_client=http), seen


class Provider:
    """The Events Provider: answers at once, or waits until the test lets it go.

    ``hold(n)`` makes the next answers wait; ``all_entered`` is set when ``n``
    requests are waiting, ``release`` lets all of them go.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.waiting = 0
        self._expected = 0
        self.all_entered = asyncio.Event()
        self.release = asyncio.Event()
        self.respond = self.default_response

    def hold(self, count: int = 1) -> None:
        self._expected = count

    @staticmethod
    def default_response(request: httpx.Request) -> httpx.Response:
        path, method = request.url.path, request.method
        if method == "GET" and path.endswith("/seats/"):
            return httpx.Response(200, json={"seats": ["A4", "A5"]})
        if method == "GET" and path == "/api/events/":
            return httpx.Response(200, json=provider_page([provider_event_json("p1")]))
        if method == "POST" and path.endswith("/register/"):
            return httpx.Response(201, json={"ticket_id": "t-1"})
        if method == "DELETE" and path.endswith("/unregister/"):
            return httpx.Response(200, json={"success": True})
        return httpx.Response(404)

    async def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._expected:
            self.waiting += 1
            if self.waiting >= self._expected:
                self.all_entered.set()
            try:
                await self.release.wait()
            finally:
                self.waiting -= 1
        return self.respond(request)
