from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Annotated, Any
from urllib.parse import quote

import httpx
from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    Field,
    ValidationError,
)

from app.config import get_provider_settings

EVENTS_PATH = "/api/events/"
DEFAULT_TIMEOUT = 30.0


class ProviderError(Exception):
    """Base class for Events Provider failures. Never contains the API key."""


class ProviderUnavailableError(ProviderError):
    """Network failure, timeout, or a response that is not valid contract data."""


class ProviderRejectedError(ProviderError):
    """The provider answered with a non-2xx status."""

    def __init__(self, status_code: int, body: str):
        super().__init__(f"Events Provider returned HTTP {status_code}")
        self.status_code = status_code
        self.body = body[:500]


def _as_utc(value: datetime) -> datetime:
    # Assumption: a timestamp without an offset is UTC.
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


ExternalId = Annotated[str, BeforeValidator(lambda v: str(v) if v is not None else v)]
UtcDatetime = Annotated[datetime, AfterValidator(_as_utc)]


class ProviderPlace(BaseModel):
    id: ExternalId
    name: str
    city: str
    address: str
    seats_pattern: str


class ProviderEvent(BaseModel):
    """Event as returned by ``GET /api/events/``; mirrors the public contract."""

    id: ExternalId
    name: str
    place: ProviderPlace
    event_time: UtcDatetime
    registration_deadline: UtcDatetime | None = None
    status: str
    number_of_visitors: int | None = 0
    changed_at: UtcDatetime
    created_at: UtcDatetime

    def to_row(self) -> dict[str, Any]:
        """Flat column values for the ``events`` table."""
        return {
            "id": self.id,
            "name": self.name,
            "place_id": self.place.id,
            "place_name": self.place.name,
            "place_city": self.place.city,
            "place_address": self.place.address,
            "place_seats_pattern": self.place.seats_pattern,
            "event_time": self.event_time,
            "registration_deadline": self.registration_deadline,
            "status": self.status,
            "number_of_visitors": self.number_of_visitors or 0,
            "changed_at": self.changed_at,
            "created_at": self.created_at,
        }


class ProviderSeats(BaseModel):
    """``GET /api/events/{id}/seats/``: free seats only."""

    seats: list[str]


class ProviderRegistration(BaseModel):
    """``POST /api/events/{id}/register/``: the new ticket."""

    ticket_id: ExternalId = Field(min_length=1)


class ProviderUnregistration(BaseModel):
    """``DELETE /api/events/{id}/unregister/``: ``{"success": true}``."""

    success: bool


class EventsPage(BaseModel):
    next: str | None = None
    previous: str | None = None
    results: list[ProviderEvent]


def _next_target(url: str) -> str:
    """Path and query of a ``next`` link, to be requested on our ``base_url``.

    The scheme and host of ``next`` (public name, no port, http instead of
    https, ...) are ignored, so the API key only ever goes to ``base_url``.
    """
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        raise ProviderUnavailableError(
            "Events Provider returned a bad next link"
        ) from None
    if parsed.path != EVENTS_PATH:
        raise ProviderUnavailableError(
            "Events Provider returned an unexpected next link"
        )
    return parsed.raw_path.decode("ascii")


@dataclass
class EventsProviderClient:
    """All HTTP traffic to the Events Provider lives here.

    Redirects are never followed, whatever the ``http_client`` was built with:
    the API key must not leave ``base_url``. Nothing is retried here.
    """

    base_url: str
    api_key: str
    timeout: float = DEFAULT_TIMEOUT
    http_client: httpx.AsyncClient | None = None

    def __post_init__(self) -> None:
        if self.http_client is None:
            self.http_client = httpx.AsyncClient(
                base_url=self.base_url, timeout=self.timeout, follow_redirects=False
            )

    async def aclose(self) -> None:
        await self.http_client.aclose()

    async def get_events_page(
        self, *, changed_at: date | None = None, url: str | None = None
    ) -> EventsPage:
        """Fetch one page: the first (by ``changed_at``) or a ready ``next`` URL."""
        if url is not None:
            response = await self._request("GET", _next_target(url), error_body=True)
        else:
            params = {"changed_at": changed_at.isoformat()} if changed_at else None
            response = await self._request(
                "GET", EVENTS_PATH, params=params, error_body=True
            )
        try:
            return EventsPage.model_validate_json(response.content)
        except ValidationError as exc:
            raise ProviderUnavailableError(
                "Events Provider returned an invalid events page "
                f"({exc.error_count()} validation errors)"
            ) from None

    async def get_event_seats(self, event_id: str) -> list[str]:
        """Free seats of a published event (the provider path needs the final ``/``)."""
        response = await self._request(
            "GET", f"{EVENTS_PATH}{quote(event_id, safe='')}/seats/", error_body=True
        )
        try:
            return ProviderSeats.model_validate_json(response.content).seats
        except ValidationError as exc:
            raise ProviderUnavailableError(
                "Events Provider returned invalid seats "
                f"({exc.error_count()} validation errors)"
            ) from None

    async def register(
        self,
        event_id: str,
        *,
        first_name: str,
        last_name: str,
        seat: str,
        email: str,
    ) -> str:
        """Register a participant and return the provider's ``ticket_id``.

        Never retried here: after a timeout the registration may exist already, and
        a second POST could create another one. Errors carry no personal data.
        """
        response = await self._request(
            "POST",
            f"{EVENTS_PATH}{quote(event_id, safe='')}/register/",
            json={
                "first_name": first_name,
                "last_name": last_name,
                "seat": seat,
                "email": email,
            },
        )
        try:
            return ProviderRegistration.model_validate_json(response.content).ticket_id
        except ValidationError as exc:
            raise ProviderUnavailableError(
                "Events Provider returned an invalid registration "
                f"({exc.error_count()} validation errors)"
            ) from None

    async def unregister(self, event_id: str, ticket_id: str) -> None:
        """Cancel a registration; returns only if the provider confirmed it.

        Never retried here: the provider's idempotency is not confirmed, so after a
        timeout the outcome stays unknown and is reported as such.
        """
        response = await self._request(
            "DELETE",
            f"{EVENTS_PATH}{quote(event_id, safe='')}/unregister/",
            json={"ticket_id": ticket_id},
        )
        try:
            confirmed = ProviderUnregistration.model_validate_json(
                response.content
            ).success
        except ValidationError as exc:
            raise ProviderUnavailableError(
                "Events Provider returned an invalid cancellation response "
                f"({exc.error_count()} validation errors)"
            ) from None
        if confirmed is not True:
            raise ProviderUnavailableError(
                "Events Provider did not confirm cancellation"
            )

    async def _request(
        self,
        method: str,
        url: str,
        *,
        error_body: bool = False,
        params: dict[str, str] | None = None,
        json: dict[str, str] | None = None,
    ) -> httpx.Response:
        """One request; a non-2xx answer (3xx included) is ``ProviderRejectedError``.

        ``error_body`` keeps the response text in that error. It is off for
        register/unregister, whose answers may echo the participant's data.
        """
        try:
            response = await self.http_client.request(
                method,
                url,
                params=params,
                json=json,
                headers={"x-api-key": self.api_key},
                timeout=self.timeout,
                follow_redirects=False,  # the API key must not leave ``base_url``
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Events Provider request failed: {type(exc).__name__}"
            ) from None
        if not response.is_success:
            raise ProviderRejectedError(
                response.status_code, response.text if error_body else ""
            )
        return response


class EventsPaginator:
    """Iterate (``async for``) over every event of every page, following ``next``.

    Pages are fetched one after another, only when the previous one is consumed.
    """

    def __init__(self, client: EventsProviderClient, changed_at: date):
        self.client = client
        self.changed_at = changed_at

    async def __aiter__(self) -> AsyncIterator[ProviderEvent]:
        page = await self.client.get_events_page(changed_at=self.changed_at)
        while True:
            for event in page.results:
                yield event
            if not page.next:
                return
            page = await self.client.get_events_page(url=page.next)


# The one client of the process: the routes and the background sync share it, and
# with it its HTTP connections. Created on first use, closed at shutdown.
_shared_client: EventsProviderClient | None = None


def get_shared_client() -> EventsProviderClient:
    """The process-wide client; ``ConfigError`` while the API key is missing."""
    global _shared_client
    if _shared_client is None:
        settings = get_provider_settings()
        _shared_client = EventsProviderClient(settings.base_url, settings.api_key)
    return _shared_client


async def close_shared_client() -> None:
    global _shared_client
    client, _shared_client = _shared_client, None
    if client is not None:
        await client.aclose()
