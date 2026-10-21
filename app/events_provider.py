from collections.abc import Iterator
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


@dataclass
class EventsProviderClient:
    """All HTTP traffic to the Events Provider lives here."""

    base_url: str
    api_key: str
    timeout: float = DEFAULT_TIMEOUT
    http_client: httpx.Client | None = None

    def __post_init__(self) -> None:
        if self.http_client is None:
            self.http_client = httpx.Client(
                base_url=self.base_url, timeout=self.timeout, follow_redirects=True
            )

    def close(self) -> None:
        self.http_client.close()

    def get_events_page(
        self, *, changed_at: date | None = None, url: str | None = None
    ) -> EventsPage:
        """Fetch one page: the first (by ``changed_at``) or a ready ``next`` URL."""
        if url is not None:
            response = self._get(self._next_target(url))
        else:
            params = {"changed_at": changed_at.isoformat()} if changed_at else None
            response = self._get(EVENTS_PATH, params=params)
        try:
            return EventsPage.model_validate_json(response.content)
        except ValidationError as exc:
            raise ProviderUnavailableError(
                "Events Provider returned an invalid events page "
                f"({exc.error_count()} validation errors)"
            ) from None

    def get_event_seats(self, event_id: str) -> list[str]:
        """Free seats of a published event (the provider path needs the final ``/``)."""
        # Redirects are not followed: the API key must not leave ``base_url``.
        response = self._get(
            f"{EVENTS_PATH}{quote(event_id, safe='')}/seats/", follow_redirects=False
        )
        try:
            return ProviderSeats.model_validate_json(response.content).seats
        except ValidationError as exc:
            raise ProviderUnavailableError(
                "Events Provider returned invalid seats "
                f"({exc.error_count()} validation errors)"
            ) from None

    def register(
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
        try:
            response = self.http_client.post(
                f"{EVENTS_PATH}{quote(event_id, safe='')}/register/",
                json={
                    "first_name": first_name,
                    "last_name": last_name,
                    "seat": seat,
                    "email": email,
                },
                headers={"x-api-key": self.api_key},
                timeout=self.timeout,
                follow_redirects=False,  # the API key must not leave ``base_url``
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Events Provider request failed: {type(exc).__name__}"
            ) from None
        if not response.is_success:
            # The body may echo the participant's data, so it is not kept.
            raise ProviderRejectedError(response.status_code, "")
        try:
            return ProviderRegistration.model_validate_json(response.content).ticket_id
        except ValidationError as exc:
            raise ProviderUnavailableError(
                "Events Provider returned an invalid registration "
                f"({exc.error_count()} validation errors)"
            ) from None

    def unregister(self, event_id: str, ticket_id: str) -> None:
        """Cancel a registration; returns only if the provider confirmed it.

        Never retried here: the provider's idempotency is not confirmed, so after a
        timeout the outcome stays unknown and is reported as such.
        """
        try:
            response = self.http_client.request(
                "DELETE",
                f"{EVENTS_PATH}{quote(event_id, safe='')}/unregister/",
                json={"ticket_id": ticket_id},
                headers={"x-api-key": self.api_key},
                timeout=self.timeout,
                follow_redirects=False,  # the API key must not leave ``base_url``
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Events Provider request failed: {type(exc).__name__}"
            ) from None
        if not response.is_success:
            raise ProviderRejectedError(response.status_code, "")
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

    @staticmethod
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

    def _get(
        self,
        url: str,
        params: dict[str, str] | None = None,
        follow_redirects: bool | None = None,
    ) -> httpx.Response:
        options: dict[str, Any] = {}
        if follow_redirects is not None:
            options["follow_redirects"] = follow_redirects
        try:
            response = self.http_client.get(
                url,
                params=params,
                headers={"x-api-key": self.api_key},
                timeout=self.timeout,
                **options,
            )
        except httpx.HTTPError as exc:
            raise ProviderUnavailableError(
                f"Events Provider request failed: {type(exc).__name__}"
            ) from None
        if not response.is_success:
            raise ProviderRejectedError(response.status_code, response.text)
        return response


class EventsPaginator:
    """Iterate over every event of every page, following the provider's ``next``."""

    def __init__(self, client: EventsProviderClient, changed_at: date):
        self.client = client
        self.changed_at = changed_at

    def __iter__(self) -> Iterator[ProviderEvent]:
        page = self.client.get_events_page(changed_at=self.changed_at)
        while True:
            yield from page.results
            if not page.next:
                return
            page = self.client.get_events_page(url=page.next)
