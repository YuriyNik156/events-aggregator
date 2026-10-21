import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.config import ConfigError, get_provider_settings
from app.database import dispose_engine, get_session_factory
from app.events_provider import EventsProviderClient, ProviderError
from app.repositories import EventRepository
from app.schemas import (
    EventDetailOut,
    EventOut,
    EventsPageOut,
    SeatsOut,
    TicketCreate,
    TicketCreatedOut,
)
from app.services import (
    EventNotFoundError,
    EventNotPublishedError,
    InvalidSeatError,
    RegistrationClosedError,
    RegistrationRejectedError,
    SeatsCache,
    SeatUnavailableError,
    SyncAlreadyRunningError,
    TicketNotSavedError,
    get_available_seats,
    register_ticket,
    seats_cache,
    sync_events,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    dispose_engine()


app = FastAPI(title="Events Aggregator", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # The LMS test wants 400 for an invalid ticket body; other routes keep the 422.
    if request.method != "POST" or request.url.path != "/api/tickets":
        return await request_validation_exception_handler(request, exc)
    # Field and kind of error only: the submitted values (email, names) are not echoed.
    errors = [{"loc": e["loc"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse({"detail": errors}, status_code=400)


def get_session() -> Iterator[Session]:
    with get_session_factory()() as session:
        yield session


def get_provider_client() -> Iterator[EventsProviderClient]:
    try:
        settings = get_provider_settings()
    except ConfigError as exc:
        logger.error("Events Provider is not configured: %s", exc)
        raise HTTPException(500, "Events Provider is not configured") from None
    client = EventsProviderClient(settings.base_url, settings.api_key)
    try:
        yield client
    finally:
        client.close()


def get_seats_cache() -> SeatsCache:
    return seats_cache


@app.get("/api/health")
def health_check():
    return {"status": "ok"}


@app.post("/api/sync/trigger")
def trigger_sync(
    session: Annotated[Session, Depends(get_session)],
    client: Annotated[EventsProviderClient, Depends(get_provider_client)],
):
    try:
        sync_events(session, client)
    except SyncAlreadyRunningError:
        raise HTTPException(409, "Synchronization is already running") from None
    except ProviderError:
        raise HTTPException(502, "Events Provider request failed") from None
    return {"success": True}


@app.get("/api/events", response_model=EventsPageOut)
def list_events(
    request: Request,
    session: Annotated[Session, Depends(get_session)],
    date_from: date | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1)] = 20,
):
    start = datetime.combine(date_from, time.min, tzinfo=UTC) if date_from else None
    events, count = EventRepository(session).list(
        date_from=start, page=page, page_size=page_size
    )

    def page_url(number: int) -> str:
        return str(request.url.include_query_params(page=number))

    return EventsPageOut(
        count=count,
        next=page_url(page + 1) if page * page_size < count else None,
        previous=page_url(page - 1) if page > 1 else None,
        results=[EventOut.from_event(e) for e in events],
    )


@app.get("/api/events/{event_id}", response_model=EventDetailOut)
def get_event(event_id: str, session: Annotated[Session, Depends(get_session)]):
    event = EventRepository(session).get(event_id)
    if event is None:
        raise HTTPException(404, "Event not found")
    return EventDetailOut.from_event(event)


@app.get("/api/events/{event_id}/seats", response_model=SeatsOut)
def get_event_seats(
    event_id: str,
    session: Annotated[Session, Depends(get_session)],
    client: Annotated[EventsProviderClient, Depends(get_provider_client)],
    cache: Annotated[SeatsCache, Depends(get_seats_cache)],
):
    try:
        seats = get_available_seats(session, client, cache, event_id)
    except EventNotFoundError:
        raise HTTPException(404, "Event not found") from None
    except EventNotPublishedError:
        raise HTTPException(400, "Event is not published") from None
    except ProviderError as exc:
        logger.warning("Seats request for event %s failed: %s", event_id, exc)
        raise HTTPException(502, "Events Provider request failed") from None
    return SeatsOut(event_id=event_id, available_seats=seats)


@app.post("/api/tickets", status_code=201, response_model=TicketCreatedOut)
def create_ticket(
    body: TicketCreate,
    session: Annotated[Session, Depends(get_session)],
    client: Annotated[EventsProviderClient, Depends(get_provider_client)],
    cache: Annotated[SeatsCache, Depends(get_seats_cache)],
):
    try:
        ticket_id = register_ticket(session, client, cache, **body.model_dump())
    except EventNotFoundError:
        raise HTTPException(404, "Event not found") from None
    except EventNotPublishedError:
        raise HTTPException(400, "Event is not published") from None
    except RegistrationClosedError:
        raise HTTPException(400, "Registration is closed") from None
    except InvalidSeatError:
        raise HTTPException(400, "Seat does not exist") from None
    except SeatUnavailableError:
        raise HTTPException(409, "Seat is not available") from None
    except RegistrationRejectedError:
        raise HTTPException(
            409, "Events Provider rejected the registration (seat may be taken)"
        ) from None
    except TicketNotSavedError as exc:
        raise HTTPException(
            500,
            "Registration was confirmed by Events Provider but could not be saved; "
            f"do not retry, reference ticket_id={exc.ticket_id}",
        ) from None
    except ProviderError as exc:
        # The outcome may be unknown (e.g. timeout); the message has no personal data.
        logger.warning("Registration for event %s failed: %s", body.event_id, exc)
        raise HTTPException(502, "Events Provider request failed") from None
    return TicketCreatedOut(ticket_id=ticket_id)
