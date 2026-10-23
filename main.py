import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, time
from typing import Annotated
from urllib.parse import urlsplit

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import (
    DEFAULT_SYNC_INTERVAL,
    ConfigError,
    get_public_base_url,
    get_sync_interval,
)
from app.database import dispose_async_engine, get_async_session_factory
from app.events_provider import (
    EventsProviderClient,
    ProviderError,
    close_shared_client,
    get_shared_client,
)
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
    CancellationInProgressError,
    CancellationNotSavedError,
    CancellationRejectedError,
    EventNotFoundError,
    EventNotPublishedError,
    EventStartedError,
    InvalidSeatError,
    RegistrationClosedError,
    RegistrationRejectedError,
    SeatsCache,
    SeatUnavailableError,
    SyncAlreadyRunningError,
    TicketNotFoundError,
    TicketNotSavedError,
    cancel_ticket,
    get_available_seats,
    register_ticket,
    seats_cache,
    sync_events,
)
from app.sync_worker import SyncWorker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Fail at startup on a bad PUBLIC_BASE_URL, not with a 500 on every request.
    public_base_url = get_public_base_url()
    if public_base_url and public_base_url.startswith("http://"):
        logger.warning("PUBLIC_BASE_URL is not HTTPS; fine for local runs only")
    sync_interval = get_sync_interval()  # also fails at startup when invalid
    worker: SyncWorker | None = None
    if sync_interval == 0:
        logger.info("Background sync is off (SYNC_INTERVAL_SECONDS=0)")
    else:
        if sync_interval > DEFAULT_SYNC_INTERVAL:
            logger.warning(
                "SYNC_INTERVAL_SECONDS=%d is above 24 hours: "
                "the daily sync requirement is not met",
                sync_interval,
            )
        worker = SyncWorker(sync_interval)
        worker.start()
    yield
    if worker is not None and not await worker.stop():
        # Not even a cancelled sync has unwound: it may still use the client and the
        # engine, so they stay open (the process is going away anyway).
        logger.warning(
            "Background sync is still stopping; the client and the engine stay open"
        )
        return
    try:
        await close_shared_client()
    finally:
        await dispose_async_engine()


app = FastAPI(title="Events Aggregator", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    # The LMS test wants 400 for an invalid ticket body; other routes keep the 422.
    if request.method != "POST" or request.url.path != "/api/tickets":
        return await request_validation_exception_handler(request, exc)
    # Field and kind of error only: the submitted values (email, names) are not echoed.
    errors = [{"loc": e["loc"], "type": e["type"]} for e in exc.errors()]
    return JSONResponse({"detail": errors}, status_code=400)


# The dependencies are ``async`` on purpose: a plain ``def`` one would be run in the
# thread pool for nothing.


async def get_session() -> AsyncIterator[AsyncSession]:
    """A session of its own for every request, closed (rolled back) at its end."""
    async with get_async_session_factory()() as session:
        yield session


async def get_provider_client() -> EventsProviderClient:
    try:
        return get_shared_client()
    except ConfigError as exc:
        logger.error("Events Provider is not configured: %s", exc)
        raise HTTPException(500, "Events Provider is not configured") from None


async def get_seats_cache() -> SeatsCache:
    return seats_cache


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/sync/trigger")
async def trigger_sync(
    session: Annotated[AsyncSession, Depends(get_session)],
    client: Annotated[EventsProviderClient, Depends(get_provider_client)],
):
    try:
        await sync_events(session, client)
    except SyncAlreadyRunningError:
        raise HTTPException(409, "Synchronization is already running") from None
    except ProviderError:
        raise HTTPException(502, "Events Provider request failed") from None
    return {"success": True}


@app.get("/api/events", response_model=EventsPageOut)
async def list_events(
    request: Request,
    session: Annotated[AsyncSession, Depends(get_session)],
    date_from: date | None = None,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1)] = 20,
):
    start = datetime.combine(date_from, time.min, tzinfo=UTC) if date_from else None
    events, count = await EventRepository(session).list(
        date_from=start, page=page, page_size=page_size
    )

    public = urlsplit(get_public_base_url() or "")

    def page_url(number: int) -> str:
        url = request.url.include_query_params(page=number)
        if public.netloc:  # configured origin wins over what the proxy forwarded
            url = url.replace(scheme=public.scheme, netloc=public.netloc)
        return str(url)

    return EventsPageOut(
        count=count,
        next=page_url(page + 1) if page * page_size < count else None,
        previous=page_url(page - 1) if page > 1 else None,
        results=[EventOut.from_event(e) for e in events],
    )


@app.get("/api/events/{event_id}", response_model=EventDetailOut)
async def get_event(
    event_id: str, session: Annotated[AsyncSession, Depends(get_session)]
):
    event = await EventRepository(session).get(event_id)
    if event is None:
        raise HTTPException(404, "Event not found")
    return EventDetailOut.from_event(event)


@app.get("/api/events/{event_id}/seats", response_model=SeatsOut)
async def get_event_seats(
    event_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    client: Annotated[EventsProviderClient, Depends(get_provider_client)],
    cache: Annotated[SeatsCache, Depends(get_seats_cache)],
):
    try:
        seats = await get_available_seats(session, client, cache, event_id)
    except EventNotFoundError:
        raise HTTPException(404, "Event not found") from None
    except EventNotPublishedError:
        raise HTTPException(400, "Event is not published") from None
    except ProviderError as exc:
        logger.warning("Seats request for event %s failed: %s", event_id, exc)
        raise HTTPException(502, "Events Provider request failed") from None
    return SeatsOut(event_id=event_id, available_seats=seats)


@app.post("/api/tickets", status_code=201, response_model=TicketCreatedOut)
async def create_ticket(
    body: TicketCreate,
    session: Annotated[AsyncSession, Depends(get_session)],
    client: Annotated[EventsProviderClient, Depends(get_provider_client)],
    cache: Annotated[SeatsCache, Depends(get_seats_cache)],
):
    try:
        ticket_id = await register_ticket(session, client, cache, **body.model_dump())
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


@app.delete("/api/tickets/{ticket_id}")
async def delete_ticket(
    ticket_id: str,
    session: Annotated[AsyncSession, Depends(get_session)],
    client: Annotated[EventsProviderClient, Depends(get_provider_client)],
    cache: Annotated[SeatsCache, Depends(get_seats_cache)],
):
    try:
        await cancel_ticket(session, client, cache, ticket_id)
    except TicketNotFoundError:
        raise HTTPException(404, "Ticket not found") from None
    except EventNotFoundError:
        raise HTTPException(404, "Event not found") from None
    except EventStartedError:
        raise HTTPException(400, "Event has already started") from None
    except CancellationInProgressError:
        raise HTTPException(409, "Cancellation is already in progress") from None
    except CancellationRejectedError:
        raise HTTPException(409, "Events Provider rejected the cancellation") from None
    except CancellationNotSavedError as exc:
        raise HTTPException(
            500,
            "Cancellation was confirmed by Events Provider but could not be saved; "
            f"do not retry, reference ticket_id={exc.ticket_id}",
        ) from None
    except ProviderError as exc:
        # The outcome may be unknown (e.g. timeout); the ticket stays active.
        logger.warning("Cancellation of ticket %s failed: %s", ticket_id, exc)
        raise HTTPException(502, "Events Provider request failed") from None
    return {"success": True}
