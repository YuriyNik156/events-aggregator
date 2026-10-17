import logging
from collections.abc import Iterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, time
from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from sqlalchemy.orm import Session

from app.config import ConfigError, get_provider_settings
from app.database import dispose_engine, get_session_factory
from app.events_provider import EventsProviderClient, ProviderError
from app.repositories import EventRepository
from app.schemas import EventOut, EventsPageOut
from app.services import SyncAlreadyRunningError, sync_events

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    dispose_engine()


app = FastAPI(title="Events Aggregator", lifespan=lifespan)


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
