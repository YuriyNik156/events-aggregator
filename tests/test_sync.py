import threading
from datetime import UTC, date, datetime
from unittest.mock import Mock

import pytest
from sqlalchemy import func, select

from app.events_provider import EventsPage, ProviderUnavailableError
from app.models import Event
from app.repositories import EventRepository, SyncMetadataRepository
from app.services import (
    SyncAlreadyRunningError,
    SyncStoppedError,
    _sync_lock,
    sync_events,
)
from tests.conftest import INITIAL, provider_event_json, provider_page


def page(*events: dict, next_url: str | None = None) -> EventsPage:
    return EventsPage.model_validate(provider_page(list(events), next_url=next_url))


def client_with(*pages) -> Mock:
    client = Mock()
    client.get_events_page.side_effect = list(pages)
    return client


def event_count(session) -> int:
    return session.scalar(select(func.count()).select_from(Event))


def stored_cursor(session) -> datetime | None:
    value = SyncMetadataRepository(session).get().last_changed_at
    return value.replace(tzinfo=UTC) if value and value.tzinfo is None else value


def test_first_sync_uses_initial_cursor_and_saves_metadata(session):
    client = client_with(page(provider_event_json("e1"), provider_event_json("e2")))

    result = sync_events(session, client)

    client.get_events_page.assert_called_once_with(changed_at=INITIAL)
    assert result.processed == 2
    assert event_count(session) == 2
    meta = SyncMetadataRepository(session).get()
    assert meta.sync_status == "success"
    assert meta.last_sync_time is not None
    assert stored_cursor(session) == datetime(2026, 1, 5, 7, tzinfo=UTC)
    assert result.changed_at_cursor == date(2026, 1, 5)


def test_all_pages_are_saved(session):
    client = client_with(
        page(provider_event_json("e1"), next_url="http://x/?page=2"),
        page(provider_event_json("e2")),
    )

    assert sync_events(session, client).processed == 2
    assert event_count(session) == 2


def test_second_sync_uses_stored_cursor_and_updates_without_duplicates(session):
    sync_events(session, client_with(page(provider_event_json("e1", name="Old"))))
    changed = provider_event_json(
        "e1", name="New", changed_at="2026-02-01T00:00:00+00:00"
    )
    client = client_with(page(changed))

    sync_events(session, client)

    client.get_events_page.assert_called_once_with(changed_at=date(2026, 1, 5))
    assert event_count(session) == 1
    assert EventRepository(session).get("e1").name == "New"
    assert stored_cursor(session).date() == date(2026, 2, 1)


def test_sync_without_events_keeps_cursor(session):
    sync_events(session, client_with(page(provider_event_json("e1"))))
    before = stored_cursor(session)

    result = sync_events(session, client_with(page()))

    assert result.processed == 0
    assert stored_cursor(session) == before
    assert SyncMetadataRepository(session).get().sync_status == "success"


def test_cursor_never_moves_back(session):
    sync_events(session, client_with(page(provider_event_json("e1"))))
    before = stored_cursor(session)
    older = provider_event_json("e2", changed_at="2025-01-01T00:00:00+00:00")

    sync_events(session, client_with(page(older)))

    assert stored_cursor(session) == before


def test_pagination_error_keeps_cursor_and_events(session):
    sync_events(session, client_with(page(provider_event_json("e1"))))
    before = stored_cursor(session)
    client = client_with(
        page(
            provider_event_json("e2", changed_at="2026-03-01T00:00:00+00:00"),
            next_url="http://x/?page=2",
        ),
        ProviderUnavailableError("boom"),
    )

    with pytest.raises(ProviderUnavailableError):
        sync_events(session, client)

    assert event_count(session) == 1  # page 1 of the failed run was rolled back
    assert stored_cursor(session) == before
    assert SyncMetadataRepository(session).get().sync_status == "error"


def test_failed_first_sync_leaves_no_cursor(session):
    with pytest.raises(ProviderUnavailableError):
        sync_events(session, client_with(ProviderUnavailableError("boom")))

    meta = SyncMetadataRepository(session).get()
    assert meta.sync_status == "error"
    assert meta.last_changed_at is None
    assert meta.last_sync_time is None


def test_save_error_keeps_cursor(session, monkeypatch):
    sync_events(session, client_with(page(provider_event_json("e1"))))
    before = stored_cursor(session)

    def broken_upsert(self, data):
        raise RuntimeError("db down")

    monkeypatch.setattr(EventRepository, "upsert", broken_upsert)
    with pytest.raises(RuntimeError):
        sync_events(session, client_with(page(provider_event_json("e2"))))

    assert stored_cursor(session) == before


def test_parallel_sync_is_refused(session):
    assert _sync_lock.acquire(blocking=False)
    try:
        client = client_with(page())
        with pytest.raises(SyncAlreadyRunningError):
            sync_events(session, client)
        client.get_events_page.assert_not_called()
    finally:
        _sync_lock.release()


def test_stop_before_start_changes_nothing(session):
    sync_events(session, client_with(page(provider_event_json("e1"))))
    before = stored_cursor(session)
    stop = threading.Event()
    stop.set()
    newer = provider_event_json("e2", changed_at="2026-03-01T00:00:00+00:00")

    with pytest.raises(SyncStoppedError):
        sync_events(session, client_with(page(newer)), stop)

    assert event_count(session) == 1
    assert stored_cursor(session) == before
    assert SyncMetadataRepository(session).get().sync_status == "success"  # not error


def test_stop_between_events_rolls_back(session, monkeypatch):
    stop = threading.Event()
    upsert = EventRepository.upsert

    def upsert_then_stop(self, data):
        result = upsert(self, data)
        stop.set()  # as if shutdown began while the first event was saved
        return result

    monkeypatch.setattr(EventRepository, "upsert", upsert_then_stop)
    client = client_with(page(provider_event_json("e1"), provider_event_json("e2")))

    with pytest.raises(SyncStoppedError):
        sync_events(session, client, stop)

    assert event_count(session) == 0
    assert SyncMetadataRepository(session).get() is None  # no error row either
    assert not _sync_lock.locked()


def test_unset_stop_does_not_change_the_sync(session):
    result = sync_events(
        session, client_with(page(provider_event_json("e1"))), threading.Event()
    )

    assert result.processed == 1
