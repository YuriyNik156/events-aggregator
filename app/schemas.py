from datetime import datetime

from pydantic import BaseModel

from app.models import Event


class PlaceOut(BaseModel):
    id: str
    name: str
    city: str
    address: str


class EventOut(BaseModel):
    id: str
    name: str
    place: PlaceOut
    event_time: datetime
    registration_deadline: datetime | None
    status: str
    number_of_visitors: int

    @classmethod
    def from_event(cls, event: Event) -> "EventOut":
        return cls(
            id=event.id,
            name=event.name,
            place=PlaceOut(
                id=event.place_id,
                name=event.place_name,
                city=event.place_city,
                address=event.place_address,
            ),
            event_time=event.event_time,
            registration_deadline=event.registration_deadline,
            status=event.status,
            number_of_visitors=event.number_of_visitors,
        )


class PlaceDetailOut(PlaceOut):
    seats_pattern: str


class EventDetailOut(EventOut):
    place: PlaceDetailOut

    @classmethod
    def from_event(cls, event: Event) -> "EventDetailOut":
        base = EventOut.from_event(event)
        return cls(
            **base.model_dump(exclude={"place"}),
            place=PlaceDetailOut(
                **base.place.model_dump(), seats_pattern=event.place_seats_pattern
            ),
        )


class EventsPageOut(BaseModel):
    count: int
    next: str | None
    previous: str | None
    results: list[EventOut]


class SeatsOut(BaseModel):
    event_id: str
    available_seats: list[str]
