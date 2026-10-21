from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, StringConstraints

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


NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
# EmailStr would need email-validator, which is not in uv.lock; a plain shape
# check is enough because the provider has the final say.
Email = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        pattern=r"^[^@\s]+@[^@\s.]+(\.[^@\s.]+)+$",
        max_length=254,
    ),
]


class TicketCreate(BaseModel):
    event_id: NonEmptyStr
    first_name: NonEmptyStr
    last_name: NonEmptyStr
    email: Email
    seat: NonEmptyStr


class TicketCreatedOut(BaseModel):
    ticket_id: str
