import os
from dataclasses import dataclass, field

from sqlalchemy.engine import URL, make_url
from sqlalchemy.exc import ArgumentError

DRIVER = "postgresql+psycopg"
PARTS_VARS = (
    "POSTGRES_HOST",
    "POSTGRES_PORT",
    "POSTGRES_DATABASE_NAME",
    "POSTGRES_USERNAME",
    "POSTGRES_PASSWORD",
)


class ConfigError(RuntimeError):
    """Configuration is missing or invalid. Never contains secret values."""


def _env(name: str) -> str:
    return os.environ.get(name, "").strip()


def get_database_url() -> URL:
    """Build the PostgreSQL URL from the LMS environment.

    ``POSTGRES_CONNECTION_STRING`` wins when non-empty; otherwise the URL is
    assembled from the separate ``POSTGRES_*`` variables. The returned URL
    object masks the password in ``str()``/``repr()``.
    """
    connection_string = _env("POSTGRES_CONNECTION_STRING")
    if connection_string:
        return _from_connection_string(connection_string)

    values = {name: _env(name) for name in PARTS_VARS}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise ConfigError(
            "PostgreSQL is not configured: set POSTGRES_CONNECTION_STRING or "
            "all of POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DATABASE_NAME, "
            "POSTGRES_USERNAME, POSTGRES_PASSWORD. Missing: " + ", ".join(missing)
        )
    try:
        port = int(values["POSTGRES_PORT"])
    except ValueError:
        raise ConfigError("POSTGRES_PORT must be an integer") from None
    # URL.create takes raw (unencoded) credentials and escapes them on render.
    return URL.create(
        DRIVER,
        username=values["POSTGRES_USERNAME"],
        password=values["POSTGRES_PASSWORD"],
        host=values["POSTGRES_HOST"],
        port=port,
        database=values["POSTGRES_DATABASE_NAME"],
    )


def _from_connection_string(value: str) -> URL:
    try:
        url = make_url(value)  # decodes percent-escapes once
    except ArgumentError:
        # Do not echo the value: it may contain the password.
        raise ConfigError("POSTGRES_CONNECTION_STRING is not a valid URL") from None
    if url.get_backend_name() not in ("postgresql", "postgres"):
        raise ConfigError("POSTGRES_CONNECTION_STRING must be a PostgreSQL URL")
    if url.drivername in ("postgres", "postgresql"):
        url = url.set(drivername=DRIVER)
    return url


DEFAULT_EVENTS_PROVIDER_URL = "http://events-provider.dev-2.python-labs.ru"


@dataclass(frozen=True)
class ProviderSettings:
    base_url: str
    api_key: str = field(repr=False)


def get_provider_settings() -> ProviderSettings:
    """Events Provider address and key from the environment (key is a secret)."""
    api_key = _env("EVENTS_PROVIDER_API_KEY")
    if not api_key:
        raise ConfigError("EVENTS_PROVIDER_API_KEY is not set")
    base_url = _env("EVENTS_PROVIDER_BASE_URL") or DEFAULT_EVENTS_PROVIDER_URL
    return ProviderSettings(base_url=base_url, api_key=api_key)
