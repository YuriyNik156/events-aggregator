import pytest
from sqlalchemy.engine import make_url

from app.config import (
    ConfigError,
    get_database_url,
    get_public_base_url,
    get_sync_interval,
)


def test_connection_string_has_priority(clean_db_env):
    clean_db_env.setenv(
        "POSTGRES_CONNECTION_STRING", "postgresql://u:p@dbhost:6543/main"
    )
    clean_db_env.setenv("POSTGRES_HOST", "other")

    url = get_database_url()

    assert url.host == "dbhost"
    assert url.port == 6543
    assert url.database == "main"


@pytest.mark.parametrize("scheme", ["postgresql", "postgres", "postgresql+psycopg"])
def test_scheme_normalized_to_psycopg3(clean_db_env, scheme):
    clean_db_env.setenv("POSTGRES_CONNECTION_STRING", f"{scheme}://u:p@h/db")

    assert get_database_url().drivername == "postgresql+psycopg"


def test_query_parameters_are_preserved(clean_db_env):
    clean_db_env.setenv(
        "POSTGRES_CONNECTION_STRING", "postgresql://u:p@h/db?sslmode=require"
    )

    assert get_database_url().query == {"sslmode": "require"}


def test_explicit_other_driver_is_not_overwritten(clean_db_env):
    clean_db_env.setenv("POSTGRES_CONNECTION_STRING", "postgresql+psycopg2://u:p@h/db")

    assert get_database_url().drivername == "postgresql+psycopg2"


def test_percent_encoded_credentials_are_decoded_once(clean_db_env):
    clean_db_env.setenv(
        "POSTGRES_CONNECTION_STRING", "postgresql://us%40er:p%2540w@h/db"
    )

    url = get_database_url()

    assert url.username == "us@er"
    assert url.password == "p%40w"  # %25 -> '%', not decoded a second time


def test_separate_variables_with_special_characters(clean_db_env):
    clean_db_env.setenv("POSTGRES_HOST", "dbhost")
    clean_db_env.setenv("POSTGRES_PORT", "5433")
    clean_db_env.setenv("POSTGRES_DATABASE_NAME", "appdb")
    clean_db_env.setenv("POSTGRES_USERNAME", "us@er")
    clean_db_env.setenv("POSTGRES_PASSWORD", "p@ss:w/rd%41#?")

    url = get_database_url()

    assert url.drivername == "postgresql+psycopg"
    assert (url.host, url.port, url.database) == ("dbhost", 5433, "appdb")
    # Round-trip through the rendered string must give back the raw credentials
    rendered = url.render_as_string(hide_password=False)
    again = make_url(rendered)
    assert again.username == "us@er"
    assert again.password == "p@ss:w/rd%41#?"


def test_empty_connection_string_falls_back_to_parts(clean_db_env):
    clean_db_env.setenv("POSTGRES_CONNECTION_STRING", "  ")
    clean_db_env.setenv("POSTGRES_HOST", "h")
    clean_db_env.setenv("POSTGRES_PORT", "5432")
    clean_db_env.setenv("POSTGRES_DATABASE_NAME", "d")
    clean_db_env.setenv("POSTGRES_USERNAME", "u")
    clean_db_env.setenv("POSTGRES_PASSWORD", "p")

    assert get_database_url().host == "h"


def test_missing_variables_are_listed_without_values(clean_db_env):
    clean_db_env.setenv("POSTGRES_HOST", "h")
    clean_db_env.setenv("POSTGRES_PASSWORD", "super-secret-value")

    with pytest.raises(ConfigError) as exc:
        get_database_url()

    message = str(exc.value)
    for name in ("POSTGRES_PORT", "POSTGRES_DATABASE_NAME", "POSTGRES_USERNAME"):
        assert name in message
    assert "super-secret-value" not in message


def test_no_configuration_at_all(clean_db_env):
    with pytest.raises(ConfigError, match="POSTGRES_CONNECTION_STRING"):
        get_database_url()


def test_invalid_port(clean_db_env):
    clean_db_env.setenv("POSTGRES_HOST", "h")
    clean_db_env.setenv("POSTGRES_PORT", "abc")
    clean_db_env.setenv("POSTGRES_DATABASE_NAME", "d")
    clean_db_env.setenv("POSTGRES_USERNAME", "u")
    clean_db_env.setenv("POSTGRES_PASSWORD", "p")

    with pytest.raises(ConfigError, match="POSTGRES_PORT"):
        get_database_url()


def test_invalid_connection_string_does_not_leak_it(clean_db_env):
    clean_db_env.setenv("POSTGRES_CONNECTION_STRING", "not a url, pw=topsecret")

    with pytest.raises(ConfigError) as exc:
        get_database_url()

    assert "topsecret" not in str(exc.value)


def test_non_postgres_connection_string_is_rejected(clean_db_env):
    clean_db_env.setenv("POSTGRES_CONNECTION_STRING", "mysql://u:p@h/db")

    with pytest.raises(ConfigError, match="PostgreSQL"):
        get_database_url()


def test_url_repr_masks_password(clean_db_env):
    clean_db_env.setenv("POSTGRES_CONNECTION_STRING", "postgresql://u:topsecret@h/db")

    assert "topsecret" not in repr(get_database_url())
    assert "topsecret" not in str(get_database_url())


def test_importing_database_module_does_not_need_configuration(clean_db_env):
    import app.database as database

    database.dispose_engine()
    # Only creating the engine requires configuration, and it fails clearly.
    with pytest.raises(ConfigError):
        database.get_engine()


def test_engine_is_lazy_and_does_not_connect(clean_db_env):
    import app.database as database

    clean_db_env.setenv("POSTGRES_CONNECTION_STRING", "postgresql://u:p@192.0.2.1/db")
    database.dispose_engine()
    try:
        engine = database.get_engine()  # must not open a connection
        assert engine.pool.checkedout() == 0
        assert database.get_engine() is engine
    finally:
        database.dispose_engine()


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://h.example", "https://h.example"),
        ("https://h.example/", "https://h.example"),
        ("  https://h.example:8443  ", "https://h.example:8443"),
        ("http://localhost:8000", "http://localhost:8000"),
        ("", None),
    ],
)
def test_public_base_url_valid(monkeypatch, value, expected):
    monkeypatch.setenv("PUBLIC_BASE_URL", value)
    assert get_public_base_url() == expected


def test_public_base_url_unset(monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    assert get_public_base_url() is None


@pytest.mark.parametrize(
    "value",
    [
        "h.example",
        "ftp://h.example",
        "https://",
        "https://h.example/prefix",
        "https://h.example/?a=1",
        "https://h.example/#x",
        "https://user:pw@h.example",
        "https://h.example:notaport",
    ],
)
def test_public_base_url_invalid(monkeypatch, value):
    monkeypatch.setenv("PUBLIC_BASE_URL", value)
    with pytest.raises(ConfigError) as exc:
        get_public_base_url()
    assert "pw" not in str(exc.value)


def test_sync_interval_defaults_to_24_hours(monkeypatch):
    monkeypatch.delenv("SYNC_INTERVAL_SECONDS", raising=False)

    assert get_sync_interval() == 86400


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("", 86400),
        ("  ", 86400),
        ("0", 0),
        ("60", 60),
        (" 3600 ", 3600),
        ("172800", 172800),
    ],
)
def test_sync_interval_accepts(monkeypatch, value, expected):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", value)

    assert get_sync_interval() == expected


@pytest.mark.parametrize("value", ["-1", "1", "59", "abc", "1.5", "1h"])
def test_sync_interval_rejects(monkeypatch, value):
    monkeypatch.setenv("SYNC_INTERVAL_SECONDS", value)

    with pytest.raises(ConfigError, match="SYNC_INTERVAL_SECONDS"):
        get_sync_interval()
