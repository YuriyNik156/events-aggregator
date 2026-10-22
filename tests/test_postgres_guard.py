"""The guard in front of every PostgreSQL test: it must stop a wrong database
*before* any test can touch it, and it needs no database itself."""

import pytest

from tests.conftest import checked_postgres_url

GOOD = "postgresql://u:p@localhost:5432/events_test"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    monkeypatch.delenv("POSTGRES_CONNECTION_STRING", raising=False)
    monkeypatch.delenv("REQUIRE_POSTGRES_TESTS", raising=False)


def test_without_a_url_the_tests_are_skipped(monkeypatch):
    with pytest.raises(pytest.skip.Exception):
        checked_postgres_url()


def test_without_a_url_they_fail_where_they_are_required(monkeypatch):
    monkeypatch.setenv("REQUIRE_POSTGRES_TESTS", "1")

    with pytest.raises(pytest.fail.Exception, match="required"):
        checked_postgres_url()


@pytest.mark.parametrize(
    "url",
    [GOOD, GOOD.replace("localhost", "127.0.0.1"), GOOD.replace("localhost", "[::1]")],
)
def test_a_local_test_database_is_accepted(monkeypatch, url):
    monkeypatch.setenv("TEST_DATABASE_URL", url)

    assert checked_postgres_url() == url


@pytest.mark.parametrize(
    "name",
    ["events", "events_prod", "test", "test_events", "events_test_backup", "postgres"],
)
def test_a_database_not_named_like_a_test_one_is_refused(monkeypatch, name):
    monkeypatch.setenv("TEST_DATABASE_URL", GOOD.replace("events_test", name))

    with pytest.raises(pytest.fail.Exception, match="_test"):
        checked_postgres_url()


@pytest.mark.parametrize(
    "host", ["db.example.org", "10.0.0.5", "postgres.python-labs.ru", "lms-postgres"]
)
def test_a_database_on_another_machine_is_refused(monkeypatch, host):
    monkeypatch.setenv("TEST_DATABASE_URL", GOOD.replace("localhost", host))

    with pytest.raises(pytest.fail.Exception, match="this machine"):
        checked_postgres_url()


def test_the_applications_own_database_is_refused(monkeypatch):
    monkeypatch.setenv("TEST_DATABASE_URL", GOOD)
    monkeypatch.setenv("POSTGRES_CONNECTION_STRING", GOOD.replace("u:p", "other:pw"))

    with pytest.raises(pytest.fail.Exception, match="application database"):
        checked_postgres_url()


def test_another_database_on_the_same_server_is_not_confused_with_the_app_one(
    monkeypatch,
):
    monkeypatch.setenv("TEST_DATABASE_URL", GOOD)
    monkeypatch.setenv(
        "POSTGRES_CONNECTION_STRING", "postgresql://u:p@localhost:5432/events"
    )

    assert checked_postgres_url() == GOOD


def test_the_error_messages_do_not_contain_the_password(monkeypatch):
    monkeypatch.setenv(
        "TEST_DATABASE_URL", "postgresql://u:hunter2@db.example.org/x_test"
    )

    with pytest.raises(pytest.fail.Exception) as info:
        checked_postgres_url()

    assert "hunter2" not in str(info.value)
