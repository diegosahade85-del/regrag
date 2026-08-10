"""Test fixtures.

Store tests run against a real Postgres + pgvector instance rather than a mock:
the failure modes worth catching here are SQL and pgvector behaviour, and a mock
would assert our own assumptions back at us. Each test runs inside a transaction
that is rolled back, so tests neither persist state nor see each other's.
"""

import os

import psycopg
import pytest
from dotenv import load_dotenv

load_dotenv()

def _test_url() -> str:
    if explicit := os.environ.get("TEST_DATABASE_URL"):
        return explicit
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        return ""
    # Swap only the database name — the last path segment. A plain str.replace
    # of the db name also rewrites the username, which is the same word.
    base, _, _ = url.rpartition("/")
    return f"{base}/regrag_test"


TEST_DATABASE_URL = _test_url()


@pytest.fixture
def conn():
    if not TEST_DATABASE_URL:
        pytest.skip("DATABASE_URL not configured")
    connection = psycopg.connect(TEST_DATABASE_URL)
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()
