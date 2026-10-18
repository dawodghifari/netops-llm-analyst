"""The database layer's defences: the guard, read-only access, the time limit."""

import sqlite3

import pytest

from src import db


def test_run_query_refuses_unsafe_sql(netops_db):
    with pytest.raises(ValueError):
        db.run_query("DELETE FROM costs")


def test_connection_is_read_only_even_without_the_guard(netops_db):
    con = db.connect()
    try:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            con.execute("DELETE FROM costs")
    finally:
        con.close()
    assert len(db.run_query("SELECT * FROM costs")) > 0


def test_runaway_query_is_cancelled(netops_db, monkeypatch):
    monkeypatch.setattr(db, "QUERY_TIMEOUT_SECONDS", 0.3)
    endless = "WITH RECURSIVE r(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM r) SELECT count(*) FROM r"
    with pytest.raises(TimeoutError):
        db.run_query(endless)


def test_missing_database_says_how_to_create_it(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", str(tmp_path / "absent.db"))
    with pytest.raises(FileNotFoundError, match="generate_data.py"):
        db.connect()


def test_schema_description_lists_every_table(netops_db):
    schema = db.schema_description()
    for table in ("pops", "links", "telemetry", "costs", "incidents"):
        assert f"{table}(" in schema
