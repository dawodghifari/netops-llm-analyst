"""The SQL allow-list that every model-proposed query passes through."""

import pytest

from src.db import is_safe_select


@pytest.mark.parametrize("sql", [
    "SELECT * FROM costs",
    "select pop_id from pops;",
    "  WITH t AS (SELECT 1 AS x) SELECT x FROM t  ",
    "SELECT created_at FROM incidents",  # 'create' inside a word is fine
])
def test_allows_single_read_only_queries(sql):
    assert is_safe_select(sql)


@pytest.mark.parametrize("sql", [
    "SELECT 1; DROP TABLE costs",       # stacked statement
    "DELETE FROM costs",
    "UPDATE costs SET total_bill_usd = 0",
    "INSERT INTO pops VALUES (1)",
    "DROP TABLE pops",
    "ATTACH DATABASE 'x.db' AS x",
    "PRAGMA writable_schema = 1",
    "WITH t AS (SELECT 1) DELETE FROM costs",
    "EXPLAIN SELECT 1",                 # not a SELECT/WITH at the start
    "",
])
def test_rejects_writes_stacking_and_non_selects(sql):
    assert not is_safe_select(sql)
