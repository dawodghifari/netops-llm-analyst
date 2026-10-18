"""SQLite helpers: connection, schema introspection, and safe read-only queries."""

import os
import re
import sqlite3
import time
from pathlib import Path

import pandas as pd

DB_PATH = os.environ.get(
    "NETOPS_DB",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "netops.db"),
)

# Any single query is cancelled after this long, so a runaway query (an unbounded
# WITH RECURSIVE, say) fails with an error instead of hanging the app.
QUERY_TIMEOUT_SECONDS = float(os.environ.get("NETOPS_QUERY_TIMEOUT", "10"))

# Only single read-only SELECT/WITH statements are allowed to reach the database.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|truncate)\b",
    re.IGNORECASE,
)


def connect() -> sqlite3.Connection:
    """Open the database READ-ONLY.

    The query guard below is the first line of defence; this is the second. Even if
    a write slipped past the guard, SQLite itself would refuse it. The data
    generator opens its own read-write connection.
    """
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run: python data/generate_data.py"
        )
    return sqlite3.connect(Path(DB_PATH).resolve().as_uri() + "?mode=ro", uri=True)


def table_names() -> list[str]:
    """Names of all tables currently in the database (empty list if none)."""
    con = connect()
    try:
        return [
            r[0]
            for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
        ]
    finally:
        con.close()


def schema_description() -> str:
    """A compact text description of all tables + columns, for the LLM prompt."""
    con = connect()
    lines = []
    for (table,) in con.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall():
        cols = con.execute(f"PRAGMA table_info({table})").fetchall()
        col_str = ", ".join(f"{c[1]} {c[2]}" for c in cols)
        lines.append(f"{table}({col_str})")
    con.close()
    return "\n".join(lines)


def is_safe_select(sql: str) -> bool:
    """Allow exactly one read-only SELECT/WITH statement; reject anything that writes."""
    s = sql.strip().rstrip(";").strip()
    if ";" in s:  # no stacked statements
        return False
    if _FORBIDDEN.search(s):
        return False
    return bool(re.match(r"^(select|with)\b", s, re.IGNORECASE))


def run_query(sql: str, params: tuple = ()) -> pd.DataFrame:
    """Execute a vetted read-only query and return a DataFrame.

    Values go in `params` as `?` placeholders, never into the SQL text.
    """
    if not is_safe_select(sql):
        raise ValueError("Only a single read-only SELECT/WITH query is permitted.")
    con = connect()
    deadline = time.monotonic() + QUERY_TIMEOUT_SECONDS
    # SQLite calls this every N virtual-machine steps; a non-zero return aborts.
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
    try:
        return pd.read_sql_query(sql, con, params=params)
    except Exception as exc:
        if time.monotonic() > deadline:
            raise TimeoutError(
                f"Query cancelled after {QUERY_TIMEOUT_SECONDS:g}s."
            ) from exc
        raise
    finally:
        con.close()
