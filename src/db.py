"""SQLite helpers: connection, schema introspection, and safe read-only queries."""

import os
import re
import sqlite3

import pandas as pd

DB_PATH = os.environ.get(
    "NETOPS_DB",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "netops.db"),
)

# Only single read-only SELECT/WITH statements are allowed to reach the database.
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|pragma|replace|truncate)\b",
    re.IGNORECASE,
)


def connect() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database not found at {DB_PATH}. Run: python data/generate_data.py"
        )
    return sqlite3.connect(DB_PATH)


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


def run_query(sql: str) -> pd.DataFrame:
    """Execute a vetted read-only query and return a DataFrame."""
    if not is_safe_select(sql):
        raise ValueError("Only a single read-only SELECT/WITH query is permitted.")
    con = connect()
    try:
        return pd.read_sql_query(sql, con)
    finally:
        con.close()
