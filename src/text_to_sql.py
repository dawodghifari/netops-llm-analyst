"""
Natural-language → SQL → answer pipeline.

Flow:
  1. Ask the LLM to turn the question into a single read-only SQLite query, given the
     live schema.
  2. Validate it (read-only, single statement) and execute it.
  3. Ask the LLM to summarise the result table in plain English.

The LLM client is injected, so this module is testable offline with a fake client.
"""

import re
from dataclasses import dataclass

import pandas as pd

from . import db

SQL_SYSTEM = """You are a senior data analyst writing SQLite queries for a CDN/edge-network
operations database. Given a question and the schema, return ONE read-only SQLite
SELECT (or WITH ... SELECT) query that answers it.

Rules:
- Output ONLY the SQL. No prose, no markdown fences, no explanation.
- Read-only: never INSERT/UPDATE/DELETE/DROP/CREATE/ALTER/PRAGMA.
- A single statement, no semicolons mid-query.
- Prefer explicit column names; round money to whole dollars and rates to 3 dp.
- 'month' is 'YYYY-MM'; 'date' is ISO 'YYYY-MM-DD'. Use substr(date,1,7) for monthly rollups.
- Quality KPIs: avg_latency_ms, packet_loss_pct, rebuffer_ratio_pct, availability_pct.
- Cost columns live in `costs`; traffic/quality in `telemetry`.

Schema:
{schema}
"""

SUMMARY_SYSTEM = """You are a network-operations data analyst. Given a user's question and the
resulting data table, write a concise, factual 1-3 sentence answer. Quote the key
numbers. Do not invent values beyond the table. If the table is empty, say so."""


@dataclass
class QueryResult:
    question: str
    sql: str
    data: pd.DataFrame
    summary: str


def _strip_fences(text: str) -> str:
    """Remove ```sql ... ``` fences or stray backticks the model may add."""
    text = text.strip()
    m = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    return text.strip().rstrip(";").strip()


def generate_sql(question: str, llm) -> str:
    system = SQL_SYSTEM.format(schema=db.schema_description())
    raw = llm.complete(system=system, user=question, max_tokens=600)
    return _strip_fences(raw)


def answer_question(question: str, llm, max_retries: int = 1) -> QueryResult:
    last_err = None
    sql = ""
    for attempt in range(max_retries + 1):
        prompt = question if attempt == 0 else (
            f"{question}\n\nYour previous query failed with: {last_err}\n"
            "Return a corrected single read-only SELECT."
        )
        sql = generate_sql(prompt, llm)
        if not db.is_safe_select(sql):
            last_err = "query was not a single read-only SELECT"
            continue
        try:
            data = db.run_query(sql)
            break
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
    else:
        raise RuntimeError(f"Could not produce a valid query: {last_err}")

    # Summarise (truncate very large tables before sending to the model).
    preview = data.head(50).to_csv(index=False)
    summary = llm.complete(
        system=SUMMARY_SYSTEM,
        user=f"Question: {question}\n\nResult table (CSV):\n{preview}",
        max_tokens=300,
    )
    return QueryResult(question=question, sql=sql, data=data, summary=summary)
