"""The question -> SQL -> answer pipeline, driven by a scripted fake LLM."""

import pytest

from src import text_to_sql

GOOD_SQL = "SELECT pop_id, SUM(total_bill_usd) AS total FROM costs GROUP BY pop_id ORDER BY total DESC LIMIT 3"


def test_happy_path_returns_sql_data_and_summary(netops_db, fake_llm):
    llm = fake_llm(GOOD_SQL, "TYO costs the most.")
    result = text_to_sql.answer_question("Which PoPs cost the most?", llm)
    assert result.sql == GOOD_SQL
    assert len(result.data) == 3
    assert result.summary == "TYO costs the most."
    # The schema goes into the SQL prompt; the result table into the summary prompt.
    assert "costs(" in llm.calls[0]["system"]
    assert "pop_id,total" in llm.calls[1]["user"]


def test_markdown_fences_are_stripped(netops_db, fake_llm):
    llm = fake_llm(f"```sql\n{GOOD_SQL};\n```", "ok")
    assert text_to_sql.answer_question("q", llm).sql == GOOD_SQL


def test_unsafe_sql_is_never_run_and_triggers_a_retry(netops_db, fake_llm):
    llm = fake_llm("DELETE FROM costs", GOOD_SQL, "ok")
    result = text_to_sql.answer_question("q", llm)
    assert result.sql == GOOD_SQL
    assert "not a single read-only SELECT" in llm.calls[1]["user"]


def test_sql_error_is_fed_back_to_the_model(netops_db, fake_llm):
    llm = fake_llm("SELECT nope FROM costs", GOOD_SQL, "ok")
    text_to_sql.answer_question("q", llm)
    assert "no such column" in llm.calls[1]["user"]


def test_gives_up_after_the_retry_budget(netops_db, fake_llm):
    llm = fake_llm("DROP TABLE costs", "DROP TABLE costs")
    with pytest.raises(RuntimeError, match="Could not produce a valid query"):
        text_to_sql.answer_question("q", llm, max_retries=1)
