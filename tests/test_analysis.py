"""The deterministic analysis: every number the LLM later narrates."""

import numpy as np
import pytest

from src import analysis, db


def test_capacity_never_reports_days_left_for_a_full_pop(netops_db):
    cf = analysis.capacity_forecast(30)
    full = cf[cf["current_peak_gbps"] >= cf["capacity_gbps"]]
    assert len(full) > 0, "the seeded data should include a PoP over capacity"
    assert (full["days_to_capacity"] == 0).all()
    assert (cf["days_to_capacity"].dropna() >= 0).all()


def test_capacity_projection_grows_with_the_horizon(netops_db):
    short = analysis.capacity_forecast(30).set_index("pop_id")
    long = analysis.capacity_forecast(90).set_index("pop_id")
    growing = short.index[short["growth_gbps_per_day"] > 0]
    assert (long.loc[growing, "projected_util_90d_pct"] >= short.loc[growing, "projected_util_30d_pct"]).all()


def test_capacity_trend_tracks_the_weekly_peak(netops_db):
    # Fitting the 7-day peak keeps the latest point close to its own trend. Fitting
    # raw daily peaks left it 8 to 25% above on every PoP (weekends run hotter).
    cf = analysis.capacity_forecast(30)
    gap = (cf["current_peak_gbps"] - cf["trend_peak_gbps"]).abs() / cf["capacity_gbps"]
    assert gap.median() < 0.05


def test_cost_decomposition_adds_up_to_the_bill_change(netops_db):
    months = db.run_query("SELECT DISTINCT month FROM costs ORDER BY month")["month"].tolist()
    a, b = months[-2], months[-1]
    decomposition = analysis.decompose_cost_change(a, b)
    totals = db.run_query(
        "SELECT month, SUM(total_bill_usd) AS t FROM costs WHERE month IN (?, ?) GROUP BY month",
        (a, b),
    ).set_index("month")["t"]
    assert np.isclose(decomposition["delta_usd"].sum(), totals[b] - totals[a], atol=1.0)


def test_quality_anomalies_are_beyond_the_threshold(netops_db):
    anomalies = analysis.quality_anomalies("rebuffer_ratio_pct", z_thresh=3.0)
    assert len(anomalies) > 0
    assert (anomalies["robust_z"].abs() >= 3.0).all()


def test_narration_sends_the_table_to_the_model(netops_db, fake_llm):
    llm = fake_llm("Bandwidth drove the increase.")
    table = analysis.cost_by_pop()
    assert analysis.narrate_indicator("Cost by PoP", table, llm) == "Bandwidth drove the increase."
    assert "pop_id" in llm.calls[0]["user"]


def test_unknown_metric_is_rejected_before_any_sql(netops_db):
    with pytest.raises(ValueError, match="unknown quality metric"):
        analysis.quality_anomalies("value FROM pops --")


def test_every_listed_metric_runs(netops_db):
    for metric in analysis.QUALITY_METRICS:
        analysis.quality_anomalies(metric)


def test_month_is_a_bound_parameter_not_sql(netops_db):
    # Pasted into the old f-string query this became "month = '2026-01' OR '1'='1'"
    # and returned every row. Bound as a parameter it is just a month that
    # matches nothing.
    assert analysis.cost_by_pop("2026-01' OR '1'='1").empty
