"""
Data-mining and indicator analysis over the NetOps dataset (pure pandas/numpy — no
LLM needed for the maths): quality anomalies, cost decomposition, and capacity
forecasting.

Functions return tidy DataFrames so the Streamlit app and notebook can chart them,
and the LLM can narrate them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import db


# ---- Quality governance ---------------------------------------------------
def quality_anomalies(metric: str = "rebuffer_ratio_pct", z_thresh: float = 3.0) -> pd.DataFrame:
    """Per-PoP anomalies in a quality metric, flagged by robust z-score (median/MAD).

    Robust stats (median + MAD) are used instead of mean/std so a few incident days
    don't inflate the threshold and hide the rest.
    """
    tel = db.run_query("SELECT date, pop_id, " + metric + " AS value FROM telemetry")
    out = []
    for pop_id, grp in tel.groupby("pop_id"):
        v = grp["value"].to_numpy()
        med = np.median(v)
        mad = np.median(np.abs(v - med)) or 1e-9
        robust_z = 0.6745 * (grp["value"] - med) / mad
        flagged = grp[robust_z.abs() >= z_thresh].copy()
        flagged["robust_z"] = robust_z[robust_z.abs() >= z_thresh].round(2)
        flagged["median_baseline"] = round(med, 3)
        out.append(flagged)
    res = pd.concat(out) if out else tel.iloc[0:0]
    return res.sort_values("robust_z", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# ---- Cost governance ------------------------------------------------------
def cost_by_pop(month: str | None = None) -> pd.DataFrame:
    """Total bill and component split per PoP for a month (default: latest)."""
    if month is None:
        month = db.run_query("SELECT MAX(month) AS m FROM costs")["m"].iloc[0]
    df = db.run_query(
        f"""SELECT pop_id, bandwidth_cost_usd, transit_cost_usd, hardware_amort_usd,
                   total_bill_usd
            FROM costs WHERE month = '{month}' ORDER BY total_bill_usd DESC"""
    )
    df["month"] = month
    return df


def decompose_cost_change(month_a: str, month_b: str) -> pd.DataFrame:
    """Decompose the total-bill change from month_a to month_b by PoP and component.

    Returns a tidy table of contributions (USD delta) so you can see exactly what
    drove the bill up or down — the core 'indicator decomposition' task.
    """
    a = db.run_query(f"SELECT * FROM costs WHERE month='{month_a}'").set_index("pop_id")
    b = db.run_query(f"SELECT * FROM costs WHERE month='{month_b}'").set_index("pop_id")
    comps = ["bandwidth_cost_usd", "transit_cost_usd", "hardware_amort_usd"]
    rows = []
    for pop_id in sorted(set(a.index) | set(b.index)):
        for comp in comps:
            va = float(a.loc[pop_id, comp]) if pop_id in a.index else 0.0
            vb = float(b.loc[pop_id, comp]) if pop_id in b.index else 0.0
            rows.append({"pop_id": pop_id, "component": comp, "delta_usd": round(vb - va, 2)})
    df = pd.DataFrame(rows)
    df = df[df["delta_usd"] != 0]
    return df.sort_values("delta_usd", key=lambda s: s.abs(), ascending=False).reset_index(drop=True)


# ---- Capacity planning ----------------------------------------------------
def capacity_forecast(horizon_days: int = 30) -> pd.DataFrame:
    """Per-PoP linear trend on peak traffic; project utilisation and days-to-capacity.

    The trend is fitted to the 7-day rolling maximum of daily peaks, not to the
    daily peaks themselves. Capacity is a peak problem: the question is when the
    busiest day of the week crosses capacity. Fitting raw days mixes weekdays with
    weekends (which run hotter here), so the latest day could sit far above the
    trend, and a PoP already over capacity was reported as weeks away from it.

    `current_*` is the latest 7-day peak; `trend_peak_gbps` is the fitted value on
    the same day, so the projection can be read against it like for like.
    """
    tel = db.run_query("SELECT date, pop_id, peak_traffic_gbps FROM telemetry")
    caps = db.run_query("SELECT pop_id, capacity_gbps FROM pops").set_index("pop_id")["capacity_gbps"]
    tel["date"] = pd.to_datetime(tel["date"])
    rows = []
    for pop_id, grp in tel.groupby("pop_id"):
        grp = grp.sort_values("date").set_index("date")
        weekly_peak = grp["peak_traffic_gbps"].rolling("7D", min_periods=7).max().dropna()
        t = (weekly_peak.index - weekly_peak.index.min()).days.to_numpy()
        y = weekly_peak.to_numpy()
        slope, intercept = np.polyfit(t, y, 1)  # Gbps/day
        cap = float(caps[pop_id])
        current = float(y[-1])
        trend_now = slope * t[-1] + intercept
        projected = slope * (t[-1] + horizon_days) + intercept
        # A PoP already at or over capacity has no days left, whatever the trend says.
        if current >= cap:
            days_left = 0.0
        elif slope > 0:
            days_left = max(0.0, (cap - trend_now) / slope)
        else:
            days_left = np.inf
        rows.append(
            {
                "pop_id": pop_id,
                "capacity_gbps": round(cap, 1),
                "current_peak_gbps": round(current, 1),
                "current_util_pct": round(100 * current / cap, 1),
                "trend_peak_gbps": round(trend_now, 1),
                "growth_gbps_per_day": round(slope, 3),
                f"projected_peak_{horizon_days}d_gbps": round(projected, 1),
                f"projected_util_{horizon_days}d_pct": round(100 * projected / cap, 1),
                "days_to_capacity": (round(days_left) if np.isfinite(days_left) else None),
            }
        )
    return pd.DataFrame(rows).sort_values(
        f"projected_util_{horizon_days}d_pct", ascending=False
    ).reset_index(drop=True)


# ---- LLM narration of an indicator ----------------------------------------
NARRATE_SYSTEM = """You are a network-operations analyst briefing leadership on cost/quality
governance. Given a decomposition table, write a short, factual briefing: the headline
number, the top 2-3 drivers (with figures), and one recommended follow-up. Be concise
and quote real numbers from the table only."""


def narrate_indicator(title: str, table: pd.DataFrame, llm) -> str:
    """Turn any decomposition/analysis table into a governance-style narrative via the LLM."""
    preview = table.head(40).to_csv(index=False)
    return llm.complete(
        system=NARRATE_SYSTEM,
        user=f"Indicator: {title}\n\nDecomposition table (CSV):\n{preview}",
        max_tokens=350,
    )
