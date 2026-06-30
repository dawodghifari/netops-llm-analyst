"""
NetOps LLM Analyst — local Streamlit app.

Run:
    streamlit run app/streamlit_app.py

Four tabs:
  • Ask         — natural-language question -> SQL -> table + chart + summary (LLM)
  • Cost        — cost governance: per-PoP spend + month-over-month decomposition
  • Quality     — anomaly detection on streaming quality KPIs
  • Capacity    — per-PoP traffic-growth forecast and days-to-capacity
"""

import os
import sys

import pandas as pd
import streamlit as st

# Make the `src` package importable when run from anywhere.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import analysis, db, text_to_sql  # noqa: E402
from src.llm import LLMClient  # noqa: E402

st.set_page_config(page_title="NetOps LLM Analyst", layout="wide")


@st.cache_resource
def get_llm():
    return LLMClient()


@st.cache_data
def months():
    return db.run_query("SELECT DISTINCT month FROM costs ORDER BY month")["month"].tolist()


st.title("NetOps LLM Analyst")
st.caption("LLM-powered Q&A and data-mining over CDN / edge-network cost & quality telemetry")

# DB presence check
try:
    db.connect().close()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()

with st.sidebar:
    st.subheader("Status")
    try:
        _llm = get_llm()
        st.success(f"LLM: {_llm.provider} · {_llm.model}")
    except Exception as e:  # noqa: BLE001
        st.warning(f"LLM not configured — Ask tab disabled.\n\n{e}")
    st.write("**Tables:** pops, links, telemetry, costs, incidents")
    st.code(db.schema_description(), language="text")

tab_ask, tab_cost, tab_quality, tab_capacity = st.tabs(
    ["Ask", "Cost governance", "Quality", "Capacity"]
)

# ---- Ask ------------------------------------------------------------------
with tab_ask:
    st.markdown("Ask a question in plain English. It's turned into SQL, run against the DB, and summarised.")
    examples = [
        "Which 5 PoPs had the highest total bill last month?",
        "What is the average rebuffer ratio by region?",
        "Show monthly total cost trend across all PoPs.",
        "Which PoPs had availability below 99.9 on any day?",
    ]
    picked = st.selectbox("Example questions", [""] + examples)
    question = st.text_input("Your question", value=picked)
    if st.button("Run", type="primary") and question.strip():
        try:
            llm = get_llm()
        except Exception as e:  # noqa: BLE001
            st.error(f"LLM not configured: {e}")
        else:
            with st.spinner("Thinking…"):
                res = text_to_sql.answer_question(question, llm)
            st.success(res.summary)
            with st.expander("Generated SQL"):
                st.code(res.sql, language="sql")
            st.dataframe(res.data, use_container_width=True)
            # auto-chart: if exactly one label column + one numeric column
            num = res.data.select_dtypes("number")
            if res.data.shape[1] == 2 and num.shape[1] == 1:
                label_col = [c for c in res.data.columns if c not in num.columns][0]
                st.bar_chart(res.data.set_index(label_col))

# ---- Cost -----------------------------------------------------------------
with tab_cost:
    ms = months()
    latest = analysis.cost_by_pop()
    st.subheader(f"Per-PoP spend — {latest['month'].iloc[0]}")
    st.bar_chart(latest.set_index("pop_id")["total_bill_usd"])
    st.dataframe(latest.drop(columns="month"), use_container_width=True)

    st.subheader("Month-over-month cost decomposition")
    c1, c2 = st.columns(2)
    ma = c1.selectbox("From", ms, index=0)
    mb = c2.selectbox("To", ms, index=len(ms) - 1)
    dec = analysis.decompose_cost_change(ma, mb)
    st.caption(f"Net change: ${dec['delta_usd'].sum():,.0f}")
    st.dataframe(dec, use_container_width=True)
    if st.button("Narrate this indicator (LLM)"):
        try:
            txt = analysis.narrate_indicator(f"Total-bill change {ma}→{mb}", dec, get_llm())
            st.info(txt)
        except Exception as e:  # noqa: BLE001
            st.error(f"LLM not configured: {e}")

# ---- Quality --------------------------------------------------------------
with tab_quality:
    metric = st.selectbox(
        "Quality KPI",
        ["rebuffer_ratio_pct", "packet_loss_pct", "avg_latency_ms", "availability_pct"],
    )
    z = st.slider("Anomaly threshold (robust z)", 2.0, 6.0, 3.0, 0.5)
    an = analysis.quality_anomalies(metric, z)
    st.subheader(f"{len(an)} anomalous PoP-days in {metric}")
    st.dataframe(an, use_container_width=True)

# ---- Capacity -------------------------------------------------------------
with tab_capacity:
    horizon = st.slider("Forecast horizon (days)", 7, 90, 30, 7)
    cf = analysis.capacity_forecast(horizon)
    st.subheader("Projected utilisation & days-to-capacity")
    st.dataframe(cf, use_container_width=True)
    chart_col = f"projected_util_{horizon}d_pct"
    st.bar_chart(cf.set_index("pop_id")[chart_col])
