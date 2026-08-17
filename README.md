# NetOps LLM Analyst

An **LLM-powered analytics tool for CDN / edge-network operations data** — ask questions in
plain English, get SQL-backed answers, and run data-mining over cost, streaming-quality and
capacity metrics. The question it was built to answer: how much of network **cost and quality
governance** can a language model do without ever being trusted to compute a number?

```
                    ┌──────────────── Streamlit app (local) ────────────────┐
  "Which PoPs cost  │  Ask  │  Cost governance  │  Quality  │  Capacity      │
   the most?"  ───▶ └───┬───────────────────────────────────────────────────┘
                        │
              ┌─────────▼──────────┐   schema-aware     ┌──────────────┐
              │  text-to-SQL (LLM) │ ─────────────────▶ │  SQLite DB   │
              │  read-only guard   │ ◀───── results ─── │ (PoPs/links/ │
              └─────────┬──────────┘                    │  telemetry/  │
                        │ summarise / narrate           │  costs/inci.)│
              ┌─────────▼──────────┐                    └──────────────┘
              │  LLM (Anthropic /  │     pandas/numpy data-mining:
              │  OpenAI, pluggable)│     anomalies · cost decomposition · forecast
              └────────────────────┘
```

## What it does

- **Natural-language Q&A (text-to-SQL).** Your question + the live DB schema go to an LLM,
  which returns a single **read-only** SQLite query; a safety guard blocks anything that
  writes or stacks statements, the query runs, and the LLM summarises the result.
- **Cost governance.** Per-PoP spend and a **month-over-month cost decomposition** — exactly
  which PoP and which component (bandwidth / transit / hardware) drove the bill up or down.
- **Quality governance.** Robust-z (median/MAD) **anomaly detection** on streaming KPIs
  (rebuffer ratio, packet loss, latency, availability).
- **Capacity planning.** Per-PoP traffic-growth trend, projected utilisation, and
  **days-to-capacity**.
- **LLM indicator narration.** Turn any decomposition table into a concise leadership-style
  governance briefing.

## Why this design

- **Provider-agnostic LLM** (`src/llm.py`): set `ANTHROPIC_API_KEY` *or* `OPENAI_API_KEY` and
  the app auto-detects the vendor — the rest of the code never cares which.
- **Safety first**: the model only ever proposes SQL; execution goes through a read-only
  allow-list (`src/db.is_safe_select`), so the LLM can't mutate data.
- **Maths in code, narration in the model**: all numbers come from deterministic pandas/numpy;
  the LLM only translates questions and explains results — so figures are trustworthy.

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env          # then paste ONE key (Anthropic or OpenAI) into .env
python data/generate_data.py  # creates data/netops.db (synthetic CDN dataset)
streamlit run app/streamlit_app.py
```

No API key yet? Everything except the **Ask** tab and the LLM-narration button works offline
(the data-mining is pure pandas). The notebook runs offline too.

## Project layout

```
data/generate_data.py     # synthetic CDN telemetry + costs + incidents -> SQLite
src/
  db.py                   # connection, schema introspection, read-only query guard
  llm.py                  # provider-agnostic LLM client (Anthropic / OpenAI)
  text_to_sql.py          # NL question -> SQL -> execute -> summarise
  analysis.py             # anomalies, cost decomposition, capacity forecast, narration
app/streamlit_app.py      # local 4-tab UI
notebooks/netops_analysis.ipynb   # data-mining walkthrough
```

## The data (synthetic)

12 PoPs (APAC/AMER/EMEA), a sparse backbone of inter-PoP links (BGP/OSPF/IS-IS, various transit
providers), ~6 months of daily quality telemetry, monthly cost/billing derived from traffic,
and injected incidents that visibly degrade quality. All synthetic but internally consistent
(cost tracks traffic, quality drops during incidents), so analysis and Q&A return sensible
results. No real or proprietary data.

## Limitations

- **The data is synthetic, and that flatters the anomaly detection.** Incidents were injected
  by a generator, so the anomalies are cleaner and better separated than real telemetry, where
  a rebuffer spike and a measurement artefact look alike. Robust-z on real CDN data would need
  a far more careful threshold.
- **The text-to-SQL path has been verified against a mock model, not a paid API.** The guard,
  the schema prompt and the execution path all run and are tested; what has not been measured
  is how often a real model writes the wrong query for a right-sounding answer. That number is
  the one that would decide whether this is usable, and I do not have it yet.
- The read-only guard is an allow-list over single `SELECT` statements. It stops writes and
  stacked statements. It is not a substitute for a database user with read-only grants.
- Capacity forecasting is a linear trend on ~6 months of daily data. It will not see a step
  change coming, which is the case that matters most.
