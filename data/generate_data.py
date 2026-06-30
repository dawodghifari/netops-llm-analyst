"""
Generate a realistic (synthetic) CDN / edge-network dataset into a SQLite database.

Models the kind of data an AI/LLM Data Application team works with: points of
presence (PoPs), inter-PoP links, daily quality telemetry, monthly cost/billing,
and incidents. All values are synthetic but internally consistent (cost tracks
traffic, quality degrades during incidents, etc.) so analysis and LLM Q&A produce
sensible results.

Run:
    python data/generate_data.py            # -> data/netops.db
    python data/generate_data.py --days 365 # longer history

Deterministic: a fixed seed makes the dataset reproducible.
"""

import argparse
import os
import sqlite3
from datetime import date, timedelta

import numpy as np

# Override with NETOPS_DB to write elsewhere (e.g. a fast local disk).
DB_PATH = os.environ.get("NETOPS_DB", os.path.join(os.path.dirname(__file__), "netops.db"))

# 12 PoPs across APAC/global, loosely mirroring a streaming CDN footprint.
POPS = [
    ("SYD", "Sydney", "APAC", "AU"),
    ("MEL", "Melbourne", "APAC", "AU"),
    ("SIN", "Singapore", "APAC", "SG"),
    ("TYO", "Tokyo", "APAC", "JP"),
    ("HKG", "Hong Kong", "APAC", "HK"),
    ("LAX", "Los Angeles", "AMER", "US"),
    ("IAD", "Ashburn", "AMER", "US"),
    ("GRU", "Sao Paulo", "AMER", "BR"),
    ("LON", "London", "EMEA", "GB"),
    ("FRA", "Frankfurt", "EMEA", "DE"),
    ("PAR", "Paris", "EMEA", "FR"),
    ("DXB", "Dubai", "EMEA", "AE"),
]

ROUTING_PROTOCOLS = ["BGP", "OSPF", "IS-IS"]
TRANSIT_PROVIDERS = ["Arelion", "Lumen", "NTT", "Telstra", "Cogent"]
INCIDENT_TYPES = ["fibre-cut", "congestion", "hardware-fault", "ddos", "power"]


def _create_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        DROP TABLE IF EXISTS pops;
        DROP TABLE IF EXISTS links;
        DROP TABLE IF EXISTS telemetry;
        DROP TABLE IF EXISTS costs;
        DROP TABLE IF EXISTS incidents;

        CREATE TABLE pops (
            pop_id   TEXT PRIMARY KEY,
            city     TEXT NOT NULL,
            region   TEXT NOT NULL,   -- APAC / AMER / EMEA
            country  TEXT NOT NULL,
            capacity_gbps REAL NOT NULL  -- provisioned egress capacity
        );

        CREATE TABLE links (
            link_id        TEXT PRIMARY KEY,
            src_pop        TEXT NOT NULL REFERENCES pops(pop_id),
            dst_pop        TEXT NOT NULL REFERENCES pops(pop_id),
            capacity_gbps  REAL NOT NULL,
            protocol       TEXT NOT NULL,   -- BGP / OSPF / IS-IS
            transit_provider TEXT NOT NULL
        );

        CREATE TABLE telemetry (
            date            TEXT NOT NULL,    -- ISO date (daily)
            pop_id          TEXT NOT NULL REFERENCES pops(pop_id),
            peak_traffic_gbps REAL NOT NULL,
            avg_latency_ms  REAL NOT NULL,
            packet_loss_pct REAL NOT NULL,
            rebuffer_ratio_pct REAL NOT NULL, -- streaming quality KPI
            availability_pct REAL NOT NULL
        );

        CREATE TABLE costs (
            month               TEXT NOT NULL,  -- YYYY-MM
            pop_id              TEXT NOT NULL REFERENCES pops(pop_id),
            bandwidth_cost_usd  REAL NOT NULL,
            transit_cost_usd    REAL NOT NULL,
            hardware_amort_usd  REAL NOT NULL,
            total_bill_usd      REAL NOT NULL
        );

        CREATE TABLE incidents (
            date         TEXT NOT NULL,
            pop_id       TEXT NOT NULL REFERENCES pops(pop_id),
            type         TEXT NOT NULL,
            duration_min INTEGER NOT NULL,
            traffic_impact_pct REAL NOT NULL
        );
        """
    )


def generate(days: int = 180, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    con = sqlite3.connect(DB_PATH)
    _create_schema(con)

    # ---- PoPs -------------------------------------------------------------
    pop_capacity = {}
    pop_rows = []
    for pid, city, region, country in POPS:
        cap = float(rng.integers(200, 900))  # Gbps provisioned
        pop_capacity[pid] = cap
        pop_rows.append((pid, city, region, country, cap))
    con.executemany("INSERT INTO pops VALUES (?,?,?,?,?)", pop_rows)

    # ---- Links (a sparse backbone mesh) -----------------------------------
    link_rows = []
    pop_ids = [p[0] for p in POPS]
    for i, src in enumerate(pop_ids):
        for dst in pop_ids[i + 1 :]:
            if rng.random() < 0.45:  # ~45% of pairs are directly linked
                link_rows.append(
                    (
                        f"{src}-{dst}",
                        src,
                        dst,
                        float(rng.integers(100, 400)),
                        rng.choice(ROUTING_PROTOCOLS),
                        rng.choice(TRANSIT_PROVIDERS),
                    )
                )
    con.executemany("INSERT INTO links VALUES (?,?,?,?,?,?)", link_rows)

    # ---- Daily telemetry --------------------------------------------------
    start = date.today() - timedelta(days=days)
    # Per-PoP baseline traffic as a fraction of capacity, with its own growth rate.
    base_load = {pid: rng.uniform(0.45, 0.72) for pid in pop_ids}
    growth = {pid: rng.uniform(0.0003, 0.0018) for pid in pop_ids}  # daily compounding

    # Pre-roll incidents so telemetry can reflect them.
    incident_rows = []
    incidents_by_day_pop = {}
    n_incidents = max(8, days // 12)
    for _ in range(n_incidents):
        d = start + timedelta(days=int(rng.integers(0, days)))
        pid = rng.choice(pop_ids)
        itype = rng.choice(INCIDENT_TYPES)
        dur = int(rng.integers(20, 600))
        impact = float(round(rng.uniform(5, 60), 1))
        incident_rows.append((d.isoformat(), pid, itype, dur, impact))
        incidents_by_day_pop[(d.isoformat(), pid)] = (impact, itype)

    telem_rows = []
    for n in range(days):
        d = (start + timedelta(days=n)).isoformat()
        # Weekly seasonality: weekends carry more streaming traffic.
        dow = (start + timedelta(days=n)).weekday()
        weekend_boost = 1.18 if dow >= 5 else 1.0
        for pid in pop_ids:
            cap = pop_capacity[pid]
            load_frac = base_load[pid] * (1 + growth[pid]) ** n * weekend_boost
            load_frac = min(load_frac, 1.05)
            noise = rng.normal(0, 0.03)
            peak = max(1.0, cap * (load_frac + noise))

            # Quality degrades as utilisation approaches capacity.
            util = peak / cap
            latency = 18 + 60 * max(0, util - 0.7) + rng.normal(0, 2)
            ploss = max(0.0, 0.05 + 1.8 * max(0, util - 0.85) + rng.normal(0, 0.03))
            rebuffer = max(0.0, 0.2 + 2.5 * max(0, util - 0.8) + rng.normal(0, 0.05))
            avail = 99.99

            inc = incidents_by_day_pop.get((d, pid))
            if inc:
                impact, _ = inc
                peak *= 1 - impact / 200.0
                latency += impact * 1.5
                ploss += impact / 20.0
                rebuffer += impact / 12.0
                avail -= impact / 30.0

            telem_rows.append(
                (
                    d,
                    pid,
                    round(peak, 1),
                    round(latency, 2),
                    round(ploss, 3),
                    round(rebuffer, 3),
                    round(min(avail, 100.0), 4),
                )
            )
    con.executemany("INSERT INTO telemetry VALUES (?,?,?,?,?,?,?)", telem_rows)
    con.executemany("INSERT INTO incidents VALUES (?,?,?,?,?)", incident_rows)

    # ---- Monthly costs (derived from traffic) -----------------------------
    # Aggregate telemetry to month/PoP, then price it.
    cur = con.execute(
        """
        SELECT substr(date,1,7) AS month, pop_id, AVG(peak_traffic_gbps) AS avg_peak
        FROM telemetry GROUP BY month, pop_id
        """
    )
    cost_rows = []
    for month, pid, avg_peak in cur.fetchall():
        # $/Mbps blended transit pricing varies a little by PoP/region.
        price_per_mbps = rng.uniform(0.10, 0.35)
        bandwidth_cost = avg_peak * 1000 * price_per_mbps  # Gbps -> Mbps
        transit_cost = bandwidth_cost * rng.uniform(0.25, 0.5)
        hardware_amort = pop_capacity[pid] * rng.uniform(40, 80)  # monthly amortisation
        total = bandwidth_cost + transit_cost + hardware_amort
        cost_rows.append(
            (
                month,
                pid,
                round(bandwidth_cost, 2),
                round(transit_cost, 2),
                round(hardware_amort, 2),
                round(total, 2),
            )
        )
    con.executemany("INSERT INTO costs VALUES (?,?,?,?,?,?)", cost_rows)

    con.commit()

    # Summary
    counts = {
        t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        for t in ["pops", "links", "telemetry", "costs", "incidents"]
    }
    con.close()
    print(f"Wrote {DB_PATH}")
    for t, c in counts.items():
        print(f"  {t:10s}: {c:>6,} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate the NetOps SQLite dataset.")
    p.add_argument("--days", type=int, default=180, help="days of daily telemetry (default 180)")
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    generate(days=args.days, seed=args.seed)
