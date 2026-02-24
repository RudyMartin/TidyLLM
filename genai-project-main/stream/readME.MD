Done. I added a full **Markdown docs pack** to your canvas with copy-paste files for your team:

* README, QUICKSTART, ARCHITECTURE (with mermaid), STANDARDS\_RULES
* RUNNER\_APP\_GUIDE, REPORTING\_APP\_GUIDE
* DATA\_MODEL, METRICS\_AND\_EVENTS
* SECURITY\_COMPLIANCE, COST\_PERFORMANCE, OPERATIONS\_SRE
* DEMO\_SCRIPT, FAQ, GLOSSARY, CONTRIBUTING

If you want any of these trimmed to a one-pager or localized, say the word and I’ll generate variants.

Below are single‑file masters for:

app/streamlit_app.py (runner UI — QA Validation Review only)

reporting_app.py (read‑only analytics)

infra/schema.sql (Stage‑2 full DB creation)

They assume the scaffold pieces we added exist (app/config.py, storage/s3_bus.py, workers/checker.py S3‑native, workers/standards_loader.py, app/db_utils.py, app/metrics.py). All I/O is S3‑native; the only “local” touch is the two uploads’ in‑memory buffers.

# 🎛️ Hierarchical LLM Demo — Markdown Assembly-Line (MCP-friendly)

This updated scaffold adds a **“Choose Your Name” dropdown** at the top of the Streamlit app. Each teammate selects their name, and all outputs go into their own S3 folder automatically (no collisions).

---

## 🔑 Change in `app/config.py`

Add a helper to override the run prefix dynamically:

```python
from dataclasses import dataclass
import os

@dataclass
class Settings:
    bucket: str = os.getenv("S3_BUCKET", "sagemaker-us-east-1-188494237500")
    run_prefix: str = os.getenv("RUN_PREFIX", "demo_run")  # Will be combined with username
    embed_model: str = os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1024"))
    pg_dsn: str = os.getenv("PG_DSN", "postgresql://user:pass@localhost:5432/mrm")
    use_bedrock: bool = os.getenv("USE_BEDROCK", "false").lower() == "true"
    max_gist_tokens: int = int(os.getenv("MAX_GIST_TOKENS", "120"))

settings = Settings()

# Function to update prefix per teammate

def get_run_prefix(username: str):
    base = settings.run_prefix.rstrip("/")
    return f"{base}/{username.strip().lower()}"
```

---

## 🔑 Change in `app/streamlit_app.py`

At the top of the app, add the **dropdown**:

```python
import streamlit as st
from app.config import settings, get_run_prefix
from app.demo import step_ingest, step_summarize, step_check, step_report
from storage.s3_bus import S3Bus

st.set_page_config(page_title="HierLLM Markdown Assembly-Line", layout="wide")

st.title("Hierarchical LLM Demo — Markdown Assembly-Line")
st.caption("Each teammate has their own folder • Select your name before starting")

# Dropdown for teammate name
user = st.selectbox(
    "Choose your name:",
    ["alex", "maria", "ravi"],
    index=0
)

# Build S3 bus using user-specific prefix
bus = S3Bus(settings.bucket, get_run_prefix(user) + "/WF-2025-042")

st.info(f"You are working in folder: {get_run_prefix(user)}/WF-2025-042")
```

---

## ✅ Effect

* Teammates select their **name** once.
* All outputs (chunks, summaries, findings, reports) are saved to `demo_run/<name>/WF-2025-042/…`.
* No overwriting each other’s runs.
* Clear visual confirmation with `st.info()` box showing which folder they’re writing to.

---

## Example S3 Layout After Runs

```
s3://sagemaker-us-east-1-188494237500/demo_run/alex/WF-2025-042/…
s3://sagemaker-us-east-1-188494237500/demo_run/maria/WF-2025-042/…
s3://sagemaker-us-east-1-188494237500/demo_run/ravi/WF-2025-042/…
```

---

This way, teammates just **pick their name** and click the four buttons — no risk of messing up each other’s test. ✅

---

## 🧩 Org → Team → Process (full hierarchy wiring)

### 1) DB schema updates (`infra/schema.sql` — append)

```sql
ALTER TABLE review_runs
  ADD COLUMN IF NOT EXISTS org TEXT,
  ADD COLUMN IF NOT EXISTS team TEXT,
  ADD COLUMN IF NOT EXISTS process TEXT;

CREATE INDEX IF NOT EXISTS review_runs_org ON review_runs (org);
CREATE INDEX IF NOT EXISTS review_runs_org_team ON review_runs (org, team);
CREATE INDEX IF NOT EXISTS review_runs_org_team_proc_date ON review_runs (org, team, process, review_date);
```

### 2) Streamlit selectors (add near the top of `app/streamlit_app.py`)

```python
# Org → Team → Process hierarchy
ORG_MAP = {
    "QA-Americas": {
        "teams": ["QA-East", "QA-West"],
        "processes": ["QAValidationReview"],
    },
    "QA-EMEA": {
        "teams": ["QA-UK", "QA-DE", "QA-FR"],
        "processes": ["QAValidationReview"],
    },
    "QA-APAC": {
        "teams": ["QA-SG", "QA-AU"],
        "processes": ["QAValidationReview"],
    },
}
org = st.sidebar.selectbox("Org", list(ORG_MAP.keys()), index=0)
team = st.sidebar.selectbox("Team", ORG_MAP[org]["teams"], index=0)
process = st.sidebar.selectbox("Process", ORG_MAP[org]["processes"], index=0)
st.sidebar.caption("These tags will be saved with each run and used for filters/exports.")

# convenient filter vars
org_where, team_where, proc_where = org, team, process
```

### 3) Save Run includes org/team/process (`app/db_utils.py`)

```python
# extend save_run signature + INSERT

def save_run(conn, reviewer: str, doc_id: str, review_date, ruleset_hash: str,
             report_s3_uri: str, finding_mds: list[str], metrics: dict | None = None,
             org: str | None = None, team: str | None = None, process: str | None = None) -> int:
    finding_count = len(finding_mds)
    tokens_in  = (metrics or {}).get("tokens_input")
    tokens_out = (metrics or {}).get("tokens_output")
    runtime    = (metrics or {}).get("runtime_sec")
    chunk_cnt  = (metrics or {}).get("chunk_count")
    model_calls= (metrics or {}).get("model_calls", {})
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO review_runs
              (reviewer, org, team, process, doc_id, review_date, ruleset_hash, report_s3_uri,
               runtime_sec, chunk_count, finding_count, tokens_input, tokens_output, model_calls)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (reviewer, org, team, process, doc_id, review_date, ruleset_hash, report_s3_uri,
             runtime, chunk_cnt, finding_count, tokens_in, tokens_out, json.dumps(model_calls))
        )
        run_id = cur.fetchone()[0]
        # ... insert review_findings as before ...
    conn.commit()
    return run_id
```

### 4) Save button passes org/team/process (`app/streamlit_app.py`)

```python
if st.button("💾 Save Run to DB"):
    finding_keys = st.session_state.get("finding_keys", [])
    finding_mds  = [bus.read_text(k) for k in finding_keys]
    metrics      = st.session_state.get("metrics_snapshot", {})
    try:
        conn = psycopg2.connect(settings.pg_dsn)
        run_id = save_run(
            conn,
            reviewer=user,
            doc_id="WF-2025-042",
            review_date=review_date,
            ruleset_hash=st.session_state.get("ruleset_hash"),
            report_s3_uri=f"s3://{settings.bucket}/{st.session_state['report_key']}",
            finding_mds=finding_mds,
            metrics=metrics,
            org=org, team=team, process=process,
        )
        conn.close()
        st.success(f"Saved review run #{run_id} ✅")
        st.balloons()
    except Exception as e:
        st.error(f"DB save failed: {e}")
```

### 5) Boss Panel + Burger counter honor filters

```python
# burger badge
def get_burgers_total(org=None, team=None, process=None):
    try:
        conn = psycopg2.connect(settings.pg_dsn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(SUM(finding_count),0)
            FROM review_runs
            WHERE (%s IS NULL OR org=%s)
              AND (%s IS NULL OR team=%s)
              AND (%s IS NULL OR process=%s)
            """,
            (org, org, team, team, process, process)
        )
        total = cur.fetchone()[0]
        conn.close()
        return total or 0
    except Exception:
        return 0

st.metric("🍔 Findings Checked (Burgers Served)", f"{get_burgers_total(org_where, team_where, proc_where):,}")

# last 5 runs (filtered)
conn = psycopg2.connect(settings.pg_dsn)
cur = conn.cursor()
cur.execute(
    """
    SELECT id, reviewer, org, team, process, review_date, ruleset_hash, finding_count, runtime_sec
    FROM review_runs
    WHERE (%s IS NULL OR org=%s)
      AND (%s IS NULL OR team=%s)
      AND (%s IS NULL OR process=%s)
    ORDER BY id DESC LIMIT 5
    """,
    (org_where, org_where, team_where, team_where, proc_where, proc_where)
)
rows = cur.fetchall(); conn.close()
# render rows as before

# growth chart (filtered)
conn = psycopg2.connect(settings.pg_dsn)
cur = conn.cursor()
cur.execute(
    """
    SELECT date(review_date), SUM(finding_count)
    FROM review_runs
    WHERE (%s IS NULL OR org=%s)
      AND (%s IS NULL OR team=%s)
      AND (%s IS NULL OR process=%s)
    GROUP BY 1 ORDER BY 1
    """,
    (org_where, org_where, team_where, team_where, proc_where, proc_where)
)
data = cur.fetchall(); conn.close()
# chart render as before

# export (filtered)
if st.button("Export CSV/PDF to S3"):
    try:
        conn = psycopg2.connect(settings.pg_dsn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT id, reviewer, review_date, ruleset_hash, finding_count, runtime_sec,
                   tokens_input, tokens_output, model_calls
            FROM review_runs
            WHERE review_date >= %s AND review_date <= %s
              AND (%s IS NULL OR org=%s)
              AND (%s IS NULL OR team=%s)
              AND (%s IS NULL OR process=%s)
            ORDER BY id DESC
            """,
            (start_date, end_date, org_where, org_where, team_where, team_where, proc_where, proc_where)
        )
        rows = cur.fetchall(); conn.close()
        if not rows:
            st.warning("No runs for that range/filters.")
        else:
            base_prefix = f"{get_run_prefix(user)}/audit/exports"
            csv_url, pdf_url = export_audit_to_s3(settings.bucket, base_prefix, rows)
            st.success(f"Exported {len(rows)} rows.")
            st.write(f"▶️ CSV: {csv_url}")
            st.write(f"▶️ PDF: {pdf_url}" if pdf_url else "PDF export disabled (install reportlab).")
    except Exception as e:
        st.error(f"Export failed: {e}")
```

> All S3 I/O remains S3‑native (no local files beyond the two uploads). Org/Team/Process metadata flows through **save**, **filters**, **metrics**, and **exports**.

---

## 17) Stage‑2 SQL Migrations (apply first)

> **Run these on your Postgres**. They extend `review_runs`, add event tables, and prep fast filters for reporting.

```sql
-- 17.1 Extend review_runs with Stage‑2 analytics fields
ALTER TABLE review_runs
  ADD COLUMN IF NOT EXISTS pipeline_version TEXT,
  ADD COLUMN IF NOT EXISTS doc_pages INT,
  ADD COLUMN IF NOT EXISTS doc_bytes BIGINT,
  ADD COLUMN IF NOT EXISTS input_hash TEXT,
  ADD COLUMN IF NOT EXISTS prompt_hash TEXT,
  ADD COLUMN IF NOT EXISTS model_primary TEXT,
  ADD COLUMN IF NOT EXISTS latency_p50_ms INT,
  ADD COLUMN IF NOT EXISTS latency_p95_ms INT,
  ADD COLUMN IF NOT EXISTS cost_usd NUMERIC(10,4),
  ADD COLUMN IF NOT EXISTS error_count INT,
  ADD COLUMN IF NOT EXISTS timeout_count INT,
  ADD COLUMN IF NOT EXISTS retry_count INT,
  ADD COLUMN IF NOT EXISTS evidence_count INT,
  ADD COLUMN IF NOT EXISTS severity_counts JSONB,
  ADD COLUMN IF NOT EXISTS consensus_status TEXT;

-- helpful search indexes
CREATE INDEX IF NOT EXISTS review_runs_date ON review_runs(review_date);
CREATE INDEX IF NOT EXISTS review_runs_model ON review_runs(model_primary);
CREATE INDEX IF NOT EXISTS review_runs_ruleset ON review_runs(ruleset_hash);

-- 17.2 Org→Team→Process (if not applied yet)
ALTER TABLE review_runs
  ADD COLUMN IF NOT EXISTS org TEXT,
  ADD COLUMN IF NOT EXISTS team TEXT,
  ADD COLUMN IF NOT EXISTS process TEXT;
CREATE INDEX IF NOT EXISTS review_runs_org ON review_runs(org);
CREATE INDEX IF NOT EXISTS review_runs_org_team ON review_runs(org,team);
CREATE INDEX IF NOT EXISTS review_runs_org_team_proc_date ON review_runs(org,team,process,review_date);

-- 17.3 Events tables (append-only raw + daily rollups)
CREATE TABLE IF NOT EXISTS events_raw (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_id TEXT,
  org TEXT,
  team TEXT,
  process TEXT,
  event TEXT NOT NULL,    -- e.g., auth_login, session_start, page_view, config_change,
                           -- validation_error, app_error, scoring_update, etc.
  payload JSONB           -- free-form, schema in docs; avoid PII
);
CREATE INDEX IF NOT EXISTS events_raw_ts ON events_raw(ts);
CREATE INDEX IF NOT EXISTS events_raw_keys ON events_raw(org,team,process,event);

-- daily materialized rollups (simple table + upsert proc)
CREATE TABLE IF NOT EXISTS events_daily (
  day DATE NOT NULL,
  org TEXT,
  team TEXT,
  process TEXT,
  metric TEXT NOT NULL,   -- e.g., active_users, page_views, runs, errors
  value BIGINT NOT NULL,
  PRIMARY KEY(day, org, team, process, metric)
);

-- convenience view for “🍔 burgers served”
CREATE OR REPLACE VIEW burgers_served AS
SELECT COALESCE(SUM(finding_count),0) AS total_burgers FROM review_runs;

-- 17.4 Rollup function (24h window by default)
CREATE OR REPLACE FUNCTION rollup_events_daily(p_start DATE, p_end DATE)
RETURNS VOID AS $$
BEGIN
  -- active users (distinct user_id with any event)
  INSERT INTO events_daily(day,org,team,process,metric,value)
  SELECT d::date, org, team, process, 'active_users', COUNT(DISTINCT user_id)
  FROM events_raw, LATERAL (SELECT date(ts) d) d
  WHERE date(ts) BETWEEN p_start AND p_end
  GROUP BY d::date, org, team, process
  ON CONFLICT (day,org,team,process,metric) DO UPDATE SET value=EXCLUDED.value;

  -- page views
  INSERT INTO events_daily(day,org,team,process,metric,value)
  SELECT d::date, org, team, process, 'page_views', COUNT(*)
  FROM events_raw, LATERAL (SELECT date(ts) d) d
  WHERE event='page_view' AND date(ts) BETWEEN p_start AND p_end
  GROUP BY d::date, org, team, process
  ON CONFLICT (day,org,team,process,metric) DO UPDATE SET value=EXCLUDED.value;

  -- runs recorded (from review_runs)
  INSERT INTO events_daily(day,org,team,process,metric,value)
  SELECT review_date, org, team, process, 'runs', COUNT(*)
  FROM review_runs
  WHERE review_date BETWEEN p_start AND p_end
  GROUP BY review_date, org, team, process
  ON CONFLICT (day,org,team,process,metric) DO UPDATE SET value=EXCLUDED.value;

  -- burgers served per day (sum findings)
  INSERT INTO events_daily(day,org,team,process,metric,value)
  SELECT review_date, org, team, process, 'burgers', COALESCE(SUM(finding_count),0)
  FROM review_runs
  WHERE review_date BETWEEN p_start AND p_end
  GROUP BY review_date, org, team, process
  ON CONFLICT (day,org,team,process,metric) DO UPDATE SET value=EXCLUDED.value;
END; $$ LANGUAGE plpgsql;
```

---

## 18) `app/metrics.py` — tiny metrics writer (Stage‑2 fields)

> In‑memory aggregator you call from workers and at the end snapshot to DB fields.

```python
# app/metrics.py
import time, statistics
from collections import defaultdict

class Metrics:
    def __init__(self, pipeline_version: str = "v1"):
        self.pipeline_version = pipeline_version
        self.t0 = time.time()
        self.tokens_in = 0
        self.tokens_out = 0
        self.model_calls = defaultdict(int)
        self.latencies = []  # ms
        self.errors = 0
        self.timeouts = 0
        self.retries = 0
        self.doc_pages = None
        self.doc_bytes = None
        self.input_hash = None
        self.prompt_hash = None
        self.model_primary = None
        self.evidence_count = 0
        self.severity_counts = defaultdict(int)  # {High: n, Medium: n, Low: n}
        self.cost_usd = 0.0

    def add_tokens(self, model: str, in_t: int, out_t: int, cost_usd: float = 0.0):
        self.tokens_in += int(in_t or 0)
        self.tokens_out += int(out_t or 0)
        self.model_calls[model] += 1
        self.cost_usd += float(cost_usd or 0.0)
        if not self.model_primary:
            self.model_primary = model

    def add_latency_ms(self, ms: int):
        self.latencies.append(int(ms))

    def inc_error(self):
        self.errors += 1

    def inc_timeout(self):
        self.timeouts += 1

    def inc_retry(self):
        self.retries += 1

    def set_doc_stats(self, pages: int | None, bytes_len: int | None):
        self.doc_pages = pages; self.doc_bytes = bytes_len

    def set_hashes(self, input_hash: str | None, prompt_hash: str | None):
        self.input_hash = input_hash; self.prompt_hash = prompt_hash

    def add_evidence(self, n: int):
        self.evidence_count += int(n or 0)

    def add_severity(self, level: str):
        if level:
            self.severity_counts[level] += 1

    def snapshot(self):
        rt = int(time.time() - self.t0)
        p50 = int(statistics.median(self.latencies)) if self.latencies else None
        p95 = int(sorted(self.latencies)[max(0,int(0.95*len(self.latencies))-1)]) if self.latencies else None
        return {
            "pipeline_version": self.pipeline_version,
            "runtime_sec": rt,
            "tokens_input": self.tokens_in,
            "tokens_output": self.tokens_out,
            "model_calls": dict(self.model_calls),
            "latency_p50_ms": p50,
            "latency_p95_ms": p95,
            "error_count": self.errors,
            "timeout_count": self.timeouts,
            "retry_count": self.retries,
            "doc_pages": self.doc_pages,
            "doc_bytes": self.doc_bytes,
            "input_hash": self.input_hash,
            "prompt_hash": self.prompt_hash,
            "model_primary": self.model_primary,
            "evidence_count": self.evidence_count,
            "severity_counts": dict(self.severity_counts),
            "cost_usd": round(self.cost_usd, 4),
        }
```

> In your pipeline, create one `metrics = Metrics(pipeline_version="v1.2")`, pass it to workers, then at save time: `metrics_snapshot = metrics.snapshot()`.

---

## 19) `app/db_utils.py` — save Stage‑2 fields + events API

```python
# app/db_utils.py (additions)
import json, yaml

def save_run(conn, reviewer: str, doc_id: str, review_date, ruleset_hash: str,
             report_s3_uri: str, finding_mds: list[str], metrics: dict | None = None,
             org: str | None = None, team: str | None = None, process: str | None = None) -> int:
    m = metrics or {}
    finding_count = len(finding_mds)
    severity_counts = {"High":0, "Medium":0, "Low":0}
    # gently parse severity from findings frontmatter
    for md in finding_mds:
        parts = md.split('---')
        if len(parts) >= 3:
            front = yaml.safe_load(parts[1]) or {}
            lvl = front.get("severity")
            if lvl in severity_counts:
                severity_counts[lvl] += 1
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO review_runs
              (reviewer, org, team, process, doc_id, review_date, ruleset_hash, report_s3_uri,
               runtime_sec, chunk_count, finding_count, tokens_input, tokens_output, model_calls,
               pipeline_version, doc_pages, doc_bytes, input_hash, prompt_hash, model_primary,
               latency_p50_ms, latency_p95_ms, cost_usd, error_count, timeout_count, retry_count,
               evidence_count, severity_counts, consensus_status)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,
                    %s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            RETURNING id
            """,
            (
              reviewer, org, team, process, doc_id, review_date, ruleset_hash, report_s3_uri,
              m.get("runtime_sec"), m.get("chunk_count"), finding_count,
              m.get("tokens_input"), m.get("tokens_output"), json.dumps(m.get("model_calls", {})),
              m.get("pipeline_version"), m.get("doc_pages"), m.get("doc_bytes"),
              m.get("input_hash"), m.get("prompt_hash"), m.get("model_primary"),
              m.get("latency_p50_ms"), m.get("latency_p95_ms"), m.get("cost_usd"),
              m.get("error_count"), m.get("timeout_count"), m.get("retry_count"),
              m.get("evidence_count"), json.dumps(severity_counts), m.get("consensus_status"),
            )
        )
        run_id = cur.fetchone()[0]
        # upsert findings as before (omitted here for brevity)
    conn.commit()
    return run_id

# simple events ingestion

def insert_event(conn, user_id: str, org: str | None, team: str | None, process: str | None,
                 event: str, payload: dict | None = None):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO events_raw (user_id, org, team, process, event, payload)
            VALUES (%s,%s,%s,%s,%s,%s::jsonb)
            """,
            (user_id, org, team, process, event, json.dumps(payload or {}))
        )
    conn.commit()

```

---

## 20) `app/events.py` — tiny client helpers

```python
# app/events.py
import psycopg2
from app.config import settings
from app.db_utils import insert_event

# Call these from UI moments: login, page views, run started/finished, errors, etc.

def log_page_view(user_id: str, org: str, team: str, process: str, page: str):
    conn = psycopg2.connect(settings.pg_dsn)
    insert_event(conn, user_id, org, team, process, "page_view", {"page": page})
    conn.close()

def log_run_started(user_id: str, org: str, team: str, process: str, doc_id: str):
    conn = psycopg2.connect(settings.pg_dsn)
    insert_event(conn, user_id, org, team, process, "run_started", {"doc_id": doc_id})
    conn.close()

def log_run_finished(user_id: str, org: str, team: str, process: str, doc_id: str, ok: bool, burgers: int):
    conn = psycopg2.connect(settings.pg_dsn)
    insert_event(conn, user_id, org, team, process, "run_finished", {"doc_id": doc_id, "ok": ok, "burgers": burgers})
    conn.close()

def log_error(user_id: str, org: str, team: str, process: str, kind: str, message: str):
    conn = psycopg2.connect(settings.pg_dsn)
    insert_event(conn, user_id, org, team, process, kind, {"message": message})  # kind=validation_error/app_error
    conn.close()
```

---

## 21) `reporting_app.py` — read‑only analytics app (Streamlit)

> **Purpose:** all “Sneaky Boss” features live here: KPIs, audit/export, standards diff, utilization, reliability, engagement. **S3‑native**, **DB‑backed**, **no local files**.

```python
# reporting_app.py
import streamlit as st
from datetime import date, timedelta
import psycopg2, pandas as pd, altair as alt, boto3
from app.config import settings, get_run_prefix
from app.audit_utils import list_rule_hashes, diff_rule_sets
from app.audit_export import export_audit_to_s3

st.set_page_config(page_title="QA Validation Review — Reporting", layout="wide")
st.title("Reporting — QA Validation Review")

# Filters
ORG_MAP = {
    "QA-Americas": {"teams":["QA-East","QA-West"], "processes":["QAValidationReview"]},
    "QA-EMEA": {"teams":["QA-UK","QA-DE","QA-FR"], "processes":["QAValidationReview"]},
    "QA-APAC": {"teams":["QA-SG","QA-AU"], "processes":["QAValidationReview"]},
}
cols = st.columns(5)
org = cols[0].selectbox("Org", list(ORG_MAP.keys()))
team = cols[1].selectbox("Team", ORG_MAP[org]["teams"])
process = cols[2].selectbox("Process", ORG_MAP[org]["processes"])  # fixed for demo
start_date = cols[3].date_input("Start", value=(date.today()-timedelta(days=30)))
end_date   = cols[4].date_input("End", value=date.today())

# KPI tiles
def fetch_scalar(q, params):
    try:
        conn = psycopg2.connect(settings.pg_dsn); cur = conn.cursor()
        cur.execute(q, params); x = cur.fetchone()[0]; conn.close(); return x or 0
    except Exception:
        return 0

k_burgers = fetch_scalar("""
    SELECT COALESCE(SUM(finding_count),0) FROM review_runs
    WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
""", (start_date, end_date, org, team, process))

k_runs = fetch_scalar("""
    SELECT COUNT(*) FROM review_runs
    WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
""", (start_date, end_date, org, team, process))

k_cost = fetch_scalar("""
    SELECT COALESCE(SUM(cost_usd),0) FROM review_runs
    WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
""", (start_date, end_date, org, team, process))

k_err = fetch_scalar("""
    SELECT COALESCE(SUM(error_count),0) FROM review_runs
    WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
""", (start_date, end_date, org, team, process))

c1,c2,c3,c4 = st.columns(4)
c1.metric("🍔 Findings", f"{k_burgers:,}")
c2.metric("▶️ Runs", f"{k_runs:,}")
c3.metric("💸 Cost ($)", f"{k_cost:,.2f}")
c4.metric("❗Errors", f"{k_err:,}")

st.divider()

# Tabs
Overview, Audit, StandardsDiff, UtilCost, Reliability, Engagement = st.tabs([
    "Overview","Audit Trail","Standards Diff","Utilization & Cost","Reliability","Engagement"])

with Overview:
    try:
        conn = psycopg2.connect(settings.pg_dsn); cur = conn.cursor()
        cur.execute("""
            SELECT date(review_date) d,
                   SUM(finding_count) burgers,
                   COUNT(*) runs,
                   COALESCE(SUM(tokens_input+tokens_output),0) toks,
                   COALESCE(SUM(cost_usd),0) cost
            FROM review_runs
            WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
            GROUP BY 1 ORDER BY 1
        """, (start_date, end_date, org, team, process))
        rows = cur.fetchall(); conn.close()
        if rows:
            df = pd.DataFrame(rows, columns=["date","burgers","runs","tokens","cost"])
            st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="date:T", y="burgers:Q"), use_container_width=True)
            st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="date:T", y="runs:Q"), use_container_width=True)
    except Exception as e:
        st.warning(f"Overview error: {e}")

with Audit:
    cols = st.columns(3)
    reviewer = cols[0].text_input("Reviewer contains", value="")
    model    = cols[1].text_input("Model contains", value="")
    limit    = cols[2].number_input("Limit", 1, 5000, 200)
    try:
        conn = psycopg2.connect(settings.pg_dsn); cur = conn.cursor()
        cur.execute("""
            SELECT id, review_date, reviewer, model_primary, finding_count, runtime_sec,
                   ruleset_hash, report_s3_uri
            FROM review_runs
            WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
              AND (%s='' OR reviewer ILIKE '%'||%s||'%')
              AND (%s='' OR model_primary ILIKE '%'||%s||'%')
            ORDER BY id DESC LIMIT %s
        """, (start_date,end_date,org,team,process,reviewer,reviewer,model,model,limit))
        rows = cur.fetchall(); conn.close()
        if rows:
            df = pd.DataFrame(rows, columns=["id","date","reviewer","model","🍔","runtime","hash","report_s3_uri"])
            st.dataframe(df, use_container_width=True)
            if st.button("Export CSV/PDF to S3"):
                base_prefix = f"{get_run_prefix(reviewer or 'report')}/audit/exports"
                from app.audit_export import export_audit_to_s3
                # Reuse query rows but add tokens/model_calls for CSV completeness
                conn = psycopg2.connect(settings.pg_dsn); cur = conn.cursor()
                cur.execute("""
                    SELECT id, reviewer, review_date, ruleset_hash, finding_count, runtime_sec,
                           tokens_input, tokens_output, model_calls
                    FROM review_runs
                    WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
                    ORDER BY id DESC LIMIT %s
                """, (start_date,end_date,org,team,process,limit))
                rows2 = cur.fetchall(); conn.close()
                csv_url, pdf_url = export_audit_to_s3(settings.bucket, base_prefix, rows2)
                st.success("Exported.")
                st.write(f"CSV: {csv_url}")
                if pdf_url: st.write(f"PDF: {pdf_url}")
    except Exception as e:
        st.warning(f"Audit error: {e}")

with StandardsDiff:
    st.caption("Compare a baseline rules prefix with current run's atoms")
    baseline_prefix = st.text_input("Baseline S3 prefix (atoms)", value="standards/baseline/atoms")
    current_prefix  = st.text_input("Current S3 prefix (atoms)",  value=f"demo_run/{team}/WF-2025-042/requirements/atoms")
    if st.button("Compare"):
        try:
            old = list_rule_hashes(settings.bucket, baseline_prefix)
            new = list_rule_hashes(settings.bucket, current_prefix)
            st.json(diff_rule_sets(old, new))
        except Exception as e:
            st.warning(f"Diff error: {e}")

with UtilCost:
    try:
        conn = psycopg2.connect(settings.pg_dsn); cur = conn.cursor()
        cur.execute("""
            SELECT date(review_date) d,
                   SUM(tokens_input+tokens_output) toks,
                   COALESCE(SUM(cost_usd),0) cost,
                   PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY runtime_sec) AS p50_rt,
                   PERCENTILE_DISC(0.95) WITHIN GROUP (ORDER BY runtime_sec) AS p95_rt
            FROM review_runs
            WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
            GROUP BY 1 ORDER BY 1
        """, (start_date,end_date,org,team,process))
        rows = cur.fetchall(); conn.close()
        if rows:
            df = pd.DataFrame(rows, columns=["date","tokens","cost","p50","p95"])
            st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="date:T", y="tokens:Q"), use_container_width=True)
            st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="date:T", y="cost:Q"), use_container_width=True)
    except Exception as e:
        st.warning(f"Util/Cost error: {e}")

with Reliability:
    try:
        conn = psycopg2.connect(settings.pg_dsn); cur = conn.cursor()
        cur.execute("""
            SELECT date(review_date) d,
                   COALESCE(SUM(error_count),0) errors,
                   COALESCE(SUM(timeout_count),0) timeouts,
                   COALESCE(SUM(retry_count),0) retries
            FROM review_runs
            WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
            GROUP BY 1 ORDER BY 1
        """, (start_date,end_date,org,team,process))
        rows = cur.fetchall(); conn.close()
        if rows:
            df = pd.DataFrame(rows, columns=["date","errors","timeouts","retries"])
            st.altair_chart(alt.Chart(df.melt("date")).mark_line(point=True).encode(x="date:T", y="value:Q", color="variable:N"), use_container_width=True)
    except Exception as e:
        st.warning(f"Reliability error: {e}")

with Engagement:
    try:
        conn = psycopg2.connect(settings.pg_dsn); cur = conn.cursor()
        cur.execute("""
            SELECT day, metric, value
            FROM events_daily
            WHERE day BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
            ORDER BY day, metric
        """, (start_date,end_date,org,team,process))
        rows = cur.fetchall(); conn.close()
        if rows:
            df = pd.DataFrame(rows, columns=["day","metric","value"])
            st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="day:T", y="value:Q", color="metric:N"), use_container_width=True)
    except Exception as e:
        st.warning(f"Engagement error: {e}")
```

> **Deployment tip:** Keep this as a separate App Runner/ECS service (`reporting_app.py`) with **read‑only** DB creds and S3 List/Get only.

---

## 22) Move “Sneaky Boss” features out of main app

* Remove (or hide) from `streamlit_app.py`:

  * Audit expander, standards diff, export buttons, growth charts.
* Keep only: name/org/team/process selectors, Review Date, 1→2→3→4 flow, and **Save Run to DB** + burger badge.
* All analytics live in the new `reporting_app.py`.

---

# 📦 Master Deliverables (copy‑paste ready)

Below are **single‑file masters** for:

* `app/streamlit_app.py` (runner UI — QA Validation Review only)
* `reporting_app.py` (read‑only analytics)
* `infra/schema.sql` (Stage‑2 full DB creation)

They assume the scaffold pieces we added exist (`app/config.py`, `storage/s3_bus.py`, `workers/checker.py` S3‑native, `workers/standards_loader.py`, `app/db_utils.py`, `app/metrics.py`). All I/O is **S3‑native**; the only “local” touch is the two uploads’ in‑memory buffers.

---

## MASTER: `app/streamlit_app.py`

```python
import streamlit as st
from datetime import date
import io
import re
from typing import List, Tuple

import psycopg2
import boto3
from pypdf import PdfReader
from docx import Document

from app.config import settings, get_run_prefix
from storage.s3_bus import S3Bus
from workers.standards_loader import load_standards_from_s3
from workers.checker import run_checks_s3
from app.db_utils import save_run
from app.metrics import Metrics

S3 = boto3.client("s3")

st.set_page_config(page_title="QA Validation Review — Runner", layout="wide")
st.title("QA Validation Review — Runner")

# ------------------ Identity & Scope ------------------
user = st.selectbox("Choose your name", ["alex", "maria", "ravi"], index=0)

ORG_MAP = {
    "QA-Americas": {"teams": ["QA-East", "QA-West"], "processes": ["QAValidationReview"]},
    "QA-EMEA": {"teams": ["QA-UK", "QA-DE", "QA-FR"], "processes": ["QAValidationReview"]},
    "QA-APAC": {"teams": ["QA-SG", "QA-AU"], "processes": ["QAValidationReview"]},
}
org = st.sidebar.selectbox("Org", list(ORG_MAP.keys()), index=0)
team = st.sidebar.selectbox("Team", ORG_MAP[org]["teams"], index=0)
process = st.sidebar.selectbox("Process", ORG_MAP[org]["processes"], index=0)
review_date = st.sidebar.date_input("Review date (point-in-time standards)", value=date.today())

# Per-user run folder in S3
DOC_ID = "WF-2025-042"  # demo id (keep stable for comparisons)
doc_prefix = f"{get_run_prefix(user)}/{DOC_ID}"
bus = S3Bus(settings.bucket, doc_prefix)

# ------------------ Burgers badge (filtered) ------------------

def get_burgers_total(org=None, team=None, process=None):
    try:
        conn = psycopg2.connect(settings.pg_dsn)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT COALESCE(SUM(finding_count),0)
            FROM review_runs
            WHERE (%s IS NULL OR org=%s)
              AND (%s IS NULL OR team=%s)
              AND (%s IS NULL OR process=%s)
            """,
            (org, org, team, team, process, process),
        )
        total = cur.fetchone()[0]
        conn.close()
        return total or 0
    except Exception:
        return 0

st.metric("🍔 Findings Checked (Burgers Served)", f"{get_burgers_total(org, team, process):,}")

st.caption(f"Run folder: s3://{settings.bucket}/{doc_prefix}")

# ------------------ Standards: S3 → atoms ------------------
st.sidebar.subheader("Standards")
standards_prefix = st.sidebar.text_input("S3 standards prefix", value="standards/qa/")
standards_eff = st.sidebar.date_input("Standards effective date", value=date.today())
if st.sidebar.button("Load Standards from S3 → Atoms"):
    atoms = load_standards_from_s3(settings.bucket, standards_prefix, stage=process, eff_date=standards_eff)
    count, preview_key = 0, None
    for rel_key, md in atoms:
        s3_key = f"{doc_prefix}/requirements/{rel_key}"
        S3.put_object(Bucket=settings.bucket, Key=s3_key, Body=md.encode("utf-8"), ContentType="text/markdown")
        count += 1
        if preview_key is None:
            preview_key = s3_key
    st.success(f"Created {count} rule atoms under {doc_prefix}/requirements/atoms/")
    if preview_key:
        body = S3.get_object(Bucket=settings.bucket, Key=preview_key)["Body"].read().decode("utf-8", errors="ignore")
        st.code(body[:1200], language="markdown")

st.divider()

# ------------------ Uploads (no local files written) ------------------
st.subheader("1) Upload documents")
colu1, colu2 = st.columns(2)
review_file = colu1.file_uploader("Validation Review (PDF/DOCX)", type=["pdf","docx"])
scope_file  = colu2.file_uploader("Validation Scope (PDF/DOCX)", type=["pdf","docx"])

# ------------------ Minimal in-memory text extraction ------------------

def read_pdf_bytes(b: bytes) -> str:
    out = []
    reader = PdfReader(io.BytesIO(b))
    for p in reader.pages:
        t = p.extract_text() or ""
        if t: out.append(t)
    return "

".join(out)

def read_docx_bytes(b: bytes) -> str:
    doc = Document(io.BytesIO(b))
    parts = [p.text for p in doc.paragraphs if p.text]
    for tbl in doc.tables:
        for row in tbl.rows:
            row_txt = " | ".join(c.text for c in row.cells).strip()
            if row_txt: parts.append(row_txt)
    return "
".join(parts)

def extract_text(upload) -> Tuple[str, str]:
    if upload is None:
        return "", ""
    name = upload.name
    ext = name.lower().split(".")[-1]
    b = upload.read()
    if ext == "pdf":
        return name, read_pdf_bytes(b)
    elif ext == "docx":
        return name, read_docx_bytes(b)
    else:
        return name, ""

# ------------------ 2) Ingest → Chunks ------------------
st.subheader("2) Ingest → Chunks")
chunk_count_placeholder = st.empty()

if st.button("Create Chunks from Uploads"):
    if not review_file and not scope_file:
        st.error("Please upload at least the Validation Review file.")
    else:
        # read texts
        texts = []
        ev = []
        for up in [review_file, scope_file]:
            if not up: continue
            fname, txt = extract_text(up)
            if txt:
                texts.append(f"# Source: {fname}

{txt}")
                raw_key = f"{doc_prefix}/00_source/{fname}"
                S3.put_object(Bucket=settings.bucket, Key=raw_key, Body=up.getvalue())
                ev.append(f"s3://{settings.bucket}/{raw_key}")
        full_text = "

".join(texts)

        # simple chunker: split by ~1200 chars on paragraph boundaries
        def chunks(s: str, maxlen=1200):
            parts, cur = [], []
            for para in s.split("

"):
                if sum(len(p) for p in cur)+len(para) > maxlen and cur:
                    parts.append("

".join(cur)); cur=[]
                cur.append(para)
            if cur: parts.append("

".join(cur))
            return parts

        md_chunks: List[str] = []
        for i, c in enumerate(chunks(full_text), start=1):
            md = f"---
source: upload
index: {i}
---
{c}
"
            key = f"{doc_prefix}/10_chunks/chunk_{i:04d}.md"
            S3.put_object(Bucket=settings.bucket, Key=key, Body=md.encode("utf-8"), ContentType="text/markdown")
            md_chunks.append(key)
        st.session_state["chunks"] = md_chunks
        st.session_state["evidence"] = ev
        chunk_count_placeholder.success(f"Chunks created: {len(md_chunks)} ✅")

# ------------------ 3) Summarize → Section ------------------
st.subheader("3) Summarize → Section")
from workers.summarizer import compress_chunks, build_section_md

if st.button("Summarize Chunks"):
    keys = st.session_state.get("chunks", [])
    if not keys:
        st.error("Create chunks first.")
    else:
        md_blobs = [S3.get_object(Bucket=settings.bucket, Key=k)["Body"].read().decode("utf-8", errors="ignore") for k in keys]
        body = compress_chunks(md_blobs, max_chars=8*settings.max_gist_tokens)
        chunk_ids = [k.split("/")[-1].replace(".md", "") for k in keys]
        section_md = build_section_md(DOC_ID, "3.2", chunk_ids, st.session_state.get("evidence", []), body)
        out_key = f"{doc_prefix}/20_sections/3.2.section.md"
        S3.put_object(Bucket=settings.bucket, Key=out_key, Body=section_md.encode("utf-8"), ContentType="text/markdown")
        st.session_state["section_key"] = out_key
        st.success("Section summary ready ✅")
        st.code(section_md[:1200], language="markdown")

# ------------------ 4) Check → Findings (S3-native rules) ------------------
st.subheader("4) Check → Findings")
if st.button("Run Checks"):
    sk = st.session_state.get("section_key")
    if not sk:
        st.error("Summarize first.")
    else:
        section_md = S3.get_object(Bucket=settings.bucket, Key=sk)["Body"].read().decode("utf-8", errors="ignore")
        header, body_md = section_md.split("---
# Section Gist", 1)
        evidence = [line.strip()[1:].strip() for line in header.splitlines() if line.strip().startswith("-")]
        rules_prefix = f"{doc_prefix}/requirements/atoms"
        finding_mds, ruleset_hash = run_checks_s3(body_md, evidence, settings.bucket, rules_prefix, as_of=review_date)
        keys = []
        for i, fmd in enumerate(finding_mds):
            out_key = f"{doc_prefix}/30_findings/3.2.finding.{i:02d}.mf.md"
            S3.put_object(Bucket=settings.bucket, Key=out_key, Body=fmd.encode("utf-8"), ContentType="text/markdown")
            keys.append(out_key)
        st.session_state["finding_keys"] = keys
        st.session_state["ruleset_hash"] = ruleset_hash
        st.success(f"Findings generated: {len(keys)} (hash {ruleset_hash[:8]}…) ✅")

# ------------------ 5) Assemble → Report ------------------
st.subheader("5) Assemble → Report")
if st.button("Assemble Report"):
    sk = st.session_state.get("section_key")
    fks = st.session_state.get("finding_keys", [])
    if not sk or not fks:
        st.error("Run summary and checks first.")
    else:
        section_md = S3.get_object(Bucket=settings.bucket, Key=sk)["Body"].read().decode("utf-8", errors="ignore")
        findings_md = [S3.get_object(Bucket=settings.bucket, Key=k)["Body"].read().decode("utf-8", errors="ignore") for k in fks]
        report = "
".join([
            "---
title: QA Validation Review Report
---",
            "# Executive Summary
- QA Validation Review demo run.",
            "# Section
", section_md,
            "# Findings
", "

".join(findings_md)
        ])
        out_key = f"{doc_prefix}/90_report/final.md"
        S3.put_object(Bucket=settings.bucket, Key=out_key, Body=report.encode("utf-8"), ContentType="text/markdown")
        st.session_state["report_key"] = out_key
        st.success("Final report saved to S3 ✅")
        st.code(report[:1200], language="markdown")

# ------------------ 6) Save Run to DB ------------------
st.subheader("6) Save Run to DB")
if st.button("💾 Save Run to DB"):
    try:
        finding_keys = st.session_state.get("finding_keys", [])
        finding_mds  = [S3.get_object(Bucket=settings.bucket, Key=k)["Body"].read().decode("utf-8", errors="ignore") for k in finding_keys]
        # minimal metrics snapshot
        m = Metrics(pipeline_version="v1.2")
        m.add_evidence(len(st.session_state.get("evidence", [])))
        m_snapshot = m.snapshot()
        conn = psycopg2.connect(settings.pg_dsn)
        run_id = save_run(
            conn,
            reviewer=user,
            doc_id=DOC_ID,
            review_date=review_date,
            ruleset_hash=st.session_state.get("ruleset_hash"),
            report_s3_uri=f"s3://{settings.bucket}/{st.session_state.get('report_key')}",
            finding_mds=finding_mds,
            metrics=m_snapshot,
            org=org, team=team, process=process,
        )
        conn.close()
        st.success(f"Saved review run #{run_id} ✅")
        st.balloons()
    except Exception as e:
        st.error(f"DB save failed: {e}")

# ------------------ Tiny Help ------------------
with st.expander("Help"):
    st.markdown("""
**Flow**
1) Load Standards → Atoms (optional if already generated)
2) Upload Review (+ Scope)
3) Create Chunks → Summarize → Run Checks → Assemble Report
4) Save Run to DB → burger counter ticks up
""")
```

---

## MASTER: `reporting_app.py`

```python
import streamlit as st
from datetime import date, timedelta
import psycopg2, pandas as pd, altair as alt
from app.config import settings, get_run_prefix
from app.audit_utils import list_rule_hashes, diff_rule_sets
from app.audit_export import export_audit_to_s3

st.set_page_config(page_title="QA Validation Review — Reporting", layout="wide")
st.title("Reporting — QA Validation Review")

ORG_MAP = {
    "QA-Americas": {"teams":["QA-East","QA-West"], "processes":["QAValidationReview"]},
    "QA-EMEA": {"teams":["QA-UK","QA-DE","QA-FR"], "processes":["QAValidationReview"]},
    "QA-APAC": {"teams":["QA-SG","QA-AU"], "processes":["QAValidationReview"]},
}
cols = st.columns(5)
org = cols[0].selectbox("Org", list(ORG_MAP.keys()))
team = cols[1].selectbox("Team", ORG_MAP[org]["teams"])
process = cols[2].selectbox("Process", ORG_MAP[org]["processes"])  # fixed for demo
start_date = cols[3].date_input("Start", value=(date.today()-timedelta(days=30)))
end_date   = cols[4].date_input("End", value=date.today())

# KPI helpers

def fetch_df(q, params, cols):
    conn = psycopg2.connect(settings.pg_dsn); cur = conn.cursor(); cur.execute(q, params)
    rows = cur.fetchall(); conn.close(); return pd.DataFrame(rows, columns=cols)

def fetch_scalar(q, params):
    conn = psycopg2.connect(settings.pg_dsn); cur = conn.cursor(); cur.execute(q, params)
    x = cur.fetchone()[0]; conn.close(); return x or 0

k_burgers = fetch_scalar("""
  SELECT COALESCE(SUM(finding_count),0) FROM review_runs
  WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
""", (start_date, end_date, org, team, process))

k_runs = fetch_scalar("""
  SELECT COUNT(*) FROM review_runs
  WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
""", (start_date, end_date, org, team, process))

k_cost = fetch_scalar("""
  SELECT COALESCE(SUM(cost_usd),0) FROM review_runs
  WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
""", (start_date, end_date, org, team, process))

k_errs = fetch_scalar("""
  SELECT COALESCE(SUM(error_count),0) FROM review_runs
  WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
""", (start_date, end_date, org, team, process))

c1,c2,c3,c4 = st.columns(4)
c1.metric("🍔 Findings", f"{k_burgers:,}")
c2.metric("▶️ Runs", f"{k_runs:,}")
c3.metric("💸 Cost ($)", f"{k_cost:,.2f}")
c4.metric("❗Errors", f"{k_errs:,}")

st.divider()

Overview, Audit, StandardsDiff, UtilCost, Reliability, Engagement = st.tabs([
    "Overview","Audit Trail","Standards Diff","Utilization & Cost","Reliability","Engagement"])

with Overview:
    df = fetch_df(
        """
        SELECT date(review_date) d, SUM(finding_count) burgers, COUNT(*) runs,
               COALESCE(SUM(tokens_input+tokens_output),0) toks, COALESCE(SUM(cost_usd),0) cost
        FROM review_runs
        WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
        GROUP BY 1 ORDER BY 1
        """,
        (start_date,end_date,org,team,process),
        ["date","burgers","runs","tokens","cost"]
    )
    if not df.empty:
        st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="date:T", y="burgers:Q"), use_container_width=True)
        st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="date:T", y="runs:Q"), use_container_width=True)

with Audit:
    reviewer = st.text_input("Reviewer contains", "")
    model    = st.text_input("Model contains", "")
    limit    = st.number_input("Limit", 1, 5000, 200)
    df = fetch_df(
        """
        SELECT id, review_date, reviewer, model_primary, finding_count, runtime_sec, ruleset_hash, report_s3_uri
        FROM review_runs
        WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
          AND (%s='' OR reviewer ILIKE '%'||%s||'%')
          AND (%s='' OR model_primary ILIKE '%'||%s||'%')
        ORDER BY id DESC LIMIT %s
        """,
        (start_date,end_date,org,team,process,reviewer,reviewer,model,model,limit),
        ["id","date","reviewer","model","🍔","runtime","hash","report_s3_uri"]
    )
    st.dataframe(df, use_container_width=True)
    if st.button("Export CSV/PDF to S3"):
        df2 = fetch_df(
            """
            SELECT id, reviewer, review_date, ruleset_hash, finding_count, runtime_sec, tokens_input, tokens_output, model_calls
            FROM review_runs
            WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
            ORDER BY id DESC LIMIT %s
            """,
            (start_date,end_date,org,team,process,limit),
            ["id","reviewer","review_date","ruleset_hash","finding_count","runtime_sec","tokens_input","tokens_output","model_calls"]
        )
        from app.audit_export import export_audit_to_s3
        csv_url, pdf_url = export_audit_to_s3(settings.bucket, f"{get_run_prefix(reviewer or 'report')}/audit/exports", df2.itertuples(index=False))
        st.success("Exported.")
        st.write(f"CSV: {csv_url}")
        if pdf_url: st.write(f"PDF: {pdf_url}")

with StandardsDiff:
    baseline_prefix = st.text_input("Baseline S3 prefix (atoms)", value="standards/baseline/atoms")
    current_prefix  = st.text_input("Current S3 prefix (atoms)",  value=f"demo_run/{team}/{DOC_ID}/requirements/atoms")
    if st.button("Compare"):
        old = list_rule_hashes(settings.bucket, baseline_prefix)
        new = list_rule_hashes(settings.bucket, current_prefix)
        st.json(diff_rule_sets(old, new))

with UtilCost:
    df = fetch_df(
        """
        SELECT date(review_date) d,
               SUM(tokens_input+tokens_output) toks,
               COALESCE(SUM(cost_usd),0) cost,
               PERCENTILE_DISC(0.5) WITHIN GROUP (ORDER BY runtime_sec) AS p50_rt,
               PERCENTILE_DISC(0.95) WITHIN GROUP (ORDER BY runtime_sec) AS p95_rt
        FROM review_runs
        WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
        GROUP BY 1 ORDER BY 1
        """,
        (start_date,end_date,org,team,process),
        ["date","tokens","cost","p50","p95"]
    )
    if not df.empty:
        st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="date:T", y="tokens:Q"), use_container_width=True)
        st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="date:T", y="cost:Q"), use_container_width=True)

with Reliability:
    df = fetch_df(
        """
        SELECT date(review_date) d, COALESCE(SUM(error_count),0) errors, COALESCE(SUM(timeout_count),0) timeouts, COALESCE(SUM(retry_count),0) retries
        FROM review_runs
        WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
        GROUP BY 1 ORDER BY 1
        """,
        (start_date,end_date,org,team,process),
        ["date","errors","timeouts","retries"]
    )
    if not df.empty:
        st.altair_chart(alt.Chart(df.melt("date")).mark_line(point=True).encode(x="date:T", y="value:Q", color="variable:N"), use_container_width=True)

with Engagement:
    df = fetch_df(
        """
        SELECT day, metric, value
        FROM events_daily
        WHERE day BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
        ORDER BY day, metric
        """,
        (start_date,end_date,org,team,process),
        ["day","metric","value"]
    )
    if not df.empty:
        st.altair_chart(alt.Chart(df).mark_line(point=True).encode(x="day:T", y="value:Q", color="metric:N"), use_container_width=True)
```

---

## MASTER: `infra/schema.sql`

```sql
-- Create all objects from scratch (idempotent where possible)
CREATE TABLE IF NOT EXISTS review_runs (
  id BIGSERIAL PRIMARY KEY,
  reviewer TEXT,
  org TEXT,
  team TEXT,
  process TEXT,
  doc_id TEXT NOT NULL,
  review_date DATE NOT NULL,
  ruleset_hash TEXT NOT NULL,
  report_s3_uri TEXT NOT NULL,
  -- stage‑1
  runtime_sec INT,
  chunk_count INT,
  finding_count INT,
  tokens_input INT,
  tokens_output INT,
  model_calls JSONB,
  -- stage‑2 enrichments
  pipeline_version TEXT,
  doc_pages INT,
  doc_bytes BIGINT,
  input_hash TEXT,
  prompt_hash TEXT,
  model_primary TEXT,
  latency_p50_ms INT,
  latency_p95_ms INT,
  cost_usd NUMERIC(10,4),
  error_count INT,
  timeout_count INT,
  retry_count INT,
  evidence_count INT,
  severity_counts JSONB,
  consensus_status TEXT,
  created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS review_findings (
  run_id BIGINT REFERENCES review_runs(id) ON DELETE CASCADE,
  rule_id TEXT NOT NULL,
  stage TEXT,
  severity TEXT,
  status TEXT,
  evidence JSONB,
  rule_source TEXT,
  PRIMARY KEY (run_id, rule_id)
);

CREATE INDEX IF NOT EXISTS review_runs_date ON review_runs(review_date);
CREATE INDEX IF NOT EXISTS review_runs_model ON review_runs(model_primary);
CREATE INDEX IF NOT EXISTS review_runs_ruleset ON review_runs(ruleset_hash);
CREATE INDEX IF NOT EXISTS review_runs_org ON review_runs(org);
CREATE INDEX IF NOT EXISTS review_runs_org_team ON review_runs(org, team);
CREATE INDEX IF NOT EXISTS review_runs_org_team_proc_date ON review_runs(org, team, process, review_date);

-- Event capture for reporting
CREATE TABLE IF NOT EXISTS events_raw (
  id BIGSERIAL PRIMARY KEY,
  ts TIMESTAMPTZ NOT NULL DEFAULT now(),
  user_id TEXT,
  org TEXT,
  team TEXT,
  process TEXT,
  event TEXT NOT NULL,
  payload JSONB
);
CREATE INDEX IF NOT EXISTS events_raw_ts ON events_raw(ts);
CREATE INDEX IF NOT EXISTS events_raw_keys ON events_raw(org,team,process,event);

CREATE TABLE IF NOT EXISTS events_daily (
  day DATE NOT NULL,
  org TEXT,
  team TEXT,
  process TEXT,
  metric TEXT NOT NULL,
  value BIGINT NOT NULL,
  PRIMARY KEY(day, org, team, process, metric)
);

CREATE OR REPLACE VIEW burgers_served AS
SELECT COALESCE(SUM(finding_count),0) AS total_burgers FROM review_runs;

-- Daily rollup
CREATE OR REPLACE FUNCTION rollup_events_daily(p_start DATE, p_end DATE)
RETURNS VOID AS $$
BEGIN
  INSERT INTO events_daily(day,org,team,process,metric,value)
  SELECT date(ts), org, team, process, 'active_users', COUNT(DISTINCT user_id)
  FROM events_raw
  WHERE date(ts) BETWEEN p_start AND p_end
  GROUP BY date(ts), org, team, process
  ON CONFLICT (day,org,team,process,metric) DO UPDATE SET value=EXCLUDED.value;

  INSERT INTO events_daily(day,org,team,process,metric,value)
  SELECT date(ts), org, team, process, 'page_views', COUNT(*)
  FROM events_raw
  WHERE event='page_view' AND date(ts) BETWEEN p_start AND p_end
  GROUP BY date(ts), org, team, process
  ON CONFLICT (day,org,team,process,metric) DO UPDATE SET value=EXCLUDED.value;

  INSERT INTO events_daily(day,org,team,process,metric,value)
  SELECT review_date, org, team, process, 'runs', COUNT(*)
  FROM review_runs
  WHERE review_date BETWEEN p_start AND p_end
  GROUP BY review_date, org, team, process
  ON CONFLICT (day,org,team,process,metric) DO UPDATE SET value=EXCLUDED.value;

  INSERT INTO events_daily(day,org,team,process,metric,value)
  SELECT review_date, org, team, process, 'burgers', COALESCE(SUM(finding_count),0)
  FROM review_runs
  WHERE review_date BETWEEN p_start AND p_end
  GROUP BY review_date, org, team, process
  ON CONFLICT (day,org,team,process,metric) DO UPDATE SET value=EXCLUDED.value;
END; $$ LANGUAGE plpgsql;
```

---

# 📚 Documentation Pack (Markdown, copy‑paste into separate files)

Below are ready‑to‑ship `.md` docs your team can paste into a `docs/` folder. Language is concise and non‑technical where possible. Everything matches the S3‑native, point‑in‑time, QA‑Validation‑Review‑only scope.

---

## `docs/README.md`

```markdown
# QA Validation Review — Hierarchical LLM System (Overview)

**In one sentence:** Upload two docs (Validation Review + Scope), apply point‑in‑time standards from S3, get PASS/FAIL findings and a human‑readable report — all auditable and reproducible.

## Why this matters
- Saves reviewer time; catches gaps consistently.
- Standards change each year — we can run with **rules “as of” any date**.
- No local files (except uploads). All artifacts live in **S3**; runs are logged in **Postgres**.

## What you do (quick flow)
1. Open **Runner App** → choose your name, org/team, process, and **Review Date**.
2. (Optional) **Load Standards** from S3 → creates small rule files (“atoms”).
3. Upload **Validation Review** and **Scope**.
4. Click *Create Chunks* → *Summarize* → *Run Checks* → *Assemble Report*.
5. Click **Save Run** → DB records hash, date, counts → burger counter 🍔 ticks up.
6. Use **Reporting App** for dashboards, diffs, and CSV/PDF exports.

## What’s under the hood
- **S3**: sources, chunks, summaries, rule atoms, findings, final report.
- **Postgres**: `review_runs`, `review_findings`, and analytics events.
- **LLM hierarchy**: small workers for extraction/summarizing, checker applies rules, big model only for final composition (optional).

## Links
- See `docs/QUICKSTART.md` to run a demo.
- See `docs/ARCHITECTURE.md` for diagrams.
```

---

## `docs/QUICKSTART.md`

```markdown
# Quickstart (Demo)

### Prereqs
- AWS credentials with S3 Get/Put/List for your bucket; Bedrock access optional.
- Postgres DSN (read/write) for the runner; read‑only for reporting.
- `infra/schema.sql` applied.

### Environment
Set in your deployment or `.env`:
```

S3\_BUCKET=your-bucket
RUN\_PREFIX=demo\_run
PG\_DSN=postgresql://user\:pass\@host:5432/mrm
USE\_BEDROCK=false
MAX\_GIST\_TOKENS=120

```

### Steps
1) Start the **Runner App** → select your name, Org/Team `QA‑Americas/QA‑East`, Process `QAValidationReview`, Review Date (e.g., 2025‑02‑01).
2) **Load Standards** (sidebar) from `standards/qa/` → atoms will appear under your run prefix.
3) **Upload**: `Validation_Review.pdf` and `Validation_Scope_Template.docx`.
4) Click **Create Chunks** → **Summarize** → **Run Checks** → **Assemble Report**.
5) Click **Save Run to DB** → note the run ID; watch the 🍔 counter.
6) Open **Reporting App** → filter by your Org/Team/Process → view KPIs and export CSV/PDF.
```

---

## `docs/ARCHITECTURE.md`

````markdown
# Architecture

## Diagram — high level
```mermaid
flowchart LR
  U[Reviewer] -->|Upload Review+Scope| Runner
  subgraph AWS
    S3[(S3 Artifacts)]
    PG[(Postgres)]
  end
  Standards[Standards PDFs/DOCX in S3] --> Loader
  Loader -->|Rule Atoms .md| S3
  Runner -->|Chunks & Summaries| S3
  Runner -->|Invoke Checker| Checker
  Checker -->|Findings .md| S3
  Runner -->|Assemble Report| S3
  Runner -->|Save Run Row| PG
  Reporting[[Reporting App]] -->|Read KPIs| PG
  Reporting -->|Standards Diff & Export| S3
````

## Key ideas

* **Rule atoms**: small `.md` files with YAML frontmatter (`id, stage, severity, effective_date, requires, pattern`).
* **Point‑in‑time**: select a **Review Date**; loader filters atoms by `effective_date <= date`.
* **S3‑native**: no local writes; everything saved/read from S3.
* **Audit hash**: `ruleset_hash` fingerprints the exact rule set used.

## Runner vs Reporting

* **Runner**: does the work (ingest → summarize → check → report → save run).
* **Reporting**: read‑only analytics (KPIs, diffs, exports).

````

---

## `docs/STANDARDS_RULES.md`
```markdown
# Standards & Rules (Atoms)

## What is a rule atom?
A small `.md` file with YAML header describing the rule and a short prose body.

### Example
```yaml
---
id: QA-STD-001
stage: QAValidationReview
severity: High
effective_date: "2025-01-01"
requires: ["owner_name","contact","data_lineage"]
---
Validation review must document data lineage and a named accountable owner with contact info.
````

## Loading standards from S3

* The **Standards Loader** scans your standards PDFs/DOCX in S3.
* It extracts candidate “must/shall/required” bullets and turns them into atoms.
* Atoms are written to: `s3://<bucket>/<run_prefix>/<DOC_ID>/requirements/atoms/...`

## Using the right version (Review Date)

* Runner filters atoms by `effective_date <= Review Date`.
* Change the Review Date to replay historical reviews.

## Tips

* Keep each rule short and specific.
* Use `severity: High|Medium|Low` to weight the findings.
* Prefer `requires: [...]` tokens; add `pattern:` only for specific regex checks.

````

---

## `docs/RUNNER_APP_GUIDE.md`
```markdown
# Runner App — How to Use

1. **Select identity**: your name, Org → Team → Process, and **Review Date**.
2. **Load Standards** (optional if already generated for your run folder).
3. **Upload Documents**: Validation Review (PDF/DOCX) and Scope (DOCX/PDF).
4. Click **Create Chunks** → splits text into bite‑size parts.
5. Click **Summarize** → compresses chunks into a short section.
6. Click **Run Checks** → applies rules valid as‑of the Review Date.
7. Click **Assemble Report** → builds a markdown report.
8. Click **Save Run to DB** → logs run in Postgres; burger counter increases.

### Outputs in S3
- `00_source/` raw uploads, `10_chunks/`, `20_sections/`, `30_findings/`, `90_report/final.md`.
````

---

## `docs/REPORTING_APP_GUIDE.md`

```markdown
# Reporting App — Read‑only Analytics

### Filters
- Org → Team → Process, Date range. Optional reviewer/model text filters.

### Tabs
- **Overview**: Findings/day, runs/day.
- **Audit Trail**: recent runs (hash, model, link). Export CSV/PDF to S3.
- **Standards Diff**: compare baseline atoms vs current run atoms.
- **Utilization & Cost**: tokens, cost, p50/p95 runtime.
- **Reliability**: errors/timeouts/retries trend.
- **Engagement**: active users, page views (from events rollups).

### Exports
- CSV (always), PDF (if `reportlab` installed). Links are pre‑signed.
```

---

## `docs/DATA_MODEL.md`

```markdown
# Data Model (Postgres)

## review_runs
One row per saved review. Key fields:
- Identity: `reviewer, org, team, process, doc_id, review_date`
- Audit: `ruleset_hash, report_s3_uri`
- Counts: `finding_count, chunk_count, evidence_count, severity_counts`
- Tokens/Cost: `tokens_input, tokens_output, cost_usd`
- Latency/Reliability: `runtime_sec, latency_p50_ms, latency_p95_ms, error_count, timeout_count, retry_count`
- Provenance: `pipeline_version, model_primary, input_hash, prompt_hash, doc_pages, doc_bytes`

## review_findings
One row per (run, rule).
- `rule_id, stage, severity, status, evidence, rule_source`

## Events
- **events_raw**: append‑only app events (auth/session/page_view/run_started/run_finished/config_change/validation_error/app_error/scoring_update).
- **events_daily**: rolled‑up daily metrics (`active_users, page_views, runs, burgers`).

## Helper view/function
- `burgers_served` view; `rollup_events_daily(start, end)` function.
```

---

## `docs/METRICS_AND_EVENTS.md`

```markdown
# Metrics & Events

## Run‑level metrics (saved on Save Run)
- `finding_count, tokens_input, tokens_output, cost_usd, runtime_sec`
- `latency_p50_ms, latency_p95_ms, error_count, timeout_count, retry_count`
- `model_primary, model_calls, pipeline_version`
- `evidence_count, severity_counts`

## Event types (examples)
- `auth_login` `{ user_id }`
- `session_start` `{ user_id, tab }`
- `page_view` `{ page }`
- `run_started` `{ doc_id }`
- `run_finished` `{ doc_id, ok, burgers }`
- `config_change` `{ key, old, new }`
- `validation_error` `{ message }`
- `app_error` `{ message }`
- `scoring_update` `{ before, after }`

## Daily rollups
Use `SELECT rollup_events_daily('2025-01-01','2025-12-31');` to backfill, or call nightly.
```

---

## `docs/SECURITY_COMPLIANCE.md`

```markdown
# Security & Compliance

- **No local files** (except uploads in memory). All artifacts live in S3.
- **Least privilege IAM**: List/Get/Put limited to `demo_run/*` and standards prefixes.
- **RBAC idea**: Reviewer (write runs), Manager (view all runs), Auditor (read‑only, reporting app only).
- **PII**: don’t log raw document text in DB; only hashes/URIs.
- **Pre‑signed links** expire automatically.
- **Point‑in‑time**: keep `effective_date` in atoms; store `ruleset_hash` per run.
```

---

## `docs/COST_PERFORMANCE.md`

```markdown
# Cost & Performance Tips

- **Markdown as IPC**: short summaries between workers → fewer tokens.
- **Two‑tier retrieval**: filter by stage/risk in SQL before vector search.
- **Chunk smart**: 800–1200 chars, break on headings.
- **Cache & reuse**: name artifacts by hash; skip re‑work.
- **Model routing**: cheap models for extract/summarize; heavy only for final.
- **Parallel ingestion**: process chunks concurrently.
- **S3 lifecycle**: move old artifacts to infrequent/Glacier.
```

---

## `docs/OPERATIONS_SRE.md`

```markdown
# Operations & SRE

## Health
- Ping Postgres, S3 write/read, and a Bedrock call (if enabled).
- Alert on error_rate, timeout_rate, p95 runtime, and cost spikes.

## Jobs
- Nightly: `rollup_events_daily(current_date-7, current_date)`
- Weekly: export audit CSV/PDF for leadership (reporting app button)

## Backup/Restore
- DB: standard snapshots. S3: versioning on `demo_run/`.
```

---

## `docs/DEMO_SCRIPT.md`

```markdown
# Demo Script (5–7 minutes)

1) Select **name**, **Org/Team/Process**, **Review Date** (2024‑06‑30).
2) Load standards → atoms created.
3) Upload Review + Scope → chunks → summarize → checks → report.
4) Save Run → show run ID and 🍔 counter.
5) Change Review Date to 2025‑02‑01 → rerun checks → a rule now FAILS (lineage added).
6) Open Reporting: KPIs, Standards Diff (added/removed), Export CSV/PDF.
```

---

## `docs/FAQ.md`

```markdown
# FAQ

**Q: What if standards change?**  
A: Update atoms or reload from S3; pick a new Review Date. Old runs remain reproducible via `ruleset_hash`.

**Q: Can auditors reproduce results?**  
A: Yes — we store review date, ruleset hash, and report URI.

**Q: Does this replace the full validation process?**  
A: No — this demo covers **QA Validation Review** only.

**Q: Is my data safe?**  
A: No local writes; S3 access is least‑privilege; links are pre‑signed.
```

---

## `docs/GLOSSARY.md`

```markdown
# Glossary

- **Atom**: a tiny rule file (`.md` with YAML frontmatter).
- **Effective date**: rule’s start date; used for point‑in‑time reviews.
- **Ruleset hash**: fingerprint of all atoms used in a run.
- **Burger (🍔)**: one finding checked; our utilization unit.
- **Runner App**: does the review work.
- **Reporting App**: dashboards, diffs, exports (read‑only).
```

---

## `docs/CONTRIBUTING.md`

```markdown
# Contributing

- Keep atoms short; one requirement each.
- Add `effective_date` in `YYYY‑MM‑DD`.
- Prefer `requires` tokens; avoid complex regex unless needed.
- Update `CHANGELOG.md` when standards change.
- Use feature branches; PRs require one reviewer.
```

---

## ✅ COMPLETE `app/streamlit_app.py` (final, clean)

```python
import streamlit as st
from datetime import date
import io
from typing import List, Tuple

import psycopg2
import boto3
from pypdf import PdfReader
from docx import Document

from app.config import settings, get_run_prefix
from workers.standards_loader import load_standards_from_s3
from workers.summarizer import compress_chunks, build_section_md
from workers.checker import run_checks_s3
from app.db_utils import save_run
from app.metrics import Metrics

# ---- AWS S3 client ----
S3 = boto3.client("s3")

st.set_page_config(page_title="QA Validation Review — Runner", layout="wide")
st.title("QA Validation Review — Runner")

# =========================
# 0) Identity & Scope
# =========================
user = st.selectbox("Choose your name", ["alex", "maria", "ravi"], index=0)

ORG_MAP = {
    "QA-Americas": {"teams": ["QA-East", "QA-West"], "processes": ["QAValidationReview"]},
    "QA-EMEA": {"teams": ["QA-UK", "QA-DE", "QA-FR"], "processes": ["QAValidationReview"]},
    "QA-APAC": {"teams": ["QA-SG", "QA-AU"], "processes": ["QAValidationReview"]},
}
org = st.sidebar.selectbox("Org", list(ORG_MAP.keys()), index=0)
team = st.sidebar.selectbox("Team", ORG_MAP[org]["teams"], index=0)
process = st.sidebar.selectbox("Process", ORG_MAP[org]["processes"], index=0)
review_date = st.sidebar.date_input("Review date (point-in-time standards)", value=date.today())

# Per-user run folder in S3 (no local files)
DOC_ID = "WF-2025-042"  # demo id (stable for comparisons)
doc_prefix = f"{get_run_prefix(user)}/{DOC_ID}"
st.caption(f"Run folder: s3://{settings.bucket}/{doc_prefix}")

# Burger counter (team/process scoped)

def _scalar(q: str, params: tuple):
    try:
        conn = psycopg2.connect(settings.pg_dsn)
        cur = conn.cursor(); cur.execute(q, params)
        x = cur.fetchone()[0]
        conn.close()
        return x or 0
    except Exception:
        return 0

burgers = _scalar(
    """
    SELECT COALESCE(SUM(finding_count),0)
    FROM review_runs
    WHERE (%s IS NULL OR org=%s) AND (%s IS NULL OR team=%s) AND (%s IS NULL OR process=%s)
    """,
    (org, org, team, team, process, process)
)
st.metric("🍔 Findings Checked (Burgers Served)", f"{burgers:,}")

st.divider()

# =========================
# 1) Standards (S3 → atoms)
# =========================
st.sidebar.subheader("Standards")
standards_prefix = st.sidebar.text_input("S3 standards prefix", value="standards/qa/")
standards_eff = st.sidebar.date_input("Standards effective date", value=date.today())
if st.sidebar.button("Load Standards from S3 → Atoms"):
    atoms = load_standards_from_s3(settings.bucket, standards_prefix, stage=process, eff_date=standards_eff)
    count, preview_key = 0, None
    for rel_key, md in atoms:
        s3_key = f"{doc_prefix}/requirements/{rel_key}"
        S3.put_object(Bucket=settings.bucket, Key=s3_key, Body=md.encode("utf-8"), ContentType="text/markdown")
        count += 1
        if preview_key is None: preview_key = s3_key
    st.success(f"Created {count} rule atoms under {doc_prefix}/requirements/atoms/")
    if preview_key:
        body = S3.get_object(Bucket=settings.bucket, Key=preview_key)["Body"].read().decode("utf-8", errors="ignore")
        st.code(body[:1200], language="markdown")

# =========================
# 2) Uploads → in‑memory text
# =========================
st.subheader("1) Upload documents")
col1, col2 = st.columns(2)
review_file = col1.file_uploader("Validation Review (PDF/DOCX)", type=["pdf","docx"])
scope_file  = col2.file_uploader("Validation Scope (PDF/DOCX)", type=["pdf","docx"])


def read_pdf_bytes(b: bytes) -> str:
    out = []
    reader = PdfReader(io.BytesIO(b))
    for p in reader.pages:
        t = p.extract_text() or ""
        if t: out.append(t)
    return "\n\n".join(out)


def read_docx_bytes(b: bytes) -> str:
    doc = Document(io.BytesIO(b))
    parts = [p.text for p in doc.paragraphs if p.text]
    for tbl in doc.tables:
        for row in tbl.rows:
            row_txt = " | ".join(c.text for c in row.cells).strip()
            if row_txt: parts.append(row_txt)
    return "\n\n".join(parts)


def extract_text(upload) -> Tuple[str, str]:
    if upload is None:
        return "", ""
    name = upload.name
    ext = name.lower().split(".")[-1]
    b = upload.read()
    if ext == "pdf":
        return name, read_pdf_bytes(b)
    if ext == "docx":
        return name, read_docx_bytes(b)
    return name, ""

st.subheader("2) Ingest → Chunks")
chunk_count_placeholder = st.empty()

if st.button("Create Chunks from Uploads"):
    if not review_file and not scope_file:
        st.error("Please upload at least the Validation Review file.")
    else:
        texts: List[str] = []
        evidence_links: List[str] = []
        for up in [review_file, scope_file]:
            if not up: continue
            fname, txt = extract_text(up)
            if not txt: continue
            texts.append(f"# Source: {fname}\n\n{txt}")
            raw_key = f"{doc_prefix}/00_source/{fname}"
            S3.put_object(Bucket=settings.bucket, Key=raw_key, Body=up.getvalue())
            evidence_links.append(f"s3:/
```



