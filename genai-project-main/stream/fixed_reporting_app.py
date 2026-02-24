import streamlit as st
from datetime import date, timedelta
import psycopg2
import pandas as pd
import altair as alt
from app.config import settings, get_run_prefix
from app.audit_utils import list_rule_hashes, diff_rule_sets
from app.audit_export import export_audit_to_s3

st.set_page_config(page_title="QA Validation Review – Reporting", layout="wide")
st.title("Reporting – QA Validation Review")

# =========================
# Filters
# =========================
ORG_MAP = {
    "QA-Americas": {"teams":["QA-East","QA-West"], "processes":["QAValidationReview"]},
    "QA-EMEA": {"teams":["QA-UK","QA-DE","QA-FR"], "processes":["QAValidationReview"]},
    "QA-APAC": {"teams":["QA-SG","QA-AU"], "processes":["QAValidationReview"]},
}

cols = st.columns(5)
org = cols[0].selectbox("Org", list(ORG_MAP.keys()))
team = cols[1].selectbox("Team", ORG_MAP[org]["teams"])
process = cols[2].selectbox("Process", ORG_MAP[org]["processes"])
start_date = cols[3].date_input("Start", value=(date.today()-timedelta(days=30)))
end_date   = cols[4].date_input("End", value=date.today())

# =========================
# KPI helpers
# =========================
def fetch_df(q, params, cols):
    try:
        conn = psycopg2.connect(settings.pg_dsn)
        cur = conn.cursor()
        cur.execute(q, params)
        rows = cur.fetchall()
        conn.close()
        return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def fetch_scalar(q, params):
    try:
        conn = psycopg2.connect(settings.pg_dsn)
        cur = conn.cursor()
        cur.execute(q, params)
        x = cur.fetchone()[0]
        conn.close()
        return x or 0
    except Exception as e:
        st.error(f"Database error: {e}")
        return 0

# =========================
# KPI tiles
# =========================
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
c4.metric("⚠️ Errors", f"{k_errs:,}")

st.divider()

# =========================
# Tabs
# =========================
Overview, Audit, StandardsDiff, UtilCost, Reliability, Engagement = st.tabs([
    "Overview","Audit Trail","Standards Diff","Utilization & Cost","Reliability","Engagement"])

with Overview:
    st.subheader("Daily Trends")
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
        st.altair_chart(
            alt.Chart(df).mark_line(point=True).encode(
                x="day:T",
                y="value:Q",
                color="metric:N",
                tooltip=["day:T", "metric:N", "value:Q"]
            ).properties(title="Daily Engagement Metrics"),
            use_container_width=True
        )
        
        # Engagement summary
        metrics_summary = df.groupby("metric")["value"].sum().reset_index()
        st.subheader("Period Summary")
        st.dataframe(metrics_summary, use_container_width=True)
    else:
        st.info("No engagement data available. Events may need to be rolled up using the daily rollup function.")

# =========================
# Footer
# =========================
st.divider()
st.caption(f"QA Validation Review Reporting Dashboard • Team: {team} • Read-only analytics and exports")x="date:T",
                y="burgers:Q",
                tooltip=["date:T", "burgers:Q"]
            ).properties(title="Findings per Day"),
            use_container_width=True
        )
        st.altair_chart(
            alt.Chart(df).mark_line(point=True).encode(
                x="date:T",
                y="runs:Q",
                tooltip=["date:T", "runs:Q"]
            ).properties(title="Runs per Day"),
            use_container_width=True
        )
    else:
        st.info("No data for the selected filters and date range.")

with Audit:
    st.subheader("Audit Trail")
    
    # Additional filters
    col1, col2, col3 = st.columns(3)
    reviewer = col1.text_input("Reviewer contains", "")
    model = col2.text_input("Model contains", "")
    limit = col3.number_input("Limit", 1, 5000, 200)
    
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
    
    if not df.empty:
        st.dataframe(df, use_container_width=True)
        
        if st.button("Export CSV/PDF to S3"):
            df2 = fetch_df(
                """
                SELECT id, reviewer, review_date, ruleset_hash, finding_count, runtime_sec, 
                       tokens_input, tokens_output, model_calls
                FROM review_runs
                WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
                ORDER BY id DESC LIMIT %s
                """,
                (start_date,end_date,org,team,process,limit),
                ["id","reviewer","review_date","ruleset_hash","finding_count","runtime_sec",
                 "tokens_input","tokens_output","model_calls"]
            )
            try:
                # Use team-based export path
                base_prefix = f"{get_run_prefix('report', team)}/audit/exports"
                rows = [tuple(row) for row in df2.values]
                csv_url, pdf_url = export_audit_to_s3(settings.bucket, base_prefix, rows)
                st.success("Exported.")
                st.write(f"📄 CSV: {csv_url}")
                if pdf_url: 
                    st.write(f"📄 PDF: {pdf_url}")
            except Exception as e:
                st.error(f"Export failed: {e}")
    else:
        st.info("No audit records found for the selected criteria.")

with StandardsDiff:
    st.subheader("Standards Comparison")
    st.caption("Compare a baseline rules prefix with current run's atoms")
    
    col1, col2 = st.columns(2)
    baseline_prefix = col1.text_input("Baseline S3 prefix (atoms)", value="standards/baseline/atoms")
    # Use team-based current prefix
    current_prefix = col2.text_input("Current S3 prefix (atoms)", 
                                   value=f"{get_run_prefix('alex', team)}/WF-2025-042/requirements/atoms")
    
    if st.button("Compare Standards"):
        try:
            old = list_rule_hashes(settings.bucket, baseline_prefix)
            new = list_rule_hashes(settings.bucket, current_prefix)
            diff_result = diff_rule_sets(old, new)
            
            st.subheader("Comparison Results")
            st.json(diff_result)
            
            # Visual summary
            if diff_result:
                added = len(diff_result.get("added", []))
                removed = len(diff_result.get("removed", []))
                modified = len(diff_result.get("modified", []))
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Added Rules", added)
                col2.metric("Removed Rules", removed)
                col3.metric("Modified Rules", modified)
                
        except Exception as e:
            st.error(f"Standards comparison failed: {e}")

with UtilCost:
    st.subheader("Utilization & Cost Analysis")
    
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
        st.altair_chart(
            alt.Chart(df).mark_line(point=True).encode(
                x="date:T",
                y="tokens:Q",
                tooltip=["date:T", "tokens:Q"]
            ).properties(title="Token Usage Over Time"),
            use_container_width=True
        )
        st.altair_chart(
            alt.Chart(df).mark_line(point=True).encode(
                x="date:T",
                y="cost:Q",
                tooltip=["date:T", "cost:Q"]
            ).properties(title="Cost Over Time"),
            use_container_width=True
        )
        
        # Runtime percentiles
        rt_df = df.melt(id_vars=["date"], value_vars=["p50", "p95"], var_name="percentile", value_name="runtime_sec")
        st.altair_chart(
            alt.Chart(rt_df).mark_line(point=True).encode(
                x="date:T",
                y="runtime_sec:Q",
                color="percentile:N",
                tooltip=["date:T", "percentile:N", "runtime_sec:Q"]
            ).properties(title="Runtime Percentiles"),
            use_container_width=True
        )
    else:
        st.info("No utilization data for the selected period.")

with Reliability:
    st.subheader("System Reliability")
    
    df = fetch_df(
        """
        SELECT date(review_date) d, 
               COALESCE(SUM(error_count),0) errors, 
               COALESCE(SUM(timeout_count),0) timeouts, 
               COALESCE(SUM(retry_count),0) retries
        FROM review_runs
        WHERE review_date BETWEEN %s AND %s AND org=%s AND team=%s AND process=%s
        GROUP BY 1 ORDER BY 1
        """,
        (start_date,end_date,org,team,process),
        ["date","errors","timeouts","retries"]
    )
    
    if not df.empty:
        df_melted = df.melt("date", var_name="metric", value_name="count")
        st.altair_chart(
            alt.Chart(df_melted).mark_line(point=True).encode(
                x="date:T",
                y="count:Q",
                color="metric:N",
                tooltip=["date:T", "metric:N", "count:Q"]
            ).properties(title="Error, Timeout, and Retry Trends"),
            use_container_width=True
        )
        
        # Summary stats
        total_errors = df["errors"].sum()
        total_timeouts = df["timeouts"].sum()
        total_retries = df["retries"].sum()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Errors", total_errors)
        col2.metric("Total Timeouts", total_timeouts)
        col3.metric("Total Retries", total_retries)
    else:
        st.info("No reliability data for the selected period.")

with Engagement:
    st.subheader("User Engagement")
    
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
        st.altair_chart(
            alt.Chart(df).mark_line(point=True).encode(
                