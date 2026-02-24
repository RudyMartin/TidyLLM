

## Architecture Highlights


I've created the three complete master script files based on your comprehensive documentation:

## 1. **app/streamlit_app.py** - Complete Runner Application
This is the main user interface where reviewers:
- Select their identity (name, org, team, process)
- Load standards from S3 into rule atoms
- Upload validation documents (PDF/DOCX)
- Process through the 6-step pipeline: Chunks → Summarize → Check → Report → Save
- Track progress with the "burger counter" 🍔

Key features:
- **S3-native**: No local files except in-memory upload buffers
- **Point-in-time**: Uses review date to filter applicable rules
- **Team isolation**: Each user gets their own S3 folder
- **Full audit trail**: Saves ruleset hash, metrics, and S3 URIs

## 2. **reporting_app.py** - Complete Reporting Application  
This is the read-only analytics dashboard for managers and auditors:
- **Overview**: Daily trends for findings, runs, tokens, cost
- **Audit Trail**: Searchable run history with CSV/PDF export
- **Standards Diff**: Compare rule sets between different time periods
- **Utilization & Cost**: Token usage, cost tracking, runtime percentiles
- **Reliability**: Error rates, timeouts, retries
- **Engagement**: User activity metrics from events

Key features:
- **Filtered views**: By org, team, process, date range
- **Export capability**: CSV and PDF reports to S3
- **Performance metrics**: P50/P95 latency, cost per run
- **Standards evolution**: Track how rules change over time

## 3. **infra/schema.sql** - Complete Database Schema
This creates the full PostgreSQL schema with:

**Core Tables:**
- `review_runs`: Main table with identity, metrics, and audit fields
- `review_findings`: Individual rule results per run
- `events_raw`: Detailed event tracking
- `events_daily`: Aggregated metrics for fast reporting

**Performance Features:**
- Strategic indexes for common query patterns
- Views for convenience (`burgers_served`, `latest_runs`, `daily_summary`)
- Functions for daily rollups and maintenance
- Optional triggers for automatic severity count updates

**Analytics Support:**
- Event tracking for user engagement
- Daily rollup function for metrics aggregation
- Cleanup functions for maintenance
- Sample data section for testing

## Architecture Highlights

All three files work together to provide:

1. **S3-Native Design**: Everything stored in S3 with proper prefixes
2. **Point-in-Time Compliance**: Rules filtered by effective date
3. **Full Auditability**: Ruleset hashes, timestamps, and provenance
4. **Team Collaboration**: Org → Team → Process hierarchy
5. **Production Ready**: Error handling, metrics, monitoring hooks

The system supports your key requirements:
- ✅ No local files (except upload buffers)
- ✅ Standards versioning with effective dates
- ✅ Reproducible results via ruleset hashing
- ✅ Team isolation and burger counting
- ✅ Full reporting and analytics
- ✅ CSV/PDF export capabilities

You can copy-paste these three files directly into your project structure and they should work together seamlessly with the supporting modules you've built (config, s3_bus, workers, etc.).
