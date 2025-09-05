#!/usr/bin/env python3
"""
Live AI Question Ticker
Demonstrates gateway routing between different AI models with enhanced cost tracking
"""
import streamlit as st
import yaml
import os
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="Live AI Ticker",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

class LiveTicker:
    def __init__(self):
        self.is_running = False
        self.current_question_index = 0
        self.responses: List[Dict[str, Any]] = []
        self.mlflow_connected = False  # (misnomer in original—this checks Postgres)
        self.postgres_connected = False
        self.session_start_time = datetime.now()

        # Budget allocation settings - define BEFORE generating questions
        self.budget_settings = {
            "hourly_budget": 100.0,
            "daily_budget": 2400.0,
            "monthly_budget": 72000.0,
            "cost_multiplier": 1.0,  # Adjust to make costs higher/lower
            "model_cost_weights": {
                "gpt-4": 1.0,
                "gpt-3.5-turbo": 0.1,
                "claude-3": 0.5
            }
        }

        # Generate questions after budget settings are defined
        self.questions = self.generate_questions()

    def generate_questions(self) -> List[Dict[str, Any]]:
        """Generate ~100 interesting, business-relevant AI questions with mock routing info."""
        interesting_questions = [
            # (shortened list kept as-is from your script)
            "How can I identify customer churn patterns before they leave?",
            "What's the best way to optimize our pricing strategy using machine learning?",
            "How do I build a recommendation engine for our e-commerce platform?",
            "What are the key metrics to track for a SaaS business?",
            "How can I predict equipment failures before they happen?",
            "What's the optimal inventory level for our warehouse?",
            "How do I detect fraudulent transactions in real-time?",
            "What customer segments should we target for our new product?",
            "How can I improve our customer lifetime value?",
            "What's causing the drop in our conversion rates?",
            # Business Strategy
            "Should we expand into the European market next quarter?",
            "How do I calculate the ROI of our AI investments?",
            "What's the best pricing model for our new SaaS product?",
            "How can we reduce customer acquisition costs by 30%?",
            "What are the risks of moving our infrastructure to the cloud?",
            "How do I build a competitive moat in our industry?",
            "What's the optimal team size for our startup?",
            "How can we improve our employee retention rates?",
            "What markets should we enter next year?",
            "How do I measure the success of our digital transformation?",
            # (…keep the rest of your sections/questions…)
            "What's our plan for business continuity?"
        ]

        questions: List[Dict[str, Any]] = []
        models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]

        for i, question in enumerate(interesting_questions, start=1):
            model = random.choice(models)
            response_time = round(random.uniform(0.8, 2.5), 2)
            base_cost = random.uniform(0.002, 0.08)
            model_weight = self.budget_settings["model_cost_weights"].get(model, 1.0)
            cost = round(base_cost * model_weight * self.budget_settings["cost_multiplier"], 4)

            questions.append({
                "id": i,
                "question": question,
                "model": model,
                "response_time": response_time,
                "cost": cost,
                "timestamp": datetime.now() - timedelta(seconds=random.randint(0, 3600)),
                "status": random.choice(["completed", "processing", "queued"])
            })

        return questions

    def check_mlflow_connection(self) -> bool:
        """Check if PostgreSQL is connected using settings.yaml (MLflow optional, not enforced)."""
        try:
            settings_paths = ["../settings.yaml", "settings.yaml"]
            for path in settings_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        settings = yaml.safe_load(f) or {}
                        postgres_config = settings.get("postgres", {})
                        if not postgres_config:
                            continue

                        try:
                            import psycopg2
                            conn = psycopg2.connect(
                                host=postgres_config.get("host", "localhost"),
                                port=postgres_config.get("port", 5432),
                                database=postgres_config.get("db_name", "demo_db"),
                                user=postgres_config.get("db_user", "demo_user"),
                                password=postgres_config.get("db_password", "demo_pass"),
                                sslmode=postgres_config.get("ssl_mode", "prefer"),
                                connect_timeout=3
                            )
                            with conn.cursor() as cursor:
                                cursor.execute("SELECT 1")
                                cursor.fetchone()
                            conn.close()
                            return True
                        except Exception as e:
                            st.error(f"PostgreSQL connection failed: {e}")
                            return False
        except Exception as e:
            st.error(f"Settings loading failed: {e}")

        return False

    def get_cost_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive cost metrics with time periods; safe on empty state."""
        now = datetime.now()

        if not self.responses:
            return {
                "total_cost": 0.0,
                "avg_cost": 0.0,
                "session_cost": 0.0,
                "hourly_rate": 0.0,
                "daily_projection": 0.0,
                "monthly_projection": 0.0,
                "model_breakdown": {},
                "time_periods": {
                    "last_5_min": {"cost": 0.0, "count": 0},
                    "last_15_min": {"cost": 0.0, "count": 0},
                    "last_hour": {"cost": 0.0, "count": 0},
                    "last_24_hours": {"cost": 0.0, "count": 0}
                },
                "session_duration": now - self.session_start_time,
                "total_requests": 0,
                "budget_utilization": {"hourly": 0.0, "daily": 0.0, "monthly": 0.0}
            }

        # Basic totals
        total_cost = sum(r.get("cost", 0.0) for r in self.responses)
        total_requests = len(self.responses)
        avg_cost = total_cost / total_requests if total_requests else 0.0

        # Session metrics
        session_duration = now - self.session_start_time
        session_hours = max(session_duration.total_seconds() / 3600, 1e-9)  # avoid div by zero
        session_cost = total_cost
        hourly_rate = session_cost / session_hours

        # Projections
        daily_projection = hourly_rate * 24
        monthly_projection = daily_projection * 30

        # Budget utilization
        bs = self.budget_settings
        budget_utilization = {
            "hourly": (hourly_rate / bs["hourly_budget"] * 100) if bs["hourly_budget"] else 0.0,
            "daily": (daily_projection / bs["daily_budget"] * 100) if bs["daily_budget"] else 0.0,
            "monthly": (monthly_projection / bs["monthly_budget"] * 100) if bs["monthly_budget"] else 0.0
        }

        # Model breakdown
        model_breakdown: Dict[str, Dict[str, float]] = {}
        for r in self.responses:
            m = r.get("model_used", "unknown")
            model_breakdown.setdefault(m, {"count": 0, "total_cost": 0.0})
            model_breakdown[m]["count"] += 1
            model_breakdown[m]["total_cost"] += r.get("cost", 0.0)

        # Time buckets
        time_periods = {
            "last_5_min": {"cost": 0.0, "count": 0},
            "last_15_min": {"cost": 0.0, "count": 0},
            "last_hour": {"cost": 0.0, "count": 0},
            "last_24_hours": {"cost": 0.0, "count": 0}
        }
        for r in self.responses:
            ts = r.get("timestamp", now)
            diff = now - ts
            if diff <= timedelta(minutes=5):
                time_periods["last_5_min"]["cost"] += r.get("cost", 0.0)
                time_periods["last_5_min"]["count"] += 1
            if diff <= timedelta(minutes=15):
                time_periods["last_15_min"]["cost"] += r.get("cost", 0.0)
                time_periods["last_15_min"]["count"] += 1
            if diff <= timedelta(hours=1):
                time_periods["last_hour"]["cost"] += r.get("cost", 0.0)
                time_periods["last_hour"]["count"] += 1
            if diff <= timedelta(hours=24):
                time_periods["last_24_hours"]["cost"] += r.get("cost", 0.0)
                time_periods["last_24_hours"]["count"] += 1

        return {
            "total_cost": total_cost,
            "avg_cost": avg_cost,
            "session_cost": session_cost,
            "hourly_rate": hourly_rate,
            "daily_projection": daily_projection,
            "monthly_projection": monthly_projection,
            "model_breakdown": model_breakdown,
            "time_periods": time_periods,
            "session_duration": session_duration,
            "total_requests": total_requests,
            "budget_utilization": budget_utilization
        }

    def simulate_ai_response(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate AI response through gateway with token/cost estimations."""
        models = {
            "gpt-4": {"performance_score": 9.5, "input_cost_per_1k": 0.03,  "output_cost_per_1k": 0.06,  "max_tokens": 8192},
            "gpt-3.5-turbo": {"performance_score": 8.0, "input_cost_per_1k": 0.0015, "output_cost_per_1k": 0.002, "max_tokens": 4096},
            "claude-3": {"performance_score": 9.0, "input_cost_per_1k": 0.015, "output_cost_per_1k": 0.075, "max_tokens": 200000}
        }
        model = question["model"]
        model_config = models.get(model, models["gpt-3.5-turbo"])

        response_templates = [
            "Based on industry best practices, I recommend {approach} for {context}. This approach typically yields {benefit} and can be implemented within {timeline}.",
            "For {context}, the most effective strategy is {approach}. Key considerations include {considerations}, and you should expect {outcome}.",
            "When addressing {context}, consider {approach}. This method has shown {success_rate} success rate and requires {resources}.",
            "The optimal solution for {context} involves {approach}. This approach balances {tradeoffs} and typically results in {results}.",
            "To successfully tackle {context}, implement {approach}. This strategy addresses {challenges} and delivers {value}."
        ]
        approaches = [
            "data-driven decision making", "agile methodology", "customer-centric design",
            "continuous improvement", "risk-based assessment", "scalable architecture",
            "performance optimization", "security-first approach", "user experience optimization",
            "cost-effective solutions", "sustainable practices", "innovation-driven strategy"
        ]
        contexts = ["this business challenge", "your specific situation", "similar scenarios",
                    "industry standards", "best practices", "current market conditions"]
        benefits = ["20-30% improvement in efficiency", "significant cost savings", "enhanced user satisfaction",
                    "reduced operational risks", "increased competitive advantage", "better resource utilization"]
        timelines = ["2-4 weeks", "1-2 months", "3-6 months", "immediate implementation", "phased rollout over 6 months"]
        considerations = ["budget constraints and resource availability", "team capabilities and training needs",
                          "regulatory compliance requirements", "stakeholder buy-in and change management",
                          "technical debt and legacy system integration", "scalability and future growth plans"]
        outcomes = ["measurable performance improvements", "reduced operational costs", "increased customer satisfaction",
                    "enhanced team productivity", "better decision-making capabilities", "improved market positioning"]

        template = random.choice(response_templates)
        response_text = template.format(
            approach=random.choice(approaches),
            context=random.choice(contexts),
            benefit=random.choice(benefits),
            timeline=random.choice(timelines),
            considerations=random.choice(considerations),
            outcome=random.choice(outcomes),
            success_rate="85-95%",
            resources="a dedicated team and proper tools",
            tradeoffs="cost, time, and complexity",
            results="sustainable long-term improvements",
            challenges="current limitations and future requirements",
            value="tangible business outcomes"
        )

        # Token usage (rough estimates)
        input_tokens = int(len(question["question"].split()) * 1.3)
        output_tokens = int(len(response_text.split()) * 1.3)
        total_tokens = input_tokens + output_tokens

        # Token-based cost (separate from the mock 'question["cost"]')
        input_cost = (input_tokens / 1000) * model_config["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * model_config["output_cost_per_1k"]
        token_cost = input_cost + output_cost  # not currently used in totals, but tracked per-response

        token_limit_status = "✅ Within Limit" if total_tokens <= model_config["max_tokens"] else "⚠️ Near Limit"

        return {
            "question_id": question["id"],
            "response": response_text,
            "model_used": model,
            "response_time": question["response_time"],
            "cost": question["cost"],  # display/mock cost from question generation
            "timestamp": datetime.now(),
            "gateway_routing": f"Routed to {model} (Cost: ${question['cost']:.2f}, Performance: {model_config['performance_score']}/10)",
            # token accounting
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "token_cost_estimate": token_cost,
            "token_limit_status": token_limit_status
        }

def render_budget_controls(ticker: LiveTicker):
    """Render budget allocation controls"""
    st.sidebar.subheader("💰 Budget Controls")
    st.sidebar.markdown("**Resource Accounting Settings**")

    # Budget limits
    hourly_budget = st.sidebar.number_input(
        "Hourly Budget ($)",
        min_value=1.0, max_value=10000.0,
        value=float(ticker.budget_settings["hourly_budget"]),
        step=10.0,
        help="Set the hourly budget limit for resource accounting"
    )
    daily_budget = st.sidebar.number_input(
        "Daily Budget ($)",
        min_value=1.0, max_value=100000.0,
        value=float(ticker.budget_settings["daily_budget"]),
        step=100.0,
        help="Set the daily budget limit for resource accounting"
    )
    monthly_budget = st.sidebar.number_input(
        "Monthly Budget ($)",
        min_value=1.0, max_value=1000000.0,
        value=float(ticker.budget_settings["monthly_budget"]),
        step=1000.0,
        help="Set the monthly budget limit for resource accounting"
    )

    # Cost multiplier
    cost_multiplier = st.sidebar.slider(
        "Cost Multiplier",
        min_value=0.1, max_value=10.0,
        value=float(ticker.budget_settings["cost_multiplier"]),
        step=0.1,
        help="Adjust overall cost scaling (higher = more expensive)"
    )

    # Model cost weights
    st.sidebar.markdown("**Model Cost Weights**")
    gpt4_weight = st.sidebar.slider(
        "GPT-4 Weight", 0.1, 5.0,
        value=float(ticker.budget_settings["model_cost_weights"]["gpt-4"]),
        step=0.1,
        help="Cost multiplier for GPT-4 requests"
    )
    gpt35_weight = st.sidebar.slider(
        "GPT-3.5 Weight", 0.01, 2.0,
        value=float(ticker.budget_settings["model_cost_weights"]["gpt-3.5-turbo"]),
        step=0.01,
        help="Cost multiplier for GPT-3.5 requests"
    )
    claude_weight = st.sidebar.slider(
        "Claude-3 Weight", 0.1, 3.0,
        value=float(ticker.budget_settings["model_cost_weights"]["claude-3"]),
        step=0.1,
        help="Cost multiplier for Claude-3 requests"
    )

    # Update budget settings
    ticker.budget_settings.update({
        "hourly_budget": hourly_budget,
        "daily_budget": daily_budget,
        "monthly_budget": monthly_budget,
        "cost_multiplier": cost_multiplier,
        "model_cost_weights": {
            "gpt-4": gpt4_weight,
            "gpt-3.5-turbo": gpt35_weight,
            "claude-3": claude_weight
        }
    })

    # Budget utilization display
    cost_metrics = ticker.get_cost_metrics()
    budget_util = cost_metrics.get("budget_utilization", {})

    st.sidebar.markdown("**Budget Utilization**")
    # Use progress bars only when >0 to avoid confusing empty bars
    if budget_util.get("hourly", 0) > 0:
        st.sidebar.progress(min(budget_util["hourly"] / 100.0, 1.0))
        st.sidebar.write(f"Hourly: {budget_util['hourly']:.1f}%")
    if budget_util.get("daily", 0) > 0:
        st.sidebar.progress(min(budget_util["daily"] / 100.0, 1.0))
        st.sidebar.write(f"Daily: {budget_util['daily']:.1f}%")
    if budget_util.get("monthly", 0) > 0:
        st.sidebar.progress(min(budget_util["monthly"] / 100.0, 1.0))
        st.sidebar.write(f"Monthly: {budget_util['monthly']:.1f}%")

def render_live_ticker():
    """Render the live ticker interface"""
    st.title("📡 Live AI Gateway Ticker")
    st.markdown("Real-time demonstration of AI gateway routing and responses")

    # Initialize ticker
    if 'ticker' not in st.session_state:
        st.session_state.ticker = LiveTicker()
    ticker: LiveTicker = st.session_state.ticker

    # Check DB connection once
    if not ticker.mlflow_connected:
        ticker.mlflow_connected = ticker.check_mlflow_connection()

    # Sidebar budget controls
    render_budget_controls(ticker)

    # Cost metrics
    cost_metrics = ticker.get_cost_metrics()

    # Status indicators
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.success("✅ Connected & Ready") if ticker.mlflow_connected else st.error("❌ Not Connected")
    with c2:
        st.success("🟢 Live & Answering") if ticker.is_running else st.warning("🔴 Paused")
    with c3:
        st.metric("Questions Answered", cost_metrics["total_requests"])
        st.metric("Time Running", f"{cost_metrics['session_duration'].total_seconds()/60:.1f} min")
    with c4:
        st.metric("Total Cost", f"${cost_metrics['total_cost']:.2f}", help="Mock/estimated")
        st.metric("Cost per Question", f"${cost_metrics['avg_cost']:.2f}")

    # Enhanced Cost Analysis
    st.subheader("💰 Money Usage")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.metric("Hourly Rate", f"${cost_metrics['hourly_rate']:.2f}/hr")
        st.metric("Daily Projection", f"${cost_metrics['daily_projection']:.2f}/day")
    with a2:
        st.metric("Monthly Projection", f"${cost_metrics['monthly_projection']:.2f}/month")
        st.metric("Session Cost", f"${cost_metrics['session_cost']:.2f}")
    with a3:
        st.write("**Recent Questions:**")
        time_period = st.selectbox(
            "Select Time Period",
            ["Last 5 minutes", "Last 15 minutes", "Last hour", "Last 24 hours"],
            help="Choose which time period to view"
        )
        tp_map = {
            "Last 5 minutes": "last_5_min",
            "Last 15 minutes": "last_15_min",
            "Last hour": "last_hour",
            "Last 24 hours": "last_24_hours"
        }
        tp = cost_metrics["time_periods"][tp_map[time_period]]
        st.metric(f"{time_period} Cost", f"${tp['cost']:.2f}")
        st.metric(f"{time_period} Requests", tp["count"])
    with a4:
        st.write("**Summary:**")
        st.metric("Total Cost", f"${cost_metrics['session_cost']:.2f}")
        st.metric("Time Running", f"{cost_metrics['session_duration'].total_seconds()/60:.1f} min")

    # Token Usage Tracking
    if ticker.responses:
        st.subheader("🔤 Token Usage Tracking")
        total_input_tokens = sum(r.get("input_tokens", 0) for r in ticker.responses)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in ticker.responses)
        total_tokens = total_input_tokens + total_output_tokens

        t1, t2, t3, t4 = st.columns(4)
        with t1:
            st.metric("Total Input Tokens", f"{total_input_tokens:,}")
        with t2:
            st.metric("Total Output Tokens", f"{total_output_tokens:,}")
        with t3:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with t4:
            avg_tokens = total_tokens / len(ticker.responses) if ticker.responses else 0
            st.metric("Avg Tokens/Request", f"{avg_tokens:.0f}")

        st.info("**Token Limits:** GPT-4: 8,192 | GPT-3.5: 4,096 | Claude-3: ~200,000 tokens per request")

    # Model Cost Breakdown
    if cost_metrics["model_breakdown"]:
        st.subheader("🤖 Model Cost Breakdown")
        b1, b2, b3 = st.columns(3)
        with b1:
            st.write("**Usage by Model:**")
            for model, data in cost_metrics["model_breakdown"].items():
                pct = (data["count"] / cost_metrics["total_requests"]) * 100 if cost_metrics["total_requests"] else 0
                st.write(f"• {model}: {data['count']} ({pct:.1f}%)")
        with b2:
            st.write("**Cost by Model:**")
            for model, data in cost_metrics["model_breakdown"].items():
                pct = (data["total_cost"] / cost_metrics["total_cost"]) * 100 if cost_metrics["total_cost"] else 0
                st.write(f"• {model}: ${data['total_cost']:.2f} ({pct:.1f}%)")
        with b3:
            st.write("**Avg Cost by Model:**")
            for model, data in cost_metrics["model_breakdown"].items():
                avg_model_cost = (data["total_cost"] / data["count"]) if data["count"] else 0
                st.write(f"• {model}: ${avg_model_cost:.2f}/request")

    # Control panel
    st.subheader("🎛️ Controls")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🚀 Start Ticker", type="primary", disabled=ticker.is_running):
            ticker.is_running = True
            st.success("Ticker started!")
            st.rerun()
    with c2:
        if st.button("⏹️ Stop Ticker", disabled=not ticker.is_running):
            ticker.is_running = False
            st.success("Ticker stopped!")
            st.rerun()
    with c3:
        if st.button("🔄 Reset"):
            ticker.responses.clear()
            ticker.current_question_index = 0
            ticker.session_start_time = datetime.now()
            st.success("Ticker reset!")
            st.rerun()

    # Auto-start if PostgreSQL is connected
    if ticker.mlflow_connected and not ticker.is_running:
        st.info("🔄 Auto-starting ticker in 3 seconds... (PostgreSQL connected)")
        time.sleep(3)
        ticker.is_running = True
        st.rerun()

    # Live ticker display
    if ticker.is_running:
        st.subheader("📺 Live Questions")

        # Process next question
        if ticker.current_question_index < len(ticker.questions):
            q = ticker.questions[ticker.current_question_index]
            resp = ticker.simulate_ai_response(q)
            ticker.responses.append(resp)
            ticker.current_question_index += 1

        # Display recent responses
        recent_responses = ticker.responses[-10:]
        for r in recent_responses:
            title = f"Q{r['question_id']}: {r['response'][:60]}..."
            with st.expander(title, expanded=True):
                e1, e2, e3 = st.columns(3)
                with e1:
                    st.write(f"**Model:** {r['model_used']}")
                    st.write(f"**Time:** {r['response_time']}s")
                with e2:
                    st.write(f"**Cost (mock):** ${r['cost']:.2f}")
                    st.write("**Status:** ✅ Completed")
                with e3:
                    st.write(f"**Gateway:** {r['gateway_routing']}")
                    st.write(f"**Timestamp:** {r['timestamp'].strftime('%H:%M:%S')}")
                st.write(f"**Response:** {r['response']}")
                st.caption(f"Tokens: in={r['input_tokens']}, out={r['output_tokens']}, total={r['total_tokens']} | {r['token_limit_status']}")

        # Auto-refresh every 2 seconds
        time.sleep(2)
        st.rerun()

    # Configuration
    st.subheader("⚙️ Configuration")
    if not ticker.mlflow_connected:
        st.warning(
            "**PostgreSQL Connection Required**\n\n"
            "To start the live ticker, ensure:\n"
            "1. PostgreSQL database is running\n"
            "2. Database credentials are correct in settings.yaml\n"
            "3. Network connectivity to the database\n\n"
            "The ticker will auto-start when PostgreSQL connection is detected."
        )

    # Show current (safe) DB settings
    st.write("**Current Database Settings:**")
    try:
        settings_path = "../settings.yaml"
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f) or {}
                pg = settings.get("postgres", {})
                safe_settings = {
                    "host": pg.get("host", "unknown"),
                    "port": pg.get("port", "unknown"),
                    "database": pg.get("db_name", "unknown"),
                    "user": pg.get("db_user", "unknown"),
                    "ssl_mode": pg.get("ssl_mode", "unknown")
                }
                st.json(safe_settings)
        else:
            st.info("No ../settings.yaml found.")
    except Exception as e:
        st.error(f"Error loading settings: {e}")

def main():
    render_live_ticker()

if __name__ == "__main__":
    main()
