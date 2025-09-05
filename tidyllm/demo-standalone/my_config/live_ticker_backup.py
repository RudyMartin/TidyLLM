#!/usr/bin/env python3
"""
Live AI Question Ticker
Demonstrates gateway routing between different AI models with enhanced cost tracking
"""
import streamlit as st
import yaml
import os
import json
import time
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

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
        self.responses = []
        self.mlflow_connected = False
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
        
    def generate_questions(self) -> List[Dict[str, str]]:
        """Generate 100 interesting, business-relevant AI questions"""
        interesting_questions = [
            # Data Science & Analytics
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
            
            # Technology & Engineering
            "How can I optimize our database queries for better performance?",
            "What's the best architecture for our microservices?",
            "How do I implement zero-downtime deployments?",
            "What security measures should we add to our API?",
            "How can I reduce our cloud infrastructure costs?",
            "What's the best way to handle our data migration?",
            "How do I implement real-time analytics?",
            "What monitoring tools should we use for our system?",
            "How can I improve our CI/CD pipeline?",
            "What's the optimal caching strategy for our application?",
            
            # Marketing & Growth
            "How can I increase our organic search traffic?",
            "What's the best content strategy for our B2B audience?",
            "How do I optimize our email marketing campaigns?",
            "What social media platforms should we focus on?",
            "How can I improve our conversion funnel?",
            "What's the ROI of our influencer marketing campaigns?",
            "How do I build a viral growth strategy?",
            "What customer feedback should we prioritize?",
            "How can I reduce our customer support tickets?",
            "What's the best way to launch our new product?",
            
            # Finance & Operations
            "How can I optimize our cash flow management?",
            "What's the best way to forecast our revenue?",
            "How do I calculate the true cost of our products?",
            "What are the tax implications of our international expansion?",
            "How can I reduce our operational costs?",
            "What's the optimal payment terms for our suppliers?",
            "How do I build a financial model for our startup?",
            "What are the risks in our current investment portfolio?",
            "How can I improve our budgeting process?",
            "What's the best way to raise our next funding round?",
            
            # HR & Leadership
            "How can I improve our remote team productivity?",
            "What's the best way to handle difficult employee situations?",
            "How do I build a strong company culture?",
            "What training programs should we invest in?",
            "How can I reduce our employee turnover?",
            "What's the optimal compensation structure for our team?",
            "How do I manage conflict in our leadership team?",
            "What are the key skills we need to hire for?",
            "How can I improve our performance review process?",
            "What's the best way to onboard new employees?",
            
            # Product & UX
            "How can I improve our user onboarding experience?",
            "What features should we prioritize in our roadmap?",
            "How do I conduct effective user research?",
            "What's the best way to A/B test our new features?",
            "How can I reduce our app's crash rate?",
            "What's the optimal user interface for our dashboard?",
            "How do I measure user engagement effectively?",
            "What are the most requested features from our users?",
            "How can I improve our mobile app performance?",
            "What's the best way to handle user feedback?",
            
            # Sales & Customer Success
            "How can I improve our sales team's performance?",
            "What's the best way to qualify leads?",
            "How do I handle difficult customer negotiations?",
            "What's the optimal sales process for our product?",
            "How can I increase our average deal size?",
            "What are the best practices for customer success?",
            "How do I reduce our sales cycle length?",
            "What's the best way to upsell existing customers?",
            "How can I improve our customer satisfaction scores?",
            "What are the key objections we need to address?",
            
            # Legal & Compliance
            "How do I ensure GDPR compliance for our data?",
            "What are the legal risks of our current contracts?",
            "How can I protect our intellectual property?",
            "What are the compliance requirements for our industry?",
            "How do I handle data breach notifications?",
            "What's the best way to structure our terms of service?",
            "How can I reduce our legal liability?",
            "What are the tax implications of our business model?",
            "How do I ensure our privacy policy is compliant?",
            "What are the risks of our current employment practices?",
            
            # Innovation & R&D
            "How can I foster innovation in our organization?",
            "What emerging technologies should we invest in?",
            "How do I build a successful R&D team?",
            "What's the best way to manage our IP portfolio?",
            "How can I accelerate our product development cycle?",
            "What are the risks of being an early adopter?",
            "How do I measure the success of our innovation projects?",
            "What's the optimal budget allocation for R&D?",
            "How can I attract top research talent?",
            "What are the key trends in our industry?",
            
            # Crisis Management
            "How do I handle a PR crisis effectively?",
            "What's our disaster recovery plan?",
            "How can I prepare for economic downturns?",
            "What are the risks of our current supply chain?",
            "How do I manage cybersecurity threats?",
            "What's the best way to communicate during a crisis?",
            "How can I protect our business from lawsuits?",
            "What are the risks of our current partnerships?",
            "How do I handle negative customer reviews?",
            "What's our plan for business continuity?"
        ]
        
        questions = []
        for i, question in enumerate(interesting_questions):
            # Randomly assign to different AI models
            models = ["gpt-4", "gpt-3.5-turbo", "claude-3"]
            model = random.choice(models)
            
            # Generate realistic response time and cost based on budget settings
            response_time = round(random.uniform(0.8, 2.5), 2)
            base_cost = random.uniform(0.002, 0.08)
            model_weight = self.budget_settings["model_cost_weights"].get(model, 1.0)
            cost = round(base_cost * model_weight * self.budget_settings["cost_multiplier"], 4)
            
            questions.append({
                "id": i + 1,
                "question": question,
                "model": model,
                "response_time": response_time,
                "cost": cost,
                "timestamp": datetime.now() - timedelta(seconds=random.randint(0, 3600)),
                "status": random.choice(["completed", "processing", "queued"])
            })
        
        return questions
    
    def check_mlflow_connection(self) -> bool:
        """Check if PostgreSQL is connected (MLflow optional)"""
        try:
            settings_paths = [
                "../settings.yaml",
                "settings.yaml",
            ]
            
            for path in settings_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        settings = yaml.safe_load(f)
                        
                        # Check PostgreSQL connection (MLflow is optional)
                        postgres_config = settings.get("postgres", {})
                        if not postgres_config:
                            continue
                        
                        # Try to connect to PostgreSQL
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
                            
                            # Test the connection
                            cursor = conn.cursor()
                            cursor.execute("SELECT 1")
                            cursor.fetchone()
                            cursor.close()
                            conn.close()
                            
                            return True
                            
                        except Exception as e:
                            st.error(f"PostgreSQL connection failed: {e}")
                            return False
                            
        except Exception as e:
            st.error(f"Settings loading failed: {e}")
        
        return False
    
    def get_cost_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive cost metrics with time periods"""
        if not self.responses:
        # Calculate token usage
        input_tokens = len(question["question"].split()) * 1.3  # Rough estimate
        output_tokens = len(response_text.split()) * 1.3  # Rough estimate
        total_tokens = input_tokens + output_tokens
        
        # Calculate token-based cost
        input_cost = (input_tokens / 1000) * model_config["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * model_config["output_cost_per_1k"]
        token_cost = input_cost + output_cost
        
        # Check token limits
        token_limit_status = "✅ Within Limit" if total_tokens <= model_config["max_tokens"] else "⚠️ Near Limit"
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
                "session_duration": datetime.now() - self.session_start_time,
                "total_requests": 0,
                "budget_utilization": {
                    "hourly": 0.0,
                    "daily": 0.0,
                    "monthly": 0.0
                }
            }
        
        # Basic metrics
        total_cost = sum(r["cost"] for r in self.responses)
        avg_cost = total_cost / len(self.responses)
        
        # Session metrics
        session_duration = datetime.now() - self.session_start_time
        session_hours = session_duration.total_seconds() / 3600
        session_cost = total_cost
        hourly_rate = session_cost / session_hours if session_hours > 0 else 0
        
        # Projections
        daily_projection = hourly_rate * 24
        monthly_projection = daily_projection * 30
        
        # Budget utilization
        budget_utilization = {
            "hourly": (hourly_rate / self.budget_settings["hourly_budget"]) * 100 if self.budget_settings["hourly_budget"] > 0 else 0,
            "daily": (daily_projection / self.budget_settings["daily_budget"]) * 100 if self.budget_settings["daily_budget"] > 0 else 0,
            "monthly": (monthly_projection / self.budget_settings["monthly_budget"]) * 100 if self.budget_settings["monthly_budget"] > 0 else 0
        }
        
        # Model breakdown
        model_breakdown = {}
        for response in self.responses:
            model = response["model_used"]
            if model not in model_breakdown:
                model_breakdown[model] = {"count": 0, "total_cost": 0.0}
            model_breakdown[model]["count"] += 1
            model_breakdown[model]["total_cost"] += response["cost"]
        
        # Time period analysis
        now = datetime.now()
        time_periods = {
            "last_5_min": {"cost": 0.0, "count": 0},
            "last_15_min": {"cost": 0.0, "count": 0},
            "last_hour": {"cost": 0.0, "count": 0},
            "last_24_hours": {"cost": 0.0, "count": 0}
        }
        
        for response in self.responses:
            response_time = response["timestamp"]
            time_diff = now - response_time
            
            if time_diff <= timedelta(minutes=5):
                time_periods["last_5_min"]["cost"] += response["cost"]
                time_periods["last_5_min"]["count"] += 1
            if time_diff <= timedelta(minutes=15):
                time_periods["last_15_min"]["cost"] += response["cost"]
                time_periods["last_15_min"]["count"] += 1
            if time_diff <= timedelta(hours=1):
                time_periods["last_hour"]["cost"] += response["cost"]
                time_periods["last_hour"]["count"] += 1
            if time_diff <= timedelta(hours=24):
                time_periods["last_24_hours"]["cost"] += response["cost"]
                time_periods["last_24_hours"]["count"] += 1
        
        # Calculate token usage
        input_tokens = len(question["question"].split()) * 1.3  # Rough estimate
        output_tokens = len(response_text.split()) * 1.3  # Rough estimate
        total_tokens = input_tokens + output_tokens
        
        # Calculate token-based cost
        input_cost = (input_tokens / 1000) * model_config["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * model_config["output_cost_per_1k"]
        token_cost = input_cost + output_cost
        
        # Check token limits
        token_limit_status = "✅ Within Limit" if total_tokens <= model_config["max_tokens"] else "⚠️ Near Limit"
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
            "total_requests": len(self.responses),
            "budget_utilization": budget_utilization
        }
    
    def simulate_ai_response(self, question: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate AI response through gateway"""
        models = {
            "gpt-4": {"cost_per_1k": 0.03, "performance_score": 9.5, "input_cost_per_1k": 0.03, "output_cost_per_1k": 0.06, "max_tokens": 8192},
            "gpt-3.5-turbo": {"cost_per_1k": 0.002, "performance_score": 8.0, "input_cost_per_1k": 0.0015, "output_cost_per_1k": 0.002, "max_tokens": 4096},
            "claude-3": {"cost_per_1k": 0.015, "performance_score": 9.0, "input_cost_per_1k": 0.015, "output_cost_per_1k": 0.075, "max_tokens": 200000}
        }
        
        model = question["model"]
        model_config = models.get(model, models["gpt-3.5-turbo"])
        
        # Generate realistic, business-focused responses
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
        
        contexts = [
            "this business challenge", "your specific situation", "similar scenarios",
            "industry standards", "best practices", "current market conditions"
        ]
        
        benefits = [
            "20-30% improvement in efficiency", "significant cost savings", "enhanced user satisfaction",
            "reduced operational risks", "increased competitive advantage", "better resource utilization"
        ]
        
        timelines = [
            "2-4 weeks", "1-2 months", "3-6 months", "immediate implementation", "phased rollout over 6 months"
        ]
        
        considerations = [
            "budget constraints and resource availability", "team capabilities and training needs",
            "regulatory compliance requirements", "stakeholder buy-in and change management",
            "technical debt and legacy system integration", "scalability and future growth plans"
        ]
        
        outcomes = [
            "measurable performance improvements", "reduced operational costs", "increased customer satisfaction",
            "enhanced team productivity", "better decision-making capabilities", "improved market positioning"
        ]
        
        template = random.choice(response_templates)
        approach = random.choice(approaches)
        context = random.choice(contexts)
        benefit = random.choice(benefits)
        timeline = random.choice(timelines)
        consideration = random.choice(considerations)
        outcome = random.choice(outcomes)
        
        response_text = template.format(
            approach=approach,
            context=context,
            benefit=benefit,
            timeline=timeline,
            considerations=consideration,
            outcome=outcome,
            success_rate="85-95%",
            resources="dedicated team and proper tools",
            tradeoffs="cost, time, and complexity",
            results="sustainable long-term improvements",
            challenges="current limitations and future requirements",
            value="tangible business outcomes"
        )
        
        # Calculate token usage
        input_tokens = len(question["question"].split()) * 1.3  # Rough estimate
        output_tokens = len(response_text.split()) * 1.3  # Rough estimate
        total_tokens = input_tokens + output_tokens
        
        # Calculate token-based cost
        input_cost = (input_tokens / 1000) * model_config["input_cost_per_1k"]
        output_cost = (output_tokens / 1000) * model_config["output_cost_per_1k"]
        token_cost = input_cost + output_cost
        
        # Check token limits
        token_limit_status = "✅ Within Limit" if total_tokens <= model_config["max_tokens"] else "⚠️ Near Limit"
                return {
            "question_id": question["id"],
            "response": response_text,
            "model_used": model,
            "response_time": question["response_time"],
            "cost": question["cost"],
            "timestamp": datetime.now(),
            "gateway_routing": f"Routed to {model} (Cost: ${question['cost']:.2f}, Performance: {model_config['performance_score']}/10)"
        }

def render_budget_controls(ticker):
    """Render budget allocation controls"""
    st.sidebar.subheader("💰 Budget Controls")
    st.sidebar.markdown("**Resource Accounting Settings**")
    
    # Budget limits
    hourly_budget = st.sidebar.number_input(
        "Hourly Budget ($)", 
        min_value=1.0, 
        max_value=10000.0, 
        value=float(ticker.budget_settings["hourly_budget"]),
        step=10.0,
        help="Set the hourly budget limit for resource accounting"
    )
    
    daily_budget = st.sidebar.number_input(
        "Daily Budget ($)", 
        min_value=1.0, 
        max_value=100000.0, 
        value=float(ticker.budget_settings["daily_budget"]),
        step=100.0,
        help="Set the daily budget limit for resource accounting"
    )
    
    monthly_budget = st.sidebar.number_input(
        "Monthly Budget ($)", 
        min_value=1.0, 
        max_value=1000000.0, 
        value=float(ticker.budget_settings["monthly_budget"]),
        step=1000.0,
        help="Set the monthly budget limit for resource accounting"
    )
    
    # Cost multiplier
    cost_multiplier = st.sidebar.slider(
        "Cost Multiplier", 
        min_value=0.1, 
        max_value=10.0, 
        value=float(ticker.budget_settings["cost_multiplier"]),
        step=0.1,
        help="Adjust overall cost scaling (higher = more expensive)"
    )
    
    # Model cost weights
    st.sidebar.markdown("**Model Cost Weights**")
    gpt4_weight = st.sidebar.slider(
        "GPT-4 Weight", 
        min_value=0.1, 
        max_value=5.0, 
        value=float(ticker.budget_settings["model_cost_weights"]["gpt-4"]),
        step=0.1,
        help="Cost multiplier for GPT-4 requests"
    )
    
    gpt35_weight = st.sidebar.slider(
        "GPT-3.5 Weight", 
        min_value=0.01, 
        max_value=2.0, 
        value=float(ticker.budget_settings["model_cost_weights"]["gpt-3.5-turbo"]),
        step=0.01,
        help="Cost multiplier for GPT-3.5 requests"
    )
    
    claude_weight = st.sidebar.slider(
        "Claude-3 Weight", 
        min_value=0.1, 
        max_value=3.0, 
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
    if budget_util.get("hourly", 0) > 0:
        st.sidebar.progress(min(budget_util["hourly"] / 100, 1.0))
        st.sidebar.write(f"Hourly: {budget_util['hourly']:.1f}%")
    
    if budget_util.get("daily", 0) > 0:
        st.sidebar.progress(min(budget_util["daily"] / 100, 1.0))
        st.sidebar.write(f"Daily: {budget_util['daily']:.1f}%")
    
    if budget_util.get("monthly", 0) > 0:
        st.sidebar.progress(min(budget_util["monthly"] / 100, 1.0))
        st.sidebar.write(f"Monthly: {budget_util['monthly']:.1f}%")

def render_live_ticker():
    """Render the live ticker interface"""
    st.title("📡 Live AI Gateway Ticker")
    st.markdown("Real-time demonstration of AI gateway routing and responses")
    
    # Initialize ticker
    if 'ticker' not in st.session_state:
        st.session_state.ticker = LiveTicker()
    
    ticker = st.session_state.ticker
    
    # Check MLflow connection
    if not ticker.mlflow_connected:
        ticker.mlflow_connected = ticker.check_mlflow_connection()
    
    # Render budget controls in sidebar
    render_budget_controls(ticker)
    
    # Enhanced cost metrics
    cost_metrics = ticker.get_cost_metrics()
    
    # Status indicators with enhanced cost display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if ticker.mlflow_connected:
            st.success("✅ Connected & Ready")
        else:
            st.error("❌ Not Connected")
    
    with col2:
        if ticker.is_running:
            st.success("🟢 Live & Answering")
        else:
            st.warning("🔴 Paused")
    
    with col3:
        st.metric("Questions Answered", cost_metrics["total_requests"])
        st.metric("Time Running", f"{cost_metrics['session_duration'].total_seconds()/60:.1f} min")
    
    with col4:
        st.metric("Total Cost", f"${cost_metrics['total_cost']:.2f}", help="Not actual, estimates")
        st.metric("Cost per Question", f"${cost_metrics['avg_cost']:.2f}")
    
    # Enhanced Cost Analysis Section
    st.subheader("💰 Money Usage")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Hourly Rate", f"${cost_metrics['hourly_rate']:.2f}/hr")
        st.metric("Daily Projection", f"${cost_metrics['daily_projection']:.2f}/day")
    
    with col2:
        st.metric("Monthly Projection", f"${cost_metrics['monthly_projection']:.2f}/month")
        st.metric("Session Cost", f"${cost_metrics['session_cost']:.2f}")
    
    with col3:
        st.write("**Recent Questions:**")
        time_period = st.selectbox(
            "Select Time Period",
            ["Last 5 minutes", "Last 15 minutes", "Last hour", "Last 24 hours"],
            help="Choose which time period to view"
        )
        
        if time_period == "Last 5 minutes":
            cost = cost_metrics["time_periods"]["last_5_min"]["cost"]
            count = cost_metrics["time_periods"]["last_5_min"]["count"]
        elif time_period == "Last 15 minutes":
            cost = cost_metrics["time_periods"]["last_15_min"]["cost"]
            count = cost_metrics["time_periods"]["last_15_min"]["count"]
        elif time_period == "Last hour":
            cost = cost_metrics["time_periods"]["last_hour"]["cost"]
            count = cost_metrics["time_periods"]["last_hour"]["count"]
        else:  # Last 24 hours
            cost = cost_metrics["time_periods"]["last_24_hours"]["cost"]
            count = cost_metrics["time_periods"]["last_24_hours"]["count"]
        
        st.metric(f"{time_period} Cost", f"${cost:.2f}")
        st.metric(f"{time_period} Requests", count)
    
    with col4:
        st.write("**Summary:**")
        st.metric("Total Cost", f"${cost_metrics['session_cost']:.2f}")
        st.metric("Time Running", f"{cost_metrics['session_duration'].total_seconds()/60:.1f} min")
    
    # Model Cost Breakdown
    # Token Usage Tracking
    if ticker.responses:
        st.subheader("🔤 Token Usage Tracking")
        
        # Calculate total token usage
        total_input_tokens = sum(r.get("input_tokens", 0) for r in ticker.responses)
        total_output_tokens = sum(r.get("output_tokens", 0) for r in ticker.responses)
        total_tokens = total_input_tokens + total_output_tokens
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Input Tokens", f"{total_input_tokens:,}")
        with col2:
            st.metric("Total Output Tokens", f"{total_output_tokens:,}")
        with col3:
            st.metric("Total Tokens", f"{total_tokens:,}")
        with col4:
            avg_tokens_per_request = total_tokens / len(ticker.responses) if ticker.responses else 0
            st.metric("Avg Tokens/Request", f"{avg_tokens_per_request:.0f}")
        
        # Token limits info
        st.info("**Token Limits:** GPT-4: 8,192 | GPT-3.5: 4,096 | Claude-3: 200,000 tokens per request")
            if cost_metrics["model_breakdown"]:
        st.subheader("🤖 Model Cost Breakdown")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Usage by Model:**")
            for model, data in cost_metrics["model_breakdown"].items():
                percentage = (data["count"] / cost_metrics["total_requests"]) * 100
                st.write(f"• {model}: {data['count']} ({percentage:.1f}%)")
        
        with col2:
            st.write("**Cost by Model:**")
            for model, data in cost_metrics["model_breakdown"].items():
                percentage = (data["total_cost"] / cost_metrics["total_cost"]) * 100
                st.write(f"• {model}: ${data['total_cost']:.2f} ({percentage:.1f}%)")
        
        with col3:
            st.write("**Avg Cost by Model:**")
            for model, data in cost_metrics["model_breakdown"].items():
                avg_model_cost = data["total_cost"] / data["count"]
                st.write(f"• {model}: ${avg_model_cost:.2f}/request")
    
    # Control panel
    st.subheader("🎛️ Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🚀 Start Ticker", type="primary", disabled=ticker.is_running):
            ticker.is_running = True
            st.success("Ticker started!")
            st.rerun()
    
    with col2:
        if st.button("⏹️ Stop Ticker", disabled=not ticker.is_running):
            ticker.is_running = False
            st.success("Ticker stopped!")
            st.rerun()
    
    with col3:
        if st.button("🔄 Reset"):
            ticker.responses = []
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
        
        # Process new questions
        if ticker.current_question_index < len(ticker.questions):
            question = ticker.questions[ticker.current_question_index]
            response = ticker.simulate_ai_response(question)
            ticker.responses.append(response)
            ticker.current_question_index += 1
        
        # Display recent responses
        recent_responses = ticker.responses[-10:]  # Show last 10
        
        for response in recent_responses:
            with st.expander(f"Q{response['question_id']}: {response['response'][:50]}...", expanded=True):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Model:** {response['model_used']}")
                    st.write(f"**Time:** {response['response_time']}s")
                
                with col2:
                    st.write(f"**Cost:** ${response['cost']:.2f}")
                    st.write(f"**Status:** ✅ Completed")
                
                with col3:
                    st.write(f"**Gateway:** {response['gateway_routing']}")
                    st.write(f"**Timestamp:** {response['timestamp'].strftime('%H:%M:%S')}")
                
                st.write(f"**Response:** {response['response']}")
        
        # Auto-refresh every 2 seconds
        time.sleep(2)
        st.rerun()
    
    # Configuration
    st.subheader("⚙️ Configuration")
    
    if not ticker.mlflow_connected:
        st.warning("""
        **PostgreSQL Connection Required**
        
        To start the live ticker, ensure:
        1. PostgreSQL database is running
        2. Database credentials are correct in settings.yaml
        3. Network connectivity to the database
        
        The ticker will auto-start when PostgreSQL connection is detected.
        """)
    
    # Show current settings
    st.write("**Current Database Settings:**")
    try:
        settings_path = "../settings.yaml"
        if os.path.exists(settings_path):
            with open(settings_path, 'r') as f:
                settings = yaml.safe_load(f)
                postgres_settings = settings.get("postgres", {})
                # Hide sensitive info
                safe_settings = {
                    "host": postgres_settings.get("host", "unknown"),
                    "port": postgres_settings.get("port", "unknown"),
                    "database": postgres_settings.get("db_name", "unknown"),
                    "user": postgres_settings.get("db_user", "unknown"),
                    "ssl_mode": postgres_settings.get("ssl_mode", "unknown")
                }
                st.json(safe_settings)
    except Exception as e:
        st.error(f"Error loading settings: {e}")

def main():
    """Main application"""
    render_live_ticker()

if __name__ == "__main__":
    main()
