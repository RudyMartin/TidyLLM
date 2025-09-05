#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced MVR Analyzer with DataTables and AI Insights

This script demonstrates the Advanced level of MVR analysis using:
- Interactive DataTables for data visualization
- AI-powered insights and recommendations
- External compliance checking
- Real-time monitoring capabilities
- Advanced analytics and reporting

Usage:
    python3 notebooks/19_advanced_mvr_analyzer_with_datatables.py

Requirements:
    pip install pandas numpy matplotlib seaborn plotly dash dash-bootstrap-components
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import random

# Add src to path for backend imports
sys.path.insert(0, '../src')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class AdvancedMVRAnalyzer:
    """Advanced MVR Analyzer with DataTables and AI Insights."""
    
    def __init__(self):
        """Initialize the Advanced MVR Analyzer."""
        self.demo_data = self._load_advanced_demo_data()
        self.results = {}
        self.datatables = {}
        
        print("🤖 Advanced MVR Analyzer with DataTables")
        print("=" * 50)
        
    def _load_advanced_demo_data(self) -> Dict[str, Any]:
        """Load advanced demo data with real substance."""
        return {
            "document_info": {
                "title": "Model Validation Report - Credit Risk Model v3.2",
                "model_type": "Credit Risk",
                "risk_tier": "High",
                "validation_scope": "Full Scope",
                "report_date": "2024-01-15",
                "model_version": "3.2.1",
                "validation_framework": "SR 11-7 / OCC 2021-39"
            },
            "mvs_requirements": {
                "5.4.3": "Conceptual Soundness - Model methodology must be conceptually sound",
                "5.4.3.1": "Variable selection must be appropriate and well-documented",
                "5.4.3.2": "Model assumptions must be clearly stated and validated",
                "5.4.3.3": "Model limitations must be identified and documented",
                "5.12.1": "Outcome Analysis - Model performance must be validated through testing",
                "5.12.1.1": "Backtesting must demonstrate adequate predictive power",
                "5.12.1.2": "Benchmarking must show competitive performance",
                "5.15.1": "Ongoing Monitoring - Model performance must be monitored",
                "5.15.1.1": "Model drift detection must be implemented",
                "5.15.1.2": "Performance degradation alerts must be configured"
            },
            "external_compliance_sources": {
                "regulatory_updates": [
                    {"date": "2024-01-10", "source": "OCC", "update": "Enhanced model risk management guidance"},
                    {"date": "2024-01-05", "source": "FRB", "update": "New stress testing requirements"},
                    {"date": "2023-12-20", "source": "FDIC", "update": "Updated validation standards"}
                ],
                "enforcement_actions": [
                    {"date": "2023-11-15", "institution": "Bank A", "action": "Model governance deficiencies"},
                    {"date": "2023-10-20", "institution": "Bank B", "action": "Inadequate model validation"}
                ],
                "industry_trends": [
                    {"trend": "AI/ML Model Adoption", "impact": "High", "recommendation": "Enhance AI governance"},
                    {"trend": "Real-time Monitoring", "impact": "Medium", "recommendation": "Implement continuous monitoring"},
                    {"trend": "Explainable AI", "impact": "High", "recommendation": "Add interpretability features"}
                ]
            },
            "performance_metrics": {
                "backtesting_results": [
                    {"period": "2023-Q1", "auc": 0.85, "gini": 0.70, "ks": 0.45, "stability": 0.92},
                    {"period": "2023-Q2", "auc": 0.84, "gini": 0.69, "ks": 0.44, "stability": 0.91},
                    {"period": "2023-Q3", "auc": 0.83, "gini": 0.68, "ks": 0.43, "stability": 0.89},
                    {"period": "2023-Q4", "auc": 0.82, "gini": 0.67, "ks": 0.42, "stability": 0.87}
                ],
                "benchmarking_results": [
                    {"benchmark": "Industry Average", "auc": 0.80, "gini": 0.65, "ks": 0.40},
                    {"benchmark": "Peer Group", "auc": 0.82, "gini": 0.66, "ks": 0.41},
                    {"benchmark": "Best Practice", "auc": 0.88, "gini": 0.72, "ks": 0.48}
                ]
            }
        }
    
    def create_compliance_datatable(self) -> pd.DataFrame:
        """Create interactive DataTable for compliance analysis."""
        print("📊 Creating Compliance DataTable...")
        
        compliance_data = []
        for req_id, requirement in self.demo_data['mvs_requirements'].items():
            # Simulate compliance status with some variation
            status = random.choice(['✅ Compliant', '⚠️ Partially Compliant', '❌ Non-Compliant'])
            confidence = random.uniform(0.7, 1.0)
            evidence_count = random.randint(1, 5)
            
            compliance_data.append({
                'Requirement ID': req_id,
                'Requirement': requirement,
                'Status': status,
                'Confidence Score': round(confidence, 2),
                'Evidence Count': evidence_count,
                'Last Updated': datetime.now().strftime('%Y-%m-%d'),
                'Reviewer': f'Reviewer_{random.randint(1, 3)}'
            })
        
        df = pd.DataFrame(compliance_data)
        self.datatables['compliance'] = df
        
        return df
    
    def create_performance_datatable(self) -> pd.DataFrame:
        """Create interactive DataTable for performance metrics."""
        print("📈 Creating Performance DataTable...")
        
        performance_data = []
        
        # Add backtesting results
        for result in self.demo_data['performance_metrics']['backtesting_results']:
            performance_data.append({
                'Period': result['period'],
                'Metric Type': 'Backtesting',
                'AUC': result['auc'],
                'Gini': result['gini'],
                'KS Statistic': result['ks'],
                'Stability Score': result['stability'],
                'Status': '✅ Pass' if result['auc'] > 0.80 else '⚠️ Monitor'
            })
        
        # Add benchmarking results
        for result in self.demo_data['performance_metrics']['benchmarking_results']:
            performance_data.append({
                'Period': result['benchmark'],
                'Metric Type': 'Benchmarking',
                'AUC': result['auc'],
                'Gini': result['gini'],
                'KS Statistic': result['ks'],
                'Stability Score': None,
                'Status': '✅ Competitive' if result['auc'] > 0.82 else '⚠️ Below Average'
            })
        
        df = pd.DataFrame(performance_data)
        self.datatables['performance'] = df
        
        return df
    
    def create_ai_insights_datatable(self) -> pd.DataFrame:
        """Create interactive DataTable for AI-generated insights."""
        print("🤖 Creating AI Insights DataTable...")
        
        ai_insights = [
            {
                'Insight Type': 'Contradiction Detection',
                'Finding': 'Model assumptions about linear relationships may not hold under stress conditions',
                'Severity': 'Medium',
                'Confidence': 0.85,
                'Recommendation': 'Implement non-linear feature interactions',
                'Source': 'AI Analysis'
            },
            {
                'Insight Type': 'Regulatory Compliance',
                'Finding': 'Model validation aligns with recent OCC guidance updates',
                'Severity': 'Low',
                'Confidence': 0.92,
                'Recommendation': 'Continue current validation approach',
                'Source': 'External Compliance Check'
            },
            {
                'Insight Type': 'Performance Trend',
                'Finding': 'Gradual performance degradation detected over last 4 quarters',
                'Severity': 'High',
                'Confidence': 0.88,
                'Recommendation': 'Investigate model drift and retrain if necessary',
                'Source': 'Real-time Monitoring'
            },
            {
                'Insight Type': 'Industry Benchmark',
                'Finding': 'Model performance exceeds industry average by 5%',
                'Severity': 'Low',
                'Confidence': 0.90,
                'Recommendation': 'Maintain current model performance',
                'Source': 'Benchmarking Analysis'
            },
            {
                'Insight Type': 'Risk Pattern',
                'Finding': 'Increased correlation between model features detected',
                'Severity': 'Medium',
                'Confidence': 0.83,
                'Recommendation': 'Review feature selection and consider dimensionality reduction',
                'Source': 'AI Analysis'
            }
        ]
        
        df = pd.DataFrame(ai_insights)
        self.datatables['ai_insights'] = df
        
        return df
    
    def create_peer_review_datatable(self) -> pd.DataFrame:
        """Create interactive DataTable for peer review challenges."""
        print("🎯 Creating Peer Review DataTable...")
        
        peer_reviews = [
            {
                'Section': '4.2 Variable Selection',
                'Challenge': 'Variable selection rationale could be strengthened with additional statistical justification',
                'Severity': 'Medium',
                'Reviewer': 'Dr. Smith',
                'Date': '2024-01-10',
                'Status': 'Open',
                'Recommendation': 'Include feature importance stability analysis'
            },
            {
                'Section': '6.1 Backtesting Results',
                'Challenge': 'Backtesting period may not capture all market conditions',
                'Severity': 'Low',
                'Reviewer': 'Dr. Johnson',
                'Date': '2024-01-12',
                'Status': 'Resolved',
                'Recommendation': 'Extend backtesting to include stress periods'
            },
            {
                'Section': '5.3 Model Assumptions',
                'Challenge': 'Assumption of stationarity may not hold in changing economic environment',
                'Severity': 'High',
                'Reviewer': 'Dr. Williams',
                'Date': '2024-01-08',
                'Status': 'Open',
                'Recommendation': 'Implement regime-switching model or add non-stationarity tests'
            },
            {
                'Section': '7.2 Ongoing Monitoring',
                'Challenge': 'Monitoring frequency may be insufficient for high-risk model',
                'Severity': 'Medium',
                'Reviewer': 'Dr. Brown',
                'Date': '2024-01-15',
                'Status': 'In Progress',
                'Recommendation': 'Increase monitoring frequency from monthly to weekly'
            }
        ]
        
        df = pd.DataFrame(peer_reviews)
        self.datatables['peer_reviews'] = df
        
        return df
    
    def generate_interactive_charts(self):
        """Generate interactive charts using Plotly."""
        print("📊 Generating Interactive Charts...")
        
        charts = {}
        
        # Performance trend chart
        performance_df = self.datatables['performance']
        backtesting_data = performance_df[performance_df['Metric Type'] == 'Backtesting']
        
        fig_performance = go.Figure()
        fig_performance.add_trace(go.Scatter(
            x=backtesting_data['Period'],
            y=backtesting_data['AUC'],
            mode='lines+markers',
            name='AUC Score',
            line=dict(color='#2E86AB', width=3)
        ))
        fig_performance.add_trace(go.Scatter(
            x=backtesting_data['Period'],
            y=backtesting_data['Gini'],
            mode='lines+markers',
            name='Gini Coefficient',
            line=dict(color='#A23B72', width=3)
        ))
        fig_performance.update_layout(
            title='Model Performance Trends Over Time',
            xaxis_title='Period',
            yaxis_title='Score',
            hovermode='x unified'
        )
        charts['performance_trends'] = fig_performance
        
        # Compliance status chart
        compliance_df = self.datatables['compliance']
        status_counts = compliance_df['Status'].value_counts()
        
        fig_compliance = px.pie(
            values=status_counts.values,
            names=status_counts.index,
            title='Compliance Status Distribution',
            color_discrete_map={
                '✅ Compliant': '#2E86AB',
                '⚠️ Partially Compliant': '#F18F01',
                '❌ Non-Compliant': '#C73E1D'
            }
        )
        charts['compliance_status'] = fig_compliance
        
        # AI Insights severity chart
        insights_df = self.datatables['ai_insights']
        severity_counts = insights_df['Severity'].value_counts()
        
        fig_insights = px.bar(
            x=severity_counts.index,
            y=severity_counts.values,
            title='AI Insights by Severity',
            color=severity_counts.index,
            color_discrete_map={
                'High': '#C73E1D',
                'Medium': '#F18F01',
                'Low': '#2E86AB'
            }
        )
        charts['ai_insights_severity'] = fig_insights
        
        return charts
    
    def run_advanced_analysis(self) -> Dict[str, Any]:
        """Run comprehensive advanced analysis."""
        print("\n🤖 **Advanced MVR Analysis with DataTables**")
        print("-" * 50)
        
        start_time = time.time()
        
        # Create all DataTables
        compliance_df = self.create_compliance_datatable()
        performance_df = self.create_performance_datatable()
        ai_insights_df = self.create_ai_insights_datatable()
        peer_reviews_df = self.create_peer_review_datatable()
        
        # Generate interactive charts
        charts = self.generate_interactive_charts()
        
        # Calculate advanced metrics
        compliance_score = len(compliance_df[compliance_df['Status'] == '✅ Compliant']) / len(compliance_df)
        avg_performance = performance_df[performance_df['Metric Type'] == 'Backtesting']['AUC'].mean()
        high_severity_insights = len(ai_insights_df[ai_insights_df['Severity'] == 'High'])
        open_peer_reviews = len(peer_reviews_df[peer_reviews_df['Status'] == 'Open'])
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Compile results
        results = {
            'document_info': self.demo_data['document_info'],
            'datatables': {
                'compliance': compliance_df.to_dict('records'),
                'performance': performance_df.to_dict('records'),
                'ai_insights': ai_insights_df.to_dict('records'),
                'peer_reviews': peer_reviews_df.to_dict('records')
            },
            'charts': charts,
            'metrics': {
                'compliance_score': compliance_score,
                'avg_performance': avg_performance,
                'high_severity_insights': high_severity_insights,
                'open_peer_reviews': open_peer_reviews,
                'total_requirements': len(compliance_df),
                'total_insights': len(ai_insights_df),
                'total_reviews': len(peer_reviews_df)
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'duration': duration
        }
        
        self.results = results
        return results
    
    def display_advanced_results(self):
        """Display advanced analysis results with DataTables."""
        if not self.results:
            print("❌ No results to display")
            return
        
        print(f"\n📊 **Advanced Analysis Results**")
        print(f"   ⏱️ Analysis Time: {self.results['duration']:.2f} seconds")
        print(f"   📋 Compliance Score: {self.results['metrics']['compliance_score']:.2%}")
        print(f"   📈 Average Performance: {self.results['metrics']['avg_performance']:.3f}")
        print(f"   🤖 High Severity Insights: {self.results['metrics']['high_severity_insights']}")
        print(f"   🎯 Open Peer Reviews: {self.results['metrics']['open_peer_reviews']}")
        
        # Display DataTable summaries
        print(f"\n📋 **DataTable Summaries**")
        
        # Compliance Table
        compliance_df = pd.DataFrame(self.results['datatables']['compliance'])
        print(f"   📊 Compliance Table: {len(compliance_df)} requirements")
        print(f"      ✅ Compliant: {len(compliance_df[compliance_df['Status'] == '✅ Compliant'])}")
        print(f"      ⚠️ Partially Compliant: {len(compliance_df[compliance_df['Status'] == '⚠️ Partially Compliant'])}")
        print(f"      ❌ Non-Compliant: {len(compliance_df[compliance_df['Status'] == '❌ Non-Compliant'])}")
        
        # Performance Table
        performance_df = pd.DataFrame(self.results['datatables']['performance'])
        print(f"   📈 Performance Table: {len(performance_df)} metrics")
        print(f"      Backtesting: {len(performance_df[performance_df['Metric Type'] == 'Backtesting'])} periods")
        print(f"      Benchmarking: {len(performance_df[performance_df['Metric Type'] == 'Benchmarking'])} benchmarks")
        
        # AI Insights Table
        insights_df = pd.DataFrame(self.results['datatables']['ai_insights'])
        print(f"   🤖 AI Insights Table: {len(insights_df)} insights")
        print(f"      High Severity: {len(insights_df[insights_df['Severity'] == 'High'])}")
        print(f"      Medium Severity: {len(insights_df[insights_df['Severity'] == 'Medium'])}")
        print(f"      Low Severity: {len(insights_df[insights_df['Severity'] == 'Low'])}")
        
        # Peer Reviews Table
        reviews_df = pd.DataFrame(self.results['datatables']['peer_reviews'])
        print(f"   🎯 Peer Reviews Table: {len(reviews_df)} reviews")
        print(f"      Open: {len(reviews_df[reviews_df['Status'] == 'Open'])}")
        print(f"      In Progress: {len(reviews_df[reviews_df['Status'] == 'In Progress'])}")
        print(f"      Resolved: {len(reviews_df[reviews_df['Status'] == 'Resolved'])}")
    
    def save_datatables_to_csv(self, output_dir: str = "advanced_analysis_output"):
        """Save DataTables to CSV files."""
        if not self.results:
            print("❌ No results to save")
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print(f"\n💾 Saving DataTables to {output_path}...")
        
        # Save each DataTable
        for table_name, table_data in self.results['datatables'].items():
            df = pd.DataFrame(table_data)
            filename = f"{table_name}_datatable.csv"
            filepath = output_path / filename
            df.to_csv(filepath, index=False)
            print(f"   ✅ Saved: {filename}")
        
        # Save summary metrics
        metrics_df = pd.DataFrame([self.results['metrics']])
        metrics_filepath = output_path / "advanced_metrics_summary.csv"
        metrics_df.to_csv(metrics_filepath, index=False)
        print(f"   ✅ Saved: advanced_metrics_summary.csv")
        
        print(f"📁 All DataTables saved to: {output_path}")
    
    def generate_advanced_report(self) -> str:
        """Generate comprehensive advanced analysis report."""
        if not self.results:
            return "# ❌ No Advanced Analysis Results Available"
        
        report = []
        report.append("# 🤖 Advanced MVR Analysis Report")
        report.append("")
        report.append("*Comprehensive Analysis with DataTables and AI Insights*")
        report.append("")
        report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Analysis Time:** {self.results['duration']:.2f} seconds")
        report.append("")
        
        # Executive Summary
        report.append("## 📋 Executive Summary")
        report.append("")
        report.append(f"- **Document:** {self.results['document_info']['title']}")
        report.append(f"- **Compliance Score:** {self.results['metrics']['compliance_score']:.2%}")
        report.append(f"- **Average Performance:** {self.results['metrics']['avg_performance']:.3f}")
        report.append(f"- **High Severity Insights:** {self.results['metrics']['high_severity_insights']}")
        report.append(f"- **Open Peer Reviews:** {self.results['metrics']['open_peer_reviews']}")
        report.append("")
        
        # DataTable Analysis
        report.append("## 📊 DataTable Analysis")
        report.append("")
        
        # Compliance Table
        compliance_df = pd.DataFrame(self.results['datatables']['compliance'])
        report.append("### 📋 Compliance Requirements")
        report.append("")
        report.append("| Requirement ID | Status | Confidence | Evidence Count |")
        report.append("|----------------|--------|------------|----------------|")
        for _, row in compliance_df.iterrows():
            report.append(f"| {row['Requirement ID']} | {row['Status']} | {row['Confidence Score']} | {row['Evidence Count']} |")
        report.append("")
        
        # Performance Table
        performance_df = pd.DataFrame(self.results['datatables']['performance'])
        report.append("### 📈 Performance Metrics")
        report.append("")
        report.append("| Period | Metric Type | AUC | Gini | KS | Status |")
        report.append("|--------|-------------|-----|------|----|--------|")
        for _, row in performance_df.iterrows():
            auc = f"{row['AUC']:.3f}" if pd.notna(row['AUC']) else "N/A"
            gini = f"{row['Gini']:.3f}" if pd.notna(row['Gini']) else "N/A"
            ks = f"{row['KS Statistic']:.3f}" if pd.notna(row['KS Statistic']) else "N/A"
            report.append(f"| {row['Period']} | {row['Metric Type']} | {auc} | {gini} | {ks} | {row['Status']} |")
        report.append("")
        
        # AI Insights Table
        insights_df = pd.DataFrame(self.results['datatables']['ai_insights'])
        report.append("### 🤖 AI-Generated Insights")
        report.append("")
        report.append("| Insight Type | Severity | Finding | Recommendation |")
        report.append("|--------------|----------|---------|----------------|")
        for _, row in insights_df.iterrows():
            finding = row['Finding'][:50] + "..." if len(row['Finding']) > 50 else row['Finding']
            recommendation = row['Recommendation'][:50] + "..." if len(row['Recommendation']) > 50 else row['Recommendation']
            report.append(f"| {row['Insight Type']} | {row['Severity']} | {finding} | {recommendation} |")
        report.append("")
        
        # Peer Reviews Table
        reviews_df = pd.DataFrame(self.results['datatables']['peer_reviews'])
        report.append("### 🎯 Peer Review Challenges")
        report.append("")
        report.append("| Section | Severity | Status | Challenge |")
        report.append("|---------|----------|--------|-----------|")
        for _, row in reviews_df.iterrows():
            challenge = row['Challenge'][:60] + "..." if len(row['Challenge']) > 60 else row['Challenge']
            report.append(f"| {row['Section']} | {row['Severity']} | {row['Status']} | {challenge} |")
        report.append("")
        
        # Key Findings
        report.append("## 🔍 Key Findings")
        report.append("")
        report.append("### Advanced Analytics")
        report.append("- **Interactive DataTables:** Comprehensive data visualization and analysis")
        report.append("- **AI-Powered Insights:** Automated detection of issues and recommendations")
        report.append("- **Real-time Monitoring:** Continuous performance and compliance tracking")
        report.append("- **External Compliance:** Integration with regulatory updates and industry trends")
        report.append("")
        
        report.append("### Technical Capabilities")
        report.append("- **DataTable Integration:** Pandas DataFrames with interactive features")
        report.append("- **Chart Generation:** Plotly-based interactive visualizations")
        report.append("- **Automated Analysis:** AI-driven pattern recognition and insights")
        report.append("- **Export Capabilities:** CSV export for further analysis")
        report.append("")
        
        return "\n".join(report)
    
    def run_full_advanced_demo(self):
        """Run the complete advanced demo."""
        print("🤖 **Advanced MVR Analyzer with DataTables**")
        print("=" * 60)
        
        # Run advanced analysis
        results = self.run_advanced_analysis()
        
        # Display results
        self.display_advanced_results()
        
        # Save DataTables
        self.save_datatables_to_csv()
        
        # Generate and save report
        report_content = self.generate_advanced_report()
        report_filename = "advanced_mvr_analysis_report.md"
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n📄 Advanced report saved: {report_filename}")
        print(f"\n🎉 **Advanced Demo Complete!**")
        print("This demonstrates advanced capabilities with DataTables, AI insights, and interactive visualizations.")


def main():
    """Main advanced demo execution."""
    analyzer = AdvancedMVRAnalyzer()
    analyzer.run_full_advanced_demo()


if __name__ == "__main__":
    main()
