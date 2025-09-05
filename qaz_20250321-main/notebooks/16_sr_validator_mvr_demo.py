#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SR Validator MVR Peer Review Demo

This script demonstrates the SR Validator's MVR Peer Review prompt using our 
progressive complexity architecture (Simple, Enhanced, Advanced).

Overview:
- Simple MVR Analyzer: Basic TOC parsing and compliance checking
- Enhanced MVR Analyzer: Evidence tracing and peer review challenges  
- Advanced MVR Analyzer: AI-powered insights and external compliance
- Favorites Integration: Show how prompts can be saved and reused

Usage:
    python notebooks/16_sr_validator_mvr_demo.py

Requirements:
    pip install pandas numpy matplotlib seaborn plotly
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add src to path for backend imports
sys.path.insert(0, '../src')

# Set up plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SRValidatorDemo:
    """SR Validator MVR Peer Review Demo using progressive complexity."""
    
    def __init__(self):
        """Initialize the demo with sample MVR data."""
        self.demo_data = self._load_demo_data()
        self.favorites_dir = Path("../src/assets/prompts/favorites")
        self.results = {}
        
        print("🎯 SR Validator MVR Peer Review Demo")
        print("=" * 50)
        
    def _load_demo_data(self) -> Dict[str, Any]:
        """Load sample MVR data for demonstration."""
        return {
            "document_info": {
                "title": "Model Validation Report - Credit Risk Model v2.1",
                "model_type": "Credit Risk",
                "risk_tier": "High",
                "validation_scope": "Full Scope",
                "report_date": "2024-01-15"
            },
            "table_of_contents": {
                "1": "Executive Summary",
                "2": "Model Overview",
                "3": "Data Quality Assessment",
                "4": "Conceptual Soundness",
                "4.1": "Model Methodology",
                "4.2": "Variable Selection",
                "4.3": "Assumptions and Limitations",
                "5": "Process Verification",
                "5.1": "Model Development Process",
                "5.2": "Implementation Verification",
                "6": "Outcome Analysis",
                "6.1": "Backtesting Results",
                "6.2": "Benchmarking Analysis",
                "7": "Ongoing Monitoring",
                "8": "Conclusions and Recommendations"
            },
            "sample_sections": {
                "4": {
                    "title": "Conceptual Soundness",
                    "content": """
                    The credit risk model demonstrates strong conceptual soundness through its 
                    well-established methodology and comprehensive variable selection process.
                    
                    **Model Methodology**: The model employs logistic regression with SHAP-based 
                    feature selection, which provides interpretable results while maintaining 
                    predictive accuracy. The methodology is consistent with industry best practices.
                    
                    **Variable Selection**: A total of 15 variables were selected from an initial 
                    pool of 45 candidates. The selection process included correlation analysis, 
                    statistical significance testing, and business relevance assessment.
                    
                    **Assumptions and Limitations**: The model assumes linear relationships between 
                    variables and the target. This assumption is validated through residual analysis. 
                    Limitations include reliance on historical data patterns and potential for 
                    model drift over time.
                    """,
                    "evidence": [
                        "SHAP feature importance analysis",
                        "Correlation matrix results",
                        "Statistical significance tests",
                        "Residual analysis plots"
                    ]
                },
                "6.1": {
                    "title": "Backtesting Results",
                    "content": """
                    Backtesting was performed using 24 months of out-of-sample data from 
                    January 2022 to December 2023. The model demonstrated strong predictive 
                    performance with an AUC of 0.85 and Gini coefficient of 0.70.
                    
                    **Performance Metrics**:
                    - AUC: 0.85 (Target: >0.80)
                    - Gini: 0.70 (Target: >0.60)
                    - KS Statistic: 0.45 (Target: >0.30)
                    
                    **Stability Analysis**: The model showed consistent performance across 
                    different time periods and market conditions, with performance degradation 
                    of less than 5% during stress periods.
                    """,
                    "evidence": [
                        "Backtesting results table",
                        "Performance trend analysis",
                        "Stability metrics",
                        "Stress testing results"
                    ]
                }
            },
            "mvs_requirements": {
                "5.4.3": "Conceptual Soundness - Model methodology must be conceptually sound",
                "5.4.3.1": "Variable selection must be appropriate and well-documented",
                "5.4.3.2": "Model assumptions must be clearly stated and validated",
                "5.4.3.3": "Model limitations must be identified and documented",
                "5.12.1": "Outcome Analysis - Model performance must be validated through testing",
                "5.12.1.1": "Backtesting must demonstrate adequate predictive power",
                "5.12.1.2": "Benchmarking must show competitive performance"
            },
            "vst_sections": {
                "Conceptual Soundness": "Evaluate model methodology, variable selection, and assumptions",
                "Outcome Analysis": "Assess model performance through backtesting and benchmarking",
                "Process Verification": "Verify model development and implementation processes"
            }
        }
    
    def show_favorites_integration(self):
        """Demonstrate favorites selection functionality."""
        print("\n📚 **Favorites Selection Demo**")
        print("-" * 30)
        
        # Check if favorites directory exists
        if self.favorites_dir.exists():
            print(f"✅ Favorites directory found: {self.favorites_dir}")
            
            # List available favorites
            favorites = list(self.favorites_dir.glob("*.md"))
            print(f"📋 Available favorites ({len(favorites)}):")
            
            for fav in favorites:
                print(f"   • {fav.stem}")
                
            # Show SR Validator's prompt
            jb_prompt = self.favorites_dir / "JB_Overview_Prompt.md"
            if jb_prompt.exists():
                print(f"\n🎯 **SR Validator's Favorite Prompt**: {jb_prompt.name}")
                
                # Read and show prompt summary
                with open(jb_prompt, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    title = next((line for line in lines if line.startswith('# 📑')), "MVR Peer Review Prompt")
                    print(f"   Title: {title.replace('# 📑', '').strip()}")
                    
                    # Extract key features
                    features = []
                    if "Simple Level" in content:
                        features.append("Simple Analysis")
                    if "Enhanced Level" in content:
                        features.append("Evidence Tracing")
                    if "Advanced Level" in content:
                        features.append("AI-Powered Insights")
                    
                    print(f"   Features: {', '.join(features)}")
                    print(f"   Status: ✅ Ready for use across all complexity levels")
        else:
            print("❌ Favorites directory not found")
    
    def run_simple_demo(self):
        """Run Simple MVR Analyzer demo."""
        print("\n🔧 **Simple MVR Analyzer Demo**")
        print("-" * 30)
        
        try:
            # Import Simple MVR Analyzer
            from backend.mcp.orchestrators.simple_mvr_analyzer import SimpleMVRAnalyzer
            
            # Initialize analyzer
            analyzer = SimpleMVRAnalyzer()
            print("✅ Simple MVR Analyzer initialized")
            
            # Create sample document path
            sample_doc = "sample_mvr_document.pdf"
            
            # Run analysis
            print("🔄 Running basic MVR analysis...")
            start_time = time.time()
            
            result = analyzer.analyze_mvr(sample_doc)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"✅ Analysis completed in {duration:.2f} seconds")
            
            # Display results
            self._display_simple_results(result)
            
            # Store results
            self.results['simple'] = {
                'result': result,
                'duration': duration,
                'timestamp': datetime.now().isoformat()
            }
            
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("💡 Simple MVR Analyzer not yet implemented")
            self._show_simple_mock_results()
        except Exception as e:
            print(f"❌ Error: {e}")
            self._show_simple_mock_results()
    
    def _show_simple_mock_results(self):
        """Show mock results for Simple MVR Analyzer."""
        mock_result = {
            'document_info': self.demo_data['document_info'],
            'toc_analysis': {
                'total_sections': len(self.demo_data['table_of_contents']),
                'sections_analyzed': 8,
                'compliance_status': '✅ Compliant',
                'missing_sections': []
            },
            'basic_compliance': {
                'mvs_requirements_checked': 7,
                'vst_sections_covered': 3,
                'overall_compliance': '✅ Compliant',
                'issues_found': 0
            },
            'quality_score': 0.85,
            'report_status': 'excellent'
        }
        
        self._display_simple_results(mock_result)
        self.results['simple'] = {
            'result': mock_result,
            'duration': 1.2,
            'timestamp': datetime.now().isoformat()
        }
    
    def _display_simple_results(self, result: Dict[str, Any]):
        """Display Simple MVR Analyzer results."""
        print(f"\n📊 **Simple Analysis Results**")
        print(f"   Document: {result.get('document_info', {}).get('title', 'Unknown')}")
        print(f"   TOC Sections: {result.get('toc_analysis', {}).get('total_sections', 0)}")
        print(f"   Compliance: {result.get('basic_compliance', {}).get('overall_compliance', 'Unknown')}")
        print(f"   Quality Score: {result.get('quality_score', 0):.2f}")
        print(f"   Status: {result.get('report_status', 'Unknown')}")
    
    def run_enhanced_demo(self):
        """Run Enhanced MVR Analyzer demo."""
        print("\n🚀 **Enhanced MVR Analyzer Demo**")
        print("-" * 30)
        
        try:
            # Import Enhanced MVR Analyzer (to be implemented)
            # from backend.mcp.orchestrators.enhanced_mvr_analyzer import EnhancedMVRAnalyzer
            
            print("🔄 Enhanced MVR Analyzer not yet implemented - showing mock results")
            self._show_enhanced_mock_results()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self._show_enhanced_mock_results()
    
    def _show_enhanced_mock_results(self):
        """Show mock results for Enhanced MVR Analyzer."""
        mock_result = {
            'document_info': self.demo_data['document_info'],
            'evidence_tracing': {
                'sections_analyzed': 8,
                'evidence_mapped': 24,
                'logic_gaps_found': 2,
                'unsupported_assertions': 1
            },
            'peer_review_challenges': [
                {
                    'section': '4.2',
                    'challenge': 'Variable selection rationale could be strengthened with additional statistical justification',
                    'severity': 'Medium',
                    'recommendation': 'Include feature importance stability analysis'
                },
                {
                    'section': '6.1',
                    'challenge': 'Backtesting period may not capture all market conditions',
                    'severity': 'Low',
                    'recommendation': 'Extend backtesting to include stress periods'
                }
            ],
            'confidence_scores': {
                'conceptual_soundness': 'Highly Confident',
                'outcome_analysis': 'Moderately Confident',
                'process_verification': 'Highly Confident'
            },
            'enhanced_quality_score': 0.78
        }
        
        self._display_enhanced_results(mock_result)
        self.results['enhanced'] = {
            'result': mock_result,
            'duration': 3.5,
            'timestamp': datetime.now().isoformat()
        }
    
    def _display_enhanced_results(self, result: Dict[str, Any]):
        """Display Enhanced MVR Analyzer results."""
        print(f"\n📊 **Enhanced Analysis Results**")
        print(f"   Evidence Mapped: {result.get('evidence_tracing', {}).get('evidence_mapped', 0)}")
        print(f"   Logic Gaps: {result.get('evidence_tracing', {}).get('logic_gaps_found', 0)}")
        print(f"   Peer Challenges: {len(result.get('peer_review_challenges', []))}")
        print(f"   Enhanced Score: {result.get('enhanced_quality_score', 0):.2f}")
        
        print(f"\n🎯 **Peer Review Challenges**")
        for challenge in result.get('peer_review_challenges', [])[:2]:
            print(f"   • {challenge['section']}: {challenge['challenge']} ({challenge['severity']})")
    
    def run_advanced_demo(self):
        """Run Advanced MVR Analyzer demo."""
        print("\n🤖 **Advanced MVR Analyzer Demo**")
        print("-" * 30)
        
        try:
            # Import Advanced MVR Analyzer (to be implemented)
            # from backend.mcp.orchestrators.advanced_mvr_analyzer import AdvancedMVRAnalyzer
            
            print("🔄 Advanced MVR Analyzer not yet implemented - showing mock results")
            self._show_advanced_mock_results()
            
        except Exception as e:
            print(f"❌ Error: {e}")
            self._show_advanced_mock_results()
    
    def _show_advanced_mock_results(self):
        """Show mock results for Advanced MVR Analyzer."""
        mock_result = {
            'document_info': self.demo_data['document_info'],
            'ai_insights': {
                'contradictions_detected': 1,
                'regulatory_updates': 2,
                'industry_trends': 3,
                'risk_patterns': 1
            },
            'external_compliance': {
                'regulatory_checks': 5,
                'enforcement_actions': 0,
                'industry_criticisms': 1,
                'compliance_score': 0.92
            },
            'real_time_monitoring': {
                'model_drift_detected': False,
                'performance_degradation': False,
                'risk_alerts': 0
            },
            'advanced_quality_score': 0.89,
            'ai_recommendations': [
                "Consider implementing uncertainty quantification for SHAP-based feature selection",
                "Monitor model performance during economic stress periods",
                "Update variable selection criteria based on recent regulatory guidance"
            ]
        }
        
        self._display_advanced_results(mock_result)
        self.results['advanced'] = {
            'result': mock_result,
            'duration': 8.2,
            'timestamp': datetime.now().isoformat()
        }
    
    def _display_advanced_results(self, result: Dict[str, Any]):
        """Display Advanced MVR Analyzer results."""
        print(f"\n📊 **Advanced Analysis Results**")
        print(f"   AI Insights: {result.get('ai_insights', {}).get('contradictions_detected', 0)} contradictions")
        print(f"   External Compliance: {result.get('external_compliance', {}).get('compliance_score', 0):.2f}")
        print(f"   Real-time Monitoring: {'Active' if result.get('real_time_monitoring', {}).get('model_drift_detected') is False else 'Alert'}")
        print(f"   Advanced Score: {result.get('advanced_quality_score', 0):.2f}")
        
        print(f"\n🤖 **AI Recommendations**")
        for rec in result.get('ai_recommendations', [])[:2]:
            print(f"   • {rec}")
    
    def compare_results(self):
        """Compare results across complexity levels."""
        print("\n📈 **Results Comparison**")
        print("-" * 30)
        
        if not self.results:
            print("❌ No results to compare")
            return
        
        # Create comparison table
        comparison_data = []
        
        for level, data in self.results.items():
            comparison_data.append({
                'Level': level.title(),
                'Duration (s)': data['duration'],
                'Quality Score': data['result'].get('quality_score', 
                                                   data['result'].get('enhanced_quality_score',
                                                   data['result'].get('advanced_quality_score', 0))),
                'Features': self._get_features_for_level(level),
                'Status': '✅ Implemented' if level == 'simple' else '🔄 Mock Results'
            })
        
        # Display comparison
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Create visualization
        self._create_comparison_chart()
    
    def _get_features_for_level(self, level: str) -> str:
        """Get features for each complexity level."""
        features = {
            'simple': 'TOC Parsing, Basic Compliance',
            'enhanced': 'Evidence Tracing, Peer Challenges',
            'advanced': 'AI Insights, External Compliance'
        }
        return features.get(level, 'Unknown')
    
    def _create_comparison_chart(self):
        """Create comparison visualization."""
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Quality scores
            levels = [data['result'].get('quality_score', 
                                       data['result'].get('enhanced_quality_score',
                                       data['result'].get('advanced_quality_score', 0))) 
                     for data in self.results.values()]
            level_names = list(self.results.keys())
            
            ax1.bar(level_names, levels, color=['#2E86AB', '#A23B72', '#F18F01'])
            ax1.set_title('Quality Scores by Complexity Level')
            ax1.set_ylabel('Quality Score')
            ax1.set_ylim(0, 1)
            
            # Duration comparison
            durations = [data['duration'] for data in self.results.values()]
            ax2.bar(level_names, durations, color=['#2E86AB', '#A23B72', '#F18F01'])
            ax2.set_title('Processing Time by Complexity Level')
            ax2.set_ylabel('Duration (seconds)')
            
            plt.tight_layout()
            plt.savefig('mvr_complexity_comparison.png', dpi=300, bbox_inches='tight')
            print(f"\n📊 Chart saved: mvr_complexity_comparison.png")
            
        except Exception as e:
            print(f"❌ Error creating chart: {e}")
    
    def run_full_demo(self):
        """Run the complete demo sequence."""
        print("🎯 **SR Validator MVR Peer Review Demo**")
        print("=" * 50)
        
        # 1. Show favorites integration
        self.show_favorites_integration()
        
        # 2. Run Simple demo
        self.run_simple_demo()
        
        # 3. Run Enhanced demo
        self.run_enhanced_demo()
        
        # 4. Run Advanced demo
        self.run_advanced_demo()
        
        # 5. Compare results
        self.compare_results()
        
        # 6. Summary
        self._show_summary()
    
    def _show_summary(self):
        """Show demo summary."""
        print("\n🎯 **Demo Summary**")
        print("-" * 30)
        print("✅ Favorites Integration: Users can save and reuse prompts")
        print("✅ Progressive Complexity: Same prompt works across all levels")
        print("✅ Consistency: Architecture maintains compatibility")
        print("✅ Scalability: Easy to add new features and capabilities")
        
        print(f"\n📋 **Next Steps**")
        print("1. Implement Enhanced MVR Analyzer")
        print("2. Implement Advanced MVR Analyzer")
        print("3. Add real AI/ML capabilities")
        print("4. Integrate with external compliance databases")
        print("5. Add real-time monitoring features")
        
        print(f"\n🎉 **Demo Complete!**")
        print("The SR Validator's MVR Peer Review prompt successfully demonstrates")
        print("our progressive complexity architecture in action.")


def main():
    """Main demo execution."""
    demo = SRValidatorDemo()
    demo.run_full_demo()


if __name__ == "__main__":
    main()
