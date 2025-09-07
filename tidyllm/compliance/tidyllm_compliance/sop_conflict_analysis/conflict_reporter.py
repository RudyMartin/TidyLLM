"""
SOP Conflict Reporter Implementation
===================================

Compliance-owned conflict detection and resolution reporting.
Integrates YRSN analysis for regulatory-grade guidance validation.

Part of tidyllm-compliance: Automated compliance with complete transparency
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import csv

from .yrsn_analyzer import YRSNNoiseAnalyzer
from .fallback_strategy import ComplianceSOPFallback

class SOPConflictReporter:
    """Compliance-owned SOP conflict detection and reporting system"""
    
    def __init__(self, output_dir: str = "sop_conflict_reports", base_path: str = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize compliance components
        self.yrsn_analyzer = YRSNNoiseAnalyzer()
        self.fallback_strategy = ComplianceSOPFallback(base_path)
        self.fallback_strategy.yrsn_analyzer = self.yrsn_analyzer
        
        print("=" * 60)
        print("COMPLIANCE SOP CONFLICT REPORTER")
        print("=" * 60)
        print(f"Output directory: {self.output_dir}")
        print(f"Report timestamp: {self.timestamp}")
        print("Compliance validation: YRSN analysis enabled")
        print("=" * 60)
    
    def generate_compliance_report(self, queries: List[str] = None, authoritative_date: str = "2025-09-05") -> Dict[str, Any]:
        """Generate comprehensive compliance conflict report"""
        
        # Default compliance queries
        if not queries:
            queries = [
                'What is the official session management pattern for TidyLLM?',
                'Which embedding system should be used: tidyllm-sentence or tidyllm-vectorqa?'
            ]
        
        compliance_results = []
        
        for query in queries:
            # Retrieve guidance using compliance fallback strategy
            guidance_result = self.fallback_strategy.retrieve_compliant_guidance(query, authoritative_date)
            
            # Validate guidance quality using YRSN analysis
            if guidance_result.guidance_content:
                yrsn_validation = self.yrsn_analyzer.analyze_guidance_quality(
                    guidance_result.guidance_content, 
                    query
                )
                
                compliance_results.append({
                    'query': query,
                    'compliance_status': guidance_result.compliance_status,
                    'guidance_content': guidance_result.guidance_content,
                    'sources': guidance_result.sources,
                    'confidence_level': guidance_result.confidence_level,
                    'retrieval_method': guidance_result.retrieval_method,
                    'yrsn_noise_score': yrsn_validation.noise_percentage,
                    'quality_assessment': yrsn_validation.quality_assessment,
                    'actionable_content_ratio': yrsn_validation.actionable_content_ratio,
                    'specific_guidance_found': yrsn_validation.specific_guidance_found
                })
            else:
                compliance_results.append({
                    'query': query,
                    'compliance_status': 'NO_GUIDANCE_FOUND',
                    'guidance_content': [],
                    'sources': [],
                    'confidence_level': 'NONE',
                    'retrieval_method': 'none',
                    'yrsn_noise_score': 100.0,
                    'quality_assessment': 'COMPLIANCE FAILURE - No guidance found',
                    'actionable_content_ratio': 0.0,
                    'specific_guidance_found': 0
                })
        
        # Generate comprehensive compliance reports
        reports = {
            'compliance_summary': self._generate_compliance_summary(compliance_results),
            'detailed_analysis': self._generate_detailed_analysis(compliance_results),
            'yrsn_validation': self._generate_yrsn_validation_report(compliance_results),
            'recommendations': self._generate_compliance_recommendations(compliance_results)
        }
        
        # Save compliance reports
        self._save_compliance_reports(reports)
        self._generate_compliance_csv(compliance_results)
        self._create_compliance_dashboard(reports)
        
        return reports
    
    def _generate_compliance_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate executive compliance summary"""
        total_queries = len(results)
        compliant_queries = len([r for r in results if r['yrsn_noise_score'] < 50])
        high_risk_queries = len([r for r in results if r['yrsn_noise_score'] >= 70])
        
        return {
            'report_type': 'Compliance Executive Summary',
            'generated_at': datetime.now().isoformat(),
            'compliance_metrics': {
                'total_queries_analyzed': total_queries,
                'compliant_responses': compliant_queries,
                'high_risk_responses': high_risk_queries,
                'overall_compliance_rate': (compliant_queries / total_queries * 100) if total_queries > 0 else 0
            },
            'compliance_status': 'PASS' if high_risk_queries == 0 else 'NEEDS_REVIEW',
            'recommendation': 'Review high-risk responses for compliance' if high_risk_queries > 0 else 'All responses meet compliance standards'
        }
    
    def _generate_detailed_analysis(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate detailed compliance analysis"""
        return {
            'report_type': 'Detailed Compliance Analysis',
            'generated_at': datetime.now().isoformat(),
            'analysis_scope': {
                'queries_processed': len(results),
                'yrsn_validation_enabled': True,
                'fallback_strategy_used': True
            },
            'query_breakdown': [
                {
                    'query_id': f"COMP-{i:03d}",
                    'query': result['query'],
                    'compliance_details': {
                        'status': result['compliance_status'],
                        'yrsn_noise_score': result['yrsn_noise_score'],
                        'quality_assessment': result['quality_assessment'],
                        'actionable_content_ratio': result['actionable_content_ratio'],
                        'guidance_sources': result['sources'],
                        'retrieval_method': result['retrieval_method'],
                        'confidence_level': result['confidence_level']
                    },
                    'guidance_content': result['guidance_content'][:3] if result['guidance_content'] else [],
                    'compliance_recommendation': self._get_compliance_recommendation(result)
                }
                for i, result in enumerate(results, 1)
            ]
        }
    
    def _generate_yrsn_validation_report(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate YRSN validation compliance report"""
        yrsn_scores = [r['yrsn_noise_score'] for r in results]
        
        return {
            'report_type': 'YRSN Validation Report',
            'generated_at': datetime.now().isoformat(),
            'yrsn_metrics': {
                'average_noise_score': sum(yrsn_scores) / len(yrsn_scores) if yrsn_scores else 100,
                'highest_noise_score': max(yrsn_scores) if yrsn_scores else 100,
                'lowest_noise_score': min(yrsn_scores) if yrsn_scores else 100,
                'queries_above_50_percent_noise': len([s for s in yrsn_scores if s > 50]),
                'queries_above_70_percent_noise': len([s for s in yrsn_scores if s > 70])
            },
            'validation_details': [
                {
                    'query': result['query'],
                    'noise_score': result['yrsn_noise_score'],
                    'quality_level': self._categorize_quality(result['yrsn_noise_score']),
                    'actionable_ratio': result['actionable_content_ratio'],
                    'specific_guidance_count': result['specific_guidance_found']
                }
                for result in results
            ]
        }
    
    def _generate_compliance_recommendations(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate compliance recommendations"""
        high_noise_results = [r for r in results if r['yrsn_noise_score'] >= 70]
        
        recommendations = []
        
        if high_noise_results:
            recommendations.append({
                'priority': 'HIGH',
                'action': 'Review High-Noise Responses',
                'description': f'{len(high_noise_results)} queries have >70% noise and require guidance improvement',
                'affected_queries': [r['query'] for r in high_noise_results]
            })
        
        no_guidance_results = [r for r in results if r['compliance_status'] == 'NO_GUIDANCE_FOUND']
        if no_guidance_results:
            recommendations.append({
                'priority': 'CRITICAL',
                'action': 'Provide Missing Guidance',
                'description': f'{len(no_guidance_results)} queries have no guidance available',
                'affected_queries': [r['query'] for r in no_guidance_results]
            })
        
        return {
            'report_type': 'Compliance Recommendations',
            'generated_at': datetime.now().isoformat(),
            'recommendations': recommendations,
            'next_steps': [
                'Review all HIGH and CRITICAL priority recommendations',
                'Update SOP documentation to reduce noise scores',
                'Implement automated YRSN monitoring in CI/CD pipeline'
            ]
        }
    
    def _get_compliance_recommendation(self, result: Dict) -> str:
        """Get compliance recommendation for a specific result"""
        noise_score = result['yrsn_noise_score']
        
        if noise_score >= 90:
            return "CRITICAL: No actionable guidance found - immediate SOP update required"
        elif noise_score >= 70:
            return "HIGH RISK: Minimal actionable content - SOP improvement needed"
        elif noise_score >= 50:
            return "MODERATE RISK: Some actionable content - consider guidance enhancement"
        elif noise_score >= 30:
            return "ACCEPTABLE: Good actionable content - monitor for improvements"
        else:
            return "EXCELLENT: High-quality actionable guidance - compliant"
    
    def _categorize_quality(self, noise_score: float) -> str:
        """Categorize quality level based on noise score"""
        if noise_score >= 90:
            return "CRITICAL_FAILURE"
        elif noise_score >= 70:
            return "HIGH_RISK"
        elif noise_score >= 50:
            return "MODERATE_RISK"
        elif noise_score >= 30:
            return "ACCEPTABLE"
        else:
            return "EXCELLENT"
    
    def _save_compliance_reports(self, reports: Dict[str, Any]):
        """Save compliance reports in JSON format"""
        for report_name, report_data in reports.items():
            filename = f"compliance_{report_name}_{self.timestamp}.json"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            print(f"[SAVED] {report_name}: {filename}")
    
    def _generate_compliance_csv(self, results: List[Dict]):
        """Generate CSV report for compliance analysis"""
        filename = f"compliance_analysis_{self.timestamp}.csv"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'Query', 'Compliance_Status', 'YRSN_Noise_Score', 'Quality_Assessment',
                'Actionable_Ratio', 'Guidance_Count', 'Sources', 'Retrieval_Method'
            ])
            
            for result in results:
                writer.writerow([
                    result['query'],
                    result['compliance_status'],
                    result['yrsn_noise_score'],
                    result['quality_assessment'],
                    result['actionable_content_ratio'],
                    result['specific_guidance_found'],
                    '; '.join(result['sources'][:3]),
                    result['retrieval_method']
                ])
        
        print(f"[SAVED] CSV Analysis: {filename}")
    
    def _create_compliance_dashboard(self, reports: Dict[str, Any]):
        """Create compliance dashboard in markdown format"""
        filename = f"compliance_dashboard_{self.timestamp}.md"
        filepath = self.output_dir / filename
        
        summary = reports['compliance_summary']
        
        dashboard = f"""
# Compliance SOP Analysis Dashboard
## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 📊 Compliance Metrics
- **Queries Analyzed:** {summary['compliance_metrics']['total_queries_analyzed']}
- **Compliant Responses:** {summary['compliance_metrics']['compliant_responses']}
- **High Risk Responses:** {summary['compliance_metrics']['high_risk_responses']}
- **Overall Compliance Rate:** {summary['compliance_metrics']['overall_compliance_rate']:.1f}%

## 🚨 Compliance Status: {summary['compliance_status']}

## 📋 Recommendation
{summary['recommendation']}

## 🎯 Next Steps
Review detailed analysis reports for specific guidance improvements.
"""
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(dashboard)
        
        print(f"[SAVED] Dashboard: {filename}")


def main():
    """Generate compliance SOP conflict reports"""
    reporter = SOPConflictReporter()
    reports = reporter.generate_compliance_report()
    
    print(f"\n{'='*60}")
    print("COMPLIANCE REPORTING COMPLETE")
    print("="*60)
    print(f"Reports generated: {len(reports)} different formats")
    print(f"Output directory: {reporter.output_dir}")
    print(f"Timestamp: {reporter.timestamp}")
    
    return reports


if __name__ == "__main__":
    main()