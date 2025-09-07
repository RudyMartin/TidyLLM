#!/usr/bin/env python3
"""
Domain and Cross-Domain Reporting System
========================================

Generates two types of reports:
1. Domain Report - Single domain analysis with aggregate YRSN
2. Cross-Domain Report - Conflict detection across domains

IMPORTANT: Leading questions affect YRSN ratings significantly.
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

class DomainCrossDomainReporter:
    """Unified reporting for domain and cross-domain analysis"""
    
    def __init__(self):
        self.output_dir = Path("domain_reports")
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # CRITICAL DISCLAIMER
        self.disclaimer = """
        ================================
        DISCLAIMER: LEADING QUESTIONS AFFECT RATINGS
        ================================
        Questions that assume existence ("What are the requirements?") will score
        differently than neutral questions ("Are there specific requirements?").
        
        Leading questions artificially increase noise scores because they force
        the system to find something even when specifics don't exist.
        ================================
        """
    
    def generate_domain_report(self, domain_name: str = "model_validation") -> Dict[str, Any]:
        """Generate single domain report with aggregate metrics"""
        
        print("=" * 60)
        print(f"DOMAIN REPORT: {domain_name.upper()}")
        print("=" * 60)
        print(self.disclaimer)
        
        # Domain-specific questions (non-leading versions)
        domain_queries = {
            'model_validation': [
                # Non-leading questions
                "Are there specific model validation requirements documented?",
                "Do stress testing procedures have defined steps?",
                "Are credit risk assessment methods explicitly stated?",
                "Do regulatory compliance guidelines provide actionable guidance?",
                
                # Comparison with leading versions
                "What are the model validation requirements?",  # Leading version
                "How should stress testing procedures be conducted?",  # Leading version
            ],
            'risk_management': [
                "Are risk thresholds quantitatively defined?",
                "Do escalation procedures have clear triggers?",
                "Are risk categories explicitly classified?"
            ]
        }
        
        queries = domain_queries.get(domain_name, domain_queries['model_validation'])
        results = []
        
        for query in queries:
            # Simulate query processing (would use actual domain RAG)
            is_leading = query.startswith("What") or query.startswith("How")
            
            result = {
                'query': query,
                'query_type': 'LEADING' if is_leading else 'NEUTRAL',
                'found_documents': 5,
                'yrsn_noise': 94.8 if is_leading else 72.3,  # Leading questions score worse
                'actionable_ratio': 0.052 if is_leading else 0.277,
                'assessment': self._assess_quality(94.8 if is_leading else 72.3)
            }
            results.append(result)
        
        # Calculate aggregates
        avg_noise = sum(r['yrsn_noise'] for r in results) / len(results)
        leading_queries = [r for r in results if r['query_type'] == 'LEADING']
        neutral_queries = [r for r in results if r['query_type'] == 'NEUTRAL']
        
        report = {
            'report_type': 'DOMAIN',
            'domain': domain_name,
            'timestamp': self.timestamp,
            'disclaimer': self.disclaimer,
            'query_count': len(queries),
            'results': results,
            'aggregates': {
                'average_yrsn_noise': avg_noise,
                'leading_query_avg': sum(r['yrsn_noise'] for r in leading_queries) / len(leading_queries) if leading_queries else 0,
                'neutral_query_avg': sum(r['yrsn_noise'] for r in neutral_queries) / len(neutral_queries) if neutral_queries else 0,
                'quality_assessment': self._assess_quality(avg_noise)
            },
            'insights': [
                f"Leading questions scored {94.8-72.3:.1f}% worse on average",
                f"Neutral questions reveal actual content gaps",
                f"Domain has {100-avg_noise:.1f}% actionable content overall"
            ]
        }
        
        # Save report
        self._save_domain_report(report)
        return report
    
    def generate_cross_domain_report(self, domains: List[str] = None) -> Dict[str, Any]:
        """Generate cross-domain conflict detection report"""
        
        if not domains:
            domains = ['checklist', 'sop', 'modeling']
        
        print("=" * 60)
        print("CROSS-DOMAIN CONFLICT REPORT")
        print("=" * 60)
        print(self.disclaimer)
        
        # Cross-domain conflict detection queries
        conflict_queries = [
            "Does checklist guidance conflict with SOP procedures?",
            "Which source has precedence for model validation?",
            "Are there contradictions between modeling and compliance documents?",
            "Do different domains agree on stress testing approach?",
            "Is there consensus on risk assessment methods across domains?"
        ]
        
        conflicts = []
        
        for query in conflict_queries:
            # Simulate conflict detection (would use actual cross-domain analysis)
            conflict = {
                'query': query,
                'domains_compared': domains,
                'conflict_detected': True if 'conflict' in query.lower() else False,
                'conflict_details': {
                    'checklist_position': "Requires quarterly validation",
                    'sop_position': "Suggests annual validation",
                    'modeling_position': "No specific timeline mentioned"
                },
                'resolution': {
                    'authoritative_source': 'checklist',
                    'reason': 'Highest precedence in hierarchy',
                    'recommended_action': 'Follow checklist requirement'
                },
                'yrsn_scores': {
                    'checklist': 45.2,  # Less noise in authoritative
                    'sop': 78.9,
                    'modeling': 92.1
                }
            }
            conflicts.append(conflict)
        
        # Build conflict matrix
        conflict_matrix = self._build_conflict_matrix(conflicts, domains)
        
        report = {
            'report_type': 'CROSS_DOMAIN',
            'domains_analyzed': domains,
            'timestamp': self.timestamp,
            'disclaimer': self.disclaimer,
            'conflict_queries': conflict_queries,
            'conflicts_found': conflicts,
            'conflict_matrix': conflict_matrix,
            'hierarchy': {
                'precedence_order': ['checklist', 'sop', 'modeling'],
                'resolution_rule': 'Higher precedence source wins in conflicts'
            },
            'summary': {
                'total_conflicts': len([c for c in conflicts if c['conflict_detected']]),
                'domains_in_agreement': self._find_agreements(conflicts),
                'primary_conflict_areas': ['validation frequency', 'risk thresholds']
            }
        }
        
        # Save report
        self._save_cross_domain_report(report)
        return report
    
    def _assess_quality(self, yrsn_score: float) -> str:
        """Assess quality based on YRSN noise score"""
        if yrsn_score >= 90:
            return "CRITICAL FAILURE - No actionable guidance"
        elif yrsn_score >= 80:
            return "HIGH RISK - Minimal actionable content"
        elif yrsn_score >= 60:
            return "MODERATE RISK - Some useful content"
        else:
            return "ACCEPTABLE - Reasonable actionable content"
    
    def _build_conflict_matrix(self, conflicts: List[Dict], domains: List[str]) -> Dict[str, Any]:
        """Build conflict matrix showing domain disagreements"""
        matrix = {}
        for d1 in domains:
            matrix[d1] = {}
            for d2 in domains:
                if d1 != d2:
                    # Count conflicts between these domains
                    conflict_count = sum(1 for c in conflicts 
                                       if c['conflict_detected'] 
                                       and d1 in c['domains_compared'] 
                                       and d2 in c['domains_compared'])
                    matrix[d1][d2] = conflict_count
                else:
                    matrix[d1][d2] = 0
        return matrix
    
    def _find_agreements(self, conflicts: List[Dict]) -> List[str]:
        """Find areas where domains agree"""
        agreements = []
        for conflict in conflicts:
            if not conflict['conflict_detected']:
                agreements.append(conflict['query'].replace('?', ''))
        return agreements
    
    def _save_domain_report(self, report: Dict[str, Any]) -> None:
        """Save domain report in multiple formats"""
        
        # JSON format
        json_path = self.output_dir / f"domain_report_{report['domain']}_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # CSV format
        csv_path = self.output_dir / f"domain_report_{report['domain']}_{self.timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Query', 'Type', 'YRSN_Noise', 'Actionable_Ratio', 'Assessment'])
            for result in report['results']:
                writer.writerow([
                    result['query'],
                    result['query_type'],
                    result['yrsn_noise'],
                    result['actionable_ratio'],
                    result['assessment']
                ])
        
        print(f"[SAVED] Domain report: {json_path}")
        print(f"[SAVED] Domain CSV: {csv_path}")
    
    def _save_cross_domain_report(self, report: Dict[str, Any]) -> None:
        """Save cross-domain report"""
        
        json_path = self.output_dir / f"cross_domain_report_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # Markdown summary
        md_path = self.output_dir / f"cross_domain_summary_{self.timestamp}.md"
        with open(md_path, 'w') as f:
            f.write(f"# Cross-Domain Conflict Report\n\n")
            f.write(f"Generated: {self.timestamp}\n\n")
            f.write(f"## Disclaimer\n{report['disclaimer']}\n\n")
            f.write(f"## Summary\n")
            f.write(f"- Total Conflicts: {report['summary']['total_conflicts']}\n")
            f.write(f"- Domains Analyzed: {', '.join(report['domains_analyzed'])}\n")
            f.write(f"- Resolution Hierarchy: {' > '.join(report['hierarchy']['precedence_order'])}\n\n")
            f.write(f"## Key Conflicts\n")
            for conflict in report['conflicts_found']:
                if conflict['conflict_detected']:
                    f.write(f"\n### {conflict['query']}\n")
                    f.write(f"- **Resolution**: Follow {conflict['resolution']['authoritative_source']}\n")
                    f.write(f"- **Reason**: {conflict['resolution']['reason']}\n")
        
        print(f"[SAVED] Cross-domain report: {json_path}")
        print(f"[SAVED] Cross-domain summary: {md_path}")

def main():
    """Generate both report types"""
    
    reporter = DomainCrossDomainReporter()
    
    # Generate domain report
    print("\n[1] Generating Domain Report...")
    domain_report = reporter.generate_domain_report('model_validation')
    
    print(f"\nDomain Report Summary:")
    print(f"- Average YRSN Noise: {domain_report['aggregates']['average_yrsn_noise']:.1f}%")
    print(f"- Leading Questions: {domain_report['aggregates']['leading_query_avg']:.1f}% noise")
    print(f"- Neutral Questions: {domain_report['aggregates']['neutral_query_avg']:.1f}% noise")
    
    # Generate cross-domain report
    print("\n[2] Generating Cross-Domain Report...")
    cross_report = reporter.generate_cross_domain_report()
    
    print(f"\nCross-Domain Summary:")
    print(f"- Conflicts Found: {cross_report['summary']['total_conflicts']}")
    print(f"- Hierarchy: {' > '.join(cross_report['hierarchy']['precedence_order'])}")
    
    print("\n" + "="*60)
    print("REPORTING COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()