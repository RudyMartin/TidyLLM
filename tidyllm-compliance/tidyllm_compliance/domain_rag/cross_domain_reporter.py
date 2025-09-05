#!/usr/bin/env python3
"""
Cross-Domain Reporter for tidyllm-compliance
===========================================

Generates comprehensive reports for domain and cross-domain analysis:
1. Domain Reports - Single domain analysis with aggregate YRSN metrics
2. Cross-Domain Reports - Conflict detection across regulatory domains
3. Compliance Summary Reports - Overall regulatory compliance status

Integrates with hierarchical domain RAG system for complete regulatory oversight.
Includes YRSN validation, evidence assessment, and precedence resolution.

Part of tidyllm-compliance: Professional regulatory compliance platform
"""

import json
import csv
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Import compliance validation components
try:
    from ..sop_conflict_analysis.yrsn_analyzer import YRSNNoiseAnalyzer
    from ..evidence.validation import EvidenceValidator
    from .hierarchical_builder import HierarchicalDomainRAGBuilder
except ImportError:
    # Fallback for standalone usage
    print("[WARNING] Running without full compliance validation - install tidyllm-compliance for complete functionality")
    YRSNNoiseAnalyzer = None
    EvidenceValidator = None
    HierarchicalDomainRAGBuilder = None

class CrossDomainReporter:
    """
    Generate comprehensive cross-domain compliance reports.
    
    Features:
    - Domain-specific analysis with YRSN validation
    - Cross-domain conflict detection and resolution
    - Regulatory precedence analysis across domains
    - Evidence validation for document authenticity
    - Comprehensive compliance status reporting
    """
    
    def __init__(self, 
                 bucket_name: str = "nsc-mvp1",
                 knowledge_base_prefix: str = "knowledge_base",
                 output_directory: str = "compliance_reports"):
        
        self.bucket_name = bucket_name
        self.kb_prefix = knowledge_base_prefix
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Initialize hierarchical domain RAG builder
        if HierarchicalDomainRAGBuilder:
            self.domain_builder = HierarchicalDomainRAGBuilder(
                bucket_name=bucket_name,
                knowledge_base_prefix=knowledge_base_prefix,
                enable_compliance_validation=True
            )
        else:
            self.domain_builder = None
            print("[WARNING] Hierarchical Domain RAG not available")
        
        # Initialize validation components
        self.yrsn_analyzer = YRSNNoiseAnalyzer() if YRSNNoiseAnalyzer else None
        self.evidence_validator = EvidenceValidator() if EvidenceValidator else None
        
        print(f"[REPORTER] Initialized cross-domain reporter")
        print(f"[REPORTER] Output directory: {self.output_dir}")
        print(f"[REPORTER] Bucket: {bucket_name}, KB Prefix: {knowledge_base_prefix}")
        
        # Critical disclaimer for YRSN analysis
        self.disclaimer = """
        ================================
        DISCLAIMER: LEADING QUESTIONS AFFECT RATINGS
        ================================
        
        YRSN (Yes/Relevant/Specific/No-fluff) ratings are significantly
        influenced by how questions are phrased:
        
        - Leading questions ("What are the specific requirements for...") 
          tend to produce LOWER noise scores (better ratings)
        - Neutral questions ("Tell me about...") tend to produce 
          HIGHER noise scores (worse ratings)
        
        This is expected behavior - specific questions get specific answers.
        Use consistent question phrasing for comparative analysis.
        """
    
    def generate_domain_report(self, domain: str, 
                             test_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive report for a single domain.
        
        Args:
            domain: Domain name (e.g., 'model_validation', 'risk_management')
            test_queries: Custom queries for testing (optional)
            
        Returns:
            Complete domain analysis report
        """
        print(f"\n{'='*60}")
        print(f"GENERATING DOMAIN REPORT: {domain.upper()}")
        print(f"{'='*60}")
        
        # Default test queries if none provided
        if not test_queries:
            test_queries = self._get_default_queries_for_domain(domain)
        
        domain_report = {
            'report_type': 'domain_analysis',
            'domain': domain,
            'generated_at': datetime.now().isoformat(),
            'total_queries_tested': len(test_queries),
            'queries_and_results': [],
            'aggregates': {},
            'compliance_summary': {},
            'recommendations': [],
            'disclaimer': self.disclaimer
        }
        
        # Process each query through hierarchical domain RAG
        query_results = []
        yrsn_scores = []
        evidence_scores = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n[QUERY {i}/{len(test_queries)}] Processing: {query[:60]}...")
            
            try:
                if self.domain_builder:
                    # Use hierarchical domain RAG
                    result = self.domain_builder.query_hierarchical_guidance(query, domain)
                    
                    query_result = {
                        'query': query,
                        'query_type': self._classify_query_type(query),
                        'hierarchical_result': result,
                        'primary_source': result['guidance_hierarchy']['primary_guidance']['source'],
                        'authority_level': result['guidance_hierarchy']['primary_guidance']['authority_level'],
                        'precedence_resolution': result['guidance_hierarchy']['precedence_resolution']
                    }
                    
                    # Extract compliance metrics
                    if result.get('compliance_validation'):
                        compliance = result['compliance_validation']
                        
                        if compliance.get('yrsn_validation'):
                            yrsn_data = compliance['yrsn_validation']
                            if yrsn_data.get('average_noise_score') is not None:
                                query_result['yrsn_noise_score'] = yrsn_data['average_noise_score']
                                yrsn_scores.append(yrsn_data['average_noise_score'])
                        
                        if compliance.get('evidence_validation'):
                            evidence_data = compliance['evidence_validation']
                            query_result['evidence_validity'] = evidence_data.get('primary_guidance_validity')
                            if evidence_data.get('quality_score') is not None:
                                evidence_scores.append(evidence_data['quality_score'])
                        
                        query_result['compliance_status'] = compliance.get('overall_compliance_status')
                    
                else:
                    # Fallback processing without hierarchical RAG
                    query_result = self._process_query_fallback(query, domain)
                
                query_results.append(query_result)
                
            except Exception as e:
                print(f"[ERROR] Query processing failed: {e}")
                query_results.append({
                    'query': query,
                    'error': str(e),
                    'status': 'failed'
                })
        
        domain_report['queries_and_results'] = query_results
        
        # Calculate aggregates
        if yrsn_scores:
            domain_report['aggregates']['yrsn_metrics'] = {
                'average_noise_score': sum(yrsn_scores) / len(yrsn_scores),
                'best_score': min(yrsn_scores),
                'worst_score': max(yrsn_scores),
                'total_queries_with_yrsn': len(yrsn_scores)
            }
        
        if evidence_scores:
            domain_report['aggregates']['evidence_metrics'] = {
                'average_quality_score': sum(evidence_scores) / len(evidence_scores),
                'total_queries_with_evidence': len(evidence_scores)
            }
        
        # Generate compliance summary
        domain_report['compliance_summary'] = self._generate_compliance_summary(query_results)
        
        # Generate recommendations
        domain_report['recommendations'] = self._generate_domain_recommendations(query_results)
        
        # Save reports
        self._save_domain_report(domain_report)
        
        print(f"\n[COMPLETE] Domain report generated for {domain}")
        print(f"Queries processed: {len(query_results)}")
        if yrsn_scores:
            print(f"Average YRSN noise: {domain_report['aggregates']['yrsn_metrics']['average_noise_score']:.1f}%")
        
        return domain_report
    
    def generate_cross_domain_report(self, domains: List[str],
                                   cross_domain_queries: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate cross-domain conflict analysis report.
        
        Args:
            domains: List of domains to analyze
            cross_domain_queries: Queries that span multiple domains
            
        Returns:
            Cross-domain conflict analysis report
        """
        print(f"\n{'='*60}")
        print(f"GENERATING CROSS-DOMAIN REPORT")
        print(f"Domains: {', '.join(domains)}")
        print(f"{'='*60}")
        
        if not cross_domain_queries:
            cross_domain_queries = self._get_cross_domain_queries(domains)
        
        cross_report = {
            'report_type': 'cross_domain_analysis',
            'domains_analyzed': domains,
            'generated_at': datetime.now().isoformat(),
            'cross_domain_queries': cross_domain_queries,
            'conflict_analysis': [],
            'domain_consistency': {},
            'precedence_conflicts': [],
            'recommendations': [],
            'disclaimer': self.disclaimer
        }
        
        # Analyze each cross-domain query
        for query in cross_domain_queries:
            print(f"\n[CROSS-DOMAIN] Processing: {query[:60]}...")
            
            query_analysis = {
                'query': query,
                'domain_responses': {},
                'conflicts_detected': [],
                'consistency_score': 0.0,
                'precedence_resolution': None
            }
            
            # Get response from each domain
            domain_responses = {}
            for domain in domains:
                try:
                    if self.domain_builder:
                        result = self.domain_builder.query_hierarchical_guidance(query, domain)
                        domain_responses[domain] = {
                            'primary_guidance': result['guidance_hierarchy']['primary_guidance'],
                            'authority_level': result['guidance_hierarchy']['primary_guidance']['authority_level'],
                            'source_tier': result['guidance_hierarchy']['primary_guidance']['source'],
                            'compliance_status': result.get('compliance_validation', {}).get('overall_compliance_status')
                        }
                    else:
                        domain_responses[domain] = self._process_query_fallback(query, domain)
                except Exception as e:
                    domain_responses[domain] = {'error': str(e)}
            
            query_analysis['domain_responses'] = domain_responses
            
            # Detect conflicts and inconsistencies
            conflicts = self._detect_cross_domain_conflicts(domain_responses, query)
            query_analysis['conflicts_detected'] = conflicts
            query_analysis['consistency_score'] = self._calculate_consistency_score(domain_responses)
            
            # Determine precedence resolution
            if conflicts:
                query_analysis['precedence_resolution'] = self._resolve_cross_domain_precedence(domain_responses)
            
            cross_report['conflict_analysis'].append(query_analysis)
        
        # Generate overall consistency metrics
        cross_report['domain_consistency'] = self._calculate_domain_consistency(cross_report['conflict_analysis'])
        
        # Identify precedence conflicts
        cross_report['precedence_conflicts'] = self._identify_precedence_conflicts(cross_report['conflict_analysis'])
        
        # Generate recommendations
        cross_report['recommendations'] = self._generate_cross_domain_recommendations(cross_report)
        
        # Save report
        self._save_cross_domain_report(cross_report)
        
        print(f"\n[COMPLETE] Cross-domain report generated")
        print(f"Domains analyzed: {len(domains)}")
        print(f"Queries processed: {len(cross_domain_queries)}")
        print(f"Conflicts detected: {len(cross_report['precedence_conflicts'])}")
        
        return cross_report
    
    def _get_default_queries_for_domain(self, domain: str) -> List[str]:
        """Get default test queries for a specific domain."""
        query_sets = {
            'model_validation': [
                "What are the model validation requirements for regulatory compliance?",
                "How should model performance be documented and monitored?",
                "What is the process for ongoing model validation?",
                "How should model limitations be assessed and documented?",
                "What are the requirements for independent model validation?"
            ],
            'risk_management': [
                "How should model risk be assessed and managed?",
                "What are the governance requirements for risk oversight?", 
                "How should risk appetite be defined and monitored?",
                "What is the process for risk incident reporting?",
                "How should third-party risk be evaluated?"
            ],
            'default': [
                "What are the key compliance requirements?",
                "How should documentation be maintained?",
                "What is the process for regulatory reporting?",
                "How should governance oversight be implemented?",
                "What are the audit and review requirements?"
            ]
        }
        
        return query_sets.get(domain, query_sets['default'])
    
    def _get_cross_domain_queries(self, domains: List[str]) -> List[str]:
        """Generate queries that span multiple domains."""
        return [
            "How do model validation requirements integrate with risk management processes?",
            "What is the relationship between governance oversight and operational procedures?",
            "How should regulatory reporting coordinate across different functional areas?",
            "What are the cross-functional requirements for audit and compliance?",
            "How do documentation standards apply across different regulatory domains?"
        ]
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query as leading, neutral, or specific for YRSN analysis."""
        query_lower = query.lower()
        
        if any(phrase in query_lower for phrase in ['what are the specific', 'what are the exact', 'list all']):
            return 'leading_specific'
        elif any(phrase in query_lower for phrase in ['how should', 'what is the process', 'what are the requirements']):
            return 'leading_procedural'
        elif any(phrase in query_lower for phrase in ['tell me about', 'describe', 'explain']):
            return 'neutral_exploratory'
        else:
            return 'standard'
    
    def _process_query_fallback(self, query: str, domain: str) -> Dict[str, Any]:
        """Fallback processing when hierarchical domain RAG is not available."""
        return {
            'query': query,
            'domain': domain,
            'status': 'fallback_processing',
            'note': 'Hierarchical domain RAG not available - using fallback processing',
            'yrsn_noise_score': None,
            'evidence_validity': None,
            'compliance_status': 'unknown'
        }
    
    def _detect_cross_domain_conflicts(self, domain_responses: Dict[str, Any], query: str) -> List[Dict[str, Any]]:
        """Detect conflicts between domain responses."""
        conflicts = []
        
        # Simple conflict detection based on different authority levels or guidance
        domain_guidances = {}
        for domain, response in domain_responses.items():
            if 'error' not in response and 'primary_guidance' in response:
                guidance = response['primary_guidance'].get('guidance', '')
                authority = response.get('authority_level', 'unknown')
                domain_guidances[domain] = {'guidance': guidance, 'authority': authority}
        
        # Check for conflicting guidance
        domain_names = list(domain_guidances.keys())
        for i, domain1 in enumerate(domain_names):
            for domain2 in domain_names[i+1:]:
                guidance1 = domain_guidances[domain1]['guidance']
                guidance2 = domain_guidances[domain2]['guidance']
                
                # Simple conflict detection (could be enhanced)
                if len(guidance1) > 50 and len(guidance2) > 50:
                    # Check for contradictory keywords
                    conflict_indicators = self._check_conflict_indicators(guidance1, guidance2)
                    if conflict_indicators:
                        conflicts.append({
                            'domains': [domain1, domain2],
                            'conflict_type': 'guidance_contradiction',
                            'conflict_indicators': conflict_indicators,
                            'domain1_authority': domain_guidances[domain1]['authority'],
                            'domain2_authority': domain_guidances[domain2]['authority']
                        })
        
        return conflicts
    
    def _check_conflict_indicators(self, guidance1: str, guidance2: str) -> List[str]:
        """Check for conflicting indicators between two guidance texts."""
        conflict_pairs = [
            (['required', 'mandatory', 'must'], ['optional', 'may', 'recommended']),
            (['approve', 'accept', 'allow'], ['reject', 'deny', 'prohibit']),
            (['increase', 'expand', 'enhance'], ['decrease', 'reduce', 'limit'])
        ]
        
        indicators = []
        guidance1_lower = guidance1.lower()
        guidance2_lower = guidance2.lower()
        
        for positive_terms, negative_terms in conflict_pairs:
            pos_in_1 = any(term in guidance1_lower for term in positive_terms)
            neg_in_2 = any(term in guidance2_lower for term in negative_terms)
            
            pos_in_2 = any(term in guidance2_lower for term in positive_terms)
            neg_in_1 = any(term in guidance1_lower for term in negative_terms)
            
            if (pos_in_1 and neg_in_2) or (pos_in_2 and neg_in_1):
                indicators.append(f"Contradictory terms: {positive_terms[0]} vs {negative_terms[0]}")
        
        return indicators
    
    def _calculate_consistency_score(self, domain_responses: Dict[str, Any]) -> float:
        """Calculate consistency score across domain responses."""
        valid_responses = [resp for resp in domain_responses.values() 
                          if 'error' not in resp and 'primary_guidance' in resp]
        
        if len(valid_responses) < 2:
            return 1.0  # Perfect consistency with single or no responses
        
        # Simple consistency calculation based on authority levels
        authority_levels = [resp.get('authority_level', 'unknown') for resp in valid_responses]
        unique_authorities = set(authority_levels)
        
        # More consistent if all responses come from same authority level
        consistency_score = 1.0 - (len(unique_authorities) - 1) * 0.3
        return max(0.0, min(1.0, consistency_score))
    
    def _resolve_cross_domain_precedence(self, domain_responses: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve precedence conflicts across domains."""
        # Authority level precedence mapping
        authority_precedence = {
            'regulatory_requirement': 1.0,
            'standard_procedure': 0.8,
            'technical_guidance': 0.6,
            'unknown': 0.0
        }
        
        valid_responses = [(domain, resp) for domain, resp in domain_responses.items() 
                          if 'error' not in resp and 'authority_level' in resp]
        
        if not valid_responses:
            return {'resolution_method': 'no_valid_responses', 'winning_domain': None}
        
        # Find highest authority level
        best_domain = None
        best_precedence = -1
        
        for domain, response in valid_responses:
            authority = response.get('authority_level', 'unknown')
            precedence = authority_precedence.get(authority, 0.0)
            
            if precedence > best_precedence:
                best_precedence = precedence
                best_domain = domain
        
        return {
            'resolution_method': 'authority_precedence',
            'winning_domain': best_domain,
            'winning_authority_level': domain_responses[best_domain]['authority_level'] if best_domain else None,
            'precedence_score': best_precedence
        }
    
    def _generate_compliance_summary(self, query_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate compliance summary from query results."""
        total_queries = len(query_results)
        successful_queries = len([r for r in query_results if 'error' not in r])
        
        compliance_statuses = [r.get('compliance_status') for r in query_results 
                             if r.get('compliance_status')]
        compliant_count = len([s for s in compliance_statuses if 'COMPLIANT' in str(s)])
        
        return {
            'total_queries_processed': total_queries,
            'successful_queries': successful_queries,
            'success_rate': successful_queries / total_queries if total_queries > 0 else 0,
            'compliance_rate': compliant_count / len(compliance_statuses) if compliance_statuses else 0,
            'total_with_compliance_status': len(compliance_statuses)
        }
    
    def _generate_domain_recommendations(self, query_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on domain analysis."""
        recommendations = []
        
        # Analyze YRSN scores
        yrsn_scores = [r.get('yrsn_noise_score') for r in query_results 
                      if r.get('yrsn_noise_score') is not None]
        
        if yrsn_scores:
            avg_yrsn = sum(yrsn_scores) / len(yrsn_scores)
            if avg_yrsn > 70:
                recommendations.append(f"HIGH PRIORITY: Average YRSN noise score is {avg_yrsn:.1f}% - Review guidance quality and specificity")
            elif avg_yrsn > 50:
                recommendations.append(f"MEDIUM PRIORITY: Average YRSN noise score is {avg_yrsn:.1f}% - Consider enhancing guidance clarity")
        
        # Analyze compliance status
        compliance_issues = [r.get('compliance_status') for r in query_results 
                           if r.get('compliance_status') and 'REVIEW_REQUIRED' in str(r.get('compliance_status'))]
        
        if compliance_issues:
            recommendations.append(f"COMPLIANCE ATTENTION: {len(compliance_issues)} queries require compliance review")
        
        # Default recommendations
        if not recommendations:
            recommendations.append("Domain analysis complete - no critical issues identified")
        
        return recommendations
    
    def _calculate_domain_consistency(self, conflict_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overall domain consistency metrics."""
        if not conflict_analysis:
            return {'status': 'no_analysis_available'}
        
        consistency_scores = [analysis.get('consistency_score', 0) for analysis in conflict_analysis]
        conflicts_detected = sum(len(analysis.get('conflicts_detected', [])) for analysis in conflict_analysis)
        
        return {
            'average_consistency_score': sum(consistency_scores) / len(consistency_scores) if consistency_scores else 0,
            'total_conflicts_detected': conflicts_detected,
            'queries_with_conflicts': len([a for a in conflict_analysis if a.get('conflicts_detected')]),
            'overall_consistency_rating': 'HIGH' if sum(consistency_scores) / len(consistency_scores) > 0.8 else 
                                        'MEDIUM' if sum(consistency_scores) / len(consistency_scores) > 0.6 else 'LOW'
        }
    
    def _identify_precedence_conflicts(self, conflict_analysis: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify queries with precedence conflicts that need resolution."""
        precedence_conflicts = []
        
        for analysis in conflict_analysis:
            if analysis.get('conflicts_detected'):
                precedence_conflicts.append({
                    'query': analysis['query'],
                    'conflicts': analysis['conflicts_detected'],
                    'proposed_resolution': analysis.get('precedence_resolution'),
                    'consistency_score': analysis.get('consistency_score', 0)
                })
        
        return precedence_conflicts
    
    def _generate_cross_domain_recommendations(self, cross_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations for cross-domain issues."""
        recommendations = []
        
        consistency = cross_report.get('domain_consistency', {})
        conflicts = cross_report.get('precedence_conflicts', [])
        
        if consistency.get('overall_consistency_rating') == 'LOW':
            recommendations.append("CRITICAL: Low cross-domain consistency detected - Review domain alignment and precedence rules")
        
        if len(conflicts) > 0:
            recommendations.append(f"ATTENTION: {len(conflicts)} precedence conflicts require resolution")
        
        if consistency.get('total_conflicts_detected', 0) > 5:
            recommendations.append("HIGH PRIORITY: Multiple domain conflicts detected - Consider governance review")
        
        if not recommendations:
            recommendations.append("Cross-domain analysis complete - domains appear well-aligned")
        
        return recommendations
    
    def _save_domain_report(self, report: Dict[str, Any]) -> None:
        """Save domain report in multiple formats."""
        domain = report['domain']
        
        # JSON report
        json_path = self.output_dir / f"domain_report_{domain}_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # CSV summary
        csv_path = self.output_dir / f"domain_report_{domain}_{self.timestamp}.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Query', 'Primary_Source', 'Authority_Level', 'YRSN_Noise', 'Compliance_Status'])
            
            for result in report['queries_and_results']:
                if 'error' not in result:
                    writer.writerow([
                        result.get('query', ''),
                        result.get('primary_source', ''),
                        result.get('authority_level', ''),
                        result.get('yrsn_noise_score', ''),
                        result.get('compliance_status', '')
                    ])
        
        print(f"[SAVED] Domain report: {json_path}")
        print(f"[SAVED] Domain CSV: {csv_path}")
    
    def _save_cross_domain_report(self, report: Dict[str, Any]) -> None:
        """Save cross-domain report in multiple formats."""
        # JSON report
        json_path = self.output_dir / f"cross_domain_report_{self.timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[SAVED] Cross-domain report: {json_path}")

# Example usage and demonstration
def demo_cross_domain_reporting():
    """
    Demonstrate cross-domain reporting functionality.
    """
    reporter = CrossDomainReporter(
        bucket_name="nsc-mvp1",
        knowledge_base_prefix="knowledge_base",
        output_directory="compliance_reports"
    )
    
    print("\nCross-Domain Reporter Demo")
    print("=" * 40)
    
    # Test domains
    test_domains = ['model_validation', 'risk_management']
    
    try:
        # Generate domain report
        print("\n1. Generating domain report for model_validation...")
        domain_report = reporter.generate_domain_report('model_validation')
        
        if domain_report:
            print(f"Domain report generated with {len(domain_report['queries_and_results'])} queries")
            
            # Show summary
            if domain_report.get('aggregates', {}).get('yrsn_metrics'):
                yrsn = domain_report['aggregates']['yrsn_metrics']
                print(f"Average YRSN noise: {yrsn['average_noise_score']:.1f}%")
        
        # Generate cross-domain report
        print("\n2. Generating cross-domain report...")
        cross_report = reporter.generate_cross_domain_report(test_domains)
        
        if cross_report:
            print(f"Cross-domain report generated for {len(test_domains)} domains")
            consistency = cross_report.get('domain_consistency', {})
            print(f"Overall consistency: {consistency.get('overall_consistency_rating', 'unknown')}")
            print(f"Conflicts detected: {consistency.get('total_conflicts_detected', 0)}")
        
    except Exception as e:
        print(f"Demo failed: {e}")
        print(f"Note: This requires S3 bucket access and domain RAG system")

if __name__ == "__main__":
    demo_cross_domain_reporting()