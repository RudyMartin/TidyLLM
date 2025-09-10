#!/usr/bin/env python3
"""
Hierarchical Domain RAG Builder for tidyllm-compliance
=====================================================

Builds tiered regulatory knowledge systems with built-in precedence hierarchy:

Tier 1 - AUTHORITATIVE: Checklist folder (regulatory requirements)
Tier 2 - STANDARD: SOP folder (standard operating procedures)  
Tier 3 - TECHNICAL: Modeling folder (technical guidance)

Features:
- Automatic precedence resolution for conflicting guidance
- YRSN noise analysis integration for quality validation
- S3-first architecture for corporate deployment
- Cross-domain conflict detection and resolution
- Evidence validation for document authenticity

Part of tidyllm-compliance: Professional regulatory compliance platform
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# Add tidyllm admin directory for credential management
sys.path.append(str(Path(__file__).parent.parent.parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent.parent.parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()

import boto3

# Import compliance validation components
try:
    from ..sop_conflict_analysis.yrsn_analyzer import YRSNNoiseAnalyzer
    from ..evidence.validation import EvidenceValidator
    from ..sop_golden_answers.sop_validator import SOPValidator
except ImportError:
    # Fallback for standalone usage
    print("[WARNING] Running without full compliance validation - install tidyllm-compliance for complete functionality")
    YRSNNoiseAnalyzer = None
    EvidenceValidator = None
    SOPValidator = None

class HierarchicalDomainRAGBuilder:
    """
    Build hierarchical domain RAG system with regulatory precedence.
    
    Architecture:
    - Tier 1 (Authoritative): Checklist folder - Regulatory requirements
    - Tier 2 (Standard): SOP folder - Standard operating procedures
    - Tier 3 (Technical): Modeling folder - Technical guidance
    
    All tiers integrate YRSN validation and evidence assessment.
    """
    
    def __init__(self, 
                 bucket_name: str = "nsc-mvp1",
                 knowledge_base_prefix: str = "knowledge_base",
                 enable_compliance_validation: bool = True):
        
        self.bucket_name = bucket_name
        self.kb_prefix = knowledge_base_prefix
        # AUDIT COMPLIANCE: Use UnifiedSessionManager instead of direct boto3
        try:
            from tidyllm.infrastructure.session.unified import UnifiedSessionManager
            session_manager = UnifiedSessionManager()
            self.s3_client = session_manager.get_s3_client()
        except ImportError:
            # NO FALLBACK - UnifiedSessionManager is required
            raise RuntimeError("HierarchicalBuilder: UnifiedSessionManager is required for S3 access")
        
        # Initialize compliance validators
        self.yrsn_analyzer = YRSNNoiseAnalyzer() if YRSNNoiseAnalyzer and enable_compliance_validation else None
        self.evidence_validator = EvidenceValidator() if EvidenceValidator and enable_compliance_validation else None
        self.sop_validator = SOPValidator() if SOPValidator and enable_compliance_validation else None
        
        # Define hierarchy configuration
        self.hierarchy_config = {
            'authoritative': {
                'tier': 1,
                'precedence': 1.0,
                'folder': 'checklist',
                's3_prefix': f"{knowledge_base_prefix}/checklist/",
                'authority_level': 'regulatory_requirement',
                'description': 'Regulatory checklists and mandated requirements',
                'conflict_resolution': 'highest_precedence'
            },
            'standard': {
                'tier': 2, 
                'precedence': 0.8,
                'folder': 'sop',
                's3_prefix': f"{knowledge_base_prefix}/sop/",
                'authority_level': 'standard_procedure',
                'description': 'Standard operating procedures and established workflows',
                'conflict_resolution': 'temporal_with_precedence'
            },
            'technical': {
                'tier': 3,
                'precedence': 0.6,
                'folder': 'modeling',
                's3_prefix': f"{knowledge_base_prefix}/modeling/",
                'authority_level': 'technical_guidance',
                'description': 'Technical modeling guidance and reference materials',
                'conflict_resolution': 'supplementary_only'
            }
        }
        
        print(f"[DOMAIN_RAG] Initialized hierarchical builder for bucket: {bucket_name}")
        print(f"[DOMAIN_RAG] Knowledge base prefix: {knowledge_base_prefix}")
        if self.yrsn_analyzer:
            print(f"[COMPLIANCE] YRSN validation enabled")
        if self.evidence_validator:
            print(f"[COMPLIANCE] Evidence validation enabled")
    
    def query_hierarchical_guidance(self, query: str, domain: str) -> Dict[str, Any]:
        """
        Query hierarchical domain RAG with precedence-based conflict resolution.
        
        Args:
            query: The compliance query
            domain: Domain context (e.g., 'model_validation', 'risk_management')
            
        Returns:
            Hierarchical guidance response with precedence resolution
        """
        print(f"\n{'='*60}")
        print(f"HIERARCHICAL DOMAIN RAG QUERY")
        print(f"Query: {query}")
        print(f"Domain: {domain}")
        print(f"{'='*60}")
        
        # Step 1: Search each tier for relevant guidance
        tier_results = {}
        
        for tier_name, config in self.hierarchy_config.items():
            print(f"\n[TIER_{config['tier']}] Searching {tier_name} level ({config['folder']})...")
            tier_results[tier_name] = self._search_tier(query, domain, config)
        
        # Step 2: Apply precedence resolution
        resolved_guidance = self._resolve_hierarchical_conflicts(tier_results, query)
        
        # Step 3: Comprehensive compliance validation
        compliance_validation = None
        if any([self.yrsn_analyzer, self.evidence_validator, self.sop_validator]):
            compliance_validation = self._validate_hierarchical_guidance(
                resolved_guidance, query, tier_results
            )
        
        # Step 4: Generate final response
        response = {
            'query': query,
            'domain': domain,
            'timestamp': datetime.now().isoformat(),
            'hierarchy_applied': True,
            
            # Hierarchical results
            'guidance_hierarchy': {
                'primary_guidance': resolved_guidance['primary'],
                'supporting_guidance': resolved_guidance['supporting'],
                'precedence_resolution': resolved_guidance['resolution_method']
            },
            
            # Tier-by-tier results
            'tier_results': tier_results,
            
            # Compliance validation
            'compliance_validation': compliance_validation,
            
            # Metadata
            'processing_metadata': {
                'hierarchy_config_used': self.hierarchy_config,
                'total_tiers_searched': len(tier_results),
                's3_bucket': self.bucket_name,
                'knowledge_base_prefix': self.kb_prefix
            }
        }
        
        print(f"\n[COMPLETE] Hierarchical guidance resolution complete")
        print(f"Primary guidance source: {resolved_guidance['primary']['source']}")
        print(f"Resolution method: {resolved_guidance['resolution_method']}")
        
        return response
    
    def _search_tier(self, query: str, domain: str, tier_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search a specific tier for relevant guidance.
        
        Args:
            query: Compliance query
            domain: Domain context
            tier_config: Configuration for this tier
            
        Returns:
            Tier search results with YRSN validation
        """
        try:
            # List documents in this tier's S3 prefix
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=tier_config['s3_prefix']
            )
            
            if 'Contents' not in response:
                return {
                    'tier_name': tier_config['folder'],
                    'authority_level': tier_config['authority_level'],
                    'documents_found': 0,
                    'guidance_results': [],
                    'tier_available': False
                }
            
            # Process documents in this tier
            guidance_results = []
            documents_processed = 0
            
            for obj in response['Contents']:
                if obj['Key'].endswith('.pdf') or obj['Key'].endswith('.txt') or obj['Key'].endswith('.md'):
                    try:
                        # Load document content
                        doc_response = self.s3_client.get_object(Bucket=self.bucket_name, Key=obj['Key'])
                        content = doc_response['Body'].read().decode('utf-8', errors='ignore')
                        
                        # Extract relevant guidance (simplified for demo)
                        if self._is_relevant_to_query(content, query):
                            guidance_summary = self._extract_guidance_summary(content, query)
                            
                            # YRSN validation if available
                            yrsn_score = None
                            if self.yrsn_analyzer:
                                yrsn_result = self.yrsn_analyzer.analyze_guidance_quality([guidance_summary], query)
                                yrsn_score = yrsn_result.noise_percentage
                            
                            guidance_results.append({
                                'document_key': obj['Key'],
                                'guidance_summary': guidance_summary,
                                'yrsn_noise_score': yrsn_score,
                                'document_size': obj['Size'],
                                'last_modified': obj['LastModified'].isoformat(),
                                'authority_level': tier_config['authority_level']
                            })
                            
                        documents_processed += 1
                        
                        # Limit processing for demo
                        if documents_processed >= 5:
                            break
                            
                    except Exception as e:
                        print(f"[WARNING] Failed to process {obj['Key']}: {e}")
                        continue
            
            return {
                'tier_name': tier_config['folder'],
                'tier_number': tier_config['tier'],
                'authority_level': tier_config['authority_level'],
                'precedence_score': tier_config['precedence'],
                'documents_found': len([obj for obj in response['Contents'] 
                                      if not obj['Key'].endswith('/')]),
                'documents_processed': documents_processed,
                'guidance_results': guidance_results,
                'tier_available': len(guidance_results) > 0
            }
            
        except Exception as e:
            return {
                'tier_name': tier_config['folder'],
                'authority_level': tier_config['authority_level'],
                'error': str(e),
                'tier_available': False,
                'documents_found': 0,
                'guidance_results': []
            }
    
    def _resolve_hierarchical_conflicts(self, tier_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Resolve conflicts between tiers using precedence hierarchy.
        
        Args:
            tier_results: Results from all tiers
            query: Original query for context
            
        Returns:
            Resolved guidance with precedence explanation
        """
        # Find tiers with available guidance
        available_tiers = [(name, results) for name, results in tier_results.items() 
                          if results.get('tier_available', False)]
        
        if not available_tiers:
            return {
                'primary': {
                    'source': 'none',
                    'guidance': 'No guidance available across all tiers',
                    'authority_level': 'none'
                },
                'supporting': [],
                'resolution_method': 'no_guidance_available'
            }
        
        # Sort by precedence (highest first)
        available_tiers.sort(key=lambda x: self.hierarchy_config[x[0]]['precedence'], reverse=True)
        
        # Primary guidance from highest precedence tier
        primary_tier_name, primary_results = available_tiers[0]
        primary_guidance = primary_results['guidance_results'][0] if primary_results['guidance_results'] else None
        
        # Supporting guidance from lower tiers (if available)
        supporting_guidance = []
        for tier_name, results in available_tiers[1:]:
            if results['guidance_results']:
                supporting_guidance.extend([
                    {
                        'tier': tier_name,
                        'authority_level': results['authority_level'],
                        'guidance': result['guidance_summary'],
                        'yrsn_score': result.get('yrsn_noise_score'),
                        'precedence_score': self.hierarchy_config[tier_name]['precedence']
                    }
                    for result in results['guidance_results'][:2]  # Max 2 per tier
                ])
        
        resolution_method = f"hierarchical_precedence_tier_{self.hierarchy_config[primary_tier_name]['tier']}"
        
        if not primary_guidance:
            resolution_method = 'no_primary_guidance_fallback_to_available'
            # Use best available from any tier
            for _, results in available_tiers:
                if results['guidance_results']:
                    primary_guidance = results['guidance_results'][0]
                    break
        
        return {
            'primary': {
                'source': primary_tier_name,
                'guidance': primary_guidance['guidance_summary'] if primary_guidance else 'No specific guidance available',
                'authority_level': primary_results['authority_level'],
                'yrsn_score': primary_guidance.get('yrsn_noise_score') if primary_guidance else None,
                'precedence_score': self.hierarchy_config[primary_tier_name]['precedence']
            },
            'supporting': supporting_guidance,
            'resolution_method': resolution_method,
            'total_guidance_sources': len(available_tiers)
        }
    
    def _validate_hierarchical_guidance(self, 
                                      resolved_guidance: Dict[str, Any], 
                                      query: str, 
                                      tier_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive compliance validation of hierarchical guidance.
        
        Args:
            resolved_guidance: The resolved guidance from precedence resolution
            query: Original query
            tier_results: All tier search results
            
        Returns:
            Comprehensive compliance validation results
        """
        validation_results = {
            'validation_timestamp': datetime.now().isoformat(),
            'query_validated': query
        }
        
        # YRSN validation across all tiers
        if self.yrsn_analyzer:
            tier_yrsn_scores = []
            for tier_name, results in tier_results.items():
                if results.get('tier_available') and results.get('guidance_results'):
                    for guidance in results['guidance_results']:
                        if guidance.get('yrsn_noise_score') is not None:
                            tier_yrsn_scores.append({
                                'tier': tier_name,
                                'score': guidance['yrsn_noise_score'],
                                'authority_level': guidance['authority_level']
                            })
            
            validation_results['yrsn_validation'] = {
                'tier_scores': tier_yrsn_scores,
                'average_noise_score': sum(s['score'] for s in tier_yrsn_scores) / len(tier_yrsn_scores) if tier_yrsn_scores else None,
                'best_tier_score': min(s['score'] for s in tier_yrsn_scores) if tier_yrsn_scores else None,
                'validation_status': 'PASS' if tier_yrsn_scores and min(s['score'] for s in tier_yrsn_scores) < 50 else 'NEEDS_REVIEW'
            }
        
        # Evidence validation for primary guidance
        if self.evidence_validator and resolved_guidance['primary']['guidance'] != 'No specific guidance available':
            evidence_result = self.evidence_validator.validate_document(resolved_guidance['primary']['guidance'])
            validation_results['evidence_validation'] = {
                'primary_guidance_validity': evidence_result['overall_validity'],
                'authenticity_score': evidence_result['authenticity_score'],
                'completeness_score': evidence_result['completeness_score'],
                'quality_score': evidence_result['quality_score']
            }
        
        # Overall compliance determination
        compliance_issues = []
        if validation_results.get('yrsn_validation', {}).get('validation_status') == 'NEEDS_REVIEW':
            compliance_issues.append('high_yrsn_noise_detected')
        
        if validation_results.get('evidence_validation', {}).get('primary_guidance_validity') in ['low_confidence', 'insufficient_evidence']:
            compliance_issues.append('evidence_validation_concerns')
        
        validation_results['overall_compliance_status'] = 'COMPLIANT' if not compliance_issues else f"REVIEW_REQUIRED: {', '.join(compliance_issues)}"
        
        return validation_results
    
    def _is_relevant_to_query(self, content: str, query: str) -> bool:
        """Simple relevance check for guidance content."""
        query_words = query.lower().split()
        content_lower = content.lower()
        
        # Check if query terms appear in content
        matches = sum(1 for word in query_words if word in content_lower)
        return matches >= len(query_words) * 0.3  # 30% term overlap
    
    def _extract_guidance_summary(self, content: str, query: str) -> str:
        """
        Extract relevant guidance summary from document content.
        
        Args:
            content: Full document content
            query: Query context for extraction
            
        Returns:
            Summarized guidance relevant to query
        """
        # Simple extraction - find paragraphs containing query terms
        query_terms = query.lower().split()
        paragraphs = content.split('\n\n')
        
        relevant_paragraphs = []
        for para in paragraphs:
            para_lower = para.lower()
            if any(term in para_lower for term in query_terms):
                relevant_paragraphs.append(para.strip())
        
        if relevant_paragraphs:
            # Return first few relevant paragraphs
            summary = ' '.join(relevant_paragraphs[:3])
            return summary[:500] + '...' if len(summary) > 500 else summary
        else:
            # Fallback to document beginning
            return content[:300] + '...' if len(content) > 300 else content
    
    def get_hierarchy_status(self) -> Dict[str, Any]:
        """
        Get current status of hierarchical domain RAG system.
        
        Returns:
            Status information for all tiers
        """
        status = {
            'system_status': 'active',
            'bucket': self.bucket_name,
            'knowledge_base_prefix': self.kb_prefix,
            'compliance_validation_enabled': {
                'yrsn_analysis': self.yrsn_analyzer is not None,
                'evidence_validation': self.evidence_validator is not None,
                'sop_validation': self.sop_validator is not None
            },
            'hierarchy_status': {},
            'checked_at': datetime.now().isoformat()
        }
        
        # Check each tier
        for tier_name, config in self.hierarchy_config.items():
            try:
                response = self.s3_client.list_objects_v2(
                    Bucket=self.bucket_name,
                    Prefix=config['s3_prefix']
                )
                
                document_count = len([obj for obj in response.get('Contents', []) 
                                    if not obj['Key'].endswith('/')])
                
                status['hierarchy_status'][tier_name] = {
                    'tier_number': config['tier'],
                    'precedence_score': config['precedence'],
                    'authority_level': config['authority_level'],
                    'document_count': document_count,
                    's3_prefix': config['s3_prefix'],
                    'status': 'available' if document_count > 0 else 'empty'
                }
                
            except Exception as e:
                status['hierarchy_status'][tier_name] = {
                    'tier_number': config['tier'],
                    'status': 'error',
                    'error_message': str(e)
                }
        
        return status

# Example usage and demonstration
def demo_hierarchical_domain_rag():
    """
    Demonstrate hierarchical domain RAG with compliance validation.
    """
    builder = HierarchicalDomainRAGBuilder(
        bucket_name="nsc-mvp1",
        knowledge_base_prefix="knowledge_base",
        enable_compliance_validation=True
    )
    
    # Show system status
    print("\nHierarchical Domain RAG Status:")
    status = builder.get_hierarchy_status()
    
    for tier_name, tier_status in status['hierarchy_status'].items():
        print(f"\nTier {tier_status.get('tier_number', '?')}: {tier_name.title()}")
        print(f"  Authority Level: {tier_status.get('authority_level', 'unknown')}")
        print(f"  Document Count: {tier_status.get('document_count', 0)}")
        print(f"  Status: {tier_status.get('status', 'unknown')}")
    
    # Example queries
    test_queries = [
        "What are the model validation requirements for regulatory compliance?",
        "How should risk management procedures be documented?", 
        "What is the process for ongoing model monitoring?"
    ]
    
    print(f"\nTesting hierarchical guidance resolution...")
    
    for i, query in enumerate(test_queries[:1], 1):  # Test first query only for demo
        print(f"\n[TEST {i}] Query: {query}")
        
        try:
            result = builder.query_hierarchical_guidance(query, "model_validation")
            
            print(f"\nResults Summary:")
            print(f"  Primary Source: {result['guidance_hierarchy']['primary_guidance']['source']}")
            print(f"  Authority Level: {result['guidance_hierarchy']['primary_guidance']['authority_level']}")
            print(f"  Resolution Method: {result['guidance_hierarchy']['precedence_resolution']}")
            
            if result.get('compliance_validation'):
                compliance = result['compliance_validation']
                print(f"  Compliance Status: {compliance.get('overall_compliance_status')}")
                
                if compliance.get('yrsn_validation'):
                    yrsn = compliance['yrsn_validation']
                    print(f"  YRSN Status: {yrsn.get('validation_status')}")
                    if yrsn.get('average_noise_score'):
                        print(f"  Average Noise Score: {yrsn['average_noise_score']:.1f}%")
            
        except Exception as e:
            print(f"  Error: {e}")
            print(f"  Note: This demo requires S3 bucket access and documents")

if __name__ == "__main__":
    demo_hierarchical_domain_rag()