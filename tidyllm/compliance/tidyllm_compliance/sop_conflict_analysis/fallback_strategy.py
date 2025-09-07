"""
Compliance SOP Fallback Strategy Implementation
==============================================

Compliance-owned fallback system for SOP guidance retrieval.
Integrates with risk management domainRAG for comprehensive coverage.

Part of tidyllm-compliance: Automated compliance with complete transparency
"""

import os
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass

@dataclass
class ComplianceResult:
    """Results from compliance SOP guidance retrieval"""
    guidance_content: List[str]
    sources: List[str]
    confidence_level: str
    retrieval_method: str  # 'primary_sop', 'risk_management_fallback', 'none'
    compliance_status: str

class ComplianceSOPFallback:
    """Compliance-owned fallback system for SOP guidance"""
    
    def __init__(self, base_path: str = None):
        self.base_path = Path(base_path) if base_path else Path(".")
        self.yrsn_analyzer = None  # Will be set by importing module
        
    def retrieve_compliant_guidance(self, query: str, authoritative_date: str = None) -> ComplianceResult:
        """
        Retrieve SOP guidance using compliance-approved fallback strategy
        
        Args:
            query: The compliance query requiring guidance
            authoritative_date: Preferred authoritative date (optional)
            
        Returns:
            ComplianceResult with retrieved guidance and compliance status
        """
        # Step 1: Check primary SOP domain
        primary_result = self._check_primary_sop_domain(query, authoritative_date)
        
        if primary_result.guidance_content:
            return primary_result
            
        # Step 2: Fallback to risk management domainRAG
        fallback_result = self._check_risk_management_fallback(query)
        
        if fallback_result.guidance_content:
            return fallback_result
            
        # Step 3: No guidance found
        return ComplianceResult(
            guidance_content=[],
            sources=[],
            confidence_level='NONE',
            retrieval_method='none',
            compliance_status='GUIDANCE_NOT_FOUND'
        )
    
    def _check_primary_sop_domain(self, query: str, authoritative_date: str = None) -> ComplianceResult:
        """Check primary SOP domain (docs/date structure) for compliance guidance"""
        guidance_content = []
        sources = []
        
        # Default to most recent date if not specified
        if not authoritative_date:
            authoritative_date = self._get_most_recent_date()
            
        docs_path = self.base_path / "docs" / authoritative_date
        
        if not docs_path.exists():
            return ComplianceResult(
                guidance_content=[],
                sources=[f"ERROR: Date folder {authoritative_date} not found"],
                confidence_level='NONE',
                retrieval_method='primary_sop',
                compliance_status='PATH_NOT_FOUND'
            )
        
        keywords = self._extract_compliance_keywords(query)
        
        for doc_file in docs_path.glob("*.md"):
            try:
                with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check if document contains relevant compliance guidance
                if any(keyword.lower() in content.lower() for keyword in keywords):
                    relevant_sections = self._extract_compliance_sections(content, keywords)
                    if relevant_sections:
                        guidance_content.extend(relevant_sections)
                        sources.append(doc_file.name)
                        
            except Exception as e:
                continue
        
        return ComplianceResult(
            guidance_content=guidance_content,
            sources=sources,
            confidence_level='HIGH' if len(sources) > 2 else 'MEDIUM' if sources else 'NONE',
            retrieval_method='primary_sop',
            compliance_status='FOUND' if guidance_content else 'NOT_FOUND'
        )
    
    def _check_risk_management_fallback(self, query: str) -> ComplianceResult:
        """Check risk management domainRAG fallback for compliance guidance"""
        try:
            # Import risk management system
            sys.path.append(str(self.base_path))
            from risk_management_sop_drop_zone import RiskDocumentProcessor
            
            risk_processor = RiskDocumentProcessor()
            
            # Check risk processor for guidance
            guidance_result = risk_processor._check_existing_guidance(query)
            
            if guidance_result:
                return ComplianceResult(
                    guidance_content=[guidance_result.get('content', '')],
                    sources=[f"Risk_Management_{guidance_result.get('source', 'Unknown')}"],
                    confidence_level='MEDIUM',
                    retrieval_method='risk_management_fallback',
                    compliance_status='FOUND'
                )
            else:
                # Try keyword-based search
                keywords = self._extract_compliance_keywords(query)
                for keyword in keywords:
                    fallback_result = risk_processor._check_existing_guidance(keyword)
                    if fallback_result:
                        return ComplianceResult(
                            guidance_content=[fallback_result.get('content', '')],
                            sources=[f"Risk_Management_Keyword_{keyword}"],
                            confidence_level='LOW',
                            retrieval_method='risk_management_fallback',
                            compliance_status='FOUND'
                        )
                        
        except Exception as e:
            # Risk management system unavailable
            pass
        
        return ComplianceResult(
            guidance_content=[],
            sources=[],
            confidence_level='NONE',
            retrieval_method='risk_management_fallback',
            compliance_status='SYSTEM_UNAVAILABLE'
        )
    
    def _extract_compliance_keywords(self, query: str) -> List[str]:
        """Extract compliance-relevant keywords from query"""
        import re
        
        compliance_stop_words = {'what', 'is', 'the', 'should', 'be', 'used', 'how', 'for', 'which', 'or', 'and'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in compliance_stop_words and len(w) > 3]
        
        # Add compliance-specific terms
        if 'session' in query.lower():
            keywords.extend(['session', 'manager', 'unified', 'management'])
        if 'embedding' in query.lower():
            keywords.extend(['embedding', 'sentence', 'vectorqa', 'system'])
        if 'compliance' in query.lower():
            keywords.extend(['compliance', 'validation', 'audit', 'standard'])
            
        return keywords
    
    def _extract_compliance_sections(self, content: str, keywords: List[str]) -> List[str]:
        """Extract sections that contain actual compliance guidance"""
        lines = content.split('\n')
        relevant_sections = []
        
        for i, line in enumerate(lines):
            if any(keyword.lower() in line.lower() for keyword in keywords):
                # Extract surrounding context for compliance guidance
                start = max(0, i - 2)
                end = min(len(lines), i + 3)
                section = '\n'.join(lines[start:end]).strip()
                
                # Only include if it contains actionable guidance
                if any(indicator in section.lower() for indicator in ['use', 'should', 'must', 'required']):
                    relevant_sections.append(section)
                    
        return relevant_sections[:5]  # Top 5 most relevant sections
    
    def _get_most_recent_date(self) -> str:
        """Get the most recent date folder for compliance purposes"""
        docs_path = self.base_path / "docs"
        if not docs_path.exists():
            return "2025-09-05"  # Default fallback
            
        date_folders = [f.name for f in docs_path.iterdir() if f.is_dir() and f.name.startswith("2025")]
        return max(date_folders) if date_folders else "2025-09-05"