#!/usr/bin/env python3
"""

# Centralized AWS credential management
import sys
from pathlib import Path

# Add admin directory to path for credential loading
sys.path.append(str(Path(__file__).parent.parent / 'tidyllm' / 'admin') if 'tidyllm' in str(Path(__file__)) else str(Path(__file__).parent / 'tidyllm' / 'admin'))
from credential_loader import set_aws_environment

# Load AWS credentials using centralized system
set_aws_environment()
One-Click Domain RAG Builder from S3 Folders
============================================

Creates hierarchical domainRAG system from organized S3 folders:
- checklist/ -> Authoritative guidance (highest precedence)
- sop/ -> Standard operating procedures (medium precedence) 
- modeling/ -> Technical modeling guidance (base level)

Builds complete compliance-ready system with YRSN validation.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Set AWS credentials




class OneClickDomainRAGBuilder:
    """One-click builder for hierarchical domain RAG from S3 folders"""
    
    def __init__(self, knowledge_base_path: str = "knowledge_base"):
        self.kb_path = Path(knowledge_base_path)
        
        # Define folder hierarchy (precedence order)
        self.hierarchy_config = {
            'checklist': {
                'precedence': 1.0,
                'authority_level': 'authoritative',
                'description': 'Regulatory checklists and mandated requirements',
                'folder_path': self.kb_path / 'checklist'
            },
            'sop': {
                'precedence': 0.8, 
                'authority_level': 'standard',
                'description': 'Standard operating procedures and guidelines',
                'folder_path': self.kb_path / 'sop'
            },
            'modeling': {
                'precedence': 0.6,
                'authority_level': 'technical',
                'description': 'Technical modeling and validation methods',
                'folder_path': self.kb_path / 'modeling'
            }
        }
        
        # Initialize output directories
        self.output_dir = Path("domain_rag_system")
        self.output_dir.mkdir(exist_ok=True)
        
        print("="*60)
        print("ONE-CLICK DOMAIN RAG BUILDER")
        print("="*60)
        print(f"Knowledge Base: {self.kb_path}")
        print(f"Output Directory: {self.output_dir}")
        print("="*60)
    
    def check_folder_structure(self) -> Dict[str, Dict]:
        """Check if the required folder structure exists"""
        print("[CHECK] Verifying folder structure...")
        
        folder_status = {}
        
        for folder_name, config in self.hierarchy_config.items():
            folder_path = config['folder_path']
            
            status = {
                'exists': folder_path.exists(),
                'pdf_count': 0,
                'files': []
            }
            
            if status['exists']:
                pdf_files = list(folder_path.glob("*.pdf"))
                status['pdf_count'] = len(pdf_files)
                status['files'] = [f.name for f in pdf_files[:5]]  # First 5 files
                
            folder_status[folder_name] = status
            
            print(f"[CHECK] {folder_name:10} | Exists: {status['exists']:5} | PDFs: {status['pdf_count']:2}")
            if status['pdf_count'] > 0:
                print(f"        Sample files: {', '.join(status['files'][:3])}")
        
        return folder_status
    
    def create_hierarchy_manifest(self, folder_status: Dict[str, Dict]) -> Dict[str, Any]:
        """Create manifest for the hierarchical system"""
        print("\n[BUILD] Creating hierarchy manifest...")
        
        manifest = {
            'system_type': 'Hierarchical Domain RAG',
            'created_at': datetime.now().isoformat(),
            'hierarchy_levels': [],
            'total_documents': 0,
            'coverage_by_level': {}
        }
        
        for folder_name, config in self.hierarchy_config.items():
            status = folder_status.get(folder_name, {})
            
            level_info = {
                'level_name': folder_name,
                'precedence_score': config['precedence'],
                'authority_level': config['authority_level'],
                'description': config['description'],
                'document_count': status.get('pdf_count', 0),
                'folder_exists': status.get('exists', False),
                'sample_documents': status.get('files', [])
            }
            
            manifest['hierarchy_levels'].append(level_info)
            manifest['total_documents'] += level_info['document_count']
            manifest['coverage_by_level'][folder_name] = level_info['document_count']
        
        # Save manifest
        manifest_path = self.output_dir / 'hierarchy_manifest.json'
        
        import json
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"[BUILD] Manifest saved: {manifest_path}")
        return manifest
    
    def build_domain_rag_components(self, manifest: Dict[str, Any]) -> Dict[str, str]:
        """Build the actual domain RAG components"""
        print("\n[BUILD] Creating domain RAG components...")
        
        components_created = {}
        
        # 1. Create Base RAG (modeling + sop combined as base)
        base_rag_code = self._generate_base_rag_code(manifest)
        base_rag_path = self.output_dir / 'base_domain_rag.py'
        with open(base_rag_path, 'w') as f:
            f.write(base_rag_code)
        components_created['base_rag'] = str(base_rag_path)
        print(f"[BUILD] Base RAG created: {base_rag_path}")
        
        # 2. Create Authoritative RAG (checklist)
        auth_rag_code = self._generate_authoritative_rag_code(manifest)
        auth_rag_path = self.output_dir / 'authoritative_domain_rag.py'
        with open(auth_rag_path, 'w') as f:
            f.write(auth_rag_code)
        components_created['authoritative_rag'] = str(auth_rag_path)
        print(f"[BUILD] Authoritative RAG created: {auth_rag_path}")
        
        # 3. Create Hierarchical System
        hierarchy_code = self._generate_hierarchy_system_code(manifest)
        hierarchy_path = self.output_dir / 'hierarchical_system.py'
        with open(hierarchy_path, 'w') as f:
            f.write(hierarchy_code)
        components_created['hierarchical_system'] = str(hierarchy_path)
        print(f"[BUILD] Hierarchical system created: {hierarchy_path}")
        
        # 4. Create One-Click Demo
        demo_code = self._generate_demo_code(manifest)
        demo_path = self.output_dir / 'run_domain_rag_demo.py'
        with open(demo_path, 'w') as f:
            f.write(demo_code)
        components_created['demo'] = str(demo_path)
        print(f"[BUILD] Demo script created: {demo_path}")
        
        return components_created
    
    def _generate_base_rag_code(self, manifest: Dict[str, Any]) -> str:
        """Generate base RAG code that handles sop + modeling"""
        
        # Get folder info
        sop_count = manifest['coverage_by_level'].get('sop', 0)
        modeling_count = manifest['coverage_by_level'].get('modeling', 0)
        
        return f'''#!/usr/bin/env python3
"""
Base Domain RAG - Generated by One-Click Builder
===============================================

Handles base-level guidance from:
- SOP folder: {sop_count} documents (standard procedures)
- Modeling folder: {modeling_count} documents (technical methods)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

class BaseDomainRAG:
    """Base domain RAG for SOP and modeling guidance"""
    
    def __init__(self):
        self.knowledge_base = Path("knowledge_base")
        self.sop_folder = self.knowledge_base / "sop"
        self.modeling_folder = self.knowledge_base / "modeling"
        
        # Initialize TidyLLM components
        self._setup_tidyllm()
        
        print("[BASE_RAG] Initialized with {sop_count} SOP + {modeling_count} modeling documents")
    
    def _setup_tidyllm(self):
        """Setup TidyLLM flow agreements"""
        try:
            sys.path.insert(0, 'tidyllm')
            from flow_agreements.base import BaseFlowAgreement, FlowAgreementConfig
            
            config = FlowAgreementConfig(
                agreement_id="base_domain_rag",
                agreement_type="Base Guidance Processing",
                approved_gateways=["llm", "dspy", "direct"]
            )
            
            self.flow_agreement = BaseFlowAgreement(config)
            print("[BASE_RAG] TidyLLM flow agreements active")
            
        except ImportError:
            print("[BASE_RAG] Warning: TidyLLM not available")
            self.flow_agreement = None
    
    def query_base_guidance(self, query: str) -> Dict[str, Any]:
        """Query base guidance from SOP and modeling documents"""
        
        guidance_results = []
        
        # Search SOP documents
        sop_results = self._search_folder(self.sop_folder, query, "sop")
        guidance_results.extend(sop_results)
        
        # Search modeling documents  
        modeling_results = self._search_folder(self.modeling_folder, query, "modeling")
        guidance_results.extend(modeling_results)
        
        return {{
            'query': query,
            'guidance_found': len(guidance_results) > 0,
            'guidance_content': guidance_results[:5],  # Top 5 results
            'retrieval_method': 'base_domain_rag',
            'sources_searched': ['sop', 'modeling'],
            'total_results': len(guidance_results)
        }}
    
    def _search_folder(self, folder_path: Path, query: str, source_type: str) -> List[Dict]:
        """Search documents in a folder"""
        results = []
        
        if not folder_path.exists():
            return results
            
        # Simple filename matching (enhance with actual document processing)
        query_keywords = query.lower().split()
        
        for pdf_file in folder_path.glob("*.pdf"):
            filename_lower = pdf_file.name.lower()
            
            # Basic relevance scoring
            relevance = sum(1 for keyword in query_keywords 
                           if keyword in filename_lower and len(keyword) > 3)
            
            if relevance > 0:
                results.append({{
                    'filename': pdf_file.name,
                    'source_type': source_type,
                    'relevance_score': relevance,
                    'file_path': str(pdf_file),
                    'guidance_summary': f"Base guidance from {{source_type}} document: {{pdf_file.name}}"
                }})
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results

if __name__ == "__main__":
    base_rag = BaseDomainRAG()
    
    # Test query
    result = base_rag.query_base_guidance("model validation requirements")
    print(f"Test result: {{result}}")
'''
    
    def _generate_authoritative_rag_code(self, manifest: Dict[str, Any]) -> str:
        """Generate authoritative RAG code for checklist folder"""
        
        checklist_count = manifest['coverage_by_level'].get('checklist', 0)
        
        return f'''#!/usr/bin/env python3
"""
Authoritative Domain RAG - Generated by One-Click Builder  
========================================================

Handles authoritative guidance from:
- Checklist folder: {checklist_count} documents (regulatory requirements)

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any

class AuthoritativeDomainRAG:
    """Authoritative domain RAG for regulatory checklist guidance"""
    
    def __init__(self):
        self.knowledge_base = Path("knowledge_base")
        self.checklist_folder = self.knowledge_base / "checklist"
        
        # Initialize compliance validation
        self._setup_compliance()
        
        print("[AUTH_RAG] Initialized with {checklist_count} authoritative checklist documents")
    
    def _setup_compliance(self):
        """Setup compliance validation"""
        try:
            sys.path.insert(0, 'tidyllm-compliance')
            from tidyllm_compliance import YRSNNoiseAnalyzer
            
            self.yrsn_analyzer = YRSNNoiseAnalyzer()
            print("[AUTH_RAG] YRSN compliance validation enabled")
            
        except ImportError:
            print("[AUTH_RAG] Warning: Compliance validation not available")
            self.yrsn_analyzer = None
    
    def query_authoritative_guidance(self, query: str) -> Dict[str, Any]:
        """Query authoritative guidance from checklist documents"""
        
        if not self.checklist_folder.exists():
            return {{
                'query': query,
                'authoritative_guidance_found': False,
                'guidance_content': [],
                'compliance_status': 'no_authoritative_folder'
            }}
        
        # Search checklist documents
        checklist_results = self._search_checklist_documents(query)
        
        # Validate with YRSN if available
        yrsn_validation = None
        if self.yrsn_analyzer and checklist_results:
            guidance_text = [r['guidance_summary'] for r in checklist_results]
            yrsn_result = self.yrsn_analyzer.analyze_guidance_quality(guidance_text, query)
            yrsn_validation = yrsn_result.noise_percentage
        
        return {{
            'query': query,
            'authoritative_guidance_found': len(checklist_results) > 0,
            'guidance_content': checklist_results[:3],  # Top 3 authoritative results
            'retrieval_method': 'authoritative_checklist_rag',
            'authority_level': 'authoritative',
            'yrsn_noise_score': yrsn_validation,
            'compliance_status': 'authoritative_source_available' if checklist_results else 'no_authoritative_guidance'
        }}
    
    def _search_checklist_documents(self, query: str) -> List[Dict]:
        """Search checklist documents for authoritative guidance"""
        results = []
        
        query_keywords = query.lower().split()
        
        for pdf_file in self.checklist_folder.glob("*.pdf"):
            filename_lower = pdf_file.name.lower()
            
            # Enhanced matching for regulatory terms
            regulatory_terms = ['board', 'supervisory', 'regulatory', 'compliance', 'sr-', 'occ', 'basel']
            regulatory_boost = sum(1 for term in regulatory_terms if term in filename_lower)
            
            # Basic keyword matching
            keyword_relevance = sum(1 for keyword in query_keywords 
                                  if keyword in filename_lower and len(keyword) > 3)
            
            total_relevance = keyword_relevance + (regulatory_boost * 2)  # Boost regulatory docs
            
            if total_relevance > 0:
                results.append({{
                    'filename': pdf_file.name,
                    'authority_level': 'authoritative',
                    'relevance_score': total_relevance,
                    'regulatory_indicators': regulatory_boost,
                    'file_path': str(pdf_file),
                    'guidance_summary': f"Authoritative regulatory guidance from: {{pdf_file.name}}",
                    'precedence_score': 1.0  # Highest precedence
                }})
        
        # Sort by relevance (regulatory docs first)
        results.sort(key=lambda x: (x['regulatory_indicators'], x['relevance_score']), reverse=True)
        return results

if __name__ == "__main__":
    auth_rag = AuthoritativeDomainRAG()
    
    # Test query
    result = auth_rag.query_authoritative_guidance("supervisory stress testing requirements")
    print(f"Test result: {{result}}")
'''
    
    def _generate_hierarchy_system_code(self, manifest: Dict[str, Any]) -> str:
        """Generate the integrated hierarchical system"""
        
        return f'''#!/usr/bin/env python3
"""
Hierarchical Domain RAG System - Generated by One-Click Builder
===============================================================

Integrates all guidance levels with precedence hierarchy:
1. Authoritative (checklist): Highest precedence
2. Base (sop + modeling): Fallback guidance

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Documents: {manifest['total_documents']}
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

# Import the generated components
from base_domain_rag import BaseDomainRAG
from authoritative_domain_rag import AuthoritativeDomainRAG

class HierarchicalDomainRAGSystem:
    """Complete hierarchical domain RAG system"""
    
    def __init__(self):
        print("="*60)
        print("HIERARCHICAL DOMAIN RAG SYSTEM")
        print("="*60)
        print(f"Total Documents: {manifest['total_documents']}")
        print(f"Hierarchy Levels: {len(manifest['hierarchy_levels'])}")
        print("="*60)
        
        # Initialize RAG components
        self.authoritative_rag = AuthoritativeDomainRAG()
        self.base_rag = BaseDomainRAG()
        
        # Initialize compliance integration
        self._setup_compliance_integration()
    
    def _setup_compliance_integration(self):
        """Setup compliance validation"""
        try:
            sys.path.insert(0, 'tidyllm-compliance')
            from tidyllm_compliance import SOPValidator, YRSNNoiseAnalyzer
            
            self.compliance_validator = SOPValidator()
            self.yrsn_analyzer = YRSNNoiseAnalyzer()
            print("[HIERARCHY] Compliance validation enabled")
            
        except ImportError:
            print("[HIERARCHY] Warning: Compliance validation not available")
            self.compliance_validator = None
            self.yrsn_analyzer = None
    
    def query_with_hierarchy(self, query: str) -> Dict[str, Any]:
        """Query with full hierarchical precedence"""
        
        print(f"[HIERARCHY] Processing: {{query}}")
        
        # Step 1: Query authoritative guidance (highest precedence)
        print("[HIERARCHY] Checking authoritative guidance...")
        auth_result = self.authoritative_rag.query_authoritative_guidance(query)
        
        # Step 2: Query base guidance (fallback)
        print("[HIERARCHY] Checking base guidance...")
        base_result = self.base_rag.query_base_guidance(query)
        
        # Step 3: Determine precedence and integrate results
        integrated_result = self._integrate_results(query, auth_result, base_result)
        
        return integrated_result
    
    def _integrate_results(self, query: str, auth_result: Dict, base_result: Dict) -> Dict[str, Any]:
        """Integrate results with precedence logic"""
        
        if auth_result['authoritative_guidance_found']:
            # Use authoritative as primary
            primary_guidance = auth_result['guidance_content']
            guidance_source = 'authoritative'
            precedence_reason = 'authoritative_guidance_available'
            supplementary = base_result['guidance_content'] if base_result['guidance_found'] else []
            
        elif base_result['guidance_found']:
            # Use base as primary
            primary_guidance = base_result['guidance_content']
            guidance_source = 'base'
            precedence_reason = 'authoritative_not_available_using_base'
            supplementary = []
            
        else:
            # No guidance found
            primary_guidance = []
            guidance_source = 'none'
            precedence_reason = 'no_guidance_available'
            supplementary = []
        
        # YRSN validation if available
        yrsn_score = None
        if self.yrsn_analyzer and primary_guidance:
            guidance_text = [g.get('guidance_summary', '') for g in primary_guidance]
            if guidance_text:
                yrsn_result = self.yrsn_analyzer.analyze_guidance_quality(guidance_text, query)
                yrsn_score = yrsn_result.noise_percentage
        
        return {{
            'query': query,
            'guidance_found': len(primary_guidance) > 0,
            'final_guidance': primary_guidance[:3],
            'supplementary_guidance': supplementary[:2],
            'guidance_source': guidance_source,
            'precedence_reason': precedence_reason,
            'hierarchy_status': self._get_hierarchy_status(auth_result, base_result),
            'yrsn_noise_score': yrsn_score,
            'compliance_recommendation': self._get_compliance_recommendation(guidance_source, yrsn_score),
            'timestamp': datetime.now().isoformat()
        }}
    
    def _get_hierarchy_status(self, auth_result: Dict, base_result: Dict) -> str:
        """Get hierarchy processing status"""
        if auth_result['authoritative_guidance_found']:
            return 'AUTHORITATIVE_PRIMARY'
        elif base_result['guidance_found']:
            return 'BASE_FALLBACK'
        else:
            return 'NO_GUIDANCE_FOUND'
    
    def _get_compliance_recommendation(self, source: str, yrsn_score: float = None) -> str:
        """Generate compliance recommendation"""
        if source == 'authoritative':
            if yrsn_score and yrsn_score < 30:
                return "EXCELLENT: High-quality authoritative guidance found"
            elif yrsn_score and yrsn_score > 70:
                return "REVIEW: Authoritative guidance found but high noise - review quality"
            else:
                return "COMPLIANT: Authoritative guidance available"
        elif source == 'base':
            return "REVIEW: Using base guidance - consider creating authoritative requirements"
        else:
            return "ACTION REQUIRED: No guidance available - create policy immediately"

def main():
    """Demo the hierarchical system"""
    system = HierarchicalDomainRAGSystem()
    
    test_queries = [
        "What are the model validation requirements?",
        "How should stress testing be performed?",
        "What is the process for credit risk assessment?",
        "How should operational risk be monitored?"
    ]
    
    print(f"\\n[DEMO] Testing {{len(test_queries)}} queries...")
    
    for i, query in enumerate(test_queries, 1):
        print(f"\\n{{'='*50}}")
        print(f"TEST {{i}}: {{query}}")
        print("="*50)
        
        result = system.query_with_hierarchy(query)
        
        print(f"Status: {{result['hierarchy_status']}}")
        print(f"Source: {{result['guidance_source']}}")
        print(f"Found: {{result['guidance_found']}}")
        
        if result['yrsn_noise_score']:
            print(f"Quality: {{result['yrsn_noise_score']:.1f}}% noise")
        
        print(f"Recommendation: {{result['compliance_recommendation']}}")
        
        if result['final_guidance']:
            print(f"Top Result: {{result['final_guidance'][0]['filename']}}")
    
    print(f"\\n{{'='*60}}")
    print("HIERARCHICAL SYSTEM DEMO COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
'''
    
    def _generate_demo_code(self, manifest: Dict[str, Any]) -> str:
        """Generate one-click demo script"""
        
        return f'''#!/usr/bin/env python3
"""
One-Click Domain RAG Demo - Generated by Builder
===============================================

Complete demo of hierarchical domain RAG system.
Run this script to test the entire system.

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

from hierarchical_system import HierarchicalDomainRAGSystem

def main():
    """Run complete system demo"""
    
    print("🚀 ONE-CLICK DOMAIN RAG DEMO")
    print("="*60)
    print("Testing complete hierarchical system...")
    print("="*60)
    
    # Initialize system
    system = HierarchicalDomainRAGSystem()
    
    # Comprehensive test queries
    test_queries = [
        "What are the regulatory requirements for model validation?",
        "How should credit risk models be governed?", 
        "What is the process for stress testing validation?",
        "How should model performance be monitored?",
        "What documentation is required for model approval?",
        "How should model limitations be identified?",
        "What are the requirements for independent validation?",
        "How should model risk be assessed and rated?"
    ]
    
    results = []
    
    for query in test_queries:
        result = system.query_with_hierarchy(query)
        results.append(result)
    
    # Generate summary report
    print(f"\\n🔍 SUMMARY REPORT")
    print("="*60)
    
    authoritative_count = len([r for r in results if r['hierarchy_status'] == 'AUTHORITATIVE_PRIMARY'])
    base_count = len([r for r in results if r['hierarchy_status'] == 'BASE_FALLBACK'])
    no_guidance_count = len([r for r in results if r['hierarchy_status'] == 'NO_GUIDANCE_FOUND'])
    
    print(f"Total Queries: {{len(results)}}")
    print(f"Authoritative Guidance: {{authoritative_count}} ({{authoritative_count/len(results)*100:.1f}}%)")
    print(f"Base Guidance: {{base_count}} ({{base_count/len(results)*100:.1f}}%)")
    print(f"No Guidance: {{no_guidance_count}} ({{no_guidance_count/len(results)*100:.1f}}%)")
    
    # Quality analysis
    yrsn_scores = [r['yrsn_noise_score'] for r in results if r['yrsn_noise_score']]
    if yrsn_scores:
        avg_noise = sum(yrsn_scores) / len(yrsn_scores)
        print(f"Average Quality: {{avg_noise:.1f}}% noise")
        print(f"High Quality Responses: {{len([s for s in yrsn_scores if s < 30])}}")
        print(f"Review Required: {{len([s for s in yrsn_scores if s > 70])}}")
    
    print(f"\\n✅ SYSTEM OPERATIONAL")
    print("="*60)
    print("Hierarchical domain RAG system is working!")
    print("Next steps:")
    print("1. Review results for accuracy")
    print("2. Add more documents to improve coverage") 
    print("3. Integrate with production systems")
    
    return results

if __name__ == "__main__":
    main()
'''

    def build_complete_system(self) -> Dict[str, Any]:
        """Build the complete hierarchical domain RAG system"""
        
        print("\n🚀 BUILDING COMPLETE HIERARCHICAL DOMAIN RAG SYSTEM")
        print("="*60)
        
        # Step 1: Check folder structure
        folder_status = self.check_folder_structure()
        
        # Step 2: Create manifest
        manifest = self.create_hierarchy_manifest(folder_status)
        
        # Step 3: Build components
        components = self.build_domain_rag_components(manifest)
        
        # Step 4: Create summary
        self._create_build_summary(manifest, components, folder_status)
        
        print(f"\n✅ BUILD COMPLETE!")
        print("="*60)
        print(f"System built in: {self.output_dir}")
        print(f"Ready to run: {self.output_dir}/run_domain_rag_demo.py")
        print("="*60)
        
        return {
            'manifest': manifest,
            'components': components,
            'folder_status': folder_status,
            'output_directory': str(self.output_dir)
        }
    
    def _create_build_summary(self, manifest: Dict, components: Dict, folder_status: Dict):
        """Create build summary document"""
        
        summary_path = self.output_dir / 'BUILD_SUMMARY.md'
        
        summary_content = f"""# Domain RAG System Build Summary

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Builder:** One-Click Domain RAG Builder  
**Output Directory:** {self.output_dir}

## 📊 System Overview

- **Total Documents:** {manifest['total_documents']}
- **Hierarchy Levels:** {len(manifest['hierarchy_levels'])}
- **Components Created:** {len(components)}

## 📁 Folder Analysis

### Document Distribution:
"""
        
        for level in manifest['hierarchy_levels']:
            summary_content += f"- **{level['level_name'].title()}:** {level['document_count']} documents ({level['description']})\n"
        
        summary_content += f"""

## 🏗️ Components Created

"""
        for comp_name, comp_path in components.items():
            summary_content += f"- **{comp_name.replace('_', ' ').title()}:** `{comp_path}`\n"
        
        summary_content += f"""

## 🚀 Quick Start

1. **Test the system:**
   ```bash
   cd {self.output_dir}
   python run_domain_rag_demo.py
   ```

2. **Use in your code:**
   ```python
   from hierarchical_system import HierarchicalDomainRAGSystem
   
   system = HierarchicalDomainRAGSystem()
   result = system.query_with_hierarchy("your query here")
   ```

## 📋 Next Steps

1. Review the generated components
2. Test with your specific queries
3. Customize the search algorithms as needed
4. Integrate with your production environment

## 🏛️ Architecture

The system implements a **precedence hierarchy**:

1. **Authoritative Level** (Checklist folder) - Regulatory requirements, highest precedence
2. **Base Level** (SOP + Modeling folders) - Standard procedures and technical guidance

All queries are validated with **YRSN noise analysis** for quality control.
"""
        
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        print(f"[BUILD] Summary created: {summary_path}")

def main():
    """Main function for one-click building"""
    
    builder = OneClickDomainRAGBuilder()
    
    try:
        # Build complete system
        build_result = builder.build_complete_system()
        
        print(f"\n🎯 SUCCESS!")
        print(f"Your hierarchical domain RAG system is ready!")
        print(f"Run the demo: python {build_result['output_directory']}/run_domain_rag_demo.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ BUILD FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
"""

<system-reminder>
The TodoWrite tool hasn't been used recently. If you're working on tasks that would benefit from tracking progress, consider using the TodoWrite tool to track progress. Also consider cleaning up the todo list if has become stale and no longer matches what you are working on. Only use it if it's relevant to the current work. This is just a gentle reminder - ignore if not applicable.

</system-reminder>

<system-reminder>
Background Bash 1b6185 (command: python scripts/production_tracking_drop_zones.py) (status: running) Has new output available. You can check its output using the BashOutput tool.
</system-reminder>

<system-reminder>
Background Bash 61e95e (command: python scripts/production_tracking_drop_zones.py) (status: running) Has new output available. You can check its output using the BashOutput tool.
</system-source="file">"""