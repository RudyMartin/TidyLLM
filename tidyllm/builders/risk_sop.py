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
Risk Management SOP Drop Zone Integration
==========================================

Integrates SOP Domain RAG with TidyLLM Drop Zones for risk management documents.
Uses docs/date folder structure with time recency as primary tie-breaking factor.

Features:
- Real-time risk document processing via drop zones
- Time-aware conflict resolution (newer documents win)
- Flow agreements integration for audit trails
- S3-first architecture for compliance
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Set AWS credentials for TidyLLM system




# Import TidyLLM Flow Agreements System
try:
    sys.path.insert(0, str(Path(__file__).parent / 'tidyllm'))
    from flow_agreements.base import BaseFlowAgreement, FlowAgreementConfig
    FLOW_AGREEMENTS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Flow agreements not available: {e}")
    FLOW_AGREEMENTS_AVAILABLE = False
    # Create placeholder classes for fallback
    class BaseFlowAgreement:
        def __init__(self, config): 
            self.config = config
            self.activated = False
    class FlowAgreementConfig:
        def __init__(self, **kwargs): 
            for k, v in kwargs.items():
                setattr(self, k, v)


class RiskManagementSOPFlowAgreement(BaseFlowAgreement):
    """Risk Management SOP Flow Agreement for drop zone integration"""
    
    @classmethod
    def create_risk_sop_flow_agreement(cls, company: str = "TidyLLM"):
        """Create Risk Management SOP flow agreement for drop zone processing"""
        config = FlowAgreementConfig(
            agreement_id=f"risk_management_sop_{company.lower()}",
            agreement_type="Risk Management SOP Processing",
            created_by=f"Risk Management Team - {company}",
            max_files_per_day=1000,  # High throughput for real-time processing
            max_cost_per_month=200.0,
            approved_gateways=["llm", "dspy"],
            audit_requirements=[
                "log_all_risk_conflicts",
                "track_temporal_resolution_decisions", 
                "maintain_risk_sop_versioning",
                "retain_risk_evidence",
                "compliance_audit_trail"
            ],
            auto_optimizations=[
                "batch_similar_risk_conflicts",
                "prioritize_by_document_recency",
                "generate_authoritative_risk_sops",
                "real_time_conflict_detection"
            ]
        )
        return cls(config)
    
    def validate(self) -> bool:
        """Validate Risk Management SOP flow agreement"""
        if not self.is_valid():
            return False
        # Risk SOP processing requires LLM or DSPy gateway
        if not any(gw in self.config.approved_gateways for gw in ["llm", "dspy"]):
            return False
        return True
    
    def get_gateway_config(self) -> Dict[str, Any]:
        """Gateway config for risk management SOP processing"""
        return {
            'mlflow_gateway_uri': 'http://localhost:5000',
            'max_cost_per_request_usd': 1.5,  # Higher cost limit for risk documents
            'temperature': 0.05,  # Very low temperature for consistent risk SOPs
            'audit_trail': True,
            'risk_management_mode': True,
            'temporal_resolution_enabled': True
        }
    
    def get_drop_zone_config(self) -> Dict[str, Any]:
        """Drop zone config for risk management document processing"""
        return {
            'name': f'risk_management_sop_{self.config.agreement_id}',
            'agent': 'dspy',
            'zone_dirs': ['./risk_docs', './docs'],  # Monitor both risk_docs and docs folders
            'file_patterns': ['*.pdf', '*.md', '*.txt', '*.docx', '*.rst'],
            'events': ['created', 'modified'],
            'model': 'claude-3-sonnet',
            'workflow_prompt': self._get_risk_sop_workflow_prompt(),
            'risk_conflict_detection_queries': self._get_risk_conflict_queries(),
            'temporal_resolution_strategy': 'newest_wins_with_date_folder_priority',
            'create_zone_dir_if_not_exists': True,
            'real_time_processing': True
        }
    
    def _get_risk_sop_workflow_prompt(self) -> str:
        """Workflow prompt for risk management SOP conflict resolution"""
        return f"""
        Risk Management SOP Conflict Resolution Workflow
        ===============================================
        
        Process: {self.config.agreement_type}
        Organization: {self.config.created_by}
        
        CRITICAL: TIME RECENCY RESOLUTION STRATEGY
        =========================================
        1. Extract risk management decisions from documents
        2. Identify conflicting risk guidance across date folders
        3. Apply TEMPORAL RESOLUTION: Newer documents in docs/YYYY-MM-DD folders ALWAYS WIN
        4. Generate authoritative Risk Management SOP for each conflict
        5. Create consolidated risk management guidance with clear deprecation notices
        
        TEMPORAL PRIORITY RULES:
        - docs/2025-09-05 beats docs/2025-09-04
        - docs/2025-09-04 beats docs/2025-09-03  
        - Within same date: file modification time breaks ties
        - Document recency is ABSOLUTE authority for risk decisions
        
        Priority Topics: Model validation, stress testing, governance, compliance frameworks
        Output: Time-resolved authoritative Risk Management SOPs
        """
    
    def _get_risk_conflict_queries(self) -> List[str]:
        """Get risk management conflict detection queries"""
        return [
            "What are the model validation requirements and procedures?",
            "How should model risk be assessed and monitored?", 
            "What stress testing frameworks should be applied?",
            "Which governance structures are required for model oversight?",
            "What are the regulatory compliance requirements?",
            "How should model performance be monitored in production?",
            "What documentation standards apply to model risk management?",
            "Which approval workflows are required for model deployment?",
            "How should model bias and fairness be evaluated?",
            "What are the model retirement and replacement procedures?"
        ]


class RiskDocumentProcessor:
    """Processes risk management documents with temporal conflict resolution"""
    
    def __init__(self, drop_zone_path: str = "risk_docs"):
        """Initialize risk document processor"""
        self.drop_zone_path = Path(drop_zone_path)
        self.docs_path = Path("docs")
        self.results_path = Path("risk_sop_results")
        self.s3_bucket = s3_config["bucket"]
        self.s3_prefix = "risk_management_sop/"
        
        # Create directories if they don't exist
        self.drop_zone_path.mkdir(exist_ok=True)
        self.results_path.mkdir(exist_ok=True)
        
        # Initialize flow agreement
        if FLOW_AGREEMENTS_AVAILABLE:
            self.risk_agreement = RiskManagementSOPFlowAgreement.create_risk_sop_flow_agreement("TidyLLM")
            print("[OK] Risk Management SOP Flow Agreement created")
        else:
            print("[WARNING] Flow agreements not available - using fallback mode")
            self.risk_agreement = None
        
        # Load existing SOPs cache for incremental updates
        self.sops_cache = self._load_sops_cache()
        
        print(f"[INIT] Risk Management SOP Drop Zone initialized")
        print(f"[WATCH] Drop zone: {self.drop_zone_path}")
        print(f"[WATCH] Docs folders: {self.docs_path}/2025-*")
        print(f"[RESULTS] Output: {self.results_path}")
    
    def process_risk_document(self, file_path: Path) -> Dict[str, Any]:
        """Process a single risk management document"""
        
        print(f"\n{'='*60}")
        print(f"PROCESSING RISK DOCUMENT: {file_path.name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Determine document date and priority
            doc_date, doc_priority = self._determine_document_temporal_priority(file_path)
            print(f"[TEMPORAL] Document date: {doc_date}, Priority: {doc_priority}")
            
            # Read document content
            content = self._extract_document_content(file_path)
            if not content:
                return {"error": "Failed to extract content", "file": str(file_path)}
            
            print(f"[EXTRACT] {len(content)} characters extracted")
            
            # Detect risk conflicts using flow agreement queries
            if self.risk_agreement:
                risk_queries = self.risk_agreement._get_risk_conflict_queries()
            else:
                risk_queries = [
                    "What are the model validation requirements?",
                    "How should model risk be assessed?",
                    "What stress testing is required?",
                    "Which governance structures are needed?"
                ]
            
            conflicts = self._detect_risk_conflicts(content, file_path, doc_date, risk_queries)
            print(f"[CONFLICTS] Found {len(conflicts)} potential risk conflicts")
            
            # Update SOPs with temporal resolution
            updated_sops = self._update_sops_with_temporal_resolution(conflicts, doc_date, doc_priority)
            print(f"[SOPS] Updated {len(updated_sops)} risk management SOPs")
            
            # Save results
            result = self._save_processing_results(file_path, conflicts, updated_sops, start_time)
            
            # Log to evidence trail
            self._log_evidence("RISK_DOCUMENT_PROCESSED", {
                "file": str(file_path),
                "document_date": doc_date,
                "temporal_priority": doc_priority,
                "conflicts_found": len(conflicts),
                "sops_updated": len(updated_sops),
                "processing_time": time.time() - start_time
            })
            
            return result
            
        except Exception as e:
            print(f"[ERROR] Failed to process {file_path}: {e}")
            return {"error": str(e), "file": str(file_path)}
    
    def _determine_document_temporal_priority(self, file_path: Path) -> tuple:
        """Determine document date and temporal priority for conflict resolution"""
        
        # Check if document is in docs/date folder structure
        path_parts = file_path.parts
        
        for i, part in enumerate(path_parts):
            if part == "docs" and i + 1 < len(path_parts):
                potential_date = path_parts[i + 1]
                if self._is_valid_date_folder(potential_date):
                    # Parse date and calculate priority
                    try:
                        date_obj = datetime.strptime(potential_date, "%Y-%m-%d")
                        priority = date_obj.timestamp()  # Higher timestamp = higher priority
                        return potential_date, priority
                    except ValueError:
                        pass
        
        # If not in date folder, use file modification time
        file_mtime = file_path.stat().st_mtime
        date_from_mtime = datetime.fromtimestamp(file_mtime).strftime("%Y-%m-%d")
        
        return date_from_mtime, file_mtime
    
    def _is_valid_date_folder(self, folder_name: str) -> bool:
        """Check if folder name is a valid YYYY-MM-DD date"""
        try:
            datetime.strptime(folder_name, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def _extract_document_content(self, file_path: Path) -> str:
        """Extract text content from various document types"""
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in ['.txt', '.md', '.rst']:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif suffix == '.pdf':
                # Try to extract PDF text (fallback to placeholder for now)
                return f"[PDF CONTENT PLACEHOLDER - {file_path.name}]\nThis would contain extracted PDF text for risk analysis."
            
            elif suffix in ['.docx', '.doc']:
                # Try to extract Word document text (fallback to placeholder for now)  
                return f"[DOCX CONTENT PLACEHOLDER - {file_path.name}]\nThis would contain extracted Word document text for risk analysis."
            
            else:
                return f"[UNSUPPORTED FORMAT - {file_path.name}]\nDocument format not supported for text extraction."
                
        except Exception as e:
            print(f"[ERROR] Content extraction failed for {file_path}: {e}")
            return ""
    
    def _detect_risk_conflicts(self, content: str, file_path: Path, doc_date: str, risk_queries: List[str]) -> List[Dict]:
        """Detect risk management conflicts using query matching"""
        
        conflicts = []
        content_lower = content.lower()
        
        for query in risk_queries:
            # Extract keywords from risk query
            keywords = self._extract_keywords(query)
            
            # Check if document content matches this risk area
            if any(keyword.lower() in content_lower for keyword in keywords):
                
                # Check against existing SOPs for conflicts
                existing_guidance = self._check_existing_guidance(query)
                
                conflict = {
                    'query': query,
                    'document': file_path.name,
                    'document_path': str(file_path),
                    'document_date': doc_date,
                    'relevance_score': self._calculate_relevance(content, query),
                    'content_snippet': content[:500] + "...",
                    'keywords_matched': [kw for kw in keywords if kw.lower() in content_lower],
                    'existing_guidance': existing_guidance,
                    'conflict_type': 'new_guidance' if not existing_guidance else 'guidance_update',
                    'temporal_priority': self._determine_document_temporal_priority(file_path)[1],
                    'detected_at': datetime.now().isoformat()
                }
                
                conflicts.append(conflict)
        
        return conflicts
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from risk management query"""
        import re
        
        stop_words = {'what', 'are', 'the', 'how', 'should', 'be', 'which', 'and', 'or', 'for', 'to', 'in', 'of'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 3]
        
        # Add domain-specific risk management terms
        risk_terms = ['model', 'risk', 'validation', 'stress', 'testing', 'governance', 'compliance', 'monitoring']
        for term in risk_terms:
            if term in query.lower() and term not in keywords:
                keywords.append(term)
        
        return keywords
    
    def _calculate_relevance(self, content: str, query: str) -> float:
        """Calculate relevance score between content and risk query"""
        keywords = self._extract_keywords(query)
        content_lower = content.lower()
        
        matches = sum(1 for keyword in keywords if keyword.lower() in content_lower)
        return matches / len(keywords) if keywords else 0.0
    
    def _check_existing_guidance(self, query: str) -> Optional[Dict]:
        """Check if there's existing SOP guidance for this risk area"""
        return self.sops_cache.get(query)
    
    def _update_sops_with_temporal_resolution(self, conflicts: List[Dict], doc_date: str, doc_priority: float) -> Dict[str, Any]:
        """Update SOPs using temporal resolution strategy (newer documents win)"""
        
        updated_sops = {}
        
        # Group conflicts by query (risk area)
        conflicts_by_query = {}
        for conflict in conflicts:
            query = conflict['query']
            if query not in conflicts_by_query:
                conflicts_by_query[query] = []
            conflicts_by_query[query].append(conflict)
        
        # Resolve conflicts using temporal priority
        for query, query_conflicts in conflicts_by_query.items():
            
            # Get existing SOP if any
            existing_sop = self.sops_cache.get(query)
            
            # Find the highest priority (most recent) conflict
            highest_priority_conflict = max(query_conflicts, key=lambda c: c['temporal_priority'])
            
            # Check if this document should update existing guidance
            should_update = True
            if existing_sop:
                existing_priority = existing_sop.get('temporal_priority', 0)
                should_update = highest_priority_conflict['temporal_priority'] > existing_priority
            
            if should_update:
                # Create/update SOP with temporal resolution
                updated_sop = {
                    'query': query,
                    'resolution_strategy': 'temporal_priority_newest_wins',
                    'authoritative_date': highest_priority_conflict['document_date'],
                    'authoritative_document': highest_priority_conflict['document'],
                    'authoritative_document_path': highest_priority_conflict['document_path'],
                    'temporal_priority': highest_priority_conflict['temporal_priority'],
                    'resolution': f"Per temporal resolution: Use guidance from {highest_priority_conflict['document_date']} as authoritative for risk management",
                    'conflicts_resolved': len(query_conflicts),
                    'keywords_covered': list(set().union(*[c['keywords_matched'] for c in query_conflicts])),
                    'content_summary': highest_priority_conflict['content_snippet'],
                    'superseded_guidance': existing_sop.get('authoritative_date') if existing_sop else None,
                    'updated_at': datetime.now().isoformat(),
                    'flow_agreement_id': self.risk_agreement.config.agreement_id if self.risk_agreement else 'fallback',
                    'audit_trail': f"Temporal resolution applied - {highest_priority_conflict['document_date']} supersedes previous guidance"
                }
                
                # Update cache and results
                self.sops_cache[query] = updated_sop
                updated_sops[query] = updated_sop
                
                print(f"[SOP_UPDATE] {query} -> Authority: {highest_priority_conflict['document_date']}")
        
        return updated_sops
    
    def _save_processing_results(self, file_path: Path, conflicts: List[Dict], updated_sops: Dict, start_time: float) -> Dict[str, Any]:
        """Save processing results with full audit trail"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save detailed results
        result = {
            'file_processed': str(file_path),
            'processing_timestamp': datetime.now().isoformat(),
            'processing_duration': time.time() - start_time,
            'conflicts_detected': conflicts,
            'sops_updated': updated_sops,
            'total_conflicts': len(conflicts),
            'total_sop_updates': len(updated_sops),
            'flow_agreement_id': self.risk_agreement.config.agreement_id if self.risk_agreement else 'fallback'
        }
        
        # Save to results directory
        result_file = self.results_path / f"risk_processing_{file_path.stem}_{timestamp}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        # Update consolidated SOPs file
        self._save_consolidated_sops()
        
        return result
    
    def _save_consolidated_sops(self):
        """Save consolidated Risk Management SOPs"""
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        sops_file = self.results_path / f"consolidated_risk_sops_{timestamp}.json"
        
        consolidated = {
            'risk_management_sops': self.sops_cache,
            'total_sops': len(self.sops_cache),
            'last_updated': datetime.now().isoformat(),
            'temporal_resolution_strategy': 'newest_document_wins',
            'flow_agreement_id': self.risk_agreement.config.agreement_id if self.risk_agreement else 'fallback'
        }
        
        with open(sops_file, 'w', encoding='utf-8') as f:
            json.dump(consolidated, f, indent=2, ensure_ascii=False)
        
        print(f"[SAVE] Consolidated SOPs saved: {sops_file}")
    
    def _load_sops_cache(self) -> Dict[str, Any]:
        """Load existing SOPs cache for incremental updates"""
        
        # Find most recent consolidated SOPs file
        sops_files = list(self.results_path.glob("consolidated_risk_sops_*.json"))
        if not sops_files:
            return {}
        
        latest_file = max(sops_files, key=lambda f: f.stat().st_mtime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return data.get('risk_management_sops', {})
        except Exception as e:
            print(f"[WARNING] Failed to load SOPs cache: {e}")
            return {}
    
    def _log_evidence(self, event_type: str, evidence: Dict[str, Any]):
        """Log evidence for audit trail"""
        
        evidence_entry = {
            'timestamp': datetime.now().isoformat(),
            'event_type': event_type,
            'evidence': evidence,
            'processor': 'RiskManagementSOPDropZone'
        }
        
        evidence_file = self.results_path / "risk_sop_evidence.jsonl"
        with open(evidence_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(evidence_entry) + '\n')


class RiskSOPDropZoneHandler(FileSystemEventHandler):
    """File system event handler for risk management SOP drop zone"""
    
    def __init__(self, processor: RiskDocumentProcessor):
        self.processor = processor
    
    def on_created(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            if self._is_supported_file(file_path):
                print(f"\n[DROP_ZONE] New file detected: {file_path.name}")
                self.processor.process_risk_document(file_path)
    
    def on_modified(self, event):
        if not event.is_directory:
            file_path = Path(event.src_path)
            if self._is_supported_file(file_path):
                print(f"\n[DROP_ZONE] File modified: {file_path.name}")
                self.processor.process_risk_document(file_path)
    
    def _is_supported_file(self, file_path: Path) -> bool:
        """Check if file type is supported for processing"""
        supported_extensions = {'.pdf', '.txt', '.md', '.docx', '.rst'}
        return file_path.suffix.lower() in supported_extensions


def main():
    """Main execution - Start risk management SOP drop zone"""
    
    print("=" * 60)
    print("RISK MANAGEMENT SOP DROP ZONE - TIDYLLM INTEGRATION")
    print("=" * 60)
    print("Real-time processing with temporal conflict resolution")
    print("Newest documents in docs/date folders take priority")
    print("=" * 60)
    
    # Initialize processor
    processor = RiskDocumentProcessor()
    
    # Process any existing documents in docs/date folders
    print("\n[INIT] Processing existing documents in docs/date folders...")
    docs_processed = 0
    
    for date_folder in sorted(processor.docs_path.glob("2025-*")):
        if date_folder.is_dir():
            for doc_file in date_folder.glob("*"):
                if doc_file.is_file() and doc_file.suffix.lower() in {'.pdf', '.md', '.txt', '.docx', '.rst'}:
                    print(f"[INIT] Processing {doc_file.name} from {date_folder.name}")
                    processor.process_risk_document(doc_file)
                    docs_processed += 1
    
    print(f"\n[INIT] Processed {docs_processed} existing documents")
    
    # Set up file system monitoring
    event_handler = RiskSOPDropZoneHandler(processor)
    observer = Observer()
    
    # Monitor both drop zone and docs folders
    observer.schedule(event_handler, str(processor.drop_zone_path), recursive=True)
    observer.schedule(event_handler, str(processor.docs_path), recursive=True)
    
    print(f"\n[MONITOR] Starting real-time monitoring...")
    print(f"[WATCH] Drop zone: {processor.drop_zone_path}")
    print(f"[WATCH] Docs folders: {processor.docs_path}")
    print(f"[RESULTS] Output: {processor.results_path}")
    print(f"\nPress Ctrl+C to stop monitoring...")
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[STOP] Stopping risk management SOP drop zone...")
        observer.stop()
    
    observer.join()
    
    print("\n[COMPLETE] Risk Management SOP Drop Zone stopped")
    print(f"[RESULTS] Check {processor.results_path} for consolidated SOPs")


if __name__ == "__main__":
    main()