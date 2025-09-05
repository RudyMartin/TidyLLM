#!/usr/bin/env python3
"""
Human-in-the-Loop MVR Interface
==============================

Interactive interface for manual progression through MVR analysis workflows.
Allows analysts to "push" documents through each stage with SOP compliance guidance.

Integrates with:
- DROP ZONES for document management
- Universal Bracket Flows for workflow execution
- SOP Golden Answers for compliance guidance
- tidyllm-compliance for validation
"""

import os
import sys
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import threading
import time

# Add project paths
sys.path.append(str(Path(__file__).parent.parent))

# Import core components
try:
    from tidyllm.universal_flow_parser import UniversalFlowParser, BracketCommand
    FLOW_PARSER_AVAILABLE = True
except ImportError:
    FLOW_PARSER_AVAILABLE = False
    print("[WARNING] Universal Flow Parser not available")

try:
    from tidyllm_compliance.sop_golden_answers import SOPValidator
    SOP_VALIDATOR_AVAILABLE = True
except ImportError:
    SOP_VALIDATOR_AVAILABLE = False
    print("[WARNING] SOP Validator not available")

@dataclass
class DocumentCollection:
    """Manages a collection of related documents for analysis"""
    collection_id: str
    name: str
    description: str
    document_ids: List[str]
    collection_type: str  # 'mvr_analysis', 'peer_review', 'comparison_set', 'custom'
    primary_document: Optional[str]  # Main document in the collection
    metadata: Dict[str, Any]
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'collection_id': self.collection_id,
            'name': self.name,
            'description': self.description,
            'document_ids': self.document_ids,
            'collection_type': self.collection_type,
            'primary_document': self.primary_document,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

@dataclass
class DocumentState:
    """Tracks document state through MVR workflow"""
    document_id: str
    file_path: str
    document_type: str  # 'mvr', 'vst', 'research', 'peer_review'
    current_stage: str
    completed_stages: List[str]
    checklist_status: Dict[str, bool]
    processing_history: List[Dict]
    metadata: Dict[str, Any]
    collections: List[str]  # List of collection IDs this document belongs to
    created_at: datetime
    last_updated: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            **asdict(self),
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat()
        }

class HumanLoopMVRInterface:
    """Human-in-the-Loop interface for MVR analysis workflow management"""
    
    def __init__(self, workspace_path: Optional[str] = None):
        # Setup workspace
        self.workspace = Path(workspace_path) if workspace_path else Path.cwd() / 'mvr_workspace'
        self.workspace.mkdir(exist_ok=True)
        
        # Workspace directories
        self.documents_dir = self.workspace / 'documents'
        self.stages_dir = self.workspace / 'stages'
        self.reports_dir = self.workspace / 'reports' 
        self.state_dir = self.workspace / 'state'
        self.collections_dir = self.workspace / 'collections'
        
        for dir_path in [self.documents_dir, self.stages_dir, self.reports_dir, self.state_dir, self.collections_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Document and collection tracking
        self.active_documents: Dict[str, DocumentState] = {}
        self.active_collections: Dict[str, DocumentCollection] = {}
        self.load_existing_documents()
        self.load_existing_collections()
        
        # Initialize components
        self.flow_parser = UniversalFlowParser() if FLOW_PARSER_AVAILABLE else None
        self.sop_validator = SOPValidator() if SOP_VALIDATOR_AVAILABLE else None
        
        # MVR workflow configuration
        self.mvr_workflow_stages = ['mvr_tag', 'mvr_qa', 'mvr_peer', 'mvr_report']
        self.stage_descriptions = {
            'mvr_tag': 'Document Classification & Tagging',
            'mvr_qa': 'MVR vs VST Comparison Analysis',
            'mvr_peer': 'Peer Review & Triangulation',
            'mvr_report': 'Final Report Generation'
        }
        
        print(f"[INIT] Human-in-the-Loop MVR Interface initialized")
        print(f"[INIT] Workspace: {self.workspace}")
        print(f"[INIT] Active documents: {len(self.active_documents)}")
        print(f"[INIT] Active collections: {len(self.active_collections)}")
        print(f"[INIT] Flow parser available: {FLOW_PARSER_AVAILABLE}")
        print(f"[INIT] SOP validator available: {SOP_VALIDATOR_AVAILABLE}")
    
    def load_existing_documents(self):
        """Load existing document states from workspace"""
        try:
            state_files = list(self.state_dir.glob('*.json'))
            for state_file in state_files:
                with open(state_file, 'r') as f:
                    doc_data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                doc_data['created_at'] = datetime.fromisoformat(doc_data['created_at'])
                doc_data['last_updated'] = datetime.fromisoformat(doc_data['last_updated'])
                
                doc_state = DocumentState(**doc_data)
                self.active_documents[doc_state.document_id] = doc_state
            
            if state_files:
                print(f"[LOAD] Loaded {len(state_files)} existing document states")
        
        except Exception as e:
            print(f"[WARNING] Failed to load existing documents: {e}")
    
    def load_existing_collections(self):
        """Load existing document collections from workspace"""
        try:
            collection_files = list(self.collections_dir.glob('*.json'))
            for collection_file in collection_files:
                with open(collection_file, 'r') as f:
                    collection_data = json.load(f)
                
                # Convert datetime strings back to datetime objects
                collection_data['created_at'] = datetime.fromisoformat(collection_data['created_at'])
                collection_data['last_updated'] = datetime.fromisoformat(collection_data['last_updated'])
                
                collection = DocumentCollection(**collection_data)
                self.active_collections[collection.collection_id] = collection
            
            if collection_files:
                print(f"[LOAD] Loaded {len(collection_files)} existing collections")
        
        except Exception as e:
            print(f"[WARNING] Failed to load existing collections: {e}")
    
    def save_document_state(self, doc_state: DocumentState):
        """Save document state to persistent storage"""
        try:
            state_file = self.state_dir / f"{doc_state.document_id}.json"
            doc_state.last_updated = datetime.now()
            
            with open(state_file, 'w') as f:
                json.dump(doc_state.to_dict(), f, indent=2, default=str)
            
            print(f"[SAVE] Document state saved: {doc_state.document_id}")
        
        except Exception as e:
            print(f"[ERROR] Failed to save document state: {e}")
    
    def save_collection_state(self, collection: DocumentCollection):
        """Save collection state to persistent storage"""
        try:
            collection_file = self.collections_dir / f"{collection.collection_id}.json"
            collection.last_updated = datetime.now()
            
            with open(collection_file, 'w') as f:
                json.dump(collection.to_dict(), f, indent=2, default=str)
            
            print(f"[SAVE] Collection state saved: {collection.collection_id}")
        
        except Exception as e:
            print(f"[ERROR] Failed to save collection state: {e}")
    
    def create_collection(self, name: str, description: str, document_ids: List[str], 
                         collection_type: str = 'custom', primary_document: str = None) -> str:
        """Create a new document collection"""
        # Generate collection ID
        collection_id = f"COLLECTION_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Validate all documents exist
        for doc_id in document_ids:
            if doc_id not in self.active_documents:
                raise ValueError(f"Document not found: {doc_id}")
        
        # Create collection
        collection = DocumentCollection(
            collection_id=collection_id,
            name=name,
            description=description,
            document_ids=document_ids,
            collection_type=collection_type,
            primary_document=primary_document,
            metadata={
                'created_by': 'human_analyst',
                'document_count': len(document_ids),
                'types': list(set(self.active_documents[doc_id].document_type for doc_id in document_ids))
            },
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Store collection
        self.active_collections[collection_id] = collection
        self.save_collection_state(collection)
        
        # Update documents to reference collection
        for doc_id in document_ids:
            doc_state = self.active_documents[doc_id]
            if not hasattr(doc_state, 'collections'):
                doc_state.collections = []
            if collection_id not in doc_state.collections:
                doc_state.collections.append(collection_id)
            
            # Log collection membership
            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'added_to_collection',
                'collection_id': collection_id,
                'collection_name': name,
                'user': 'human_analyst'
            }
            doc_state.processing_history.append(history_entry)
            self.save_document_state(doc_state)
        
        print(f"[COLLECTION] Created: {collection_id} with {len(document_ids)} documents")
        return collection_id
    
    def add_to_collection(self, collection_id: str, document_ids: List[str]) -> Dict[str, Any]:
        """Add documents to an existing collection"""
        if collection_id not in self.active_collections:
            raise ValueError(f"Collection not found: {collection_id}")
        
        collection = self.active_collections[collection_id]
        added_count = 0
        
        for doc_id in document_ids:
            if doc_id not in self.active_documents:
                continue
                
            if doc_id not in collection.document_ids:
                collection.document_ids.append(doc_id)
                added_count += 1
                
                # Update document
                doc_state = self.active_documents[doc_id]
                if not hasattr(doc_state, 'collections'):
                    doc_state.collections = []
                if collection_id not in doc_state.collections:
                    doc_state.collections.append(collection_id)
                
                # Log addition
                history_entry = {
                    'timestamp': datetime.now().isoformat(),
                    'action': 'added_to_collection',
                    'collection_id': collection_id,
                    'collection_name': collection.name,
                    'user': 'human_analyst'
                }
                doc_state.processing_history.append(history_entry)
                self.save_document_state(doc_state)
        
        # Update collection metadata
        collection.metadata['document_count'] = len(collection.document_ids)
        collection.metadata['types'] = list(set(
            self.active_documents[doc_id].document_type for doc_id in collection.document_ids
        ))
        self.save_collection_state(collection)
        
        return {
            'success': True,
            'added_count': added_count,
            'total_documents': len(collection.document_ids)
        }
    
    def remove_from_collection(self, collection_id: str, document_ids: List[str]) -> Dict[str, Any]:
        """Remove documents from a collection"""
        if collection_id not in self.active_collections:
            raise ValueError(f"Collection not found: {collection_id}")
        
        collection = self.active_collections[collection_id]
        removed_count = 0
        
        for doc_id in document_ids:
            if doc_id in collection.document_ids:
                collection.document_ids.remove(doc_id)
                removed_count += 1
                
                # Update document
                if doc_id in self.active_documents:
                    doc_state = self.active_documents[doc_id]
                    if hasattr(doc_state, 'collections') and collection_id in doc_state.collections:
                        doc_state.collections.remove(collection_id)
                        self.save_document_state(doc_state)
        
        # Update collection metadata
        collection.metadata['document_count'] = len(collection.document_ids)
        if collection.document_ids:
            collection.metadata['types'] = list(set(
                self.active_documents[doc_id].document_type for doc_id in collection.document_ids
            ))
        else:
            collection.metadata['types'] = []
        
        self.save_collection_state(collection)
        
        return {
            'success': True,
            'removed_count': removed_count,
            'total_documents': len(collection.document_ids)
        }
    
    def get_collection_info(self, collection_id: str) -> Dict[str, Any]:
        """Get detailed information about a collection"""
        if collection_id not in self.active_collections:
            raise ValueError(f"Collection not found: {collection_id}")
        
        collection = self.active_collections[collection_id]
        
        # Get document details
        documents = []
        for doc_id in collection.document_ids:
            if doc_id in self.active_documents:
                doc_state = self.active_documents[doc_id]
                documents.append({
                    'document_id': doc_id,
                    'document_type': doc_state.document_type,
                    'current_stage': doc_state.current_stage,
                    'file_name': Path(doc_state.file_path).name,
                    'last_updated': doc_state.last_updated
                })
        
        return {
            'collection_id': collection_id,
            'name': collection.name,
            'description': collection.description,
            'collection_type': collection.collection_type,
            'primary_document': collection.primary_document,
            'document_count': len(collection.document_ids),
            'documents': documents,
            'metadata': collection.metadata,
            'created_at': collection.created_at,
            'last_updated': collection.last_updated
        }
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """List all active collections"""
        collections = []
        
        for collection_id, collection in self.active_collections.items():
            collection_info = {
                'collection_id': collection_id,
                'name': collection.name,
                'description': collection.description,
                'collection_type': collection.collection_type,
                'document_count': len(collection.document_ids),
                'document_types': collection.metadata.get('types', []),
                'primary_document': collection.primary_document,
                'created_at': collection.created_at,
                'last_updated': collection.last_updated
            }
            collections.append(collection_info)
        
        # Sort by last updated (most recent first)
        collections.sort(key=lambda x: x['last_updated'], reverse=True)
        
        return collections
    
    def get_document_collections(self, document_id: str) -> List[str]:
        """Get all collections that contain a document"""
        if document_id not in self.active_documents:
            return []
        
        doc_state = self.active_documents[document_id]
        return getattr(doc_state, 'collections', [])
    
    def register_document(self, file_path: str, document_type: str = 'mvr') -> str:
        """Register a new document for MVR workflow processing"""
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        
        # Generate document ID from filename
        document_id = file_path.stem
        if document_id.startswith('REV'):
            # Extract REV number for proper ID
            parts = document_id.split('_')
            if len(parts) > 0 and parts[0].startswith('REV'):
                document_id = parts[0]
        
        # Copy document to workspace
        workspace_file = self.documents_dir / file_path.name
        if not workspace_file.exists():
            import shutil
            shutil.copy2(str(file_path), str(workspace_file))
        
        # Initialize document state
        doc_state = DocumentState(
            document_id=document_id,
            file_path=str(workspace_file),
            document_type=document_type,
            current_stage='mvr_tag',  # Start at first stage
            completed_stages=[],
            checklist_status={},
            processing_history=[],
            metadata={
                'original_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_type': file_path.suffix
            },
            collections=[],  # Initialize empty collections list
            created_at=datetime.now(),
            last_updated=datetime.now()
        )
        
        # Store and save
        self.active_documents[document_id] = doc_state
        self.save_document_state(doc_state)
        
        print(f"[REGISTER] Document registered: {document_id}")
        print(f"[REGISTER]   Type: {document_type}")
        print(f"[REGISTER]   File: {file_path.name}")
        print(f"[REGISTER]   Current stage: {doc_state.current_stage}")
        
        return document_id
    
    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """Get current status of a document"""
        if document_id not in self.active_documents:
            raise ValueError(f"Document not found: {document_id}")
        
        doc_state = self.active_documents[document_id]
        
        status = {
            'document_id': document_id,
            'current_stage': doc_state.current_stage,
            'stage_description': self.stage_descriptions.get(doc_state.current_stage, 'Unknown'),
            'completed_stages': doc_state.completed_stages,
            'progress_percentage': (len(doc_state.completed_stages) / len(self.mvr_workflow_stages)) * 100,
            'checklist_status': doc_state.checklist_status,
            'last_updated': doc_state.last_updated,
            'processing_history': doc_state.processing_history[-5:],  # Last 5 entries
            'next_actions': self._get_next_actions(doc_state)
        }
        
        return status
    
    def _get_next_actions(self, doc_state: DocumentState) -> List[str]:
        """Get recommended next actions for document"""
        actions = []
        
        if doc_state.current_stage in self.mvr_workflow_stages:
            actions.append(f"Complete {self.stage_descriptions[doc_state.current_stage]}")
            actions.append(f"Get SOP guidance for {doc_state.current_stage}")
            actions.append("Advance to next stage")
        
        if doc_state.current_stage == 'mvr_qa':
            actions.append("Compare with VST document")
            actions.append("Generate comparison report")
        
        if doc_state.current_stage == 'mvr_peer':
            actions.append("Perform triangulation analysis")
            actions.append("Resolve any disagreements")
        
        return actions
    
    def get_sop_guidance(self, document_id: str, question: Optional[str] = None) -> Dict[str, Any]:
        """Get SOP guidance for current document stage"""
        if not SOP_VALIDATOR_AVAILABLE:
            return {
                'guidance': 'SOP Validator not available',
                'confidence': 0.0,
                'recommendations': ['Install tidyllm-compliance module for SOP guidance']
            }
        
        if document_id not in self.active_documents:
            raise ValueError(f"Document not found: {document_id}")
        
        doc_state = self.active_documents[document_id]
        
        # Read document content for context
        try:
            with open(doc_state.file_path, 'r', encoding='utf-8', errors='ignore') as f:
                document_text = f.read()[:5000]  # First 5000 chars for context
        except:
            document_text = "Document content not accessible"
        
        # Prepare context for SOP validation
        context = {
            'workflow_stage': doc_state.current_stage,
            'document_text': document_text,
            'document_type': doc_state.document_type,
            'completed_checklist_items': [k for k, v in doc_state.checklist_status.items() if v],
            'document_id': document_id
        }
        
        # Get SOP guidance
        if question:
            # Interactive chat mode
            sop_response = self.sop_validator.chat_with_sop(question, context)
            return {
                'mode': 'chat',
                'question': question,
                'guidance': sop_response['sop_guidance'],
                'confidence': sop_response['confidence'],
                'compliance_status': sop_response['compliance_status'],
                'recommendations': sop_response['recommendations'],
                'checklist_items': sop_response.get('checklist_items', [])
            }
        else:
            # Stage-specific guidance
            validation_result = self.sop_validator.validate_with_sop_precedence(
                f"What are the requirements for {doc_state.current_stage}?",
                context
            )
            return {
                'mode': 'stage_guidance',
                'stage': doc_state.current_stage,
                'guidance': validation_result.sop_answers[0].answer if validation_result.sop_answers else 'No specific guidance available',
                'confidence': validation_result.sop_score,
                'compliance_status': validation_result.overall_compliance,
                'recommendations': validation_result.recommendations,
                'stage_guidance': validation_result.stage_guidance
            }
    
    def update_checklist(self, document_id: str, checklist_updates: Dict[str, bool]) -> Dict[str, Any]:
        """Update checklist status for document"""
        if document_id not in self.active_documents:
            raise ValueError(f"Document not found: {document_id}")
        
        doc_state = self.active_documents[document_id]
        
        # Update checklist status
        doc_state.checklist_status.update(checklist_updates)
        
        # Log the update
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'checklist_update',
            'stage': doc_state.current_stage,
            'updates': checklist_updates,
            'user': 'human_analyst'
        }
        doc_state.processing_history.append(history_entry)
        
        # Save state
        self.save_document_state(doc_state)
        
        # Check if stage is complete
        if SOP_VALIDATOR_AVAILABLE:
            stage_requirements = self.sop_validator.get_workflow_stage_requirements(doc_state.current_stage)
            required_items = stage_requirements.get('checklist_items', [])
            completed_items = [k for k, v in doc_state.checklist_status.items() if v]
            
            stage_complete = all(item in completed_items for item in required_items)
        else:
            # Fallback: assume any checklist update indicates progress
            stage_complete = len([v for v in doc_state.checklist_status.values() if v]) > 0
        
        result = {
            'document_id': document_id,
            'checklist_status': doc_state.checklist_status,
            'stage_complete': stage_complete,
            'next_actions': self._get_next_actions(doc_state)
        }
        
        print(f"[CHECKLIST] Updated {document_id}: {len(checklist_updates)} items")
        if stage_complete:
            print(f"[CHECKLIST] Stage {doc_state.current_stage} appears complete")
        
        return result
    
    def advance_workflow_stage(self, document_id: str, force: bool = False) -> Dict[str, Any]:
        """Advance document to next workflow stage"""
        if document_id not in self.active_documents:
            raise ValueError(f"Document not found: {document_id}")
        
        doc_state = self.active_documents[document_id]
        current_stage = doc_state.current_stage
        
        # Check if current stage is complete (unless forced)
        if not force and SOP_VALIDATOR_AVAILABLE:
            context = {
                'workflow_stage': current_stage,
                'completed_checklist_items': [k for k, v in doc_state.checklist_status.items() if v],
                'document_type': doc_state.document_type
            }
            
            validation_result = self.sop_validator.validate_with_sop_precedence(
                f"Is {current_stage} complete?", context
            )
            
            if validation_result.overall_compliance == 'non_compliant':
                return {
                    'success': False,
                    'message': f'Stage {current_stage} is not complete',
                    'recommendations': validation_result.recommendations,
                    'current_stage': current_stage
                }
        
        # Find next stage
        try:
            current_index = self.mvr_workflow_stages.index(current_stage)
            if current_index < len(self.mvr_workflow_stages) - 1:
                next_stage = self.mvr_workflow_stages[current_index + 1]
            else:
                next_stage = 'completed'
        except ValueError:
            # Current stage not in standard workflow
            next_stage = 'completed'
        
        # Execute workflow transition using Universal Bracket Flows
        if FLOW_PARSER_AVAILABLE and next_stage != 'completed':
            try:
                bracket_command = f"[mvr_analysis {next_stage.split('_')[1]} {Path(doc_state.file_path).name}]"
                parsed_command = self.flow_parser.parse_bracket_command(bracket_command)
                
                print(f"[WORKFLOW] Executing: {bracket_command}")
                print(f"[WORKFLOW] Parsed: {parsed_command.workflow_name} -> {parsed_command.action}")
            
            except Exception as e:
                print(f"[WARNING] Bracket flow execution failed: {e}")
        
        # Update document state
        doc_state.completed_stages.append(current_stage)
        doc_state.current_stage = next_stage
        
        # Log the advancement
        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'stage_advance',
            'from_stage': current_stage,
            'to_stage': next_stage,
            'forced': force,
            'user': 'human_analyst'
        }
        doc_state.processing_history.append(history_entry)
        
        # Save state
        self.save_document_state(doc_state)
        
        result = {
            'success': True,
            'document_id': document_id,
            'previous_stage': current_stage,
            'current_stage': next_stage,
            'stage_description': self.stage_descriptions.get(next_stage, 'Workflow Complete'),
            'progress_percentage': (len(doc_state.completed_stages) / len(self.mvr_workflow_stages)) * 100,
            'workflow_complete': next_stage == 'completed'
        }
        
        print(f"[ADVANCE] {document_id}: {current_stage} -> {next_stage}")
        if next_stage == 'completed':
            print(f"[ADVANCE] Workflow completed for {document_id}")
        
        return result
    
    def advance_collection_workflow(self, collection_id: str, force: bool = False) -> Dict[str, Any]:
        """Advance all documents in a collection to their next workflow stages"""
        if collection_id not in self.active_collections:
            raise ValueError(f"Collection not found: {collection_id}")
        
        collection = self.active_collections[collection_id]
        results = []
        successful_advances = 0
        
        print(f"[ADVANCE] Advancing collection: {collection.name} ({len(collection.document_ids)} documents)")
        
        for doc_id in collection.document_ids:
            if doc_id not in self.active_documents:
                continue
                
            try:
                result = self.advance_workflow_stage(doc_id, force=force)
                results.append({
                    'document_id': doc_id,
                    'success': result['success'],
                    'previous_stage': result.get('previous_stage'),
                    'current_stage': result.get('current_stage'),
                    'message': result.get('message', '')
                })
                
                if result['success']:
                    successful_advances += 1
                    
            except Exception as e:
                results.append({
                    'document_id': doc_id,
                    'success': False,
                    'error': str(e)
                })
        
        # Log collection advancement
        collection.metadata['last_advancement'] = datetime.now().isoformat()
        collection.metadata['advancement_results'] = {
            'total_documents': len(collection.document_ids),
            'successful': successful_advances,
            'timestamp': datetime.now().isoformat()
        }
        self.save_collection_state(collection)
        
        overall_result = {
            'success': successful_advances > 0,
            'collection_id': collection_id,
            'collection_name': collection.name,
            'total_documents': len(collection.document_ids),
            'successful_advances': successful_advances,
            'failed_advances': len(collection.document_ids) - successful_advances,
            'individual_results': results
        }
        
        print(f"[ADVANCE] Collection advancement complete: {successful_advances}/{len(collection.document_ids)} successful")
        
        return overall_result
    
    def advance_all_collections(self, force: bool = False) -> Dict[str, Any]:
        """Advance all documents in all collections"""
        if not self.active_collections:
            return {
                'success': False,
                'message': 'No active collections to advance'
            }
        
        print(f"[ADVANCE] Advancing all collections ({len(self.active_collections)} collections)")
        
        collection_results = []
        total_documents = 0
        total_successful = 0
        
        for collection_id in self.active_collections.keys():
            try:
                result = self.advance_collection_workflow(collection_id, force=force)
                collection_results.append(result)
                total_documents += result['total_documents']
                total_successful += result['successful_advances']
                
            except Exception as e:
                collection_results.append({
                    'success': False,
                    'collection_id': collection_id,
                    'error': str(e)
                })
        
        overall_result = {
            'success': total_successful > 0,
            'total_collections': len(self.active_collections),
            'total_documents': total_documents,
            'successful_advances': total_successful,
            'failed_advances': total_documents - total_successful,
            'collection_results': collection_results
        }
        
        print(f"[ADVANCE] All collections advancement complete: {total_successful}/{total_documents} documents advanced")
        
        return overall_result
    
    def generate_collection_report(self, collection_id: str) -> Dict[str, Any]:
        """Generate comprehensive report for a document collection"""
        if collection_id not in self.active_collections:
            raise ValueError(f"Collection not found: {collection_id}")
        
        collection = self.active_collections[collection_id]
        
        # Get detailed collection info
        collection_info = self.get_collection_info(collection_id)
        
        # Generate individual document reports
        document_reports = []
        for doc_id in collection.document_ids:
            if doc_id in self.active_documents:
                try:
                    doc_report = self.generate_stage_report(doc_id)
                    document_reports.append(doc_report)
                except Exception as e:
                    document_reports.append({
                        'document_id': doc_id,
                        'error': str(e)
                    })
        
        # Create comprehensive collection report
        collection_report = {
            'collection_info': collection_info,
            'document_reports': document_reports,
            'summary': {
                'total_documents': len(collection.document_ids),
                'document_types': list(set(doc['document_type'] for doc in collection_info['documents'])),
                'stages_distribution': {},
                'compliance_status': 'unknown',
                'generated_at': datetime.now().isoformat()
            }
        }
        
        # Calculate stage distribution
        for doc in collection_info['documents']:
            stage = doc['current_stage']
            collection_report['summary']['stages_distribution'][stage] = \
                collection_report['summary']['stages_distribution'].get(stage, 0) + 1
        
        # Save collection report
        report_filename = f"{collection_id}_collection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(collection_report, f, indent=2, default=str)
        
        print(f"[REPORT] Collection report generated: {report_filename}")
        
        return {
            'report_generated': True,
            'report_path': str(report_path),
            'collection_id': collection_id,
            'collection_name': collection.name,
            'document_count': len(collection.document_ids),
            'report_data': collection_report
        }

    def chat_with_sop(self, document_id: str, question: str) -> Dict[str, Any]:
        """Interactive chat with SOP during analysis"""
        return self.get_sop_guidance(document_id, question)
    
    def list_active_documents(self) -> List[Dict[str, Any]]:
        """List all active documents with their current status"""
        documents = []
        
        for doc_id, doc_state in self.active_documents.items():
            documents.append({
                'document_id': doc_id,
                'file_name': Path(doc_state.file_path).name,
                'document_type': doc_state.document_type,
                'current_stage': doc_state.current_stage,
                'stage_description': self.stage_descriptions.get(doc_state.current_stage, 'Unknown'),
                'progress_percentage': (len(doc_state.completed_stages) / len(self.mvr_workflow_stages)) * 100,
                'last_updated': doc_state.last_updated,
                'checklist_completion': len([v for v in doc_state.checklist_status.values() if v])
            })
        
        # Sort by last updated (most recent first)
        documents.sort(key=lambda x: x['last_updated'], reverse=True)
        
        return documents
    
    def pair_documents(self, mvr_doc_id: str, vst_doc_id: str) -> Dict[str, Any]:
        """Manually pair an MVR document with a VST document"""
        if mvr_doc_id not in self.active_documents:
            raise ValueError(f"MVR document not found: {mvr_doc_id}")
        if vst_doc_id not in self.active_documents:
            raise ValueError(f"VST document not found: {vst_doc_id}")
        
        mvr_state = self.active_documents[mvr_doc_id]
        vst_state = self.active_documents[vst_doc_id]
        
        # Validate document types
        if mvr_state.document_type != 'mvr':
            raise ValueError(f"Document {mvr_doc_id} is not an MVR document")
        if vst_state.document_type != 'vst':
            raise ValueError(f"Document {vst_doc_id} is not a VST document")
        
        # Create bidirectional pairing
        mvr_state.metadata['paired_document'] = vst_doc_id
        mvr_state.metadata['pair_type'] = 'MVR-primary'
        mvr_state.metadata['pairing_method'] = 'manual'
        
        vst_state.metadata['paired_document'] = mvr_doc_id
        vst_state.metadata['pair_type'] = 'VST-validation'
        vst_state.metadata['pairing_method'] = 'manual'
        
        # Log the pairing
        timestamp = datetime.now()
        
        mvr_history_entry = {
            'timestamp': timestamp.isoformat(),
            'action': 'document_paired',
            'paired_with': vst_doc_id,
            'pair_type': 'MVR-primary',
            'method': 'manual',
            'user': 'human_analyst'
        }
        mvr_state.processing_history.append(mvr_history_entry)
        
        vst_history_entry = {
            'timestamp': timestamp.isoformat(),
            'action': 'document_paired',
            'paired_with': mvr_doc_id,
            'pair_type': 'VST-validation',
            'method': 'manual',
            'user': 'human_analyst'
        }
        vst_state.processing_history.append(vst_history_entry)
        
        # Save both states
        self.save_document_state(mvr_state)
        self.save_document_state(vst_state)
        
        print(f"[PAIR] Manually paired {mvr_doc_id} with {vst_doc_id}")
        
        return {
            'success': True,
            'mvr_document': mvr_doc_id,
            'vst_document': vst_doc_id,
            'pair_type': 'MVR-VST',
            'method': 'manual',
            'timestamp': timestamp
        }
    
    def unpair_document(self, document_id: str) -> Dict[str, Any]:
        """Remove pairing from a document"""
        if document_id not in self.active_documents:
            raise ValueError(f"Document not found: {document_id}")
        
        doc_state = self.active_documents[document_id]
        paired_doc_id = doc_state.metadata.get('paired_document')
        
        if not paired_doc_id:
            return {
                'success': False,
                'message': f"Document {document_id} is not paired"
            }
        
        # Remove pairing from current document
        doc_state.metadata.pop('paired_document', None)
        doc_state.metadata.pop('pair_type', None)
        doc_state.metadata.pop('pairing_method', None)
        
        # Remove pairing from paired document if it exists
        if paired_doc_id in self.active_documents:
            paired_state = self.active_documents[paired_doc_id]
            paired_state.metadata.pop('paired_document', None)
            paired_state.metadata.pop('pair_type', None)
            paired_state.metadata.pop('pairing_method', None)
            
            # Log unpair action for paired document
            unpair_history_entry = {
                'timestamp': datetime.now().isoformat(),
                'action': 'document_unpaired',
                'unpaired_from': document_id,
                'user': 'human_analyst'
            }
            paired_state.processing_history.append(unpair_history_entry)
            self.save_document_state(paired_state)
        
        # Log unpair action
        unpair_history_entry = {
            'timestamp': datetime.now().isoformat(),
            'action': 'document_unpaired',
            'unpaired_from': paired_doc_id,
            'user': 'human_analyst'
        }
        doc_state.processing_history.append(unpair_history_entry)
        self.save_document_state(doc_state)
        
        print(f"[UNPAIR] Removed pairing from {document_id}")
        
        return {
            'success': True,
            'document': document_id,
            'previously_paired_with': paired_doc_id,
            'timestamp': datetime.now()
        }
    
    def get_paired_document(self, document_id: str) -> Optional[str]:
        """Get the ID of the document paired with this one"""
        if document_id not in self.active_documents:
            return None
        
        doc_state = self.active_documents[document_id]
        return doc_state.metadata.get('paired_document')
    
    def list_document_pairs(self) -> List[Dict[str, Any]]:
        """List all document pairs"""
        pairs = []
        processed_docs = set()
        
        for doc_id, doc_state in self.active_documents.items():
            if doc_id in processed_docs:
                continue
                
            paired_doc_id = doc_state.metadata.get('paired_document')
            if paired_doc_id and paired_doc_id in self.active_documents:
                pair_info = {
                    'mvr_document': doc_id if doc_state.document_type == 'mvr' else paired_doc_id,
                    'vst_document': paired_doc_id if doc_state.document_type == 'mvr' else doc_id,
                    'pairing_method': doc_state.metadata.get('pairing_method', 'unknown'),
                    'created_at': doc_state.created_at,
                    'pair_status': 'active'
                }
                pairs.append(pair_info)
                processed_docs.add(doc_id)
                processed_docs.add(paired_doc_id)
        
        return pairs

    def generate_stage_report(self, document_id: str) -> Dict[str, Any]:
        """Generate report for current stage"""
        if document_id not in self.active_documents:
            raise ValueError(f"Document not found: {document_id}")
        
        doc_state = self.active_documents[document_id]
        
        # Get SOP guidance for reporting
        if SOP_VALIDATOR_AVAILABLE:
            context = {
                'workflow_stage': doc_state.current_stage,
                'document_type': doc_state.document_type,
                'completed_checklist_items': [k for k, v in doc_state.checklist_status.items() if v]
            }
            
            sop_guidance = self.sop_validator.validate_with_sop_precedence(
                f"Generate report for {doc_state.current_stage}",
                context
            )
        else:
            sop_guidance = None
        
        # Create stage report
        report = {
            'document_id': document_id,
            'stage': doc_state.current_stage,
            'stage_description': self.stage_descriptions.get(doc_state.current_stage, 'Unknown'),
            'timestamp': datetime.now().isoformat(),
            'checklist_status': doc_state.checklist_status,
            'completed_items': [k for k, v in doc_state.checklist_status.items() if v],
            'pending_items': [k for k, v in doc_state.checklist_status.items() if not v],
            'processing_history': doc_state.processing_history,
            'sop_compliance': sop_guidance.overall_compliance if sop_guidance else 'unknown',
            'recommendations': sop_guidance.recommendations if sop_guidance else []
        }
        
        # Save report to workspace
        report_filename = f"{document_id}_{doc_state.current_stage}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path = self.reports_dir / report_filename
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"[REPORT] Stage report generated: {report_filename}")
        
        return {
            'report_generated': True,
            'report_path': str(report_path),
            'report_data': report
        }

def run_interactive_demo():
    """Run interactive demo of Human-in-the-Loop MVR interface"""
    print("\n" + "="*60)
    print("HUMAN-IN-THE-LOOP MVR INTERFACE DEMO")
    print("="*60)
    
    # Initialize interface
    interface = HumanLoopMVRInterface()
    
    # Demo document content
    demo_mvr_content = """
    REV12345 Motor Vehicle Record Analysis
    =====================================
    
    Document Type: MVR (Motor Vehicle Record)
    REV ID: REV12345
    Analysis Date: 2024-01-15
    
    Driver Information:
    - Name: John Doe
    - License: ABC123456
    - State: California
    
    Violations:
    - Speeding: 2023-05-10 (65 in 55 zone)
    - Parking: 2023-08-22 (Expired meter)
    
    YNSR Noise Factor: 0.15 (Low noise - high quality data)
    Classification: Standard MVR Document
    Business Purpose: Employment Verification
    """
    
    # Create demo document
    demo_file = interface.documents_dir / 'REV12345_MVR_Demo.txt'
    with open(demo_file, 'w') as f:
        f.write(demo_mvr_content)
    
    print(f"\n[DEMO] Created demo document: {demo_file.name}")
    
    # Register document
    doc_id = interface.register_document(str(demo_file), 'mvr')
    
    # Show initial status
    print(f"\n[DEMO] Initial document status:")
    status = interface.get_document_status(doc_id)
    print(f"  Document ID: {status['document_id']}")
    print(f"  Current Stage: {status['current_stage']} - {status['stage_description']}")
    print(f"  Progress: {status['progress_percentage']:.1f}%")
    
    # Demonstrate SOP guidance
    print(f"\n[DEMO] Getting SOP guidance for current stage...")
    sop_guidance = interface.get_sop_guidance(doc_id)
    print(f"  Mode: {sop_guidance['mode']}")
    print(f"  Confidence: {sop_guidance['confidence']:.1%}")
    print(f"  Guidance: {sop_guidance['guidance'][:200]}...")
    
    # Demonstrate checklist update
    print(f"\n[DEMO] Updating checklist items...")
    checklist_updates = {
        "REV00000 format ID extracted": True,
        "Document type classified (MVR/VST)": True,
        "YNSR noise analysis completed": True
    }
    checklist_result = interface.update_checklist(doc_id, checklist_updates)
    print(f"  Items updated: {len(checklist_updates)}")
    print(f"  Stage complete: {checklist_result['stage_complete']}")
    
    # Demonstrate SOP chat
    print(f"\n[DEMO] Interactive SOP chat...")
    chat_response = interface.chat_with_sop(doc_id, "What should I do if the REV number format is non-standard?")
    print(f"  Question: What should I do if the REV number format is non-standard?")
    print(f"  SOP Response: {chat_response['guidance'][:150]}...")
    
    # Demonstrate workflow advancement
    print(f"\n[DEMO] Advancing to next workflow stage...")
    advance_result = interface.advance_workflow_stage(doc_id)
    print(f"  Success: {advance_result['success']}")
    print(f"  New Stage: {advance_result['current_stage']} - {advance_result['stage_description']}")
    print(f"  Progress: {advance_result['progress_percentage']:.1f}%")
    
    # Generate stage report
    print(f"\n[DEMO] Generating stage report...")
    report_result = interface.generate_stage_report(doc_id)
    print(f"  Report generated: {report_result['report_generated']}")
    print(f"  Report path: {Path(report_result['report_path']).name}")
    
    # List active documents
    print(f"\n[DEMO] Active documents summary:")
    active_docs = interface.list_active_documents()
    for doc in active_docs:
        print(f"  - {doc['document_id']}: {doc['stage_description']} ({doc['progress_percentage']:.1f}%)")
    
    print(f"\n[DEMO] Human-in-the-Loop MVR Interface demo completed!")
    print(f"       Workspace: {interface.workspace}")
    print(f"       Documents: {len(interface.active_documents)} active")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Human-in-the-Loop MVR Interface')
    parser.add_argument('--demo', action='store_true', help='Run interactive demo')
    parser.add_argument('--workspace', type=str, help='Workspace directory path')
    
    args = parser.parse_args()
    
    if args.demo:
        run_interactive_demo()
    else:
        print("Human-in-the-Loop MVR Interface")
        print("Use --demo to run interactive demonstration")
        print("Use --workspace to specify custom workspace directory")

if __name__ == "__main__":
    main()