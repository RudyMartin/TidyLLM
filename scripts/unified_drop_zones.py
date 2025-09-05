#!/usr/bin/env python3
"""
Unified Drop Zones System
=========================

Consolidates all scattered drop zone implementations into single UnifiedSessionManager pattern.

Replaces and consolidates:
- drop_zones/working_s3_dropzones.py
- drop_zones/fixed_s3_dropzones.py  
- drop_zones/s3_chat_system.py
- drop_zones/simple_fixed_s3_dropzones.py

Uses official UnifiedSessionManager for all AWS, database, and MLflow operations.
"""

import os
import sys
import json
import time
import yaml
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import uuid

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import official UnifiedSessionManager
from scripts.start_unified_sessions import UnifiedSessionManager

@dataclass
class DocumentState:
    """Tracks document processing state"""
    document_id: str
    filename: str
    file_path: str
    doc_type: str
    status: str  # 'detected', 'processing', 'completed', 'failed'
    s3_key: Optional[str] = None
    embedding_status: str = 'pending'
    created_at: datetime = None
    updated_at: datetime = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.metadata is None:
            self.metadata = {}

@dataclass
class DocumentCollection:
    """Manages a collection of related documents for analysis"""
    collection_id: str
    name: str
    description: str
    document_ids: List[str]
    collection_type: str  # 'mvr_analysis', 'peer_review', 'comparison_set', 'custom'
    primary_document: Optional[str] = None
    metadata: Dict[str, Any] = None
    created_at: datetime = None
    last_updated: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.last_updated is None:
            self.last_updated = datetime.now()
        if self.metadata is None:
            self.metadata = {}
    
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
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None
        }

class UnifiedDropZones:
    """
    Unified Drop Zones System using UnifiedSessionManager
    
    Consolidates all scattered drop zone patterns into single implementation
    using official session management architecture.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize unified drop zones with UnifiedSessionManager"""
        
        print("[INIT] Initializing Unified Drop Zones System...")
        
        # Initialize UnifiedSessionManager (official architecture)
        self.session_mgr = UnifiedSessionManager()
        
        # Setup paths
        self.drop_zone_path = Path("drop_zones")
        self.input_path = self.drop_zone_path / "input"
        self.processed_path = self.drop_zone_path / "processed" 
        self.failed_path = self.drop_zone_path / "failed"
        self.collections_path = self.drop_zone_path / "collections"
        self.state_path = self.drop_zone_path / "state"
        
        # Create directories
        for path in [self.drop_zone_path, self.input_path, self.processed_path, 
                    self.failed_path, self.collections_path, self.state_path]:
            path.mkdir(exist_ok=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize tracking
        self.processing_results: List[Dict] = []
        self.active_documents: Dict[str, DocumentState] = {}
        self.active_collections: Dict[str, DocumentCollection] = {}
        
        # Load existing state
        self.load_existing_collections()
        self.load_existing_documents()
        
        print("[OK] Unified Drop Zones initialized with UnifiedSessionManager")
        print(f"   Drop zone: {self.drop_zone_path}")
        print(f"   S3 bucket: {self.config.get('s3_bucket', 'nsc-mvp1')}")
        print(f"   Session manager: UnifiedSessionManager (official)")
    
    def _load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """Load configuration using standard patterns"""
        
        # Default configuration
        default_config = {
            's3_bucket': 'nsc-mvp1',
            'region': 'us-east-1',
            'tracking_enabled': True,
            'mlflow_tracking': True,
            'document_types': ['pdf', 'txt', 'md', 'json'],
            'max_file_size_mb': 50
        }
        
        # Try to load from config file
        config_files = [
            config_path,
            'drop_zones.yaml',
            'drop_zones/config.yaml',
            'tidyllm/admin/embeddings_settings.yaml'
        ]
        
        for config_file in config_files:
            if config_file and Path(config_file).exists():
                try:
                    with open(config_file, 'r') as f:
                        file_config = yaml.safe_load(f) or {}
                    default_config.update(file_config)
                    print(f"[CONFIG] Loaded from: {config_file}")
                    break
                except Exception as e:
                    print(f"[WARN] Failed to load config from {config_file}: {e}")
        
        return default_config
    
    def upload_to_s3(self, file_path: str, doc_type: str) -> Optional[str]:
        """Upload file to S3 using UnifiedSessionManager"""
        
        try:
            # Generate S3 key
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = Path(file_path).name
            s3_key = f"dropzones/{timestamp}/{doc_type}/{filename}"
            
            # Upload using UnifiedSessionManager
            bucket = self.config['s3_bucket']
            success = self.session_mgr.upload_to_s3(bucket, s3_key, file_path)
            
            if success:
                print(f"[S3] Uploaded: s3://{bucket}/{s3_key}")
                
                # Log to MLflow using UnifiedSessionManager
                self.session_mgr.log_mlflow_experiment({
                    'operation': 'drop_zone_upload',
                    's3_bucket': bucket,
                    's3_key': s3_key,
                    'doc_type': doc_type,
                    'file_size_bytes': Path(file_path).stat().st_size,
                    'timestamp': timestamp
                })
                
                return s3_key
            else:
                print(f"[ERROR] S3 upload failed for {file_path}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Upload failed: {e}")
            return None
    
    def process_embeddings(self, document_id: str, file_path: str) -> bool:
        """Generate embeddings using TidyLLM native stack"""
        
        try:
            # Use TidyLLM native embeddings (following constraints)
            import tidyllm_sentence as tls
            
            # Read document content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Generate embeddings using TidyLLM native (not sentence-transformers)
            embeddings, model = tls.tfidf_fit_transform([content])
            
            # Store embeddings in PostgreSQL using UnifiedSessionManager
            self.session_mgr.execute_postgres_query(
                """INSERT INTO document_embeddings 
                   (document_id, file_path, embedding, model_type, created_at) 
                   VALUES (%s, %s, %s, %s, %s)""",
                (document_id, file_path, json.dumps(embeddings[0]), 
                 'tidyllm-sentence-tfidf', datetime.now())
            )
            
            print(f"[EMBED] Generated embeddings for {document_id}")
            return True
            
        except Exception as e:
            print(f"[ERROR] Embedding generation failed: {e}")
            return False
    
    def process_single_file(self, file_path: str, doc_type: str) -> Dict[str, Any]:
        """Complete ingest-embed-index-track-report pipeline for single file"""
        
        print(f"\n[PIPELINE] Processing: {file_path}")
        
        # Generate document ID
        document_id = f"doc_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Create document state
        doc_state = DocumentState(
            document_id=document_id,
            filename=Path(file_path).name,
            file_path=file_path,
            doc_type=doc_type,
            status='processing'
        )
        
        self.active_documents[document_id] = doc_state
        
        pipeline_result = {
            'document_id': document_id,
            'filename': doc_state.filename,
            'steps': {},
            'overall_status': 'processing',
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # STEP 1: INGEST - Upload to S3
            print(f"[STEP 1/5] INGEST - Uploading to S3...")
            s3_key = self.upload_to_s3(file_path, doc_type)
            if s3_key:
                doc_state.s3_key = s3_key
                doc_state.status = 'uploaded'
                pipeline_result['steps']['ingest'] = {'status': 'SUCCESS', 's3_key': s3_key}
                print(f"[STEP 1/5] INGEST -> SUCCESS")
            else:
                pipeline_result['steps']['ingest'] = {'status': 'FAILED', 'error': 'S3 upload failed'}
                doc_state.status = 'failed'
                print(f"[STEP 1/5] INGEST -> FAILED")
                return pipeline_result
            
            # STEP 2: EMBED - Generate embeddings
            print(f"[STEP 2/5] EMBED - Generating embeddings...")
            if self.process_embeddings(document_id, file_path):
                doc_state.embedding_status = 'completed'
                pipeline_result['steps']['embed'] = {'status': 'SUCCESS', 'model': 'tidyllm-sentence-tfidf'}
                print(f"[STEP 2/5] EMBED -> SUCCESS")
            else:
                pipeline_result['steps']['embed'] = {'status': 'FAILED', 'error': 'Embedding generation failed'}
                print(f"[STEP 2/5] EMBED -> FAILED")
            
            # STEP 3: INDEX - Store metadata
            print(f"[STEP 3/5] INDEX - Storing metadata...")
            try:
                self.session_mgr.execute_postgres_query(
                    """INSERT INTO document_metadata 
                       (document_id, filename, doc_type, s3_key, file_size, created_at) 
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (document_id, doc_state.filename, doc_type, s3_key,
                     Path(file_path).stat().st_size, datetime.now())
                )
                pipeline_result['steps']['index'] = {'status': 'SUCCESS'}
                print(f"[STEP 3/5] INDEX -> SUCCESS")
            except Exception as e:
                pipeline_result['steps']['index'] = {'status': 'FAILED', 'error': str(e)}
                print(f"[STEP 3/5] INDEX -> FAILED: {e}")
            
            # STEP 4: TRACK - MLflow experiment
            print(f"[STEP 4/5] TRACK - Logging experiment...")
            try:
                self.session_mgr.log_mlflow_experiment({
                    'document_id': document_id,
                    'filename': doc_state.filename,
                    'doc_type': doc_type,
                    's3_key': s3_key,
                    'embedding_model': 'tidyllm-sentence-tfidf',
                    'pipeline_status': 'completed',
                    'processing_time': time.time() - int(document_id.split('_')[1])
                })
                pipeline_result['steps']['track'] = {'status': 'SUCCESS', 'mlflow': 'logged'}
                print(f"[STEP 4/5] TRACK -> SUCCESS")
            except Exception as e:
                pipeline_result['steps']['track'] = {'status': 'FAILED', 'error': str(e)}
                print(f"[STEP 4/5] TRACK -> FAILED: {e}")
            
            # STEP 5: REPORT - Generate summary
            print(f"[STEP 5/5] REPORT - Generating summary...")
            report = {
                'document_id': document_id,
                'filename': doc_state.filename,
                'doc_type': doc_type,
                's3_location': f"s3://{self.config['s3_bucket']}/{s3_key}",
                'embedding_dimensions': 'variable (TF-IDF)',
                'processing_time': f"{time.time() - int(document_id.split('_')[1]):.2f}s",
                'pipeline_steps_completed': len([s for s in pipeline_result['steps'].values() if s['status'] == 'SUCCESS']),
                'total_pipeline_steps': 5
            }
            pipeline_result['steps']['report'] = {'status': 'SUCCESS', 'report': report}
            pipeline_result['overall_status'] = 'completed'
            doc_state.status = 'completed'
            print(f"[STEP 5/5] REPORT -> SUCCESS")
            
            # Move file to processed
            processed_file = self.processed_path / doc_state.filename
            Path(file_path).rename(processed_file)
            doc_state.file_path = str(processed_file)
            
            print(f"\n[PIPELINE] COMPLETED: {document_id}")
            print(f"   S3: s3://{self.config['s3_bucket']}/{s3_key}")
            print(f"   Embeddings: Generated with tidyllm-sentence")
            print(f"   Status: {doc_state.status}")
            
        except Exception as e:
            pipeline_result['overall_status'] = 'failed'
            pipeline_result['error'] = str(e)
            doc_state.status = 'failed'
            print(f"[PIPELINE] FAILED: {e}")
        
        # Save document state
        self.save_document_state(document_id)
        
        return pipeline_result
    
    def monitor_drop_zone(self, duration_minutes: int = 60):
        """Monitor drop zone for new files and process them"""
        
        print(f"\n[MONITOR] Starting drop zone monitoring for {duration_minutes} minutes...")
        print(f"[MONITOR] Watching: {self.input_path}")
        print(f"[MONITOR] Session Manager: UnifiedSessionManager (official)")
        
        start_time = time.time()
        processed_files = set()
        
        while time.time() - start_time < duration_minutes * 60:
            try:
                # Check for new files
                for file_path in self.input_path.iterdir():
                    if file_path.is_file() and str(file_path) not in processed_files:
                        
                        # Determine document type
                        doc_type = self._determine_doc_type(file_path)
                        if doc_type:
                            print(f"\n[DETECTED] New file: {file_path.name}")
                            
                            # Process through complete pipeline
                            result = self.process_single_file(str(file_path), doc_type)
                            self.processing_results.append(result)
                            processed_files.add(str(file_path))
                            
                        else:
                            print(f"[SKIP] Unsupported file type: {file_path.name}")
                
                # Wait before next check
                time.sleep(5)
                
            except KeyboardInterrupt:
                print("\n[MONITOR] Stopped by user")
                break
            except Exception as e:
                print(f"[ERROR] Monitor error: {e}")
                time.sleep(10)
        
        print(f"\n[MONITOR] Completed. Processed {len(processed_files)} files.")
        return self.processing_results
    
    def _determine_doc_type(self, file_path: Path) -> Optional[str]:
        """Determine document type from file extension"""
        extension = file_path.suffix.lower().lstrip('.')
        if extension in self.config['document_types']:
            return extension
        return None
    
    def save_document_state(self, document_id: str):
        """Save document state to disk"""
        if document_id in self.active_documents:
            state_file = self.state_path / f"{document_id}.json"
            with open(state_file, 'w') as f:
                json.dump(asdict(self.active_documents[document_id]), f, 
                         default=str, indent=2)
    
    def load_existing_documents(self):
        """Load existing document states"""
        for state_file in self.state_path.glob("*.json"):
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                doc_state = DocumentState(**data)
                self.active_documents[doc_state.document_id] = doc_state
            except Exception as e:
                print(f"[WARN] Failed to load document state {state_file}: {e}")
    
    def load_existing_collections(self):
        """Load existing collections"""
        for collection_file in self.collections_path.glob("*.json"):
            try:
                with open(collection_file, 'r') as f:
                    data = json.load(f)
                collection = DocumentCollection(**data)
                self.active_collections[collection.collection_id] = collection
            except Exception as e:
                print(f"[WARN] Failed to load collection {collection_file}: {e}")
    
    def get_processing_summary(self) -> Dict[str, Any]:
        """Get summary of processing results using UnifiedSessionManager"""
        
        try:
            # Get document counts from database
            doc_counts = self.session_mgr.execute_postgres_query(
                "SELECT doc_type, COUNT(*) as count FROM document_metadata GROUP BY doc_type"
            )
            
            # Get embedding counts
            embedding_counts = self.session_mgr.execute_postgres_query(
                "SELECT COUNT(*) as total_embeddings FROM document_embeddings"
            )
            
            summary = {
                'session_manager': 'UnifiedSessionManager (official)',
                'total_active_documents': len(self.active_documents),
                'total_collections': len(self.active_collections),
                'processing_results': len(self.processing_results),
                'document_types': dict(doc_counts) if doc_counts else {},
                'total_embeddings': embedding_counts[0][0] if embedding_counts else 0,
                's3_bucket': self.config['s3_bucket'],
                'last_updated': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            print(f"[ERROR] Summary generation failed: {e}")
            return {'error': str(e), 'session_manager': 'UnifiedSessionManager (official)'}

def main():
    """Main entry point for unified drop zones"""
    
    print("=" * 60)
    print("UNIFIED DROP ZONES SYSTEM")  
    print("Using UnifiedSessionManager (Official Architecture)")
    print("=" * 60)
    
    # Initialize unified drop zones
    drop_zones = UnifiedDropZones()
    
    # Get user input for monitoring duration
    try:
        duration = input("\nEnter monitoring duration in minutes (default: 10): ").strip()
        duration = int(duration) if duration else 10
    except ValueError:
        duration = 10
    
    print(f"\nStarting drop zone monitoring for {duration} minutes...")
    print("Drop files into: drop_zones/input/")
    print("Press Ctrl+C to stop early")
    
    # Start monitoring
    results = drop_zones.monitor_drop_zone(duration)
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    
    summary = drop_zones.get_processing_summary()
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print(f"\nTotal files processed: {len(results)}")
    for i, result in enumerate(results, 1):
        status = result.get('overall_status', 'unknown')
        filename = result.get('filename', 'unknown')
        print(f"  {i}. {filename} -> {status.upper()}")

if __name__ == "__main__":
    main()