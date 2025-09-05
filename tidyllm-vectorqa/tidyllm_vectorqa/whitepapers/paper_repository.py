"""
Paper Repository Management
==========================

Local repository system for managing downloaded papers, metadata, and collections.
Integrates with the Y=R+S+N search tracking system.
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
    try:
        from .s3_session_manager import S3SessionManager, get_s3_utils
    except ImportError:
        from s3_session_manager import S3SessionManager, get_s3_utils
except ImportError:
    S3_AVAILABLE = False
    boto3 = None
    S3SessionManager = None
    get_s3_utils = None

try:
    import psycopg2
    from psycopg2 import sql
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    psycopg2 = None

class PaperRepository:
    """Local repository for downloaded papers with metadata tracking"""
    
    def __init__(self, backend_config=None, base_path=None):
        """Initialize paper repository"""
        # Set up repository paths
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(__file__).parent / "paper_repository"
        
        self.papers_dir = self.base_path / "papers"
        self.metadata_dir = self.base_path / "metadata"
        self.collections_dir = self.base_path / "collections"
        
        # Create directory structure
        for directory in [self.papers_dir, self.metadata_dir, self.collections_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Database connection for integration with search tracking
        self.backend_config = backend_config
        if backend_config and POSTGRES_AVAILABLE:
            self.connection_params = {
                'host': backend_config.settings.postgres.host,
                'port': backend_config.settings.postgres.port,
                'database': backend_config.settings.postgres.database,
                'user': backend_config.settings.postgres.username,
                'password': backend_config.settings.postgres.password,
                'sslmode': backend_config.settings.postgres.ssl_mode
            }
            self._init_repository_tables()
        
        # Load repository index
        self.index_file = self.base_path / "repository_index.json"
        self.index = self._load_index()
    
    def _get_connection(self):
        """Get PostgreSQL connection"""
        if POSTGRES_AVAILABLE and self.backend_config:
            return psycopg2.connect(**self.connection_params)
        return None
    
    def _init_repository_tables(self):
        """Initialize repository tables in PostgreSQL"""
        try:
            conn = self._get_connection()
            if not conn:
                return
                
            with conn:
                with conn.cursor() as cursor:
                    # Downloaded papers table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS yrsn_downloaded_papers (
                            id SERIAL PRIMARY KEY,
                            paper_id TEXT UNIQUE NOT NULL,
                            title TEXT NOT NULL,
                            authors JSONB,
                            source TEXT,
                            file_path TEXT,
                            file_size INTEGER,
                            download_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            y_score FLOAT,
                            r_score FLOAT,
                            s_score FLOAT,
                            n_score FLOAT,
                            context_risk FLOAT,
                            collections JSONB,
                            tags JSONB,
                            notes TEXT,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Paper collections table
                    cursor.execute('''
                        CREATE TABLE IF NOT EXISTS yrsn_paper_collections (
                            id SERIAL PRIMARY KEY,
                            collection_name TEXT UNIQUE NOT NULL,
                            description TEXT,
                            paper_count INTEGER DEFAULT 0,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    # Create indexes
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_downloaded_papers_id 
                        ON yrsn_downloaded_papers(paper_id)
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_downloaded_papers_date 
                        ON yrsn_downloaded_papers(download_date DESC)
                    ''')
        
        except Exception as e:
            print(f"Failed to initialize repository tables: {e}")
    
    def _load_index(self):
        """Load repository index from JSON file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {"papers": {}, "collections": {}, "stats": {"total_papers": 0, "total_size": 0}}
        return {"papers": {}, "collections": {}, "stats": {"total_papers": 0, "total_size": 0}}
    
    def _save_index(self):
        """Save repository index to JSON file"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save repository index: {e}")
    
    def download_paper(self, paper_id: str, title: str, authors: List[str], 
                      source: str = "ArXiv", url: str = None, 
                      y_score: float = None, r_score: float = None, 
                      s_score: float = None, n_score: float = None,
                      context_risk: float = None) -> Dict[str, Any]:
        """Download paper and add to repository"""
        try:
            # Generate filename and paths
            safe_title = "".join(c for c in title[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{paper_id}_{safe_title.replace(' ', '_')}.pdf"
            file_path = self.papers_dir / filename
            
            # Skip if already downloaded
            if file_path.exists():
                return {"success": False, "message": "Paper already in repository", "path": str(file_path)}
            
            # Determine download URL
            if not url:
                if source == "ArXiv":
                    clean_id = paper_id.split('v')[0] if 'v' in paper_id else paper_id
                    url = f"https://arxiv.org/pdf/{clean_id}.pdf"
                else:
                    return {"success": False, "message": f"No download URL for {source}"}
            
            # Download the paper
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            # Save the PDF
            with open(file_path, 'wb') as f:
                shutil.copyfileobj(response.raw, f)
            
            file_size = file_path.stat().st_size
            
            # Create metadata
            metadata = {
                "paper_id": paper_id,
                "title": title,
                "authors": authors,
                "source": source,
                "file_path": str(file_path),
                "file_size": file_size,
                "download_date": datetime.now().isoformat(),
                "y_score": y_score,
                "r_score": r_score,
                "s_score": s_score,
                "n_score": n_score,
                "context_risk": context_risk,
                "collections": [],
                "tags": [],
                "notes": ""
            }
            
            # Save metadata to JSON
            metadata_file = self.metadata_dir / f"{paper_id}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update index
            self.index["papers"][paper_id] = {
                "title": title,
                "authors": authors,
                "source": source,
                "file_path": str(file_path),
                "file_size": file_size,
                "download_date": datetime.now().isoformat(),
                "y_score": y_score,
                "collections": []
            }
            
            self.index["stats"]["total_papers"] += 1
            self.index["stats"]["total_size"] += file_size
            self._save_index()
            
            # Save to database if available
            self._save_to_database(metadata)
            
            return {
                "success": True,
                "message": f"Downloaded {title}",
                "path": str(file_path),
                "file_size": file_size
            }
            
        except Exception as e:
            return {"success": False, "message": f"Download failed: {e}"}
    
    def _save_to_database(self, metadata):
        """Save paper metadata to PostgreSQL database"""
        try:
            conn = self._get_connection()
            if not conn:
                return
                
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute('''
                        INSERT INTO yrsn_downloaded_papers 
                        (paper_id, title, authors, source, file_path, file_size, 
                         y_score, r_score, s_score, n_score, context_risk, collections, tags, notes)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (paper_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        file_path = EXCLUDED.file_path,
                        file_size = EXCLUDED.file_size
                    ''', (
                        metadata["paper_id"], metadata["title"], json.dumps(metadata["authors"]),
                        metadata["source"], metadata["file_path"], metadata["file_size"],
                        metadata["y_score"], metadata["r_score"], metadata["s_score"],
                        metadata["n_score"], metadata["context_risk"], 
                        json.dumps(metadata["collections"]), json.dumps(metadata["tags"]), metadata["notes"]
                    ))
        except Exception as e:
            print(f"Failed to save to database: {e}")
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """Get repository statistics"""
        stats = self.index.get("stats", {"total_papers": 0, "total_size": 0})
        
        # Calculate size in MB
        size_mb = stats["total_size"] / (1024 * 1024) if stats["total_size"] > 0 else 0
        
        # Count by source
        source_counts = {}
        for paper_info in self.index.get("papers", {}).values():
            source = paper_info.get("source", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return {
            "total_papers": stats["total_papers"],
            "total_size_mb": round(size_mb, 2),
            "source_breakdown": source_counts,
            "repository_path": str(self.base_path)
        }
    
    def list_papers(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List papers in repository"""
        papers = []
        for paper_id, info in list(self.index.get("papers", {}).items())[:limit]:
            papers.append({
                "paper_id": paper_id,
                "title": info.get("title", ""),
                "authors": info.get("authors", []),
                "source": info.get("source", ""),
                "file_size_mb": round(info.get("file_size", 0) / (1024 * 1024), 2),
                "download_date": info.get("download_date", ""),
                "y_score": info.get("y_score"),
                "collections": info.get("collections", [])
            })
        
        return sorted(papers, key=lambda x: x["download_date"], reverse=True)
    
    def search_papers(self, query: str) -> List[Dict[str, Any]]:
        """Search papers in repository by title or authors"""
        query = query.lower()
        results = []
        
        for paper_id, info in self.index.get("papers", {}).items():
            title = info.get("title", "").lower()
            authors = " ".join(info.get("authors", [])).lower()
            
            if query in title or query in authors:
                results.append({
                    "paper_id": paper_id,
                    "title": info.get("title", ""),
                    "authors": info.get("authors", []),
                    "source": info.get("source", ""),
                    "file_size_mb": round(info.get("file_size", 0) / (1024 * 1024), 2),
                    "download_date": info.get("download_date", ""),
                    "y_score": info.get("y_score"),
                    "file_path": info.get("file_path", "")
                })
        
        return sorted(results, key=lambda x: x.get("y_score", 0), reverse=True)
    
    def create_collection(self, name: str, description: str = "") -> Dict[str, Any]:
        """Create a new paper collection"""
        try:
            if name in self.index.get("collections", {}):
                return {"success": False, "message": "Collection already exists"}
            
            self.index["collections"][name] = {
                "description": description,
                "papers": [],
                "created_at": datetime.now().isoformat()
            }
            
            self._save_index()
            
            # Save to database
            conn = self._get_connection()
            if conn:
                with conn:
                    with conn.cursor() as cursor:
                        cursor.execute('''
                            INSERT INTO yrsn_paper_collections (collection_name, description)
                            VALUES (%s, %s)
                        ''', (name, description))
            
            return {"success": True, "message": f"Created collection: {name}"}
        
        except Exception as e:
            return {"success": False, "message": f"Failed to create collection: {e}"}
    
    def add_to_collection(self, paper_id: str, collection_name: str) -> Dict[str, Any]:
        """Add paper to collection"""
        try:
            if paper_id not in self.index.get("papers", {}):
                return {"success": False, "message": "Paper not in repository"}
            
            if collection_name not in self.index.get("collections", {}):
                return {"success": False, "message": "Collection does not exist"}
            
            # Add to collection
            if paper_id not in self.index["collections"][collection_name]["papers"]:
                self.index["collections"][collection_name]["papers"].append(paper_id)
                
                # Add to paper's collections list
                if "collections" not in self.index["papers"][paper_id]:
                    self.index["papers"][paper_id]["collections"] = []
                if collection_name not in self.index["papers"][paper_id]["collections"]:
                    self.index["papers"][paper_id]["collections"].append(collection_name)
                
                self._save_index()
                
                return {"success": True, "message": f"Added to collection: {collection_name}"}
            else:
                return {"success": False, "message": "Paper already in collection"}
        
        except Exception as e:
            return {"success": False, "message": f"Failed to add to collection: {e}"}
    
    def remove_from_collection(self, paper_id: str, collection_name: str) -> Dict[str, Any]:
        """Remove paper from collection"""
        try:
            if paper_id not in self.index.get("papers", {}):
                return {"success": False, "message": "Paper not in repository"}
            
            if collection_name not in self.index.get("collections", {}):
                return {"success": False, "message": "Collection does not exist"}
            
            # Remove from collection
            if paper_id in self.index["collections"][collection_name]["papers"]:
                self.index["collections"][collection_name]["papers"].remove(paper_id)
                
                # Remove from paper's collections list
                if "collections" in self.index["papers"][paper_id]:
                    if collection_name in self.index["papers"][paper_id]["collections"]:
                        self.index["papers"][paper_id]["collections"].remove(collection_name)
                
                self._save_index()
                
                return {"success": True, "message": f"Removed from collection: {collection_name}"}
            else:
                return {"success": False, "message": "Paper not in this collection"}
        
        except Exception as e:
            return {"success": False, "message": f"Failed to remove from collection: {e}"}
    
    def get_collections(self) -> List[Dict[str, Any]]:
        """Get all collections"""
        collections = []
        for name, info in self.index.get("collections", {}).items():
            collections.append({
                "name": name,
                "description": info.get("description", ""),
                "paper_count": len(info.get("papers", [])),
                "created_at": info.get("created_at", "")
            })
        
        return sorted(collections, key=lambda x: x["created_at"], reverse=True)
    
    def sync_to_s3(self, bucket_name: str, prefix: str = "papers/", 
                   aws_access_key_id: str = None, aws_secret_access_key: str = None,
                   region_name: str = "us-east-1") -> Dict[str, Any]:
        """Sync repository to S3 bucket for cloud storage and collaboration"""
        try:
            # Use the enhanced S3 session manager with proven patterns
            s3_manager = S3SessionManager(region=region_name)
            s3_utils = get_s3_utils(s3_manager)
            
            # Test connection first
            connection_test = s3_manager.test_connection()
            if not connection_test["success"]:
                return connection_test
            
            # Create bucket if it doesn't exist
            bucket_result = s3_manager.create_bucket_if_not_exists(bucket_name)
            if not bucket_result["success"]:
                return bucket_result
            
            uploaded_count = 0
            total_size = 0
            errors = []
            
            # Upload papers
            for paper_id, paper_info in self.index.get("papers", {}).items():
                file_path = Path(paper_info.get("file_path", ""))
                if file_path.exists():
                    s3_key = f"{prefix}papers/{file_path.name}"
                    metadata = {
                        "paper-id": paper_id,
                        "title": paper_info.get("title", "")[:1000],  # S3 metadata limit
                        "source": paper_info.get("source", ""),
                        "y-score": str(paper_info.get("y_score", ""))
                    }
                    
                    result = s3_manager.upload_file(str(file_path), bucket_name, s3_key, metadata)
                    if result["success"]:
                        uploaded_count += 1
                        total_size += result.get("file_size", 0)
                    else:
                        errors.append(f"Failed to upload {file_path.name}: {result['message']}")
            
            # Upload metadata files
            for metadata_file in self.metadata_dir.glob("*.json"):
                s3_key = f"{prefix}metadata/{metadata_file.name}"
                result = s3_manager.upload_file(str(metadata_file), bucket_name, s3_key)
                if not result["success"]:
                    errors.append(f"Failed to upload metadata {metadata_file.name}: {result['message']}")
            
            # Upload repository index
            if self.index_file.exists():
                s3_key = f"{prefix}repository_index.json"
                result = s3_manager.upload_file(str(self.index_file), bucket_name, s3_key)
                if not result["success"]:
                    errors.append(f"Failed to upload repository index: {result['message']}")
            
            # Upload collections
            for collection_file in self.collections_dir.glob("*.json"):
                s3_key = f"{prefix}collections/{collection_file.name}"
                result = s3_manager.upload_file(str(collection_file), bucket_name, s3_key)
                if not result["success"]:
                    errors.append(f"Failed to upload collection {collection_file.name}: {result['message']}")
            
            message = f"Uploaded {uploaded_count} papers ({total_size / (1024*1024):.1f} MB) to s3://{bucket_name}/{prefix}"
            if errors:
                message += f" (with {len(errors)} errors)"
            
            return {
                "success": len(errors) == 0 or uploaded_count > 0,
                "message": message,
                "uploaded_count": uploaded_count,
                "total_size_mb": total_size / (1024*1024),
                "errors": errors,
                "s3_url": f"s3://{bucket_name}/{prefix}",
                "credential_source": s3_manager.get_credential_status()["source"]
            }
            
        except ImportError:
            return {"success": False, "message": "S3 session manager not available"}
        except Exception as e:
            return {"success": False, "message": f"Sync failed: {str(e)}"}
    
    def sync_from_s3(self, bucket_name: str, prefix: str = "papers/",
                     aws_access_key_id: str = None, aws_secret_access_key: str = None,
                     region_name: str = "us-east-1") -> Dict[str, Any]:
        """Sync from S3 bucket to local repository"""
        if not S3_AVAILABLE:
            return {"success": False, "message": "boto3 not available. Install with: pip install boto3"}
        
        try:
            # Use enhanced S3 session manager with proven credential discovery
            s3_session_manager = S3SessionManager(region=region_name)
            s3_utils = get_s3_utils(s3_session_manager)
            s3_client = s3_session_manager.get_s3_client()
            
            downloaded_count = 0
            errors = []
            
            # List all objects with the prefix
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
                if 'Contents' not in page:
                    continue
                
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    
                    # Determine local path based on S3 key structure
                    relative_path = s3_key[len(prefix):] if s3_key.startswith(prefix) else s3_key
                    local_path = self.base_path / relative_path
                    
                    # Create parent directories
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Download file if it doesn't exist locally or is newer
                    try:
                        if not local_path.exists():
                            s3_client.download_file(bucket_name, s3_key, str(local_path))
                            downloaded_count += 1
                    except Exception as e:
                        errors.append(f"Failed to download {s3_key}: {e}")
            
            # Reload index after sync
            self.index = self._load_index()
            
            message = f"Downloaded {downloaded_count} files from s3://{bucket_name}/{prefix}"
            if errors:
                message += f" (with {len(errors)} errors)"
            
            return {
                "success": True,
                "message": message,
                "downloaded_count": downloaded_count,
                "errors": errors
            }
            
        except ClientError as e:
            return {"success": False, "message": f"S3 error: {e}"}
        except Exception as e:
            return {"success": False, "message": f"Sync failed: {e}"}
    
    def generate_embedding_comparison_report(self, papers: List[str] = None) -> Dict[str, Any]:
        """Generate embedding-based comparison report for scientists using TidyLLM transformers"""
        try:
            # Import TidyLLM sentence transformers with multiple path attempts
            import sys
            from pathlib import Path
            
            # Try multiple possible paths for tidyllm-sentence and tlm
            tidyllm_paths = [
                str(Path(__file__).parent.parent.parent / 'tidyllm-sentence'),
                str(Path(__file__).parent.parent.parent.parent / 'tidyllm-sentence'),
                r'C:\Users\marti\github\tidyllm-sentence'
            ]
            
            tlm_paths = [
                str(Path(__file__).parent.parent.parent / 'tlm'),
                str(Path(__file__).parent.parent.parent.parent / 'tlm'),
                r'C:\Users\marti\github\tlm'
            ]
            
            # Add paths to sys.path
            for path in tidyllm_paths:
                if Path(path).exists():
                    sys.path.insert(0, path)
                    break
                    
            for path in tlm_paths:
                if Path(path).exists():
                    sys.path.insert(0, path)
                    break
            
            from tidyllm_sentence import transformer_fit_transform, cosine_similarity, semantic_search
            
            # Get papers to compare (default to all if none specified)
            if papers is None:
                papers = list(self.index.get("papers", {}).keys())
            
            if len(papers) < 2:
                return {
                    "success": False,
                    "message": "Need at least 2 papers for comparison",
                    "total_papers": len(papers)
                }
            
            # Extract text content from papers for embedding
            paper_texts = []
            paper_info = []
            
            for paper_id in papers:
                paper_data = self.index["papers"][paper_id]
                # Use title + abstract if available, otherwise just title
                text_content = paper_data.get("title", "")
                if paper_data.get("abstract"):
                    text_content += " " + paper_data["abstract"]
                
                paper_texts.append(text_content)
                paper_info.append({
                    "id": paper_id,
                    "title": paper_data.get("title", "Unknown"),
                    "y_score": paper_data.get("y_score", 0),
                    "source": paper_data.get("source", "Unknown")
                })
            
            # Generate embeddings using transformer-enhanced TF-IDF
            embeddings, model = transformer_fit_transform(
                paper_texts,
                attention_heads=4,
                max_seq_len=64
            )
            
            # Calculate pairwise similarities
            similarities = []
            for i in range(len(papers)):
                for j in range(i + 1, len(papers)):
                    similarity = cosine_similarity(embeddings[i], embeddings[j])
                    similarities.append({
                        "paper1": paper_info[i],
                        "paper2": paper_info[j],
                        "similarity": similarity,
                        "similarity_percent": round(similarity * 100, 1)
                    })
            
            # Sort by similarity (highest first)
            similarities.sort(key=lambda x: x["similarity"], reverse=True)
            
            # Find most similar pair
            most_similar = similarities[0] if similarities else None
            
            # Generate recommendations for each paper
            recommendations = []
            for i, paper in enumerate(paper_info):
                # Find most similar papers to this one
                similar_indices = semantic_search(embeddings[i], embeddings, top_k=min(3, len(embeddings)-1))
                similar_papers = [
                    {
                        "paper": paper_info[idx],
                        "similarity": sim,
                        "similarity_percent": round(sim * 100, 1)
                    }
                    for idx, sim in similar_indices if idx != i
                ]
                
                recommendations.append({
                    "target_paper": paper,
                    "similar_papers": similar_papers
                })
            
            # Calculate average similarity for collection
            avg_similarity = sum(s["similarity"] for s in similarities) / len(similarities) if similarities else 0
            
            report = {
                "success": True,
                "total_papers": len(papers),
                "embedding_model": "TidyLLM Transformer-enhanced TF-IDF",
                "model_details": {
                    "attention_heads": 4,
                    "max_seq_len": 64,
                    "vocab_size": model.vocab_size,
                    "embedding_dimension": model.d_model
                },
                "collection_stats": {
                    "average_similarity": round(avg_similarity, 3),
                    "average_similarity_percent": round(avg_similarity * 100, 1),
                    "total_comparisons": len(similarities),
                    "most_similar_pair": most_similar
                },
                "pairwise_similarities": similarities,
                "recommendations": recommendations,
                "analysis_summary": self._generate_analysis_summary(similarities, avg_similarity)
            }
            
            return report
            
        except ImportError as e:
            return {
                "success": False,
                "message": f"TidyLLM sentence transformers not available: {e}",
                "fallback_available": True,
                "debug_info": {
                    "paths_checked": [
                        str(Path(__file__).parent.parent.parent / 'tidyllm-sentence'),
                        str(Path(__file__).parent.parent.parent.parent / 'tidyllm-sentence'),
                        r'C:\Users\marti\github\tidyllm-sentence'
                    ],
                    "current_file": str(Path(__file__)),
                    "sys_path_added": True
                }
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Error generating comparison: {e}",
                "total_papers": len(papers) if papers else 0
            }
    
    def _generate_analysis_summary(self, similarities: List[Dict], avg_similarity: float) -> List[str]:
        """Generate human-readable analysis summary"""
        summary = []
        
        if avg_similarity > 0.7:
            summary.append("HIGH COHESION: Papers are highly similar - excellent for focused research")
        elif avg_similarity > 0.5:
            summary.append("MODERATE COHESION: Papers share common themes with some diversity")
        elif avg_similarity > 0.3:
            summary.append("DIVERSE COLLECTION: Papers span different topics - good for broad research")
        else:
            summary.append("HIGHLY DIVERSE: Papers are quite different - excellent for interdisciplinary research")
        
        if len(similarities) >= 3:
            high_sim_count = sum(1 for s in similarities if s["similarity"] > 0.6)
            if high_sim_count > len(similarities) * 0.5:
                summary.append("Multiple highly similar paper pairs detected")
        
        # Y-score integration
        y_scores = [s["paper1"]["y_score"] for s in similarities] + [s["paper2"]["y_score"] for s in similarities]
        if y_scores and any(score > 0 for score in y_scores):
            avg_y = sum(y_scores) / len(y_scores)
            summary.append(f"Average Y-score (relevance): {avg_y:.2f}")
        
        return summary

# Helper function to get default repository instance
def get_paper_repository(backend_config=None):
    """Get default paper repository instance"""
    return PaperRepository(backend_config=backend_config)