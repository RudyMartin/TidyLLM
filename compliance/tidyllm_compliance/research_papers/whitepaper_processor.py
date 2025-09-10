#!/usr/bin/env python3
"""
Whitepaper Processor for Regulatory Compliance
==============================================

PDF processing and document management for regulatory compliance research:
- Upload and process regulatory whitepapers and research documents
- Extract text and metadata for compliance analysis
- Integration with ResearchFramework for Y="+", C="-" decomposition
- S3-first architecture for corporate document management
- Citation and reference extraction for regulatory research backing
"""

import os
import json
import hashlib
import shutil
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import requests

try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    PyPDF2 = None

try:
    import boto3
    from botocore.exceptions import ClientError
    S3_AVAILABLE = True
except ImportError:
    S3_AVAILABLE = False
    boto3 = None

from dataclasses import dataclass


@dataclass
class WhitepaperMetadata:
    """Metadata structure for regulatory whitepapers"""
    paper_id: str
    title: str
    authors: List[str]
    source: str
    document_type: str  # 'whitepaper', 'research', 'regulatory_guidance', 'risk_document'
    regulatory_domain: str  # 'banking', 'securities', 'insurance', 'general'
    file_path: str
    file_size: int
    upload_date: str
    text_content: str
    page_count: int
    compliance_tags: List[str]
    regulatory_references: List[str]
    citation_count: int
    risk_level: str  # 'high', 'medium', 'low'
    validation_status: str  # 'pending', 'validated', 'flagged'
    notes: str


class WhitepaperProcessor:
    """PDF processor for regulatory compliance whitepapers"""
    
    def __init__(self, base_path: Optional[str] = None, s3_config: Optional[Dict[str, Any]] = None):
        """Initialize whitepaper processor
        
        Args:
            base_path: Local storage path for whitepapers
            s3_config: S3 configuration for cloud storage
        """
        # Set up local storage paths
        if base_path:
            self.base_path = Path(base_path)
        else:
            self.base_path = Path(__file__).parent / "whitepaper_repository"
        
        self.papers_dir = self.base_path / "papers"
        self.metadata_dir = self.base_path / "metadata" 
        self.processed_dir = self.base_path / "processed"
        self.collections_dir = self.base_path / "collections"
        
        # Create directory structure
        for directory in [self.papers_dir, self.metadata_dir, self.processed_dir, self.collections_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # S3 configuration for corporate document management
        self.s3_config = s3_config or {}
        
        # Load paper index
        self.index_file = self.base_path / "whitepaper_index.json"
        self.index = self._load_index()
        
        # Regulatory domains mapping
        self.regulatory_domains = {
            'banking': ['sr 11-7', 'model risk', 'basel', 'fed', 'occ', 'fdic'],
            'securities': ['sec', 'finra', 'mifid', 'markets', 'trading'],
            'insurance': ['naic', 'solvency', 'actuarial', 'risk management'],
            'general': ['compliance', 'regulatory', 'governance', 'audit']
        }
    
    def _load_index(self) -> Dict[str, Any]:
        """Load whitepaper index from JSON file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {
                    "papers": {}, 
                    "collections": {}, 
                    "stats": {"total_papers": 0, "total_size": 0, "by_domain": {}},
                    "validation_stats": {"pending": 0, "validated": 0, "flagged": 0}
                }
        return {
            "papers": {}, 
            "collections": {}, 
            "stats": {"total_papers": 0, "total_size": 0, "by_domain": {}},
            "validation_stats": {"pending": 0, "validated": 0, "flagged": 0}
        }
    
    def _save_index(self):
        """Save whitepaper index to JSON file"""
        try:
            with open(self.index_file, 'w', encoding='utf-8') as f:
                json.dump(self.index, f, indent=2, default=str)
        except Exception as e:
            print(f"Failed to save whitepaper index: {e}")
    
    def _extract_text_from_pdf(self, file_path: str) -> Dict[str, Any]:
        """Extract text content from PDF"""
        if not PDF_AVAILABLE:
            return {"success": False, "message": "PyPDF2 not available", "text": "", "page_count": 0}
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                page_count = len(pdf_reader.pages)
                
                for page in pdf_reader.pages:
                    text_content += page.extract_text() + "\n"
                
                return {
                    "success": True,
                    "text": text_content.strip(),
                    "page_count": page_count
                }
        except Exception as e:
            return {"success": False, "message": f"Text extraction failed: {e}", "text": "", "page_count": 0}
    
    def _determine_regulatory_domain(self, title: str, text_content: str) -> str:
        """Determine regulatory domain based on content"""
        content_lower = (title + " " + text_content[:1000]).lower()
        
        domain_scores = {}
        for domain, keywords in self.regulatory_domains.items():
            score = sum(1 for keyword in keywords if keyword in content_lower)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        return 'general'
    
    def _extract_regulatory_references(self, text_content: str) -> List[str]:
        """Extract regulatory references and citations"""
        references = []
        text_lower = text_content.lower()
        
        # Common regulatory reference patterns
        patterns = [
            'sr 11-7', 'sr11-7',
            'basel iii', 'basel ii',
            'dodd-frank',
            'sarbanes-oxley', 'sox',
            'mifid ii', 'mifid',
            'gdpr',
            'occ guidance',
            'fed guidance',
            'finra rule'
        ]
        
        for pattern in patterns:
            if pattern in text_lower:
                references.append(pattern.upper())
        
        return list(set(references))  # Remove duplicates
    
    def _assess_risk_level(self, title: str, text_content: str, regulatory_refs: List[str]) -> str:
        """Assess risk level based on content analysis"""
        risk_indicators = {
            'high': ['model risk', 'systemic risk', 'operational risk', 'credit risk', 'market risk', 'liquidity risk'],
            'medium': ['compliance', 'governance', 'audit', 'internal control'],
            'low': ['guidance', 'best practice', 'framework', 'methodology']
        }
        
        content_lower = (title + " " + text_content[:2000]).lower()
        
        # High regulatory reference count increases risk level
        if len(regulatory_refs) >= 3:
            return 'high'
        
        for level, indicators in risk_indicators.items():
            if any(indicator in content_lower for indicator in indicators):
                return level
        
        return 'medium'  # Default to medium risk
    
    def upload_whitepaper(self, 
                         file_path: str, 
                         title: str, 
                         authors: List[str],
                         document_type: str = "whitepaper",
                         source: str = "upload",
                         compliance_tags: List[str] = None,
                         notes: str = "") -> Dict[str, Any]:
        """Upload and process a whitepaper for compliance analysis
        
        Args:
            file_path: Path to the PDF file
            title: Document title
            authors: List of authors
            document_type: Type of document (whitepaper, research, regulatory_guidance, risk_document)
            source: Source of the document
            compliance_tags: Compliance-related tags
            notes: Additional notes
            
        Returns:
            Processing result with metadata
        """
        try:
            source_path = Path(file_path)
            if not source_path.exists():
                return {"success": False, "message": "File does not exist"}
            
            if not source_path.suffix.lower() == '.pdf':
                return {"success": False, "message": "Only PDF files are supported"}
            
            # Generate paper ID
            paper_id = hashlib.sha256(f"{title}_{datetime.now().isoformat()}".encode()).hexdigest()[:12]
            
            # Create safe filename
            safe_title = "".join(c for c in title[:50] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{paper_id}_{safe_title.replace(' ', '_')}.pdf"
            dest_path = self.papers_dir / filename
            
            # Copy file to repository
            shutil.copy2(source_path, dest_path)
            file_size = dest_path.stat().st_size
            
            # Extract text content
            extraction_result = self._extract_text_from_pdf(str(dest_path))
            text_content = extraction_result.get("text", "")
            page_count = extraction_result.get("page_count", 0)
            
            # Analyze content for regulatory context
            regulatory_domain = self._determine_regulatory_domain(title, text_content)
            regulatory_references = self._extract_regulatory_references(text_content)
            risk_level = self._assess_risk_level(title, text_content, regulatory_references)
            
            # Create metadata
            metadata = WhitepaperMetadata(
                paper_id=paper_id,
                title=title,
                authors=authors,
                source=source,
                document_type=document_type,
                regulatory_domain=regulatory_domain,
                file_path=str(dest_path),
                file_size=file_size,
                upload_date=datetime.now().isoformat(),
                text_content=text_content,
                page_count=page_count,
                compliance_tags=compliance_tags or [],
                regulatory_references=regulatory_references,
                citation_count=len(regulatory_references),
                risk_level=risk_level,
                validation_status="pending",
                notes=notes
            )
            
            # Save metadata
            metadata_file = self.metadata_dir / f"{paper_id}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata.__dict__, f, indent=2, default=str)
            
            # Save processed text
            processed_file = self.processed_dir / f"{paper_id}.txt"
            with open(processed_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Update index
            self.index["papers"][paper_id] = {
                "title": title,
                "authors": authors,
                "source": source,
                "document_type": document_type,
                "regulatory_domain": regulatory_domain,
                "file_path": str(dest_path),
                "file_size": file_size,
                "upload_date": datetime.now().isoformat(),
                "page_count": page_count,
                "risk_level": risk_level,
                "validation_status": "pending",
                "citation_count": len(regulatory_references),
                "compliance_tags": compliance_tags or []
            }
            
            # Update statistics
            self.index["stats"]["total_papers"] += 1
            self.index["stats"]["total_size"] += file_size
            
            domain_stats = self.index["stats"]["by_domain"].get(regulatory_domain, 0)
            self.index["stats"]["by_domain"][regulatory_domain] = domain_stats + 1
            
            self.index["validation_stats"]["pending"] += 1
            
            self._save_index()
            
            # Upload to S3 if configured
            s3_result = None
            if self.s3_config.get("enabled", False):
                s3_result = self._upload_to_s3(metadata)
            
            return {
                "success": True,
                "message": f"Successfully processed whitepaper: {title}",
                "paper_id": paper_id,
                "file_size": file_size,
                "page_count": page_count,
                "regulatory_domain": regulatory_domain,
                "risk_level": risk_level,
                "regulatory_references": regulatory_references,
                "text_extraction_success": extraction_result["success"],
                "s3_upload": s3_result
            }
            
        except Exception as e:
            return {"success": False, "message": f"Processing failed: {e}"}
    
    def _upload_to_s3(self, metadata: WhitepaperMetadata) -> Dict[str, Any]:
        """Upload whitepaper to S3 for corporate document management"""
        if not S3_AVAILABLE:
            return {"success": False, "message": "S3 not available"}
        
        try:
            bucket_name = self.s3_config.get("bucket_name")
            prefix = self.s3_config.get("prefix", "compliance/whitepapers/")
            
            if not bucket_name:
                return {"success": False, "message": "S3 bucket not configured"}
            
            # AUDIT COMPLIANCE: Use UnifiedSessionManager instead of direct boto3
            try:
                from tidyllm.infrastructure.session.unified import UnifiedSessionManager
                session_manager = UnifiedSessionManager()
                s3_client = session_manager.get_s3_client()
            except ImportError:
                # NO FALLBACK - UnifiedSessionManager is required
                raise RuntimeError("WhitepaperProcessor: UnifiedSessionManager is required for S3 access")
            
            # Upload PDF
            pdf_key = f"{prefix}papers/{Path(metadata.file_path).name}"
            s3_client.upload_file(
                metadata.file_path,
                bucket_name,
                pdf_key,
                ExtraArgs={
                    'Metadata': {
                        'paper-id': metadata.paper_id,
                        'title': metadata.title[:1000],  # S3 metadata limit
                        'regulatory-domain': metadata.regulatory_domain,
                        'risk-level': metadata.risk_level,
                        'document-type': metadata.document_type
                    }
                }
            )
            
            # Upload metadata
            metadata_key = f"{prefix}metadata/{metadata.paper_id}.json"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=metadata_key,
                Body=json.dumps(metadata.__dict__, indent=2, default=str),
                ContentType='application/json'
            )
            
            # Upload processed text
            text_key = f"{prefix}processed/{metadata.paper_id}.txt"
            s3_client.put_object(
                Bucket=bucket_name,
                Key=text_key,
                Body=metadata.text_content,
                ContentType='text/plain'
            )
            
            return {
                "success": True,
                "message": "Uploaded to S3",
                "s3_urls": {
                    "pdf": f"s3://{bucket_name}/{pdf_key}",
                    "metadata": f"s3://{bucket_name}/{metadata_key}",
                    "text": f"s3://{bucket_name}/{text_key}"
                }
            }
            
        except Exception as e:
            return {"success": False, "message": f"S3 upload failed: {e}"}
    
    def get_whitepaper(self, paper_id: str) -> Optional[WhitepaperMetadata]:
        """Get whitepaper metadata by ID"""
        metadata_file = self.metadata_dir / f"{paper_id}.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return WhitepaperMetadata(**data)
            except Exception:
                pass
        return None
    
    def list_whitepapers(self, 
                        regulatory_domain: Optional[str] = None,
                        document_type: Optional[str] = None,
                        risk_level: Optional[str] = None,
                        validation_status: Optional[str] = None,
                        limit: int = 50) -> List[Dict[str, Any]]:
        """List whitepapers with optional filtering"""
        papers = []
        
        for paper_id, info in list(self.index.get("papers", {}).items())[:limit]:
            # Apply filters
            if regulatory_domain and info.get("regulatory_domain") != regulatory_domain:
                continue
            if document_type and info.get("document_type") != document_type:
                continue
            if risk_level and info.get("risk_level") != risk_level:
                continue
            if validation_status and info.get("validation_status") != validation_status:
                continue
            
            papers.append({
                "paper_id": paper_id,
                "title": info.get("title", ""),
                "authors": info.get("authors", []),
                "source": info.get("source", ""),
                "document_type": info.get("document_type", ""),
                "regulatory_domain": info.get("regulatory_domain", ""),
                "risk_level": info.get("risk_level", ""),
                "validation_status": info.get("validation_status", ""),
                "file_size_mb": round(info.get("file_size", 0) / (1024 * 1024), 2),
                "page_count": info.get("page_count", 0),
                "upload_date": info.get("upload_date", ""),
                "citation_count": info.get("citation_count", 0),
                "compliance_tags": info.get("compliance_tags", [])
            })
        
        return sorted(papers, key=lambda x: x["upload_date"], reverse=True)
    
    def search_whitepapers(self, query: str, regulatory_domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search whitepapers by content"""
        query = query.lower()
        results = []
        
        for paper_id, info in self.index.get("papers", {}).items():
            # Domain filter
            if regulatory_domain and info.get("regulatory_domain") != regulatory_domain:
                continue
            
            # Search in title, tags, and content
            title = info.get("title", "").lower()
            tags = " ".join(info.get("compliance_tags", [])).lower()
            
            # Load processed text for content search
            processed_file = self.processed_dir / f"{paper_id}.txt"
            content = ""
            if processed_file.exists():
                try:
                    with open(processed_file, 'r', encoding='utf-8') as f:
                        content = f.read()[:2000].lower()  # First 2000 chars
                except Exception:
                    pass
            
            # Calculate relevance score
            score = 0
            if query in title:
                score += 10
            if query in tags:
                score += 5
            if query in content:
                score += 1
            
            if score > 0:
                results.append({
                    "paper_id": paper_id,
                    "title": info.get("title", ""),
                    "regulatory_domain": info.get("regulatory_domain", ""),
                    "risk_level": info.get("risk_level", ""),
                    "validation_status": info.get("validation_status", ""),
                    "relevance_score": score,
                    "file_path": info.get("file_path", ""),
                    "compliance_tags": info.get("compliance_tags", [])
                })
        
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)
    
    def update_validation_status(self, paper_id: str, status: str, notes: str = "") -> Dict[str, Any]:
        """Update validation status of a whitepaper"""
        try:
            if status not in ["pending", "validated", "flagged"]:
                return {"success": False, "message": "Invalid validation status"}
            
            # Update metadata file
            metadata_file = self.metadata_dir / f"{paper_id}.json"
            if not metadata_file.exists():
                return {"success": False, "message": "Paper not found"}
            
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            old_status = metadata.get("validation_status", "pending")
            metadata["validation_status"] = status
            if notes:
                metadata["notes"] = notes
            
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            # Update index
            if paper_id in self.index["papers"]:
                self.index["papers"][paper_id]["validation_status"] = status
            
            # Update validation statistics
            if old_status in self.index["validation_stats"]:
                self.index["validation_stats"][old_status] -= 1
            if status in self.index["validation_stats"]:
                self.index["validation_stats"][status] += 1
            
            self._save_index()
            
            return {"success": True, "message": f"Updated validation status to {status}"}
            
        except Exception as e:
            return {"success": False, "message": f"Update failed: {e}"}
    
    def get_compliance_dashboard(self) -> Dict[str, Any]:
        """Generate compliance dashboard with key metrics"""
        stats = self.index.get("stats", {})
        validation_stats = self.index.get("validation_stats", {})
        
        # Risk level distribution
        risk_distribution = {}
        domain_distribution = stats.get("by_domain", {})
        
        for paper_info in self.index.get("papers", {}).values():
            risk = paper_info.get("risk_level", "medium")
            risk_distribution[risk] = risk_distribution.get(risk, 0) + 1
        
        # Document type distribution
        doc_type_distribution = {}
        for paper_info in self.index.get("papers", {}).values():
            doc_type = paper_info.get("document_type", "whitepaper")
            doc_type_distribution[doc_type] = doc_type_distribution.get(doc_type, 0) + 1
        
        return {
            "total_papers": stats.get("total_papers", 0),
            "total_size_mb": round(stats.get("total_size", 0) / (1024 * 1024), 2),
            "validation_status": validation_stats,
            "regulatory_domains": domain_distribution,
            "risk_levels": risk_distribution,
            "document_types": doc_type_distribution,
            "repository_path": str(self.base_path),
            "s3_enabled": self.s3_config.get("enabled", False)
        }

    def create_compliance_collection(self, 
                                   name: str, 
                                   regulatory_domain: str,
                                   description: str = "",
                                   auto_populate: bool = False) -> Dict[str, Any]:
        """Create a compliance-focused collection"""
        try:
            if name in self.index.get("collections", {}):
                return {"success": False, "message": "Collection already exists"}
            
            collection = {
                "name": name,
                "regulatory_domain": regulatory_domain,
                "description": description,
                "papers": [],
                "created_at": datetime.now().isoformat(),
                "compliance_focused": True
            }
            
            # Auto-populate with papers from the regulatory domain
            if auto_populate:
                for paper_id, paper_info in self.index.get("papers", {}).items():
                    if paper_info.get("regulatory_domain") == regulatory_domain:
                        collection["papers"].append(paper_id)
            
            self.index["collections"][name] = collection
            self._save_index()
            
            # Save collection file
            collection_file = self.collections_dir / f"{name.replace(' ', '_')}.json"
            with open(collection_file, 'w', encoding='utf-8') as f:
                json.dump(collection, f, indent=2, default=str)
            
            return {
                "success": True,
                "message": f"Created compliance collection: {name}",
                "papers_added": len(collection["papers"]) if auto_populate else 0
            }
            
        except Exception as e:
            return {"success": False, "message": f"Failed to create collection: {e}"}


def get_whitepaper_processor(base_path: Optional[str] = None, s3_config: Optional[Dict[str, Any]] = None):
    """Get whitepaper processor instance"""
    return WhitepaperProcessor(base_path=base_path, s3_config=s3_config)