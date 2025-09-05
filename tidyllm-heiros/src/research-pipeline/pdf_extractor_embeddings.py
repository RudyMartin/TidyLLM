"""
Research Paper PDF Extractor with pgvector Embeddings
====================================================

Extract research papers, generate embeddings, and store in PostgreSQL with pgvector
Specifically designed for mathematical decomposition and residual risk papers
"""

import os
import sys
import requests
import PyPDF2
import psycopg2
import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import time
import io
from urllib.parse import urlparse
import logging

# OpenAI for embeddings
try:
    import openai
except ImportError:
    print("Installing openai...")
    os.system("pip install openai")
    import openai

# PDF processing
try:
    import PyPDF2
    import pdfplumber
except ImportError:
    print("Installing PDF libraries...")
    os.system("pip install PyPDF2 pdfplumber")
    import PyPDF2
    import pdfplumber

# PostgreSQL with pgvector
try:
    import psycopg2
    import psycopg2.extras
except ImportError:
    print("Installing psycopg2...")
    os.system("pip install psycopg2-binary")
    import psycopg2
    import psycopg2.extras

@dataclass
class ResearchPaper:
    """Research paper metadata and content"""
    title: str
    authors: List[str]
    abstract: str
    content: str
    url: str
    arxiv_id: Optional[str] = None
    doi: Optional[str] = None
    publication_date: Optional[datetime] = None
    keywords: List[str] = field(default_factory=list)
    paper_hash: str = ""
    extraction_date: datetime = field(default_factory=datetime.now)

@dataclass
class EmbeddingChunk:
    """Text chunk with embedding"""
    paper_id: str
    chunk_id: str
    text: str
    embedding: List[float]
    chunk_type: str  # abstract, introduction, methodology, results, conclusion
    start_page: int = 0
    end_page: int = 0
    metadata: Dict = field(default_factory=dict)

class PDFExtractor:
    """Extract text and metadata from PDF files"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def extract_from_url(self, pdf_url: str) -> Optional[str]:
        """Download and extract text from PDF URL"""
        try:
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            return self.extract_from_bytes(response.content)
            
        except Exception as e:
            self.logger.error(f"Failed to extract from URL {pdf_url}: {e}")
            return None
    
    def extract_from_file(self, file_path: str) -> Optional[str]:
        """Extract text from local PDF file"""
        try:
            with open(file_path, 'rb') as file:
                return self.extract_from_bytes(file.read())
                
        except Exception as e:
            self.logger.error(f"Failed to extract from file {file_path}: {e}")
            return None
    
    def extract_from_bytes(self, pdf_bytes: bytes) -> Optional[str]:
        """Extract text from PDF bytes using multiple methods"""
        
        # Method 1: PyPDF2
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            if len(text.strip()) > 100:  # Valid extraction
                return self.clean_text(text)
                
        except Exception as e:
            self.logger.warning(f"PyPDF2 extraction failed: {e}")
        
        # Method 2: pdfplumber (better for complex layouts)
        try:
            pdf_file = io.BytesIO(pdf_bytes)
            with pdfplumber.open(pdf_file) as pdf:
                text = ""
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                
                if len(text.strip()) > 100:
                    return self.clean_text(text)
                    
        except Exception as e:
            self.logger.warning(f"pdfplumber extraction failed: {e}")
        
        return None
    
    def clean_text(self, text: str) -> str:
        """Clean extracted text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove common PDF artifacts
        artifacts = [
            'cid:', 'fi', 'fl', 'ffi', 'ffl',  # Font encoding artifacts
            '†', '‡', '§', '¶',  # Special characters
        ]
        
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        return text

class EmbeddingGenerator:
    """Generate embeddings for research paper text"""
    
    def __init__(self, openai_api_key: str, model: str = "text-embedding-3-large"):
        """
        Initialize with OpenAI API key
        text-embedding-3-large gives 3072 dimensions by default
        We'll configure it for 1024 dimensions
        """
        openai.api_key = openai_api_key
        self.model = model
        self.embedding_dimensions = 1024
        
    def generate_embedding(self, text: str) -> List[float]:
        """Generate 1024-dimensional embedding for text"""
        try:
            response = openai.Embedding.create(
                model=self.model,
                input=text,
                dimensions=self.embedding_dimensions  # Force 1024 dimensions
            )
            
            return response['data'][0]['embedding']
            
        except Exception as e:
            self.logger.error(f"Failed to generate embedding: {e}")
            return [0.0] * self.embedding_dimensions  # Fallback zero vector
    
    def chunk_and_embed(self, paper: ResearchPaper, chunk_size: int = 1000, 
                       overlap: int = 200) -> List[EmbeddingChunk]:
        """Split paper into chunks and generate embeddings"""
        
        chunks = []
        text = paper.content
        
        # Split into overlapping chunks
        for i in range(0, len(text), chunk_size - overlap):
            chunk_text = text[i:i + chunk_size]
            
            if len(chunk_text) < 100:  # Skip very small chunks
                continue
            
            # Determine chunk type based on content
            chunk_type = self._classify_chunk(chunk_text)
            
            # Generate embedding
            embedding = self.generate_embedding(chunk_text)
            
            chunk = EmbeddingChunk(
                paper_id=paper.paper_hash,
                chunk_id=f"{paper.paper_hash}_{i}",
                text=chunk_text,
                embedding=embedding,
                chunk_type=chunk_type,
                metadata={
                    'paper_title': paper.title,
                    'authors': paper.authors,
                    'chunk_index': i // (chunk_size - overlap)
                }
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _classify_chunk(self, text: str) -> str:
        """Classify chunk type based on content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['abstract', 'summary']):
            return 'abstract'
        elif any(word in text_lower for word in ['introduction', 'background']):
            return 'introduction'
        elif any(word in text_lower for word in ['method', 'approach', 'algorithm']):
            return 'methodology'
        elif any(word in text_lower for word in ['result', 'experiment', 'evaluation']):
            return 'results'
        elif any(word in text_lower for word in ['conclusion', 'discussion', 'future']):
            return 'conclusion'
        else:
            return 'content'

class PostgreSQLVectorStore:
    """Store embeddings in PostgreSQL with pgvector"""
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.logger = logging.getLogger(__name__)
        
    def initialize_database(self):
        """Create tables and enable pgvector extension"""
        
        with psycopg2.connect(self.connection_string) as conn:
            with conn.cursor() as cur:
                # Enable pgvector extension
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create papers table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS research_papers (
                        id SERIAL PRIMARY KEY,
                        paper_hash VARCHAR(64) UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        authors TEXT[],
                        abstract TEXT,
                        content TEXT,
                        url TEXT,
                        arxiv_id VARCHAR(50),
                        doi VARCHAR(100),
                        publication_date TIMESTAMP,
                        keywords TEXT[],
                        extraction_date TIMESTAMP DEFAULT NOW(),
                        created_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Create embeddings table with 1024-dimensional vectors
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS paper_embeddings (
                        id SERIAL PRIMARY KEY,
                        paper_id VARCHAR(64) NOT NULL,
                        chunk_id VARCHAR(100) UNIQUE NOT NULL,
                        chunk_text TEXT NOT NULL,
                        embedding vector(1024) NOT NULL,
                        chunk_type VARCHAR(50),
                        start_page INTEGER DEFAULT 0,
                        end_page INTEGER DEFAULT 0,
                        metadata JSONB,
                        created_at TIMESTAMP DEFAULT NOW(),
                        FOREIGN KEY (paper_id) REFERENCES research_papers(paper_hash)
                    );
                """)
                
                # Create indexes for fast similarity search
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS paper_embeddings_embedding_idx 
                    ON paper_embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                
                # Create text search indexes
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS research_papers_title_idx 
                    ON research_papers USING gin(to_tsvector('english', title));
                """)
                
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS paper_embeddings_text_idx 
                    ON paper_embeddings USING gin(to_tsvector('english', chunk_text));
                """)
                
                conn.commit()
                
    def store_paper(self, paper: ResearchPaper) -> bool:
        """Store research paper in database"""
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO research_papers 
                        (paper_hash, title, authors, abstract, content, url, 
                         arxiv_id, doi, publication_date, keywords)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (paper_hash) DO UPDATE SET
                            title = EXCLUDED.title,
                            authors = EXCLUDED.authors,
                            abstract = EXCLUDED.abstract,
                            content = EXCLUDED.content,
                            url = EXCLUDED.url,
                            arxiv_id = EXCLUDED.arxiv_id,
                            doi = EXCLUDED.doi,
                            publication_date = EXCLUDED.publication_date,
                            keywords = EXCLUDED.keywords;
                    """, (
                        paper.paper_hash,
                        paper.title,
                        paper.authors,
                        paper.abstract,
                        paper.content,
                        paper.url,
                        paper.arxiv_id,
                        paper.doi,
                        paper.publication_date,
                        paper.keywords
                    ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to store paper: {e}")
            return False
    
    def store_embeddings(self, chunks: List[EmbeddingChunk]) -> bool:
        """Store embedding chunks in database"""
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor() as cur:
                    
                    for chunk in chunks:
                        cur.execute("""
                            INSERT INTO paper_embeddings 
                            (paper_id, chunk_id, chunk_text, embedding, chunk_type, 
                             start_page, end_page, metadata)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (chunk_id) DO UPDATE SET
                                chunk_text = EXCLUDED.chunk_text,
                                embedding = EXCLUDED.embedding,
                                chunk_type = EXCLUDED.chunk_type,
                                metadata = EXCLUDED.metadata;
                        """, (
                            chunk.paper_id,
                            chunk.chunk_id,
                            chunk.text,
                            chunk.embedding,
                            chunk.chunk_type,
                            chunk.start_page,
                            chunk.end_page,
                            json.dumps(chunk.metadata)
                        ))
                    
                    conn.commit()
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to store embeddings: {e}")
            return False
    
    def similarity_search(self, query_embedding: List[float], 
                         limit: int = 10, similarity_threshold: float = 0.7) -> List[Dict]:
        """Perform similarity search using pgvector"""
        
        try:
            with psycopg2.connect(self.connection_string) as conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    
                    cur.execute("""
                        SELECT 
                            pe.chunk_text,
                            pe.chunk_type,
                            pe.metadata,
                            rp.title,
                            rp.authors,
                            rp.url,
                            1 - (pe.embedding <=> %s) as similarity
                        FROM paper_embeddings pe
                        JOIN research_papers rp ON pe.paper_id = rp.paper_hash
                        WHERE 1 - (pe.embedding <=> %s) > %s
                        ORDER BY pe.embedding <=> %s
                        LIMIT %s;
                    """, (query_embedding, query_embedding, similarity_threshold, query_embedding, limit))
                    
                    results = cur.fetchall()
                    return [dict(row) for row in results]
                    
        except Exception as e:
            self.logger.error(f"Similarity search failed: {e}")
            return []

class ResearchPipelineManager:
    """Main pipeline manager for research paper processing"""
    
    def __init__(self, openai_api_key: str, postgres_connection: str):
        self.pdf_extractor = PDFExtractor()
        self.embedding_generator = EmbeddingGenerator(openai_api_key)
        self.vector_store = PostgreSQLVectorStore(postgres_connection)
        
        # Initialize database
        self.vector_store.initialize_database()
        
        # Research paper URLs for mathematical decomposition
        self.target_papers = [
            {
                "title": "Deep learning of dynamics and signal-noise decomposition",
                "url": "https://arxiv.org/pdf/1808.02578.pdf",
                "arxiv_id": "1808.02578"
            },
            {
                "title": "Signal and Noise: A Framework for Reducing Uncertainty",
                "url": "https://arxiv.org/pdf/2508.13144.pdf", 
                "arxiv_id": "2508.13144"
            },
            {
                "title": "Efficient Orthogonal Decomposition with Automatic Basis Extraction",
                "url": "https://arxiv.org/pdf/2404.17290.pdf",
                "arxiv_id": "2404.17290"
            },
            # Add more papers as needed
        ]
    
    def process_paper_from_url(self, paper_info: Dict) -> bool:
        """Process a single paper from URL"""
        
        print(f"Processing: {paper_info['title']}")
        
        # Extract PDF content
        content = self.pdf_extractor.extract_from_url(paper_info['url'])
        if not content:
            print(f"Failed to extract content from {paper_info['url']}")
            return False
        
        # Create paper object
        paper = ResearchPaper(
            title=paper_info['title'],
            authors=paper_info.get('authors', []),
            abstract=self.extract_abstract(content),
            content=content,
            url=paper_info['url'],
            arxiv_id=paper_info.get('arxiv_id'),
            keywords=['mathematical decomposition', 'residual risk', 'signal processing']
        )
        
        # Generate hash
        paper.paper_hash = hashlib.sha256(
            f"{paper.title}{paper.url}".encode()
        ).hexdigest()
        
        # Store paper
        if not self.vector_store.store_paper(paper):
            print(f"Failed to store paper: {paper.title}")
            return False
        
        # Generate embeddings
        chunks = self.embedding_generator.chunk_and_embed(paper)
        if not self.vector_store.store_embeddings(chunks):
            print(f"Failed to store embeddings for: {paper.title}")
            return False
        
        print(f"Successfully processed: {paper.title} ({len(chunks)} chunks)")
        return True
    
    def extract_abstract(self, content: str) -> str:
        """Extract abstract from paper content"""
        content_lower = content.lower()
        
        # Find abstract section
        abstract_start = content_lower.find('abstract')
        if abstract_start == -1:
            return ""
        
        # Find end of abstract (usually marked by next section)
        section_markers = ['introduction', '1.', 'keywords', 'background']
        abstract_end = len(content)
        
        for marker in section_markers:
            marker_pos = content_lower.find(marker, abstract_start + 8)
            if marker_pos != -1 and marker_pos < abstract_end:
                abstract_end = marker_pos
        
        abstract = content[abstract_start:abstract_end]
        return ' '.join(abstract.split())  # Clean whitespace
    
    def process_all_papers(self):
        """Process all target papers"""
        
        print("Starting research paper processing pipeline...")
        print(f"Target papers: {len(self.target_papers)}")
        
        successful = 0
        for paper_info in self.target_papers:
            if self.process_paper_from_url(paper_info):
                successful += 1
            time.sleep(2)  # Rate limiting
        
        print(f"Pipeline complete: {successful}/{len(self.target_papers)} papers processed")
    
    def search_papers(self, query: str, limit: int = 5) -> List[Dict]:
        """Search papers by query"""
        
        # Generate query embedding
        query_embedding = self.embedding_generator.generate_embedding(query)
        
        # Perform similarity search
        results = self.vector_store.similarity_search(query_embedding, limit)
        
        return results

# Example usage and testing
if __name__ == "__main__":
    # Configuration
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
    POSTGRES_CONNECTION = os.getenv("POSTGRES_CONNECTION", 
        "postgresql://user:password@localhost:5432/research_db"
    )
    
    print("Research Paper Processing Pipeline")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = ResearchPipelineManager(OPENAI_API_KEY, POSTGRES_CONNECTION)
    
    # Process papers
    pipeline.process_all_papers()
    
    # Test search
    print("\n" + "=" * 50)
    print("Testing search functionality...")
    
    test_queries = [
        "mathematical decomposition residual risk",
        "signal noise separation orthogonal",
        "empirical mode decomposition",
        "Y = R + S + N model"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = pipeline.search_papers(query, limit=3)
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['title']}")
            print(f"     Similarity: {result['similarity']:.3f}")
            print(f"     Text: {result['chunk_text'][:200]}...")