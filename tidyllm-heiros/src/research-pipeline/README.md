# Research Paper Processing Pipeline
## PDF Extraction, Embeddings & pgvector Storage

**Purpose**: Download, extract, embed, and store research papers about mathematical decomposition and residual risk analysis in PostgreSQL with pgvector for semantic search.

---

## 🚀 **Quick Start**

### **1. Prerequisites**

**PostgreSQL with pgvector:**
```bash
# Install PostgreSQL 14+
sudo apt-get install postgresql postgresql-contrib

# Install pgvector extension
git clone https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# Or use Docker:
docker run --name research-postgres -e POSTGRES_PASSWORD=mypassword -p 5432:5432 -d pgvector/pgvector:pg16
```

**Python Dependencies:**
```bash
cd src/research-pipeline
pip install -r requirements.txt
```

### **2. Database Setup**

```bash
# Create database
createdb research_papers_db

# Run setup script
psql research_papers_db < setup_database.sql
```

### **3. Configure Environment**

```bash
# Set OpenAI API key (required for embeddings)
export OPENAI_API_KEY="your-openai-api-key-here"

# Set PostgreSQL connection (optional - defaults to localhost)
export POSTGRES_CONNECTION="postgresql://user:password@localhost:5432/research_papers_db"
```

### **4. Run Pipeline**

```bash
python run_pipeline.py
```

**Expected Output:**
```
TidyLLM-HeirOS Research Paper Processing Pipeline
Target papers: 5
Processing: Deep learning of dynamics and signal-noise decomposition
Successfully processed: Deep learning of dynamics... (47 chunks)
...
✅ Pipeline completed successfully!
Papers processed: 5
Embedding chunks: 234
```

---

## 📊 **System Architecture**

```
Research Papers (PDFs)
       ↓
   PDF Extractor
       ↓
   Text Chunking
       ↓
OpenAI Embeddings (1024D)
       ↓
PostgreSQL + pgvector
       ↓
Semantic Search
```

### **Database Schema**

```sql
-- Papers table
research_papers (
    paper_hash VARCHAR(64) PRIMARY KEY,
    title TEXT,
    authors TEXT[],
    content TEXT,
    arxiv_id VARCHAR(50),
    ...
)

-- Embeddings table  
paper_embeddings (
    chunk_id VARCHAR(100) PRIMARY KEY,
    paper_id VARCHAR(64) REFERENCES research_papers,
    chunk_text TEXT,
    embedding vector(1024),  -- pgvector 1024D
    chunk_type VARCHAR(50),  -- abstract, methodology, etc.
    ...
)
```

---

## 🔍 **Target Research Areas**

### **Mathematical Decomposition Papers**

1. **Signal-Noise Decomposition**
   - ArXiv: 1808.02578 - "Deep learning of dynamics and signal-noise decomposition"
   - ArXiv: 2508.13144 - "Signal and Noise: A Framework for Reducing Uncertainty"

2. **Orthogonal Decomposition**
   - ArXiv: 2404.17290 - "Efficient Orthogonal Decomposition"
   - ArXiv: 2409.07242 - "Orthogonal Mode Decomposition"

3. **Residual Risk Analysis**
   - Bias-variance-noise decomposition
   - Error variance separation
   - Y = R + S + N models

### **Search Capabilities**

**Semantic Search:**
```python
results = pipeline.search_papers(
    "mathematical decomposition Y equals R plus S plus N",
    limit=5
)
```

**SQL Queries:**
```sql
-- Semantic search
SELECT * FROM semantic_search(
    query_embedding,
    similarity_threshold := 0.7,
    result_limit := 10
);

-- Hybrid text + semantic
SELECT * FROM hybrid_search(
    'residual risk decomposition',
    query_embedding,
    text_weight := 0.3,
    semantic_weight := 0.7
);
```

---

## 📁 **File Structure**

```
research-pipeline/
├── pdf_extractor_embeddings.py    # Main pipeline classes
├── run_pipeline.py                # Execute complete pipeline
├── setup_database.sql             # PostgreSQL schema & functions
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### **Key Classes**

**PDFExtractor:**
- Downloads PDFs from ArXiv URLs
- Extracts text using PyPDF2 and pdfplumber
- Handles common PDF extraction issues

**EmbeddingGenerator:**
- Uses OpenAI text-embedding-3-large (1024D)
- Chunks text with overlap
- Classifies chunks by type (abstract, methodology, etc.)

**PostgreSQLVectorStore:**
- Stores papers and embeddings in PostgreSQL
- Uses pgvector for similarity search
- Creates optimized indexes (IVFFlat)

**ResearchPipelineManager:**
- Orchestrates the complete pipeline
- Manages paper processing queue
- Provides search interface

---

## ⚙️ **Configuration Options**

### **Embedding Settings**
```python
embedding_generator = EmbeddingGenerator(
    openai_api_key=api_key,
    model="text-embedding-3-large",  # or text-embedding-ada-002
    embedding_dimensions=1024        # Force 1024D (default 3072)
)
```

### **Chunking Settings**
```python
chunks = embedding_generator.chunk_and_embed(
    paper=paper,
    chunk_size=1000,    # Characters per chunk
    overlap=200         # Character overlap between chunks
)
```

### **Search Settings**
```python
results = vector_store.similarity_search(
    query_embedding=embedding,
    limit=10,                    # Max results
    similarity_threshold=0.7     # Cosine similarity minimum
)
```

---

## 🔧 **Advanced Usage**

### **Add New Papers**

```python
# Add custom paper
pipeline = ResearchPipelineManager(api_key, db_connection)

custom_paper = {
    "title": "Your Research Paper",
    "url": "https://arxiv.org/pdf/XXXX.XXXXX.pdf",
    "arxiv_id": "XXXX.XXXXX",
    "authors": ["Author Name"]
}

pipeline.process_paper_from_url(custom_paper)
```

### **Custom Search Functions**

```python
# Search by paper section type
def search_methodology(query, limit=5):
    query_embedding = pipeline.embedding_generator.generate_embedding(query)
    
    with psycopg2.connect(connection_string) as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chunk_text, title, similarity
                FROM paper_embeddings pe
                JOIN research_papers rp ON pe.paper_id = rp.paper_hash
                WHERE pe.chunk_type = 'methodology'
                ORDER BY pe.embedding <=> %s
                LIMIT %s
            """, (query_embedding, limit))
            
            return cur.fetchall()
```

### **Batch Processing**

```python
# Process multiple papers concurrently
import concurrent.futures

def process_paper_batch(paper_urls):
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [
            executor.submit(pipeline.process_paper_from_url, paper)
            for paper in paper_urls
        ]
        
        results = [f.result() for f in futures]
    return results
```

---

## 📈 **Performance & Scalability**

### **Expected Performance**
- **PDF Extraction**: ~2-5 seconds per paper
- **Embedding Generation**: ~0.1 seconds per chunk (API dependent)
- **Database Storage**: ~0.01 seconds per chunk
- **Similarity Search**: ~0.1-1 seconds (depends on corpus size)

### **Optimization Tips**

**Database:**
```sql
-- Tune pgvector indexes
CREATE INDEX CONCURRENTLY paper_embeddings_embedding_idx 
ON paper_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);  -- Adjust lists based on corpus size

-- For larger corpora (>1M vectors), use HNSW
CREATE INDEX paper_embeddings_embedding_hnsw_idx 
ON paper_embeddings USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

**Embeddings:**
- Use batch embedding API calls when available
- Cache embeddings to avoid regeneration
- Consider smaller models for speed (text-embedding-ada-002)

**Memory:**
- Process papers in batches for large corpora
- Use streaming for very large PDFs
- Implement connection pooling for PostgreSQL

---

## 🧪 **Testing & Validation**

### **Unit Tests**
```python
# Test PDF extraction
def test_pdf_extraction():
    extractor = PDFExtractor()
    content = extractor.extract_from_url("https://arxiv.org/pdf/1808.02578.pdf")
    assert len(content) > 1000
    assert "decomposition" in content.lower()

# Test embedding generation
def test_embedding_generation():
    generator = EmbeddingGenerator(api_key)
    embedding = generator.generate_embedding("mathematical decomposition")
    assert len(embedding) == 1024
    assert all(isinstance(x, float) for x in embedding)
```

### **Integration Tests**
```bash
# Test complete pipeline
python -m pytest test_pipeline.py

# Test database connectivity
python -c "from pdf_extractor_embeddings import PostgreSQLVectorStore; store = PostgreSQLVectorStore('$POSTGRES_CONNECTION'); store.initialize_database()"
```

---

## 🚨 **Troubleshooting**

### **Common Issues**

**1. pgvector not installed:**
```
ERROR: extension "vector" is not available
```
**Solution:** Install pgvector extension in PostgreSQL

**2. OpenAI API rate limits:**
```
RateLimitError: You exceeded your current quota
```
**Solution:** Add retry logic with exponential backoff

**3. PDF extraction failures:**
```
Failed to extract content from URL
```
**Solution:** Try alternative PDF libraries or manual download

**4. Embedding dimension mismatch:**
```
ERROR: vector has wrong dimensions
```
**Solution:** Ensure consistent 1024D embeddings across all operations

### **Debug Mode**
```bash
# Enable debug logging
export PYTHONPATH=.
python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from run_pipeline import main
main()
"
```

---

## 📚 **References**

- [pgvector Documentation](https://github.com/pgvector/pgvector)
- [OpenAI Embeddings API](https://platform.openai.com/docs/guides/embeddings)
- [PostgreSQL Full-Text Search](https://www.postgresql.org/docs/current/textsearch.html)
- [ArXiv API](https://arxiv.org/help/api)

---

**Pipeline Status**: Ready for production  
**Last Updated**: 2025-08-30  
**Dependencies**: Python 3.8+, PostgreSQL 14+, pgvector, OpenAI API  

*"Turning research papers into searchable knowledge - one embedding at a time."*