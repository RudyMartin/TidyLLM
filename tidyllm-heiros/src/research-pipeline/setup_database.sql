-- PostgreSQL Database Setup for Research Paper Embeddings
-- Requires PostgreSQL 14+ with pgvector extension

-- Create database (run as superuser)
-- CREATE DATABASE research_papers_db;

-- Connect to the database and run the following:

-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create research papers table
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

-- Create embeddings table with 1024-dimensional vectors
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

-- Create indexes for fast similarity search
-- IVFFlat index for approximate nearest neighbor search
CREATE INDEX IF NOT EXISTS paper_embeddings_embedding_idx 
ON paper_embeddings USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- HNSW index (alternative, often faster for small-medium datasets)
-- CREATE INDEX IF NOT EXISTS paper_embeddings_embedding_hnsw_idx 
-- ON paper_embeddings USING hnsw (embedding vector_cosine_ops)
-- WITH (m = 16, ef_construction = 64);

-- Create text search indexes for full-text search
CREATE INDEX IF NOT EXISTS research_papers_title_idx 
ON research_papers USING gin(to_tsvector('english', title));

CREATE INDEX IF NOT EXISTS research_papers_abstract_idx 
ON research_papers USING gin(to_tsvector('english', abstract));

CREATE INDEX IF NOT EXISTS paper_embeddings_text_idx 
ON paper_embeddings USING gin(to_tsvector('english', chunk_text));

-- Create indexes on common query fields
CREATE INDEX IF NOT EXISTS research_papers_arxiv_idx ON research_papers(arxiv_id);
CREATE INDEX IF NOT EXISTS research_papers_doi_idx ON research_papers(doi);
CREATE INDEX IF NOT EXISTS research_papers_keywords_idx ON research_papers USING gin(keywords);
CREATE INDEX IF NOT EXISTS paper_embeddings_chunk_type_idx ON paper_embeddings(chunk_type);
CREATE INDEX IF NOT EXISTS paper_embeddings_paper_id_idx ON paper_embeddings(paper_id);

-- Create a function for semantic search
CREATE OR REPLACE FUNCTION semantic_search(
    query_embedding vector(1024),
    similarity_threshold float DEFAULT 0.7,
    result_limit int DEFAULT 10
)
RETURNS TABLE (
    paper_title text,
    chunk_text text,
    chunk_type varchar(50),
    authors text[],
    url text,
    similarity float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rp.title,
        pe.chunk_text,
        pe.chunk_type,
        rp.authors,
        rp.url,
        (1 - (pe.embedding <=> query_embedding))::float as similarity
    FROM paper_embeddings pe
    JOIN research_papers rp ON pe.paper_id = rp.paper_hash
    WHERE 1 - (pe.embedding <=> query_embedding) > similarity_threshold
    ORDER BY pe.embedding <=> query_embedding
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Create a function for hybrid search (text + semantic)
CREATE OR REPLACE FUNCTION hybrid_search(
    query_text text,
    query_embedding vector(1024),
    text_weight float DEFAULT 0.3,
    semantic_weight float DEFAULT 0.7,
    result_limit int DEFAULT 10
)
RETURNS TABLE (
    paper_title text,
    chunk_text text,
    chunk_type varchar(50),
    authors text[],
    url text,
    combined_score float
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        rp.title,
        pe.chunk_text,
        pe.chunk_type,
        rp.authors,
        rp.url,
        (
            text_weight * ts_rank(to_tsvector('english', pe.chunk_text), plainto_tsquery('english', query_text)) +
            semantic_weight * (1 - (pe.embedding <=> query_embedding))
        )::float as combined_score
    FROM paper_embeddings pe
    JOIN research_papers rp ON pe.paper_id = rp.paper_hash
    WHERE 
        to_tsvector('english', pe.chunk_text) @@ plainto_tsquery('english', query_text)
        OR (1 - (pe.embedding <=> query_embedding)) > 0.5
    ORDER BY combined_score DESC
    LIMIT result_limit;
END;
$$ LANGUAGE plpgsql;

-- Create views for easy querying
CREATE OR REPLACE VIEW paper_summary AS
SELECT 
    rp.title,
    rp.authors,
    rp.abstract,
    rp.arxiv_id,
    rp.url,
    rp.keywords,
    COUNT(pe.id) as chunk_count,
    AVG(array_length(string_to_array(pe.chunk_text, ' '), 1)) as avg_chunk_words
FROM research_papers rp
LEFT JOIN paper_embeddings pe ON rp.paper_hash = pe.paper_id
GROUP BY rp.id, rp.title, rp.authors, rp.abstract, rp.arxiv_id, rp.url, rp.keywords;

-- Create a view for embedding statistics
CREATE OR REPLACE VIEW embedding_stats AS
SELECT 
    chunk_type,
    COUNT(*) as chunk_count,
    AVG(array_length(string_to_array(chunk_text, ' '), 1)) as avg_words_per_chunk,
    MIN(array_length(string_to_array(chunk_text, ' '), 1)) as min_words,
    MAX(array_length(string_to_array(chunk_text, ' '), 1)) as max_words
FROM paper_embeddings
GROUP BY chunk_type
ORDER BY chunk_count DESC;

-- Grant permissions (adjust user as needed)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO research_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO research_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO research_user;

-- Example queries for testing:

-- 1. Find papers about mathematical decomposition
-- SELECT * FROM semantic_search(
--     (SELECT embedding FROM paper_embeddings WHERE chunk_text ILIKE '%mathematical decomposition%' LIMIT 1),
--     0.6,
--     5
-- );

-- 2. Hybrid search combining text and semantic similarity
-- SELECT * FROM hybrid_search(
--     'residual risk decomposition',
--     (SELECT embedding FROM paper_embeddings WHERE chunk_text ILIKE '%risk%' LIMIT 1),
--     0.3,
--     0.7,
--     10
-- );

-- 3. Get paper statistics
-- SELECT * FROM paper_summary;

-- 4. Get embedding statistics
-- SELECT * FROM embedding_stats;