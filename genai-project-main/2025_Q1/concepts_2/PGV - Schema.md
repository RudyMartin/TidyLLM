-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Table: pg_vec
CREATE TABLE pg_vec (
    id UUID PRIMARY KEY,
    embedding vector(1536),  -- Replace 1536 with the actual dimension of your embeddings
    embed_time TIMESTAMP WITH TIME ZONE
);

-- Table: pg_details
CREATE TABLE pg_details (
    id UUID PRIMARY KEY REFERENCES pg_vec(id),
    model_key TEXT,
    created_at TIMESTAMP WITH TIME ZONE,
    document_name TEXT,
    page_number INTEGER,
    tags JSONB
);
