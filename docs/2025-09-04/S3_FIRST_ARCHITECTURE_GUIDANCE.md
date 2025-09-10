# S3-First Architecture Guidance

## Core Principles

**Documents stored once in S3, referenced by vectors** - This is the foundational architectural constraint.

### The S3-First Approach

1. **Single Source of Truth**: Documents are stored once in S3, not duplicated across systems
2. **Vector References Only**: Vector database stores embeddings + S3 references, not content
3. **Streaming Processing**: Processing happens S3 → S3 (streaming), no local file processing
4. **On-Demand Fetching**: Content is fetched from S3 URLs only when needed
5. **Automatic Versioning**: S3 versioning tracks document updates automatically

## S3 Structure

```
Bucket: dsai-2025-asu
├── knowledge_base/
│   ├── model_validation/*.pdf
│   ├── legal/*.pdf
│   └── technical/*.pdf
```

## Database Schema

### Documents Table
- `s3_bucket`: Source bucket name
- `s3_key`: Document path in S3
- `s3_url`: Direct S3 access URL
- `s3_etag`: Version tracking
- `domain`: Knowledge domain classification

### Chunks Table
- `s3_reference`: Direct S3 URL for content
- `embedding`: Vector(1536) for similarity search
- `content_preview`: First 200 chars only
- `start_byte/end_byte`: Exact S3 content location

## Query Workflow

1. **Generate query embedding**
2. **Vector similarity search** → Returns S3 references
3. **Fetch content from S3 URLs** on-demand
4. **Generate context-aware answer**
5. **Return answer + S3 source citations**

## Key Benefits

- **Storage Efficiency**: No content duplication
- **Scalability**: Infinite S3 capacity, no local storage limits
- **Distribution**: Multiple systems access same S3 knowledge base
- **Security**: S3 IAM controls document access permissions
- **Cost**: S3 storage cheaper than duplicating in vector DB
- **Backup**: S3 built-in redundancy and backup
- **Updates**: Change documents in S3, re-embed, no data duplication

## Critical Constraints

- **MUST**: Store documents once in S3 only
- **MUST**: Vector DB references S3, never duplicates content
- **MUST**: Process S3 → S3 (streaming), no local file processing
- **MUST**: Fetch content on-demand from S3 URLs
- **MUST NOT**: Store full document content in vector database
- **MUST NOT**: Use app folders, temp directories, or local storage
- **MUST NOT**: Duplicate documents across storage systems