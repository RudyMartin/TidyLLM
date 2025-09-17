"""
Unified PostgreSQL Adapter - Single adapter implementing multiple ports
Replaces all 8 duplicate PostgreSQL adapters
"""

import json
import asyncio
import asyncpg
from typing import List, Dict, Any, Optional
from dataclasses import asdict

from ...domain.ports.outbound import (
    DocumentRepositoryPort,
    ComplianceRepositoryPort,
    Document,
    ComplianceRule
)


class UnifiedPostgreSQLAdapter(DocumentRepositoryPort, ComplianceRepositoryPort):
    """
    Single PostgreSQL adapter implementing multiple domain ports.
    NO BUSINESS LOGIC - only data translation and storage.
    """

    def __init__(self, connection_string: str):
        """Initialize with connection string only"""
        self.connection_string = connection_string
        self._pool = None

    async def _ensure_connection(self):
        """Ensure connection pool exists"""
        if not self._pool:
            self._pool = await asyncpg.create_pool(self.connection_string)

    # DocumentRepositoryPort implementation
    async def find_by_query(self, query: str, limit: int = 10) -> List[Document]:
        """Find documents matching query - NO BUSINESS LOGIC"""
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            # Pure data retrieval - no business decisions
            rows = await conn.fetch("""
                SELECT id, content, metadata, embedding
                FROM sme_document_chunks
                WHERE content ILIKE $1
                LIMIT $2
            """, f'%{query}%', limit)

            return [
                Document(
                    id=row['id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    embedding=row['embedding']
                )
                for row in rows
            ]

    async def save_document(self, document: Document) -> str:
        """Save document - pure storage operation"""
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            doc_id = await conn.fetchval("""
                INSERT INTO sme_documents (content, metadata)
                VALUES ($1, $2::jsonb)
                RETURNING id
            """, document.content, json.dumps(document.metadata))

            return str(doc_id)

    async def get_by_id(self, doc_id: str) -> Optional[Document]:
        """Get document by ID - pure retrieval"""
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT id, content, metadata, embedding
                FROM sme_document_chunks
                WHERE id = $1
            """, doc_id)

            if row:
                return Document(
                    id=row['id'],
                    content=row['content'],
                    metadata=json.loads(row['metadata']) if row['metadata'] else {},
                    embedding=row['embedding']
                )
            return None

    # ComplianceRepositoryPort implementation
    async def find_rules_by_domain(self, domain: str) -> List[ComplianceRule]:
        """Find compliance rules - NO AUTHORITY DECISIONS"""
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, authority_tier, rule_text, precedence
                FROM compliance_rules
                WHERE domain = $1
                ORDER BY precedence DESC
            """, domain)

            return [
                ComplianceRule(
                    id=row['id'],
                    authority_tier=row['authority_tier'],
                    rule_text=row['rule_text'],
                    precedence=row['precedence']
                )
                for row in rows
            ]

    async def find_by_authority_tier(self, tier: int) -> List[ComplianceRule]:
        """Find rules by tier - pure data retrieval"""
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT id, authority_tier, rule_text, precedence
                FROM compliance_rules
                WHERE authority_tier = $1
                ORDER BY precedence DESC
            """, tier)

            return [
                ComplianceRule(
                    id=row['id'],
                    authority_tier=row['authority_tier'],
                    rule_text=row['rule_text'],
                    precedence=row['precedence']
                )
                for row in rows
            ]

    async def save_rule(self, rule: ComplianceRule) -> str:
        """Save compliance rule - pure storage"""
        await self._ensure_connection()

        async with self._pool.acquire() as conn:
            rule_id = await conn.fetchval("""
                INSERT INTO compliance_rules (authority_tier, rule_text, precedence)
                VALUES ($1, $2, $3)
                RETURNING id
            """, rule.authority_tier, rule.rule_text, rule.precedence)

            return str(rule_id)

    async def close(self):
        """Clean up connection pool"""
        if self._pool:
            await self._pool.close()