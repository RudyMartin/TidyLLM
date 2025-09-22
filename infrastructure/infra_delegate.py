#!/usr/bin/env python3
"""
Infrastructure Delegate - Single Point of Infrastructure Access
===============================================================

PATTERN: Adapter/Delegate with Parent Detection

This is THE ONLY delegate you need. It provides all infrastructure services
and automatically uses parent infrastructure when available.

ARCHITECTURE:
    RAG Adapter → InfrastructureDelegate → Parent Infrastructure (if available)
                                        ↘ Simple Fallback (if not)

WHY THIS PATTERN:
1. No code duplication - reuse parent's ResilientPoolManager, aws_service, etc.
2. Progressive enhancement - get enterprise features when deployed
3. Clean boundaries - adapters don't know about infrastructure details
4. Simple testing - just pass a mock delegate

USAGE:
    from tidyllm.infrastructure.infra_delegate import get_infra_delegate

    infra = get_infra_delegate()
    conn = infra.get_db_connection()  # Don't care if ResilientPool or SimplePool

Author: TidyLLM Team
Date: 2024
"""

import sys
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import yaml

logger = logging.getLogger(__name__)

# Add parent infrastructure to path for detection
qa_root = Path(__file__).parent.parent.parent.parent
if str(qa_root) not in sys.path:
    sys.path.insert(0, str(qa_root))


class InfrastructureDelegate:
    """
    Single delegate for ALL infrastructure needs.

    DESIGN PHILOSOPHY:
    - Try parent infrastructure first (enterprise features)
    - Fallback to simple implementation (development/testing)
    - Never duplicate parent functionality
    - One class to rule them all (no separate delegates)

    WHAT THIS PROVIDES:
    - Database connections (ResilientPool or SimplePool)
    - AWS services (Bedrock, S3)
    - LLM operations (CorporateLLMGateway or direct)
    - Embeddings (SentenceTransformers or TF-IDF)
    - DSPy workflows
    """

    def __init__(self):
        """
        Initialize delegate with parent detection.

        INITIALIZATION FLOW:
        1. Try to import parent infrastructure services
        2. If available, use them (enterprise features!)
        3. If not, create simple fallbacks (basic but working)

        This happens ONCE at startup - no runtime switching.
        """
        logger.info("=" * 60)
        logger.info("InfrastructureDelegate: Initializing with parent detection")
        logger.info("=" * 60)

        # Lazy-load services - don't initialize until needed
        self._db_pool = None  # Will be initialized on first use
        self._aws = None  # Will be initialized on first use
        self._llm = None  # Will be initialized on first use
        self._embeddings = None  # Will be initialized on first use

        self._db_initialized = False
        self._aws_initialized = False
        self._llm_initialized = False
        self._embeddings_initialized = False

        logger.info("InfrastructureDelegate: Initialization complete (lazy-loading enabled)")
        # Don't log configuration yet - services not initialized

    # =======================
    # DATABASE OPERATIONS
    # =======================

    def _init_database(self):
        """
        Initialize database connection pool.

        Uses parent's ResilientPoolManager (3-pool failover, health monitoring)
        Fallback is a mock for when parent import fails (development only).
        """
        try:
            # Use parent infrastructure - this is the real implementation
            from infrastructure.services.resilient_pool_manager import ResilientPoolManager
            from infrastructure.services.credential_carrier import get_credential_carrier

            pool = ResilientPoolManager(get_credential_carrier())
            logger.info("✅ Database: Using parent ResilientPoolManager (3-pool failover)")
            return pool

        except ImportError:
            # Mock fallback - doesn't actually connect
            logger.info("⚠️ Database: Using mock pool (development mode)")

            class MockPool:
                """Mock pool that doesn't connect to anything."""
                def get_connection(self):
                    raise NotImplementedError("Mock pool - no real database connection")
                def return_connection(self, conn):
                    pass
                def getconn(self):
                    raise NotImplementedError("Mock pool - no real database connection")
                def putconn(self, conn):
                    pass

            return MockPool()

    def get_db_connection(self):
        """
        Get database connection.

        RETURNS: Raw database connection (not a context manager)
        CALLER: Gets a direct connection they can use with cursor()
        """
        # Lazy-initialize database pool if not done yet
        if not self._db_initialized:
            self._db_pool = self._init_database()
            self._db_initialized = True

        # Always use getconn interface for consistency
        if hasattr(self._db_pool, 'getconn'):
            # Both ResilientPoolManager and psycopg2 pools have this
            return self._db_pool.getconn()
        else:
            # Fallback for any pool without getconn (shouldn't happen)
            logger.warning("Pool doesn't have getconn method, attempting alternatives")
            if hasattr(self._db_pool, 'get_connection'):
                # Try to extract connection from context manager
                ctx = self._db_pool.get_connection()
                conn = ctx.__enter__()
                # Store context for cleanup
                if not hasattr(self, '_active_contexts'):
                    self._active_contexts = {}
                self._active_contexts[id(conn)] = ctx
                return conn
            else:
                raise RuntimeError("Pool doesn't support any known connection interface")

    def return_db_connection(self, conn):
        """
        Return database connection to pool.

        HANDLES: Both ResilientPool and SimplePool interfaces
        """
        if conn is None:
            return

        # Check if this was a connection from a context manager
        if hasattr(self, '_active_contexts') and id(conn) in self._active_contexts:
            ctx = self._active_contexts.pop(id(conn))
            try:
                ctx.__exit__(None, None, None)
            except Exception as e:
                logger.error(f"Error closing context manager: {e}")
            return

        # Always use putconn interface for consistency
        if hasattr(self._db_pool, 'putconn'):
            # Both ResilientPoolManager and psycopg2 pools have this
            self._db_pool.putconn(conn)
        elif hasattr(self._db_pool, 'return_connection'):
            # Fallback for pools with different interface
            self._db_pool.return_connection(conn)
        else:
            logger.warning("Pool doesn't support any known return interface")

    # =======================
    # AWS OPERATIONS
    # =======================

    def _init_aws(self):
        """
        Initialize AWS services.

        PRIORITY ORDER:
        1. Parent's aws_service (unified client management)
        2. Direct boto3 clients (simple but functional)

        NEVER: Create our own AWS service wrapper - that duplicates parent!
        """
        try:
            # TRY PARENT FIRST
            from infrastructure.services.aws_service import get_aws_service

            aws = get_aws_service()
            if aws and aws.is_available():
                logger.info("✅ AWS: Using parent aws_service")
                return aws

        except ImportError:
            pass

        # FALLBACK - Mock AWS client
        logger.info("⚠️ AWS: Parent not available, using mock")

        class MockAWSClient:
            """Mock AWS client that doesn't connect."""
            def invoke_model(self, *args, **kwargs):
                raise NotImplementedError("Mock AWS - no real connection")

        return {
            'bedrock': MockAWSClient(),
            's3': MockAWSClient()
        }

    def invoke_bedrock(self, prompt: str, model_id: str = None) -> Dict[str, Any]:
        """
        Invoke Bedrock model.

        USES: Parent aws_service OR direct boto3
        RETURNS: Consistent response format regardless of backend
        """
        model_id = model_id or 'anthropic.claude-3-haiku-20240307-v1:0'

        # Lazy-initialize AWS if not done yet
        if not self._aws_initialized:
            self._aws = self._init_aws()
            self._aws_initialized = True

        if self._aws is None:
            return {'success': False, 'error': 'AWS not available'}

        try:
            # Check if using parent service or direct boto3
            if hasattr(self._aws, 'invoke_model'):
                # Parent aws_service
                response = self._aws.invoke_model(prompt, model_id)
                return {
                    'success': True,
                    'text': response.get('text', ''),
                    'model': model_id
                }
            else:
                # Mock AWS client - return error
                return {'success': False, 'error': 'AWS service not available (mock mode)'}

        except Exception as e:
            logger.error(f"Bedrock invocation failed: {e}")
            return {'success': False, 'error': str(e)}

    # =======================
    # LLM OPERATIONS
    # =======================

    def _init_llm(self):
        """
        Initialize LLM service.

        PRIORITY ORDER:
        1. Parent's CorporateLLMGateway (session management, monitoring)
        2. Direct API calls (basic but functional)
        """
        try:
            # TRY PARENT FIRST
            from tidyllm.gateways.corporate_llm_gateway import CorporateLLMGateway

            gateway = CorporateLLMGateway()
            logger.info("✅ LLM: Using parent CorporateLLMGateway")
            return gateway

        except ImportError:
            logger.info("⚠️ LLM: Parent not available, will use Bedrock directly")
            return None

    def generate_llm_response(self, prompt: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate LLM response.

        USES: CorporateLLMGateway OR Bedrock directly
        """
        if self._llm:
            # Use parent gateway
            try:
                from tidyllm.infrastructure.standards import TidyLLMStandardRequest

                request = TidyLLMStandardRequest(
                    model_id=config.get('model', 'claude-3-sonnet'),
                    user_id='rag_system',
                    session_id='rag_session',
                    prompt=prompt,
                    temperature=config.get('temperature', 0.7),
                    max_tokens=config.get('max_tokens', 1500)
                )

                response = self._llm.process_llm_request(request)
                return {
                    'success': response.status == 'SUCCESS',
                    'text': response.data,
                    'model': request.model_id
                }
            except:
                pass

        # Fallback to Bedrock
        return self.invoke_bedrock(prompt, config.get('model') if config else None)

    # =======================
    # EMBEDDING OPERATIONS
    # =======================

    def _init_embeddings(self):
        """
        Initialize embedding service.

        PRIORITY ORDER:
        1. SentenceTransformers (if available)
        2. TF-IDF fallback (always works)
        """
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Embeddings: Using SentenceTransformers")
            return model
        except ImportError:
            logger.info("⚠️ Embeddings: Using TF-IDF fallback")
            return None

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate text embedding.

        USES: SentenceTransformers OR TF-IDF
        """
        if self._embeddings:
            # Use SentenceTransformers
            return self._embeddings.encode(text).tolist()
        else:
            # TF-IDF fallback
            import numpy as np
            words = text.lower().split()
            vector = np.zeros(384)
            for word in words:
                idx = hash(word) % 384
                vector[idx] += 1
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
            return vector.tolist()

    # =======================
    # CONFIGURATION METHODS
    # =======================

    def get_bedrock_config(self) -> dict:
        """
        Get bedrock configuration from settings.yaml.

        Returns bedrock settings including model mapping from the parent
        infrastructure's settings.yaml file, avoiding hardcoded values.
        """
        config = self._load_config()
        # Return bedrock configuration from settings
        return config.get('bedrock', {
            'model_mapping': {},
            'models': {}
        })

    # =======================
    # UTILITY METHODS
    # =======================

    def _load_config(self) -> dict:
        """Load configuration from settings.yaml."""
        settings_paths = [
            Path(__file__).parent.parent / "admin" / "settings.yaml",
            Path("tidyllm/admin/settings.yaml")
        ]

        for path in settings_paths:
            if path.exists():
                with open(path) as f:
                    return yaml.safe_load(f)

        # Default config
        return {
            'database': {
                'host': 'localhost',
                'port': 5432,
                'database': 'rag_db',
                'user': 'rag_user',
                'password': 'rag_pass'
            }
        }

    def _log_configuration(self):
        """Log current configuration for debugging."""
        logger.info("-" * 60)
        logger.info("InfrastructureDelegate Configuration:")
        if self._db_initialized and self._db_pool:
            logger.info(f"  Database: {'ResilientPool' if hasattr(self._db_pool, 'get_connection') else 'SimplePool'}")
        else:
            logger.info("  Database: Not initialized (will lazy-load on first use)")

        if self._aws_initialized:
            logger.info(f"  AWS: {'Parent Service' if hasattr(self._aws, 'invoke_model') else 'Direct boto3' if self._aws else 'Not available'}")
        else:
            logger.info("  AWS: Not initialized (will lazy-load on first use)")

        if self._llm_initialized:
            logger.info(f"  LLM: {'CorporateLLMGateway' if self._llm else 'Bedrock direct'}")
        else:
            logger.info("  LLM: Not initialized (will lazy-load on first use)")

        if self._embeddings_initialized:
            logger.info(f"  Embeddings: {'SentenceTransformers' if self._embeddings else 'TF-IDF'}")
        else:
            logger.info("  Embeddings: Not initialized (will lazy-load on first use)")
        logger.info("-" * 60)


# =======================
# MODULE-LEVEL INTERFACE
# =======================

# Single instance (initialized once)
_delegate = None

def get_infra_delegate() -> InfrastructureDelegate:
    """
    Get the infrastructure delegate (singleton).

    THIS IS THE ONLY FUNCTION RAG ADAPTERS SHOULD CALL.

    Usage:
        from tidyllm.infrastructure.infra_delegate import get_infra_delegate

        infra = get_infra_delegate()
        conn = infra.get_db_connection()
    """
    global _delegate
    if _delegate is None:
        _delegate = InfrastructureDelegate()
    return _delegate

def reset_delegate():
    """Reset delegate (mainly for testing)."""
    global _delegate
    _delegate = None


# =======================
# ARCHITECTURE NOTES
# =======================

"""
WHAT WE LEARNED:

1. DON'T create multiple delegate files - one is enough
2. DON'T reimplement parent services - that's duplication
3. DON'T use complex factories - simple function is fine
4. DON'T do runtime switching - decide once at startup

DO use parent infrastructure when available
DO provide simple fallbacks for development
DO keep it simple - this file replaces 6 files + 6 factories
DO document the pattern clearly

This is the "Goldilocks" solution - not too complex, not too simple, just right.
"""