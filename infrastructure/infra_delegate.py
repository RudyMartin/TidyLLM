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

# Note: Path setup should be done by the application using PathManager,
# not by the delegate itself (architecture principle)


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

        except ImportError as e:
            logger.debug(f"Could not import parent aws_service: {e}")

        # Try to get credentials and create a basic AWS service
        try:
            from infrastructure.services.credential_carrier import get_credential_carrier
            cred_carrier = get_credential_carrier()

            # Get AWS credentials from settings
            aws_creds = cred_carrier.get_credential('aws_bedrock')
            if not aws_creds:
                # Try environment variables
                import os
                if os.getenv('AWS_ACCESS_KEY_ID'):
                    aws_creds = {
                        'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                        'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                        'region': os.getenv('AWS_REGION', 'us-east-1')
                    }

            if aws_creds:
                # Create a simple AWS service wrapper
                logger.info("✅ AWS: Using credentials from settings/env")
                return self._create_simple_aws_service(aws_creds)

        except Exception as e:
            logger.debug(f"Could not setup AWS with credentials: {e}")

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

    def _create_simple_aws_service(self, creds):
        """Create a simple AWS service wrapper when parent is not available."""
        try:
            import boto3

            class SimpleAWSService:
                def __init__(self, credentials):
                    self.creds = credentials
                    self._bedrock_runtime = None

                def is_available(self):
                    return True

                def invoke_model(self, prompt, model_id='anthropic.claude-3-haiku-20240307-v1:0'):
                    """Simple Bedrock invocation."""
                    if not self._bedrock_runtime:
                        self._bedrock_runtime = boto3.client(
                            'bedrock-runtime',
                            region_name=self.creds.get('region', 'us-east-1'),
                            aws_access_key_id=self.creds.get('access_key_id'),
                            aws_secret_access_key=self.creds.get('secret_access_key')
                        )

                    import json
                    # Format for Claude 3
                    request_body = {
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 1000,
                        "temperature": 0.7,
                        "anthropic_version": "bedrock-2023-05-31"
                    }

                    response = self._bedrock_runtime.invoke_model(
                        modelId=model_id,
                        body=json.dumps(request_body),
                        contentType='application/json'
                    )

                    response_body = json.loads(response['body'].read())
                    # Extract text from Claude 3 response
                    if 'content' in response_body:
                        return {'text': response_body['content'][0]['text']}
                    return {'text': str(response_body)}

            return SimpleAWSService(creds)

        except ImportError:
            logger.debug("boto3 not available for simple AWS service")
            return None

    def invoke_bedrock(self, prompt: str, model_id: str = None) -> Dict[str, Any]:
        """
        Invoke Bedrock model directly - LOW LEVEL INFRASTRUCTURE CALL.

        This is the BASE LAYER that gateways call. Must NOT call back to gateways
        to avoid circular dependencies.
        """
        model_id = model_id or 'anthropic.claude-3-haiku-20240307-v1:0'

        try:
            # Get AWS credentials from infrastructure
            aws_config = self._get_aws_config()
            if not aws_config:
                return {
                    'success': False,
                    'error': 'AWS credentials not available',
                    'gateway_tracked': False
                }

            # Make direct Bedrock call (this is the base infrastructure layer)
            import boto3
            import json

            bedrock_runtime = boto3.client(
                'bedrock-runtime',
                region_name=aws_config.get('region', 'us-east-1'),
                aws_access_key_id=aws_config.get('access_key_id'),
                aws_secret_access_key=aws_config.get('secret_access_key')
            )

            # Format request for Claude
            request_body = {
                'messages': [{'role': 'user', 'content': prompt}],
                'max_tokens': 4000,
                'temperature': 0.7,
                'anthropic_version': 'bedrock-2023-05-31'
            }

            response = bedrock_runtime.invoke_model(
                modelId=model_id,
                body=json.dumps(request_body),
                contentType='application/json'
            )

            # Parse Bedrock response
            response_body = json.loads(response['body'].read())
            content = response_body['content'][0]['text']

            return {
                'success': True,
                'text': content,
                'model': model_id,
                'gateway_tracked': False  # This is direct infrastructure access
            }

        except Exception as e:
            logger.error(f"Direct Bedrock call failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'gateway_tracked': False
            }

    def _get_aws_config(self) -> Dict[str, Any]:
        """Get AWS configuration from settings."""
        try:
            from infrastructure.yaml_loader import SettingsLoader
            loader = SettingsLoader()
            return loader.get_aws_config()
        except Exception as e:
            logger.warning(f"Could not load AWS config: {e}")
            # Fallback to environment variables
            import os
            return {
                'access_key_id': os.getenv('AWS_ACCESS_KEY_ID'),
                'secret_access_key': os.getenv('AWS_SECRET_ACCESS_KEY'),
                'region': os.getenv('AWS_DEFAULT_REGION', 'us-east-1')
            }

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
        Generate LLM response - ONLY through CorporateLLMGateway.

        NO FALLBACKS TO DIRECT AWS - ALL CALLS MUST BE TRACKED!
        """
        if not self._llm:
            self._llm = self._init_llm()

        if self._llm:
            # Use gateway that tracks all calls
            try:
                from tidyllm.gateways.corporate_llm_gateway import LLMRequest

                request = LLMRequest(
                    prompt=prompt,
                    model_id=config.get('model', 'claude-3-sonnet') if config else 'claude-3-sonnet',
                    temperature=config.get('temperature', 0.7) if config else 0.7,
                    max_tokens=config.get('max_tokens', 1500) if config else 1500,
                    user_id='rag_system',
                    audit_reason='llm_generation'
                )

                response = self._llm.process_request(request)
                return {
                    'success': response.success,
                    'text': response.content if response.success else '',
                    'model': response.model_used,
                    'gateway_tracked': True
                }
            except Exception as e:
                logger.error(f"Gateway LLM generation failed: {e}")
                return {
                    'success': False,
                    'text': '',
                    'error': str(e),
                    'gateway_tracked': False
                }
        else:
            # NO DIRECT CALLS - GATEWAY REQUIRED!
            return {
                'success': False,
                'text': '',
                'error': 'CorporateLLMGateway required - no direct AWS calls allowed',
                'gateway_tracked': False
            }

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
        # Use the yaml_loader to get properly parsed bedrock config
        try:
            from infrastructure.yaml_loader import SettingsLoader
            loader = SettingsLoader()
            return loader.get_bedrock_config()
        except Exception as e:
            logger.warning(f"Could not load bedrock config via yaml_loader: {e}")
            # Fallback to direct config loading
            config = self._load_config()
            # Try credentials.bedrock_llm first (current structure), then bedrock (legacy)
            return config.get('credentials', {}).get('bedrock_llm', config.get('bedrock', {
                'model_mapping': {},
                'models': {}
            }))

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