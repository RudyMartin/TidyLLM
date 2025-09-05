#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Backend Configuration for TidyLLM Whitepapers Demo

Provides PostgreSQL integration with pgvector for embedding storage,
following the tidyLLM pattern for backend configuration.
"""

import os
import yaml
import streamlit as st
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor, Json
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

@dataclass
class PostgresConfig:
    """Configuration for PostgreSQL database connection with pgvector."""
    host: str = "localhost"
    port: int = 5432
    database: str = "research_papers_db"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_pool_size: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    
    def get_connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}?sslmode={self.ssl_mode}"

@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation and storage."""
    model: str = "text-embedding-3-large"
    dimensions: int = 1024
    batch_size: int = 50
    cache_enabled: bool = True
    similarity_threshold: float = 0.7
    max_results: int = 10

@dataclass
class ChatConfig:
    """Configuration for chat/text generation models."""
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0

@dataclass
class BackendSettings:
    """Container for all backend settings."""
    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    openai_api_key: str = ""
    aws_region: str = "us-east-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_kms_key_id: str = ""
    embedding_provider: str = "bedrock"  # "openai" or "bedrock"
    chat_provider: str = "openai"  # "openai" or "bedrock" 
    debug_mode: bool = False

class BackendConfigManager:
    """
    Manages backend configuration for TidyLLM Whitepapers Demo.
    Handles PostgreSQL connections, embedding storage, and configuration UI.
    """
    
    def __init__(self):
        self.settings = BackendSettings()
        self.connection = None
        self.config_file = Path(__file__).parent / "settings.yaml"
        self._load_from_yaml()
        self._load_from_env()
    
    def _load_from_yaml(self):
        """Load configuration from settings.yaml file if it exists."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = yaml.safe_load(f)
                    
                if config:
                    # Load PostgreSQL settings
                    if 'postgres' in config:
                        postgres_cfg = config['postgres']
                        self.settings.postgres.host = postgres_cfg.get('host', self.settings.postgres.host)
                        self.settings.postgres.port = postgres_cfg.get('port', self.settings.postgres.port)
                        self.settings.postgres.database = postgres_cfg.get('database', self.settings.postgres.database)
                        self.settings.postgres.username = postgres_cfg.get('username', self.settings.postgres.username)
                        self.settings.postgres.password = postgres_cfg.get('password', self.settings.postgres.password)
                        self.settings.postgres.ssl_mode = postgres_cfg.get('ssl_mode', self.settings.postgres.ssl_mode)
                    
                    # Load OpenAI API key
                    if 'openai' in config:
                        self.settings.openai_api_key = config['openai'].get('api_key', self.settings.openai_api_key)
                    
                    # Load AWS settings
                    if 'aws' in config:
                        aws_cfg = config['aws']
                        self.settings.aws_region = aws_cfg.get('region', self.settings.aws_region)
                        self.settings.aws_access_key_id = aws_cfg.get('access_key_id', self.settings.aws_access_key_id)
                        self.settings.aws_secret_access_key = aws_cfg.get('secret_access_key', self.settings.aws_secret_access_key)
                        self.settings.aws_kms_key_id = aws_cfg.get('kms_key_id', self.settings.aws_kms_key_id)
                    
                    # Load Embedding settings
                    if 'embeddings' in config:
                        emb_cfg = config['embeddings']
                        self.settings.embedding_provider = emb_cfg.get('provider', self.settings.embedding_provider)
                        self.settings.embeddings.model = emb_cfg.get('model', self.settings.embeddings.model)
                        self.settings.embeddings.dimensions = emb_cfg.get('dimensions', self.settings.embeddings.dimensions)
                        self.settings.embeddings.similarity_threshold = emb_cfg.get('similarity_threshold', self.settings.embeddings.similarity_threshold)
                        self.settings.embeddings.max_results = emb_cfg.get('max_results', self.settings.embeddings.max_results)
                    
                    # Load Chat settings
                    if 'chat' in config:
                        chat_cfg = config['chat']
                        self.settings.chat_provider = chat_cfg.get('provider', self.settings.chat_provider)
                        self.settings.chat.model = chat_cfg.get('model', self.settings.chat.model)
                        self.settings.chat.temperature = chat_cfg.get('temperature', self.settings.chat.temperature)
                        self.settings.chat.max_tokens = chat_cfg.get('max_tokens', self.settings.chat.max_tokens)
                        self.settings.chat.top_p = chat_cfg.get('top_p', self.settings.chat.top_p)
                        self.settings.chat.frequency_penalty = chat_cfg.get('frequency_penalty', self.settings.chat.frequency_penalty)
                        self.settings.chat.presence_penalty = chat_cfg.get('presence_penalty', self.settings.chat.presence_penalty)
                    
                    # Load debug mode
                    if 'debug_mode' in config:
                        self.settings.debug_mode = config.get('debug_mode', self.settings.debug_mode)
                        
            except Exception as e:
                st.warning(f"Could not load settings.yaml: {str(e)}")
    
    def save_to_yaml(self):
        """Save current configuration to settings.yaml file."""
        config = {
            'postgres': {
                'host': self.settings.postgres.host,
                'port': self.settings.postgres.port,
                'database': self.settings.postgres.database,
                'username': self.settings.postgres.username,
                'password': self.settings.postgres.password,
                'ssl_mode': self.settings.postgres.ssl_mode
            },
            'openai': {
                'api_key': self.settings.openai_api_key
            },
            'aws': {
                'region': self.settings.aws_region,
                'access_key_id': self.settings.aws_access_key_id,
                'secret_access_key': self.settings.aws_secret_access_key,
                'kms_key_id': self.settings.aws_kms_key_id
            },
            'embeddings': {
                'provider': self.settings.embedding_provider,
                'model': self.settings.embeddings.model,
                'dimensions': self.settings.embeddings.dimensions,
                'similarity_threshold': self.settings.embeddings.similarity_threshold,
                'max_results': self.settings.embeddings.max_results
            },
            'chat': {
                'provider': self.settings.chat_provider,
                'model': self.settings.chat.model,
                'temperature': self.settings.chat.temperature,
                'max_tokens': self.settings.chat.max_tokens,
                'top_p': self.settings.chat.top_p,
                'frequency_penalty': self.settings.chat.frequency_penalty,
                'presence_penalty': self.settings.chat.presence_penalty
            },
            'debug_mode': self.settings.debug_mode
        }
        
        try:
            with open(self.config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            st.error(f"Failed to save settings: {str(e)}")
            return False
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        # PostgreSQL configuration
        self.settings.postgres.host = os.getenv('POSTGRES_HOST', self.settings.postgres.host)
        self.settings.postgres.port = int(os.getenv('POSTGRES_PORT', str(self.settings.postgres.port)))
        self.settings.postgres.database = os.getenv('POSTGRES_DB', self.settings.postgres.database)
        self.settings.postgres.username = os.getenv('POSTGRES_USER', self.settings.postgres.username)
        self.settings.postgres.password = os.getenv('POSTGRES_PASSWORD', self.settings.postgres.password)
        
        # OpenAI API key
        self.settings.openai_api_key = os.getenv('OPENAI_API_KEY', self.settings.openai_api_key or '')
        
        # Embedding configuration
        self.settings.embeddings.model = os.getenv('EMBEDDING_MODEL', self.settings.embeddings.model)
        self.settings.embeddings.dimensions = int(os.getenv('EMBEDDING_DIMS', str(self.settings.embeddings.dimensions)))
    
    def render_config_ui(self):
        """Render Streamlit configuration UI."""
        st.header("🔧 Backend Configuration")
        
        # Note about auto-save
        st.info("💾 Configuration is automatically saved when you make changes")
        
        with st.expander("📊 Database Connection", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                self.settings.postgres.host = st.text_input(
                    "Host", 
                    value=self.settings.postgres.host,
                    help="PostgreSQL server host"
                )
                self.settings.postgres.database = st.text_input(
                    "Database", 
                    value=self.settings.postgres.database,
                    help="Database name"
                )
                self.settings.postgres.username = st.text_input(
                    "Username", 
                    value=self.settings.postgres.username,
                    help="Database username"
                )
            
            with col2:
                self.settings.postgres.port = st.number_input(
                    "Port", 
                    value=self.settings.postgres.port,
                    min_value=1, max_value=65535,
                    help="PostgreSQL server port"
                )
                self.settings.postgres.ssl_mode = st.selectbox(
                    "SSL Mode",
                    ["prefer", "require", "disable"],
                    index=["prefer", "require", "disable"].index(self.settings.postgres.ssl_mode)
                )
                self.settings.postgres.password = st.text_input(
                    "Password", 
                    value=self.settings.postgres.password,
                    type="password",
                    help="Database password"
                )
            
            # Test connection button
            if st.button("🔍 Test Database Connection"):
                self.test_database_connection()
        
        with st.expander("🧮 Embedding Configuration", expanded=False):
            # Provider selection
            self.settings.embedding_provider = st.selectbox(
                "Embedding Provider",
                ["openai", "bedrock"],
                index=["openai", "bedrock"].index(self.settings.embedding_provider),
                help="Choose between OpenAI or AWS Bedrock for embeddings"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if self.settings.embedding_provider == "openai":
                    self.settings.openai_api_key = st.text_input(
                        "OpenAI API Key",
                        value=self.settings.openai_api_key,
                        type="password",
                        help="OpenAI API key for embeddings"
                    )
                    model_options = [
                        "text-embedding-3-large",     # Latest & most capable (3072 dims)
                        "text-embedding-3-small",     # Latest & cost-effective (1536 dims)  
                        "text-embedding-ada-002"      # Legacy but still supported (1536 dims)
                    ]
                else:  # bedrock
                    # AWS credentials in left column
                    self.settings.aws_access_key_id = st.text_input(
                        "AWS Access Key ID",
                        value=self.settings.aws_access_key_id,
                        help="AWS Access Key ID for Bedrock"
                    )
                    self.settings.aws_secret_access_key = st.text_input(
                        "AWS Secret Access Key",
                        value=self.settings.aws_secret_access_key,
                        type="password",
                        help="AWS Secret Access Key for Bedrock"
                    )
                    model_options = [
                        "amazon.titan-embed-text-v1",
                        "amazon.titan-embed-text-v2:0",
                        "cohere.embed-english-v3",
                        "cohere.embed-multilingual-v3"
                    ]
            
            with col2:
                if self.settings.embedding_provider == "bedrock":
                    # AWS region and KMS in right column for Bedrock
                    self.settings.aws_region = st.text_input(
                        "AWS Region",
                        value=self.settings.aws_region,
                        help="AWS Region for Bedrock (e.g., us-east-1)"
                    )
                    self.settings.aws_kms_key_id = st.text_input(
                        "AWS KMS Key ID (Optional)",
                        value=self.settings.aws_kms_key_id,
                        help="AWS KMS Key ID for encryption (leave empty if not using KMS)"
                    )
                
                # Model selection based on provider
                if self.settings.embeddings.model not in model_options:
                    self.settings.embeddings.model = model_options[0]
                    
                self.settings.embeddings.model = st.selectbox(
                    "Embedding Model",
                    model_options,
                    index=model_options.index(self.settings.embeddings.model) if self.settings.embeddings.model in model_options else 0
                )
            
            with col2:
                self.settings.embeddings.dimensions = st.number_input(
                    "Dimensions",
                    value=self.settings.embeddings.dimensions,
                    min_value=512, max_value=3072,
                    help="Embedding dimensions"
                )
                self.settings.embeddings.similarity_threshold = st.slider(
                    "Similarity Threshold",
                    min_value=0.0, max_value=1.0,
                    value=self.settings.embeddings.similarity_threshold,
                    step=0.1,
                    help="Minimum similarity for search results"
                )
            
            # Test embedding connection button
            if st.button("🔍 Test Embedding Connection"):
                self.test_embedding_connection()
        
        with st.expander("💬 Chat Model Configuration", expanded=False):
            # Chat provider selection
            self.settings.chat_provider = st.selectbox(
                "Chat Provider",
                ["openai", "bedrock"],
                index=["openai", "bedrock"].index(self.settings.chat_provider),
                help="Choose between OpenAI or AWS Bedrock for chat/text generation"
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if self.settings.chat_provider == "openai":
                    # Use the same API key as embeddings
                    if not self.settings.openai_api_key:
                        st.info("💡 Chat models will use the same OpenAI API key as embeddings")
                    
                    chat_model_options = [
                        "gpt-4o",              # Latest multimodal model
                        "gpt-4o-mini",         # Cost-effective latest model
                        "gpt-4-turbo",         # Previous generation turbo
                        "gpt-4",               # Standard GPT-4
                        "gpt-3.5-turbo"        # Most cost-effective
                    ]
                else:  # bedrock
                    # Use same AWS credentials as embeddings
                    if not self.settings.aws_access_key_id:
                        st.info("💡 Chat models will use the same AWS credentials as embeddings")
                    
                    chat_model_options = [
                        "anthropic.claude-3-5-sonnet-20241022-v2:0",  # Latest Claude
                        "anthropic.claude-3-haiku-20240307-v1:0",     # Fast Claude
                        "amazon.titan-text-premier-v1:0",             # Amazon Titan
                        "meta.llama3-2-90b-instruct-v1:0"             # Llama 3.2
                    ]
                
                # Model selection based on provider
                if self.settings.chat.model not in chat_model_options:
                    self.settings.chat.model = chat_model_options[0]
                    
                self.settings.chat.model = st.selectbox(
                    "Chat Model",
                    chat_model_options,
                    index=chat_model_options.index(self.settings.chat.model) if self.settings.chat.model in chat_model_options else 0,
                    help="Model for text generation and analysis"
                )
            
            with col2:
                self.settings.chat.temperature = st.slider(
                    "Temperature",
                    min_value=0.0, max_value=2.0,
                    value=self.settings.chat.temperature,
                    step=0.1,
                    help="Creativity level (0.0=focused, 2.0=creative)"
                )
                
                self.settings.chat.max_tokens = st.number_input(
                    "Max Tokens",
                    min_value=100, max_value=32000,
                    value=self.settings.chat.max_tokens,
                    step=100,
                    help="Maximum length of generated text"
                )
            
            # Test chat connection button
            if st.button("🔍 Test Chat Connection"):
                self.test_chat_connection()
        
        # Auto-save configuration after any changes
        self.save_to_yaml()
        
        # Manual save/reload buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Force Save", help="Manually save configuration"):
                if self.save_to_yaml():
                    st.success("✅ Configuration saved to settings.yaml")
                else:
                    st.error("❌ Failed to save configuration")
        
        with col2:
            if st.button("🔄 Reload Configuration"):
                self._load_from_yaml()
                st.success("✅ Configuration reloaded from settings.yaml")
                st.rerun()
        
        # Show current status
        self.render_status_ui()
    
    def render_status_ui(self):
        """Render backend status information."""
        st.markdown("---")
        st.subheader("📊 Backend Status")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            postgres_status = "✅ Available" if POSTGRES_AVAILABLE else "❌ Not Available"
            st.metric("PostgreSQL Driver", postgres_status)
        
        with col2:
            numpy_status = "✅ Available" if NUMPY_AVAILABLE else "❌ Not Available"  
            st.metric("NumPy (Similarity)", numpy_status)
        
        with col3:
            api_key_status = "✅ Configured" if self.settings.openai_api_key else "❌ Missing"
            st.metric("OpenAI API Key", api_key_status)
        
        # Database connection status
        if self.connection:
            st.success("🟢 Database connected successfully")
            
            # Show database stats
            stats = self.get_database_stats()
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Papers", stats.get('papers', 0))
                with col2:
                    st.metric("Embeddings", stats.get('embeddings', 0))
                with col3:
                    st.metric("Authors", stats.get('unique_authors', 0))
                with col4:
                    st.metric("Storage Size", f"{stats.get('db_size_mb', 0):.1f} MB")
        else:
            st.info("🔵 Database not connected - click 'Test Database Connection' to connect")
    
    def test_database_connection(self):
        """Test database connection and initialize if successful."""
        if not POSTGRES_AVAILABLE:
            st.error("❌ psycopg2 not available. Install with: pip install psycopg2-binary")
            return False
        
        try:
            # Test connection
            conn = psycopg2.connect(
                host=self.settings.postgres.host,
                port=self.settings.postgres.port,
                database=self.settings.postgres.database,
                user=self.settings.postgres.username,
                password=self.settings.postgres.password,
                sslmode=self.settings.postgres.ssl_mode
            )
            
            # Initialize database tables
            self._init_database(conn)
            
            self.connection = conn
            st.success("✅ Database connection successful!")
            
            # Show database info
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT version();")
                version = cursor.fetchone()['version']
                st.info(f"📊 {version}")
            
            return True
            
        except Exception as e:
            st.error(f"❌ Database connection failed: {str(e)}")
            self.connection = None
            return False
    
    def test_embedding_connection(self):
        """Test embedding connection by generating a test embedding."""
        test_text = "This is a test embedding for TidyLLM Whitepapers"
        
        if self.settings.embedding_provider == "openai":
            return self._test_openai_embedding(test_text)
        elif self.settings.embedding_provider == "bedrock":
            return self._test_bedrock_embedding(test_text)
        else:
            st.error(f"❌ Unknown embedding provider: {self.settings.embedding_provider}")
            return False
    
    def test_chat_connection(self):
        """Test chat connection by generating a test response."""
        test_prompt = "Explain the Y=R+S+N mathematical framework in one sentence."
        
        if self.settings.chat_provider == "openai":
            return self._test_openai_chat(test_prompt)
        elif self.settings.chat_provider == "bedrock":
            return self._test_bedrock_chat(test_prompt)
        else:
            st.error(f"❌ Unknown chat provider: {self.settings.chat_provider}")
            return False
    
    def _test_openai_embedding(self, test_text: str):
        """Test OpenAI embedding connection."""
        if not self.settings.openai_api_key:
            st.error("❌ OpenAI API key is not configured")
            return False
        
        try:
            # Try to import OpenAI
            try:
                from openai import OpenAI
            except ImportError:
                st.error("❌ OpenAI library not installed. Install with: pip install openai")
                return False
            
            # Initialize OpenAI client
            client = OpenAI(api_key=self.settings.openai_api_key)
            
            with st.spinner("Testing OpenAI embedding connection..."):
                response = client.embeddings.create(
                    model=self.settings.embeddings.model,
                    input=test_text,
                    dimensions=self.settings.embeddings.dimensions if self.settings.embeddings.model.startswith("text-embedding-3") else None
                )
                
                # Get the embedding
                embedding = response.data[0].embedding
                
                # Show success message with details
                st.success("✅ OpenAI embedding connection successful!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Provider", "OpenAI")
                with col2:
                    st.metric("Model", self.settings.embeddings.model)
                with col3:
                    st.metric("Dimensions", len(embedding))
                
                # Show sample of embedding vector
                st.info(f"📊 Sample embedding vector (first 5 values): {embedding[:5]}")
                
                return True
                
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                st.error("❌ Invalid OpenAI API key")
            elif "model" in error_msg.lower():
                st.error(f"❌ Model '{self.settings.embeddings.model}' not available")
            elif "quota" in error_msg.lower():
                st.error("❌ OpenAI API quota exceeded")
            else:
                st.error(f"❌ OpenAI embedding connection failed: {error_msg}")
            return False
    
    def _test_bedrock_embedding(self, test_text: str):
        """Test AWS Bedrock embedding connection."""
        if not self.settings.aws_access_key_id or not self.settings.aws_secret_access_key:
            st.error("❌ AWS credentials are not configured")
            return False
        
        try:
            # Try to import boto3
            try:
                import boto3
                import json
            except ImportError:
                st.error("❌ boto3 not installed. Install with: pip install boto3")
                return False
            
            # Initialize Bedrock client
            client_kwargs = {
                'service_name': 'bedrock-runtime',
                'region_name': self.settings.aws_region,
                'aws_access_key_id': self.settings.aws_access_key_id,
                'aws_secret_access_key': self.settings.aws_secret_access_key
            }
            
            bedrock = boto3.client(**client_kwargs)
            
            with st.spinner("Testing AWS Bedrock embedding connection..."):
                # Format request based on model
                if self.settings.embeddings.model.startswith("amazon.titan"):
                    body = json.dumps({"inputText": test_text})
                    response = bedrock.invoke_model(
                        modelId=self.settings.embeddings.model,
                        body=body,
                        contentType='application/json',
                        accept='application/json'
                    )
                    result = json.loads(response['body'].read())
                    embedding = result['embedding']
                    
                elif self.settings.embeddings.model.startswith("cohere"):
                    body = json.dumps({
                        "texts": [test_text],
                        "input_type": "search_document"
                    })
                    response = bedrock.invoke_model(
                        modelId=self.settings.embeddings.model,
                        body=body,
                        contentType='application/json',
                        accept='application/json'
                    )
                    result = json.loads(response['body'].read())
                    embedding = result['embeddings'][0]
                else:
                    st.error(f"❌ Unsupported Bedrock model: {self.settings.embeddings.model}")
                    return False
                
                # Show success message with details
                st.success("✅ AWS Bedrock embedding connection successful!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Provider", "AWS Bedrock")
                with col2:
                    st.metric("Model", self.settings.embeddings.model)
                with col3:
                    st.metric("Dimensions", len(embedding))
                
                # Show sample of embedding vector
                st.info(f"📊 Sample embedding vector (first 5 values): {embedding[:5]}")
                
                return True
                
        except Exception as e:
            error_msg = str(e)
            if "credentials" in error_msg.lower() or "security token" in error_msg.lower():
                st.error("❌ Invalid AWS credentials")
            elif "access denied" in error_msg.lower():
                st.error("❌ AWS access denied - check IAM permissions for Bedrock")
            elif "not authorized" in error_msg.lower():
                st.error(f"❌ Not authorized to use model '{self.settings.embeddings.model}'")
            elif "region" in error_msg.lower():
                st.error(f"❌ Bedrock not available in region '{self.settings.aws_region}'")
            else:
                st.error(f"❌ Bedrock embedding connection failed: {error_msg}")
            return False
    
    def _test_openai_chat(self, test_prompt: str):
        """Test OpenAI chat connection."""
        if not self.settings.openai_api_key:
            st.error("❌ OpenAI API key is not configured")
            return False
        
        try:
            # Try to import OpenAI
            try:
                from openai import OpenAI
            except ImportError:
                st.error("❌ OpenAI library not installed. Install with: pip install openai")
                return False
            
            # Initialize OpenAI client
            client = OpenAI(api_key=self.settings.openai_api_key)
            
            with st.spinner("Testing OpenAI chat connection..."):
                response = client.chat.completions.create(
                    model=self.settings.chat.model,
                    messages=[
                        {"role": "user", "content": test_prompt}
                    ],
                    temperature=self.settings.chat.temperature,
                    max_tokens=min(self.settings.chat.max_tokens, 150)  # Limit for test
                )
                
                # Get the response
                chat_response = response.choices[0].message.content
                
                # Show success message with details
                st.success("✅ OpenAI chat connection successful!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Provider", "OpenAI")
                with col2:
                    st.metric("Model", self.settings.chat.model)
                with col3:
                    st.metric("Tokens Used", response.usage.total_tokens)
                
                # Show sample response
                st.info(f"📝 Test response: {chat_response}")
                
                return True
                
        except Exception as e:
            error_msg = str(e)
            if "api_key" in error_msg.lower():
                st.error("❌ Invalid OpenAI API key")
            elif "model" in error_msg.lower():
                st.error(f"❌ Model '{self.settings.chat.model}' not available")
            elif "quota" in error_msg.lower():
                st.error("❌ OpenAI API quota exceeded")
            else:
                st.error(f"❌ OpenAI chat connection failed: {error_msg}")
            return False
    
    def _test_bedrock_chat(self, test_prompt: str):
        """Test AWS Bedrock chat connection."""
        if not self.settings.aws_access_key_id or not self.settings.aws_secret_access_key:
            st.error("❌ AWS credentials are not configured")
            return False
        
        try:
            # Try to import boto3
            try:
                import boto3
                import json
            except ImportError:
                st.error("❌ boto3 not installed. Install with: pip install boto3")
                return False
            
            # Initialize Bedrock client
            client_kwargs = {
                'service_name': 'bedrock-runtime',
                'region_name': self.settings.aws_region,
                'aws_access_key_id': self.settings.aws_access_key_id,
                'aws_secret_access_key': self.settings.aws_secret_access_key
            }
            
            bedrock = boto3.client(**client_kwargs)
            
            with st.spinner("Testing AWS Bedrock chat connection..."):
                # Format request based on model
                if self.settings.chat.model.startswith("anthropic.claude"):
                    body = json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": min(self.settings.chat.max_tokens, 150),
                        "temperature": self.settings.chat.temperature,
                        "messages": [
                            {"role": "user", "content": test_prompt}
                        ]
                    })
                elif self.settings.chat.model.startswith("amazon.titan"):
                    body = json.dumps({
                        "inputText": test_prompt,
                        "textGenerationConfig": {
                            "maxTokenCount": min(self.settings.chat.max_tokens, 150),
                            "temperature": self.settings.chat.temperature
                        }
                    })
                elif self.settings.chat.model.startswith("meta.llama"):
                    body = json.dumps({
                        "prompt": test_prompt,
                        "max_gen_len": min(self.settings.chat.max_tokens, 150),
                        "temperature": self.settings.chat.temperature
                    })
                else:
                    st.error(f"❌ Unsupported Bedrock chat model: {self.settings.chat.model}")
                    return False
                
                response = bedrock.invoke_model(
                    modelId=self.settings.chat.model,
                    body=body,
                    contentType='application/json',
                    accept='application/json'
                )
                
                result = json.loads(response['body'].read())
                
                # Extract response based on model
                if self.settings.chat.model.startswith("anthropic.claude"):
                    chat_response = result['content'][0]['text']
                elif self.settings.chat.model.startswith("amazon.titan"):
                    chat_response = result['results'][0]['outputText']
                elif self.settings.chat.model.startswith("meta.llama"):
                    chat_response = result['generation']
                else:
                    chat_response = str(result)
                
                # Show success message with details
                st.success("✅ AWS Bedrock chat connection successful!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Provider", "AWS Bedrock")
                with col2:
                    st.metric("Model", self.settings.chat.model)
                
                # Show sample response
                st.info(f"📝 Test response: {chat_response}")
                
                return True
                
        except Exception as e:
            error_msg = str(e)
            if "credentials" in error_msg.lower() or "security token" in error_msg.lower():
                st.error("❌ Invalid AWS credentials")
            elif "access denied" in error_msg.lower():
                st.error("❌ AWS access denied - check IAM permissions for Bedrock")
            elif "not authorized" in error_msg.lower():
                st.error(f"❌ Not authorized to use model '{self.settings.chat.model}'")
            elif "region" in error_msg.lower():
                st.error(f"❌ Bedrock not available in region '{self.settings.aws_region}'")
            else:
                st.error(f"❌ Bedrock chat connection failed: {error_msg}")
            return False
    
    def _init_database(self, conn):
        """Initialize database tables for research papers and embeddings."""
        try:
            with conn.cursor() as cursor:
                # Enable pgvector extension
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                
                # Create research papers table  
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS research_papers (
                        id SERIAL PRIMARY KEY,
                        paper_hash VARCHAR(64) UNIQUE NOT NULL,
                        title TEXT NOT NULL,
                        authors TEXT[],
                        abstract TEXT,
                        content TEXT,
                        arxiv_id VARCHAR(50),
                        url TEXT,
                        published_date TIMESTAMP,
                        categories TEXT[],
                        keywords TEXT[],
                        y_score FLOAT DEFAULT 0.0,
                        decomposition_r FLOAT DEFAULT 0.0,
                        decomposition_s FLOAT DEFAULT 0.0,
                        decomposition_n FLOAT DEFAULT 0.0,
                        context_collapse_risk VARCHAR(20) DEFAULT 'unknown',
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW()
                    );
                """)
                
                # Create embeddings table with pgvector
                cursor.execute(f"""
                    CREATE TABLE IF NOT EXISTS paper_embeddings (
                        id SERIAL PRIMARY KEY,
                        paper_id VARCHAR(64) NOT NULL,
                        chunk_id VARCHAR(100) UNIQUE NOT NULL,
                        chunk_text TEXT NOT NULL,
                        embedding vector({self.settings.embeddings.dimensions}) NOT NULL,
                        chunk_type VARCHAR(50) DEFAULT 'content',
                        start_pos INTEGER DEFAULT 0,
                        end_pos INTEGER DEFAULT 0,
                        metadata JSONB DEFAULT '{{}}',
                        created_at TIMESTAMP DEFAULT NOW(),
                        FOREIGN KEY (paper_id) REFERENCES research_papers(paper_hash) ON DELETE CASCADE
                    );
                """)
                
                # Create indexes for fast search
                cursor.execute("""
                    CREATE INDEX IF NOT EXISTS idx_paper_embeddings_embedding 
                    ON paper_embeddings USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = 100);
                """)
                
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_research_papers_arxiv_id ON research_papers(arxiv_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_research_papers_y_score ON research_papers(y_score DESC);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_embeddings_chunk_type ON paper_embeddings(chunk_type);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_paper_embeddings_paper_id ON paper_embeddings(paper_id);")
                
                conn.commit()
                
        except Exception as e:
            st.error(f"❌ Failed to initialize database: {str(e)}")
            raise
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        if not self.connection:
            return {}
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                # Count papers
                cursor.execute("SELECT COUNT(*) as count FROM research_papers;")
                papers_count = cursor.fetchone()['count']
                
                # Count embeddings
                cursor.execute("SELECT COUNT(*) as count FROM paper_embeddings;")
                embeddings_count = cursor.fetchone()['count']
                
                # Count unique authors
                cursor.execute("""
                    SELECT COUNT(DISTINCT unnest(authors)) as count 
                    FROM research_papers 
                    WHERE authors IS NOT NULL;
                """)
                authors_count = cursor.fetchone()['count']
                
                # Database size
                cursor.execute("""
                    SELECT pg_size_pretty(pg_database_size(current_database())) as size,
                           pg_database_size(current_database()) / 1024.0 / 1024.0 as size_mb;
                """)
                size_info = cursor.fetchone()
                
                return {
                    'papers': papers_count,
                    'embeddings': embeddings_count,
                    'unique_authors': authors_count,
                    'db_size': size_info['size'],
                    'db_size_mb': float(size_info['size_mb'])
                }
                
        except Exception as e:
            st.error(f"❌ Failed to get database stats: {str(e)}")
            return {}
    
    def store_paper_analysis(self, title: str, authors: List[str], abstract: str, 
                           y_score: float, decomposition: Dict[str, float],
                           context_risk: str, arxiv_id: str = None) -> bool:
        """Store paper analysis results in the database."""
        if not self.connection:
            st.warning("⚠️ Database not connected")
            return False
        
        try:
            paper_hash = hashlib.sha256(f"{title}{abstract}".encode()).hexdigest()
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO research_papers (
                        paper_hash, title, authors, abstract, arxiv_id,
                        y_score, decomposition_r, decomposition_s, decomposition_n,
                        context_collapse_risk
                    ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (paper_hash) DO UPDATE SET
                        title = EXCLUDED.title,
                        authors = EXCLUDED.authors,
                        abstract = EXCLUDED.abstract,
                        y_score = EXCLUDED.y_score,
                        decomposition_r = EXCLUDED.decomposition_r,
                        decomposition_s = EXCLUDED.decomposition_s,
                        decomposition_n = EXCLUDED.decomposition_n,
                        context_collapse_risk = EXCLUDED.context_collapse_risk,
                        updated_at = NOW()
                """, (
                    paper_hash, title, authors, abstract, arxiv_id,
                    y_score, decomposition['R'], decomposition['S'], decomposition['N'],
                    context_risk
                ))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to store paper analysis: {str(e)}")
            return False
    
    def store_embedding(self, paper_hash: str, chunk_text: str, embedding: List[float],
                       chunk_type: str = 'content') -> bool:
        """Store embedding in the database."""
        if not self.connection:
            st.warning("⚠️ Database not connected")
            return False
        
        try:
            chunk_id = hashlib.sha256(f"{paper_hash}{chunk_text}".encode()).hexdigest()[:32]
            
            with self.connection.cursor() as cursor:
                cursor.execute("""
                    INSERT INTO paper_embeddings (
                        paper_id, chunk_id, chunk_text, embedding, chunk_type
                    ) VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (chunk_id) DO UPDATE SET
                        chunk_text = EXCLUDED.chunk_text,
                        embedding = EXCLUDED.embedding,
                        chunk_type = EXCLUDED.chunk_type
                """, (paper_hash, chunk_id, chunk_text, embedding, chunk_type))
                
                self.connection.commit()
                return True
                
        except Exception as e:
            st.error(f"❌ Failed to store embedding: {str(e)}")
            return False
    
    def search_similar_papers(self, query_embedding: List[float], limit: int = 10) -> List[Dict[str, Any]]:
        """Search for similar papers using vector similarity."""
        if not self.connection or not NUMPY_AVAILABLE:
            return []
        
        try:
            with self.connection.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        rp.title, rp.authors, rp.abstract, rp.arxiv_id,
                        rp.y_score, rp.decomposition_r, rp.decomposition_s, rp.decomposition_n,
                        rp.context_collapse_risk,
                        pe.chunk_text, pe.chunk_type,
                        1 - (pe.embedding <=> %s::vector) as similarity
                    FROM paper_embeddings pe
                    JOIN research_papers rp ON pe.paper_id = rp.paper_hash
                    WHERE 1 - (pe.embedding <=> %s::vector) > %s
                    ORDER BY pe.embedding <=> %s::vector
                    LIMIT %s
                """, (query_embedding, query_embedding, self.settings.embeddings.similarity_threshold, query_embedding, limit))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            st.error(f"❌ Failed to search similar papers: {str(e)}")
            return []
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration as dictionary."""
        return {
            'postgres': {
                'host': self.settings.postgres.host,
                'port': self.settings.postgres.port,
                'database': self.settings.postgres.database,
                'username': self.settings.postgres.username,
                'ssl_mode': self.settings.postgres.ssl_mode
            },
            'embeddings': {
                'model': self.settings.embeddings.model,
                'dimensions': self.settings.embeddings.dimensions,
                'similarity_threshold': self.settings.embeddings.similarity_threshold
            }
        }

# Global backend config instance
_backend_config = None

def get_backend_config() -> BackendConfigManager:
    """Get global backend configuration instance."""
    global _backend_config
    if _backend_config is None:
        _backend_config = BackendConfigManager()
    return _backend_config

def render_backend_sidebar():
    """Render backend configuration in Streamlit sidebar."""
    config = get_backend_config()
    
    with st.sidebar.expander("⚙️ Backend Configuration"):
        # Quick configuration inputs
        api_key = st.text_input(
            "OpenAI API Key",
            value=config.settings.openai_api_key,
            type="password",
            help="Your OpenAI API key for embeddings"
        )
        if api_key != config.settings.openai_api_key:
            config.settings.openai_api_key = api_key
        
        db_host = st.text_input(
            "Database Host",
            value=config.settings.postgres.host,
            help="PostgreSQL server host"
        )
        if db_host != config.settings.postgres.host:
            config.settings.postgres.host = db_host
        
        db_name = st.text_input(
            "Database Name",
            value=config.settings.postgres.database,
            help="PostgreSQL database name"
        )
        if db_name != config.settings.postgres.database:
            config.settings.postgres.database = db_name
        
        # Show KMS Key ID if using Bedrock
        if config.settings.embedding_provider == "bedrock":
            kms_key = st.text_input(
                "AWS KMS Key ID",
                value=config.settings.aws_kms_key_id,
                help="Optional KMS Key for encryption"
            )
            if kms_key != config.settings.aws_kms_key_id:
                config.settings.aws_kms_key_id = kms_key
        
        # Save button
        if st.button("💾 Save Settings", key="sidebar_save"):
            if config.save_to_yaml():
                st.success("✅ Saved to settings.yaml")
            else:
                st.error("❌ Failed to save")
        
        # Advanced configuration button
        if st.button("🔧 Advanced Config"):
            st.session_state['show_backend_config'] = True
        
        st.markdown("---")
        
        # Show quick status
        if config.connection:
            st.success("🟢 DB Connected")
        else:
            st.warning("🔴 DB Disconnected")
        
        if config.settings.openai_api_key:
            st.success("🟢 API Key Set")
        else:
            st.warning("🔴 API Key Missing")

__all__ = [
    'BackendConfigManager',
    'PostgresConfig', 
    'EmbeddingConfig',
    'BackendSettings',
    'get_backend_config',
    'render_backend_sidebar'
]