"""
Secure Credential Manager for VectorQA Sage
Loads credentials from environment variables or .env files
"""

import os
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class CredentialManager:
    """Manages application credentials and configuration securely."""
    
    def __init__(self, env_file: Optional[str] = None, config_file: Optional[str] = None):
        """
        Initialize credential manager.
        
        Args:
            env_file: Path to .env file (default: auto-detected from APP_ENV)
            config_file: Path to config YAML file (default: auto-detected from APP_ENV)
        """
        self.environment = os.getenv('APP_ENV', 'local').lower()
        self.env_file = env_file or self._get_environment_file()
        self.config_file = config_file or self._get_config_file()
        self.config_data = {}
        self._load_credentials()
        self._load_config()
    
    def _get_environment_file(self) -> str:
        """Determine environment file based on APP_ENV variable."""
        # Validate environment
        valid_envs = ['local', 'development', 'staging', 'production']
        if self.environment not in valid_envs:
            logger.warning(f"Invalid APP_ENV '{self.environment}', defaulting to 'local'")
            self.environment = 'local'
        
        env_file = f"environ_settings/.env.{self.environment}"
        logger.info(f"Using credentials file: {env_file}")
        return env_file
    
    def _get_config_file(self) -> str:
        """Determine config file based on APP_ENV variable."""
        config_file = f"environ_settings/config.{self.environment}.yaml"
        logger.info(f"Using config file: {config_file}")
        return config_file
    
    def _load_credentials(self):
        """Load credentials from environment file."""
        env_path = Path(self.env_file)
        
        if env_path.exists():
            logger.info(f"Loading credentials from {self.env_file}")
            load_dotenv(self.env_file)
        else:
            logger.warning(f"Credentials file not found: {self.env_file}")
            logger.info("Using system environment variables only")
    
    def _load_config(self):
        """Load configuration from YAML file."""
        config_path = Path(self.config_file)
        
        if config_path.exists():
            logger.info(f"Loading configuration from {self.config_file}")
            with open(config_path, 'r') as f:
                self.config_data = yaml.safe_load(f) or {}
        else:
            logger.info(f"Config file not found: {self.config_file} (optional)")
            self.config_data = {}
    
    def get_aws_credentials(self) -> Dict[str, str]:
        """Get AWS credentials."""
        return {
            "access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "region": os.getenv("AWS_REGION", "us-east-1")
        }
    
    def get_bedrock_config(self) -> Dict[str, str]:
        """Get Amazon Bedrock configuration."""
        return {
            "model_id": os.getenv("BEDROCK_MODEL_ID", "amazon.titan-embed-text-v2:0"),
            "region": os.getenv("AWS_REGION", "us-east-1")
        }
    
    def get_openai_config(self) -> Dict[str, str]:
        """Get OpenAI configuration."""
        return {
            "api_key": os.getenv("OPENAI_API_KEY")
        }
    
    def get_anthropic_config(self) -> Dict[str, str]:
        """Get Anthropic configuration."""
        return {
            "api_key": os.getenv("ANTHROPIC_API_KEY")
        }
    
    def get_cohere_config(self) -> Dict[str, str]:
        """Get Cohere configuration."""
        return {
            "api_key": os.getenv("COHERE_API_KEY")
        }
    
    def get_huggingface_config(self) -> Dict[str, str]:
        """Get Hugging Face configuration."""
        return {
            "api_key": os.getenv("HUGGINGFACE_API_KEY")
        }
    
    def get_google_config(self) -> Dict[str, str]:
        """Get Google Gemini configuration."""
        return {
            "api_key": os.getenv("GOOGLE_API_KEY")
        }
    
    def get_database_config(self) -> Dict[str, str]:
        """Get database configuration."""
        return {
            "url": os.getenv("DATABASE_URL")
        }
    
    def get_app_config(self) -> Dict[str, Any]:
        """Get application configuration (combines env vars and YAML config)."""
        # Start with YAML config
        config = self.config_data.copy()
        
        # Override with environment variables (higher priority)
        env_config = {
            "debug": os.getenv("DEBUG", str(config.get("debug", False))).lower() == "true",
            "log_level": os.getenv("LOG_LEVEL", config.get("log_level", "INFO")),
            "environment": self.environment,
            "base_url": os.getenv("BASE_URL", config.get("base_url")),
            "feature_flags": os.getenv("FEATURE_FLAGS", config.get("feature_flags", "")),
            "database_pool_size": int(os.getenv("DATABASE_POOL_SIZE", str(config.get("database_pool_size", 10))))
        }
        
        # Merge configurations
        config.update(env_config)
        return config
    
    def validate_credentials(self) -> Dict[str, bool]:
        """Validate that required credentials are present."""
        validation = {}
        
        # Check AWS credentials
        aws_creds = self.get_aws_credentials()
        validation["aws"] = all([
            aws_creds["access_key_id"],
            aws_creds["secret_access_key"]
        ])
        
        # Check other providers (optional)
        validation["openai"] = bool(self.get_openai_config()["api_key"])
        validation["anthropic"] = bool(self.get_anthropic_config()["api_key"])
        validation["cohere"] = bool(self.get_cohere_config()["api_key"])
        validation["huggingface"] = bool(self.get_huggingface_config()["api_key"])
        validation["google"] = bool(self.get_google_config()["api_key"])
        
        return validation
    
    def print_status(self):
        """Print credential status (without revealing values)."""
        validation = self.validate_credentials()
        
        print("🔐 Credential Status:")
        print("=" * 30)
        
        for provider, is_valid in validation.items():
            status = "✅" if is_valid else "❌"
            print(f"{status} {provider.upper()}")
        
        print("\n💡 To set up credentials:")
        print(f"1. Create credentials file: {self.env_file}")
        print("2. Add your API keys in format: API_KEY=your_key_here")
        print(f"3. Create config file: {self.config_file} (optional)")
        print("4. Add settings like: base_url, feature_flags, etc.")
        print("5. Set APP_ENV environment variable (local/development/staging/production)")
        print("6. Restart the application")

# Global credential manager instance
credential_manager = CredentialManager()
