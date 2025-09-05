#!/usr/bin/env python3
"""
Production Environment Settings

Environment-specific configuration for VectorQA Sage.
This file contains settings for the production environment.
"""

import os
from pathlib import Path

# Environment name
ENVIRONMENT = "production"

# Logging configuration
LOG_LEVEL = "ERROR"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Cache configuration
CACHE_DIR = "/tmp/vectorqa_cache"
CACHE_ENABLED = True

# Server configuration
SERVER_ADDRESS = "0.0.0.0"
SERVER_PORT = 8501
RUN_ON_SAVE = False
HEADLESS = True

# Backend configuration
USE_LOCAL_LLM = False
DEBUG_MODE = False

# Database configuration
DATABASE_URL = "postgresql://user:password@localhost:5432/database"

# Secrets file
SECRETS_FILE = ".env.production"

# Environment variables
ENV_VARS = {
    "VECTORQA_ENV": ENVIRONMENT,
    "LOG_LEVEL": LOG_LEVEL,
    "CACHE_DIR": CACHE_DIR,
    "DATABASE_URL": DATABASE_URL,
    "DEBUG_MODE": str(DEBUG_MODE).lower()
}

# Streamlit configuration
STREAMLIT_CONFIG = {
    "server.port": SERVER_PORT,
    "server.address": SERVER_ADDRESS,
    "browser.gatherUsageStats": False,
    "server.runOnSave": RUN_ON_SAVE,
    "server.headless": HEADLESS
}

# Backend configuration
BACKEND_CONFIG = {
    "use_local_llm": USE_LOCAL_LLM,
    "cache_enabled": CACHE_ENABLED,
    "debug_mode": DEBUG_MODE
}

def get_settings():
    """Get all settings for this environment"""
    return {
        "environment": ENVIRONMENT,
        "env_vars": ENV_VARS,
        "streamlit_config": STREAMLIT_CONFIG,
        "backend_config": BACKEND_CONFIG,
        "database_url": DATABASE_URL,
        "secrets_file": SECRETS_FILE
    }

def load_secrets():
    """Load secrets from environment file"""
    secrets_path = Path(SECRETS_FILE)
    if secrets_path.exists():
        with open(secrets_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    os.environ[key] = value
        print(f"✅ Loaded secrets from {SECRETS_FILE}")
    else:
        print(f"⚠️  Secrets file not found: {SECRETS_FILE}")

if __name__ == "__main__":
    # Test settings
    settings = get_settings()
    print(f"🌍 Environment: {settings['environment']}")
    print(f"📁 Cache dir: {settings['env_vars']['CACHE_DIR']}")
    print(f"🔧 Debug mode: {settings['backend_config']['debug_mode']}")
