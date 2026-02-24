from dataclasses import dataclass
import os

@dataclass
class Settings:
    bucket: str = os.getenv("S3_BUCKET", "sagemaker-us-east-1-188494237500")
    run_prefix: str = os.getenv("RUN_PREFIX", "demo_run")
    embed_model: str = os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    embed_dim: int = int(os.getenv("EMBED_DIM", "1024"))
    pg_dsn: str = os.getenv("PG_DSN", "postgresql://user:pass@localhost:5432/mrm")
    use_bedrock: bool = os.getenv("USE_BEDROCK", "false").lower() == "true"
    max_gist_tokens: int = int(os.getenv("MAX_GIST_TOKENS", "120"))

settings = Settings()

# Function to create team-based S3 prefixes
def get_run_prefix(username: str, team: str = None):
    """
    Create S3 prefix with team hierarchy support.
    
    Args:
        username: User identifier (e.g., 'alex', 'maria')
        team: Team identifier (e.g., 'QA-East', 'QA-West')
    
    Returns:
        S3 prefix like: demo_run/qa-east/alex or demo_run/alex
    """
    base = settings.run_prefix.rstrip("/")
    username_clean = username.strip().lower()
    
    if team:
        team_clean = team.strip().lower().replace("-", "_")
        return f"{base}/{team_clean}/{username_clean}"
    
    return f"{base}/{username_clean}"