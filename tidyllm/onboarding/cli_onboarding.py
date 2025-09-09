#!/usr/bin/env python3
"""
TidyLLM Corporate Onboarding CLI
================================

Interactive CLI wizard for setting up TidyLLM in corporate environments.
Collects configuration, validates setup, and generates production-ready configs.

Usage:
    tidyllm-onboard                    # Interactive wizard
    tidyllm-onboard --config-only      # Generate config from template
    tidyllm-onboard --validate         # Validate existing setup
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import subprocess
import shutil
from datetime import datetime

try:
    from rich.console import Console
    from rich.prompt import Prompt, Confirm
    from rich.table import Table
    from rich.panel import Panel
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    console = None

# Fallback print functions if rich not available
def rprint(text: str):
    if RICH_AVAILABLE:
        from rich import print as _rprint
        _rprint(text)
    else:
        print(text)

def print_panel(content: str, title: str = ""):
    if RICH_AVAILABLE:
        from rich.panel import Panel
        console.print(Panel(content, title=title))
    else:
        print(f"\n{'='*50}")
        if title:
            print(f"{title}")
            print('='*50)
        print(content)
        print('='*50)

def print_table(data: list, headers: list):
    if RICH_AVAILABLE:
        from rich.table import Table
        table = Table()
        for header in headers:
            table.add_column(header)
        for row in data:
            table.add_row(*row)
        console.print(table)
    else:
        print(f"\n{' | '.join(headers)}")
        print('-' * (len(' | '.join(headers)) + 10))
        for row in data:
            print(f"{' | '.join(row)}")

if RICH_AVAILABLE:
    console = Console()

class CorporateOnboarding:
    """Interactive onboarding wizard for TidyLLM corporate deployments."""
    
    def __init__(self):
        self.config = {}
        self.base_path = Path(__file__).parent
        self.template_path = self.base_path / "template.settings.yaml"
        
    def welcome_banner(self):
        """Display welcome banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║                  TidyLLM Corporate Onboarding                   ║
║                                                                  ║
║  This wizard will help you configure TidyLLM for your          ║
║  corporate environment with proper security and compliance.     ║
║                                                                  ║
║  🔒 All credentials will be stored as environment variables     ║
║  🏢 Corporate security standards will be enforced               ║
║  📋 Configuration will be validated before completion           ║
╚══════════════════════════════════════════════════════════════════╝
        """
        print_panel(banner, "Welcome")
        
    def collect_organization_info(self) -> Dict[str, Any]:
        """Collect basic organization information."""
        rprint("\n[bold blue]Organization Information[/bold blue]")
        
        if RICH_AVAILABLE:
            org_name = Prompt.ask("Organization name", default="YourCompany")
            environment = Prompt.ask(
                "Deployment environment", 
                choices=["production", "staging", "development", "corporate"],
                default="production"
            )
            aws_region = Prompt.ask("AWS region", default="us-east-1")
        else:
            org_name = input("Organization name [YourCompany]: ") or "YourCompany"
            environment = input("Deployment environment [production]: ") or "production"
            aws_region = input("AWS region [us-east-1]: ") or "us-east-1"
            
        return {
            "organization": org_name,
            "environment": environment,
            "aws_region": aws_region
        }
    
    def collect_security_preferences(self) -> Dict[str, Any]:
        """Collect security and compliance preferences."""
        rprint("\n[bold red]Security & Compliance Configuration[/bold red]")
        
        if RICH_AVAILABLE:
            enable_sso = Confirm.ask("Enable Single Sign-On (SSO)", default=True)
            enable_audit = Confirm.ask("Enable audit logging (recommended)", default=True)
            enable_encryption = Confirm.ask("Enable data encryption at rest", default=True)
            
            if enable_sso:
                sso_provider = Prompt.ask(
                    "SSO Provider", 
                    choices=["okta", "azure-ad", "ping", "other"],
                    default="okta"
                )
            else:
                sso_provider = None
                
        else:
            enable_sso = input("Enable Single Sign-On (SSO) [y/N]: ").lower().startswith('y')
            enable_audit = input("Enable audit logging [Y/n]: ").lower() != 'n'
            enable_encryption = input("Enable data encryption at rest [Y/n]: ").lower() != 'n'
            sso_provider = None
            
            if enable_sso:
                print("SSO Provider options: okta, azure-ad, ping, other")
                sso_provider = input("SSO Provider [okta]: ") or "okta"
        
        return {
            "sso_enabled": enable_sso,
            "sso_provider": sso_provider,
            "audit_logging": enable_audit,
            "encrypt_cache": enable_encryption
        }
    
    def collect_model_preferences(self) -> Dict[str, Any]:
        """Collect AI model preferences."""
        rprint("\n[bold green]AI Model Configuration[/bold green]")
        
        models = {
            "claude-3-sonnet": "Claude 3 Sonnet (Balanced performance)",
            "claude-3-haiku": "Claude 3 Haiku (Fast responses)",
            "claude-3-opus": "Claude 3 Opus (Highest quality)"
        }
        
        print_table(
            [[k, v] for k, v in models.items()],
            ["Model ID", "Description"]
        )
        
        if RICH_AVAILABLE:
            default_model = Prompt.ask(
                "Default model",
                choices=list(models.keys()),
                default="claude-3-sonnet"
            )
        else:
            default_model = input("Default model [claude-3-sonnet]: ") or "claude-3-sonnet"
            
        return {
            "default_model": default_model
        }
    
    def check_environment_variables(self) -> Dict[str, bool]:
        """Check if required environment variables are set."""
        required_vars = [
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY", 
            "TIDYLLM_DB_PASSWORD"
        ]
        
        results = {}
        for var in required_vars:
            results[var] = bool(os.getenv(var))
            
        return results
    
    def validate_aws_connectivity(self, region: str) -> Dict[str, Any]:
        """Test AWS connectivity and permissions."""
        rprint("\n[bold yellow]Validating AWS Setup...[/bold yellow]")
        
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            # Test basic AWS credentials
            try:
                session = boto3.Session(region_name=region)
                sts = session.client('sts')
                identity = sts.get_caller_identity()
                
                rprint(f"✅ AWS credentials valid - Account: {identity.get('Account')}")
                
            except (NoCredentialsError, ClientError) as e:
                rprint(f"❌ AWS credentials invalid: {e}")
                return {"valid": False, "error": str(e)}
            
            # Test Bedrock access
            try:
                bedrock = session.client('bedrock-runtime', region_name=region)
                # Try to list models (this requires permissions)
                bedrock_models = session.client('bedrock')
                models = bedrock_models.list_foundation_models()
                rprint(f"✅ Bedrock access confirmed - {len(models['modelSummaries'])} models available")
                
            except ClientError as e:
                rprint(f"⚠️ Bedrock access limited: {e}")
                
            # Test S3 access
            try:
                s3 = session.client('s3', region_name=region)
                buckets = s3.list_buckets()
                rprint(f"✅ S3 access confirmed - {len(buckets['Buckets'])} buckets accessible")
                
            except ClientError as e:
                rprint(f"⚠️ S3 access limited: {e}")
                
            return {"valid": True, "region": region}
            
        except ImportError:
            rprint("❌ boto3 not available - install with: pip install boto3")
            return {"valid": False, "error": "boto3 not installed"}
    
    def generate_configuration(self, user_prefs: Dict[str, Any]) -> Dict[str, Any]:
        """Generate complete configuration from user preferences."""
        from .config_generator import create_template_config
        
        # Combine all collected preferences
        config_input = {
            **self.config,
            **user_prefs,
            "validation_results": getattr(self, 'validation_results', {})
        }
        
        # Generate complete configuration
        return create_template_config(config_input)
    
    def save_configuration(self, config: Dict[str, Any], output_path: str):
        """Save configuration to file."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        rprint(f"✅ Configuration saved to: {output_file.absolute()}")
        
        # Also save deployment artifacts
        self.generate_deployment_artifacts(config, output_file.parent)
    
    def generate_deployment_artifacts(self, config: Dict[str, Any], output_dir: Path):
        """Generate deployment artifacts."""
        from .config_generator import generate_dockerfile, generate_docker_compose, generate_kubernetes_manifests
        
        rprint("\n[bold blue]Generating deployment artifacts...[/bold blue]")
        
        # Generate Dockerfile
        dockerfile = generate_dockerfile(config)
        with open(output_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        rprint("📦 Generated: Dockerfile")
        
        # Generate Docker Compose
        compose = generate_docker_compose(config)
        with open(output_dir / "docker-compose.yml", 'w') as f:
            f.write(compose)
        rprint("📦 Generated: docker-compose.yml")
        
        # Generate Kubernetes manifests
        k8s_manifests = generate_kubernetes_manifests(config)
        k8s_dir = output_dir / "kubernetes"
        k8s_dir.mkdir(exist_ok=True)
        
        for filename, content in k8s_manifests.items():
            with open(k8s_dir / filename, 'w') as f:
                f.write(content)
            rprint(f"📦 Generated: kubernetes/{filename}")
        
        # Generate environment file template
        env_template = self.generate_env_template(config)
        with open(output_dir / ".env.template", 'w') as f:
            f.write(env_template)
        rprint("📦 Generated: .env.template")
        
        # Generate setup instructions
        instructions = self.generate_setup_instructions(config)
        with open(output_dir / "DEPLOYMENT_INSTRUCTIONS.md", 'w') as f:
            f.write(instructions)
        rprint("📋 Generated: DEPLOYMENT_INSTRUCTIONS.md")
    
    def generate_env_template(self, config: Dict[str, Any]) -> str:
        """Generate .env template file."""
        org_name = config.get('organization', 'company').lower()
        aws_region = config.get('aws', {}).get('region', 'us-east-1')
        
        return f"""# TidyLLM Corporate Environment Variables
# ========================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Organization: {config.get('organization', 'YourOrganization')}

# REQUIRED: Database Configuration
TIDYLLM_DB_PASSWORD=your_secure_database_password_here

# REQUIRED: AWS Credentials (if not using IAM roles)
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_SESSION_TOKEN=your_aws_session_token_here_if_using_sts
AWS_DEFAULT_REGION={aws_region}

# Optional: Custom S3 bucket (defaults will be used if not set)
TIDYLLM_S3_BUCKET={org_name}-tidyllm-storage
TIDYLLM_AUDIT_BUCKET={org_name}-tidyllm-audit

# Optional: Database connection override
TIDYLLM_DB_HOST=your_postgres_host_here
TIDYLLM_DB_NAME=tidyllm_production
TIDYLLM_DB_USER=tidyllm_service

# Optional: KMS Key for encryption
TIDYLLM_KMS_KEY_ARN=arn:aws:kms:{aws_region}:account:key/your-key-id

# Optional: Logging configuration
TIDYLLM_LOG_LEVEL=INFO
TIDYLLM_LOG_FORMAT=json

# Optional: Security settings
TIDYLLM_ENCRYPT_CACHE=true
TIDYLLM_AUDIT_ENABLED=true

# Corporate proxy (if required)
# HTTP_PROXY=http://proxy.company.com:8080
# HTTPS_PROXY=http://proxy.company.com:8080
# NO_PROXY=localhost,127.0.0.1,.company.com
"""
    
    def generate_setup_instructions(self, config: Dict[str, Any]) -> str:
        """Generate deployment instructions."""
        org_name = config.get('organization', 'YourOrganization')
        aws_region = config.get('aws', {}).get('region', 'us-east-1')
        
        return f"""# TidyLLM Corporate Deployment Instructions

**Organization:** {org_name}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Environment:** {config.get('environment', 'production')}

## Pre-Deployment Checklist

### 1. Environment Variables
Copy `.env.template` to `.env` and fill in your values:
```bash
cp .env.template .env
# Edit .env with your actual credentials
```

### 2. Network Requirements
Ensure the following outbound connections are allowed:
- `bedrock-runtime.{aws_region}.amazonaws.com:443` (AWS Bedrock)
- `s3.{aws_region}.amazonaws.com:443` (S3 Storage)  
- Your PostgreSQL database host on port 5432

### 3. AWS Permissions
The AWS credentials need the following permissions:
- `bedrock:InvokeModel`
- `bedrock:ListFoundationModels`
- `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`
- `kms:Encrypt`, `kms:Decrypt` (if using KMS)

### 4. Database Setup
Create PostgreSQL database and user:
```sql
CREATE DATABASE tidyllm_production;
CREATE USER tidyllm_service WITH PASSWORD 'your_secure_password';
GRANT ALL PRIVILEGES ON DATABASE tidyllm_production TO tidyllm_service;
```

## Deployment Options

### Option 1: Docker Compose (Recommended for development)
```bash
# Start the service
docker-compose up -d

# Check logs
docker-compose logs -f tidyllm

# Stop the service  
docker-compose down
```

### Option 2: Kubernetes (Production)
```bash
# Create namespace
kubectl create namespace {org_name.lower()}

# Create secrets
kubectl create secret generic tidyllm-secrets \\
  --from-env-file=.env \\
  --namespace={org_name.lower()}

# Deploy application
kubectl apply -f kubernetes/ --namespace={org_name.lower()}

# Check status
kubectl get pods --namespace={org_name.lower()}
```

### Option 3: Docker (Standalone)
```bash
# Build image
docker build -t {org_name.lower()}/tidyllm:latest .

# Run container
docker run -d \\
  --name tidyllm-corporate \\
  --env-file .env \\
  -p 8000:8000 \\
  {org_name.lower()}/tidyllm:latest
```

## Post-Deployment Verification

### 1. Health Check
```bash
curl http://localhost:8000/health
# Should return: {{"status": "healthy"}}
```

### 2. Authentication Test
```bash
curl -H "Authorization: Bearer your-token" \\
     http://localhost:8000/api/v1/status
```

### 3. AI Model Test
```bash
curl -X POST http://localhost:8000/api/v1/chat \\
     -H "Content-Type: application/json" \\
     -H "Authorization: Bearer your-token" \\
     -d '{{"message": "Hello, TidyLLM!"}}'
```

## Monitoring Setup

### 1. Log Aggregation
Configure your log aggregation system to collect from:
- Container logs: `/var/log/tidyllm/`
- Syslog facility: `local0`
- CloudWatch log group: `tidyllm-corporate`

### 2. Health Monitoring
Set up monitoring for:
- Health endpoint: `http://localhost:8000/health`
- Metrics endpoint: `http://localhost:8000/metrics`
- Response time thresholds: < 5 seconds

### 3. Alerting
Configure alerts for:
- Service downtime
- High error rates (>5%)
- Slow response times (>5s)
- Authentication failures

## Security Considerations

### 1. Secrets Management
- Never commit `.env` files to version control
- Use Kubernetes secrets or AWS Secrets Manager in production
- Rotate credentials regularly

### 2. Network Security
- Use HTTPS in production
- Implement proper firewall rules
- Consider VPC endpoints for AWS services

### 3. Audit Logging
- Monitor all API requests
- Store audit logs for compliance requirements
- Set up log retention according to corporate policy

## Troubleshooting

### Common Issues

**1. AWS Permission Errors**
```
Error: User is not authorized to perform: bedrock:InvokeModel
```
Solution: Add the required IAM permissions to your AWS user/role.

**2. Database Connection Issues**
```
Error: could not connect to server: Connection refused
```
Solution: Check database host, port, and network connectivity.

**3. Model Not Found**
```
Error: Could not find model: anthropic.claude-3-sonnet
```
Solution: Verify the model is available in your AWS region and you have access.

### Getting Help
- Check logs: `docker-compose logs tidyllm`
- Test connectivity: `tidyllm-onboard --validate`
- Contact support: `support@tidyllm.ai`

---

**Generated by TidyLLM Corporate Onboarding Wizard**
"""
    
    def run_interactive_wizard(self):
        """Run the complete interactive onboarding wizard."""
        try:
            self.welcome_banner()
            
            # Collect information step by step
            rprint("\n[bold cyan]Step 1: Organization Setup[/bold cyan]")
            org_info = self.collect_organization_info()
            self.config.update(org_info)
            
            rprint("\n[bold cyan]Step 2: Security Configuration[/bold cyan]")
            security_config = self.collect_security_preferences()
            self.config["security"] = security_config
            
            rprint("\n[bold cyan]Step 3: AI Model Setup[/bold cyan]")
            model_config = self.collect_model_preferences()
            self.config.update(model_config)
            
            # Environment validation
            rprint("\n[bold cyan]Step 4: Environment Validation[/bold cyan]")
            env_check = self.check_environment_variables()
            
            missing_vars = [var for var, present in env_check.items() if not present]
            if missing_vars:
                rprint(f"[bold red]⚠️ Missing environment variables:[/bold red]")
                for var in missing_vars:
                    rprint(f"  - {var}")
                rprint("\n[dim]Set these before running in production[/dim]")
            else:
                rprint("✅ All required environment variables found")
            
            # AWS validation
            aws_validation = self.validate_aws_connectivity(org_info["aws_region"])
            self.validation_results = {
                "environment_variables": env_check,
                "aws_connectivity": aws_validation
            }
            
            # Generate configuration
            rprint("\n[bold cyan]Step 5: Generating Configuration[/bold cyan]")
            final_config = self.generate_configuration(self.config)
            
            # Save configuration
            output_dir = Path.cwd() / "tidyllm-corporate-config"
            output_dir.mkdir(exist_ok=True)
            
            self.save_configuration(final_config, output_dir / "settings.yaml")
            
            # Final summary
            print_panel(f"""
🎉 Corporate onboarding complete!

📁 Configuration saved to: {output_dir.absolute()}
📋 Review DEPLOYMENT_INSTRUCTIONS.md for next steps
🔧 Edit .env.template with your actual credentials

To deploy:
  cd {output_dir}
  cp .env.template .env  # Edit with your values
  docker-compose up -d

To validate setup:
  tidyllm-onboard --validate
            """, "Setup Complete!")
            
        except KeyboardInterrupt:
            rprint("\n\n[bold red]Setup cancelled by user[/bold red]")
            sys.exit(1)
        except Exception as e:
            rprint(f"\n[bold red]Setup failed: {e}[/bold red]")
            sys.exit(1)

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TidyLLM Corporate Onboarding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--config-only", 
        action="store_true",
        help="Generate configuration from template without wizard"
    )
    parser.add_argument(
        "--validate",
        action="store_true", 
        help="Validate existing configuration and environment"
    )
    parser.add_argument(
        "--output",
        default="tidyllm-corporate-config",
        help="Output directory for generated files"
    )
    
    args = parser.parse_args()
    
    onboarding = CorporateOnboarding()
    
    if args.validate:
        rprint("[bold blue]Validating TidyLLM setup...[/bold blue]")
        
        # Check environment
        env_check = onboarding.check_environment_variables()
        aws_check = onboarding.validate_aws_connectivity("us-east-1")  # Default region
        
        if all(env_check.values()) and aws_check["valid"]:
            rprint("[bold green]✅ Setup validation passed![/bold green]")
        else:
            rprint("[bold red]❌ Setup validation failed[/bold red]")
            sys.exit(1)
            
    elif args.config_only:
        rprint("[bold blue]Generating default corporate configuration...[/bold blue]")
        
        # Load template and generate default config
        default_config = {
            "organization": "YourCompany",
            "environment": "production", 
            "aws_region": "us-east-1",
            "default_model": "claude-3-sonnet",
            "security": {
                "sso_enabled": True,
                "audit_logging": True,
                "encrypt_cache": True
            }
        }
        
        from .config_generator import create_template_config
        config = create_template_config(default_config)
        
        output_dir = Path(args.output)
        onboarding.save_configuration(config, output_dir / "settings.yaml")
        
        rprint(f"[bold green]✅ Default configuration generated in {output_dir}[/bold green]")
    else:
        # Run interactive wizard
        onboarding.run_interactive_wizard()

if __name__ == "__main__":
    main()