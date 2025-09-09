#!/usr/bin/env python3
"""
TidyLLM Enhanced Corporate Onboarding CLI
=========================================

Enhanced interactive CLI wizard that integrates:
- Universal pre-flight testing framework
- SSO and temporary credential management
- PostgreSQL and MLflow validation
- Corporate proxy and network compatibility
- Comprehensive environment validation

Based on the proven test suite architecture with enterprise features.

Usage:
    python enhanced_cli_onboarding.py                    # Interactive wizard
    python enhanced_cli_onboarding.py --preflight-only   # Run pre-flight tests only
    python enhanced_cli_onboarding.py --validate         # Validate existing setup
    python enhanced_cli_onboarding.py --config-only      # Generate config from template
"""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
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

# Import our enhanced components
try:
    from enhanced_session_validator import (
        validate_corporate_aws_stack, 
        print_validation_report
    )
    from universal_preflight import (
        UniversalPreflightTest,
        run_universal_preflight,
        run_quick_preflight
    )
    from config_generator import create_template_config
except ImportError as e:
    print(f"[ERROR] Required modules not found: {e}")
    print("Ensure all onboarding files are in the same directory")
    sys.exit(1)

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

class EnhancedCorporateOnboarding:
    """
    Enhanced corporate onboarding wizard with comprehensive validation.
    
    Integrates universal pre-flight testing with interactive configuration
    for enterprise-grade TidyLLM deployments.
    """
    
    def __init__(self):
        self.config = {}
        self.base_path = Path(__file__).parent
        self.template_path = self.base_path / "template.settings.yaml"
        self.preflight_results = None
        
    def welcome_banner(self):
        """Display enhanced welcome banner."""
        banner = """
╔══════════════════════════════════════════════════════════════════╗
║              TidyLLM Enhanced Corporate Onboarding              ║
║                                                                  ║
║  Enterprise-grade configuration wizard with comprehensive       ║
║  validation for corporate environments including:               ║
║                                                                  ║
║  🔐 SSO & temporary credential management                        ║
║  🌐 Corporate proxy & network compatibility                      ║
║  🗄️ PostgreSQL & MLflow integration testing                     ║
║  ✅ Universal pre-flight validation framework                   ║
║  🏢 Enterprise security & compliance features                   ║
╚══════════════════════════════════════════════════════════════════╝
        """
        print_panel(banner, "Welcome")
    
    def run_preflight_tests(self, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Run comprehensive pre-flight tests before configuration.
        """
        rprint("\n[bold yellow]🚀 Running Pre-Flight Validation...[/bold yellow]")
        
        if quick_mode:
            rprint("[dim]Running quick AWS validation...[/dim]")
            success = run_quick_preflight()
            
            if success:
                rprint("[bold green]✅ Quick pre-flight tests passed![/bold green]")
                return {"overall_success": True, "mode": "quick"}
            else:
                rprint("[bold red]❌ Quick pre-flight tests failed![/bold red]")
                return {"overall_success": False, "mode": "quick"}
        else:
            # Run comprehensive tests
            rprint("[dim]Running comprehensive environment validation...[/dim]")
            
            # Auto-detect configuration for pre-flight
            test_runner = UniversalPreflightTest()
            results = test_runner.run_all_tests()
            
            self.preflight_results = results
            
            if results.get('test_summary', {}).get('overall_success', False):
                rprint("[bold green]✅ Pre-flight validation passed![/bold green]")
            else:
                rprint("[bold red]❌ Pre-flight validation failed![/bold red]")
                
                # Show critical issues
                assessment = results.get('deployment_assessment', {})
                if assessment.get('blockers'):
                    rprint("\n[bold red]Critical Issues:[/bold red]")
                    for blocker in assessment['blockers']:
                        rprint(f"  • {blocker}")
                
                if assessment.get('warnings'):
                    rprint("\n[bold yellow]Warnings:[/bold yellow]")
                    for warning in assessment['warnings']:
                        rprint(f"  • {warning}")
            
            return results
    
    def collect_organization_info(self) -> Dict[str, Any]:
        """Collect organization information with pre-flight context."""
        rprint("\n[bold blue]📋 Organization Configuration[/bold blue]")
        
        # Use pre-flight results to suggest defaults
        suggested_region = 'us-east-1'
        if self.preflight_results:
            cred_info = self.preflight_results.get('test_results', [])
            cred_test = next((test for test in cred_info if test.get('test_name') == 'Credential Discovery'), None)
            if cred_test and cred_test.get('active_credential', {}).get('region'):
                suggested_region = cred_test['active_credential']['region']
        
        if RICH_AVAILABLE:
            org_name = Prompt.ask("Organization name", default="YourCompany")
            environment = Prompt.ask(
                "Deployment environment", 
                choices=["production", "staging", "development", "corporate"],
                default="production"
            )
            aws_region = Prompt.ask("AWS region", default=suggested_region)
            
            # Enhanced deployment options
            deployment_type = Prompt.ask(
                "Primary deployment method",
                choices=["kubernetes", "docker-compose", "docker", "serverless"],
                default="kubernetes"
            )
            
        else:
            org_name = input("Organization name [YourCompany]: ") or "YourCompany"
            environment = input("Deployment environment [production]: ") or "production"
            aws_region = input(f"AWS region [{suggested_region}]: ") or suggested_region
            deployment_type = input("Deployment method [kubernetes]: ") or "kubernetes"
            
        return {
            "organization": org_name,
            "environment": environment,
            "aws_region": aws_region,
            "deployment_type": deployment_type
        }
    
    def collect_enhanced_security_preferences(self) -> Dict[str, Any]:
        """Collect enhanced security preferences with corporate features."""
        rprint("\n[bold red]🔐 Enhanced Security & Compliance[/bold red]")
        
        # Check if SSO was detected in pre-flight
        sso_detected = False
        proxy_detected = False
        
        if self.preflight_results:
            env_test = next((test for test in self.preflight_results.get('test_results', []) 
                           if test.get('test_name') == 'Environment Detection'), None)
            if env_test and env_test.get('environment_info'):
                sso_detected = env_test['environment_info'].get('sso_configured', False)
                proxy_detected = env_test['environment_info'].get('proxy_detected', False)
        
        if RICH_AVAILABLE:
            # SSO Configuration
            enable_sso = Confirm.ask(
                f"Enable Single Sign-On (SSO) {'[detected in environment]' if sso_detected else ''}", 
                default=True
            )
            
            if enable_sso:
                sso_provider = Prompt.ask(
                    "SSO Provider", 
                    choices=["okta", "azure-ad", "ping", "saml", "aws-sso", "other"],
                    default="aws-sso" if sso_detected else "okta"
                )
                
                # SSO-specific settings
                if sso_provider == "aws-sso":
                    sso_start_url = Prompt.ask("AWS SSO Start URL (optional)", default="")
                    sso_region = Prompt.ask("AWS SSO Region", default=self.config.get('aws_region', 'us-east-1'))
                else:
                    sso_start_url = ""
                    sso_region = ""
            else:
                sso_provider = None
                sso_start_url = ""
                sso_region = ""
            
            # Enhanced security options
            enable_audit = Confirm.ask("Enable comprehensive audit logging", default=True)
            enable_encryption = Confirm.ask("Enable data encryption at rest", default=True)
            enable_network_security = Confirm.ask("Enable network security features", default=True)
            
            # Temporary credential management
            temp_cred_refresh = Confirm.ask("Auto-refresh temporary credentials", default=True)
            
            # Corporate proxy settings
            if proxy_detected:
                configure_proxy = Confirm.ask("Configure detected corporate proxy", default=True)
            else:
                configure_proxy = Confirm.ask("Configure corporate proxy settings", default=False)
                
        else:
            enable_sso = input(f"Enable SSO {'[detected]' if sso_detected else ''} [Y/n]: ").lower() != 'n'
            sso_provider = "aws-sso" if sso_detected else "okta"
            enable_audit = input("Enable audit logging [Y/n]: ").lower() != 'n'
            enable_encryption = input("Enable encryption [Y/n]: ").lower() != 'n'
            enable_network_security = True
            temp_cred_refresh = True
            configure_proxy = proxy_detected
            sso_start_url = ""
            sso_region = ""
        
        return {
            "sso_enabled": enable_sso,
            "sso_provider": sso_provider,
            "sso_start_url": sso_start_url,
            "sso_region": sso_region,
            "audit_logging": enable_audit,
            "encrypt_cache": enable_encryption,
            "network_security": enable_network_security,
            "temp_credential_refresh": temp_cred_refresh,
            "proxy_configured": configure_proxy,
            "mask_sensitive_data": True,  # Always enabled for corporate
            "rate_limit_per_minute": 50,  # Corporate default
            "rate_limit_per_hour": 1000
        }
    
    def collect_database_configuration(self) -> Dict[str, Any]:
        """Collect PostgreSQL database configuration."""
        rprint("\n[bold cyan]🗄️ PostgreSQL Database Configuration[/bold cyan]")
        
        # Check if database was detected/tested in pre-flight
        db_detected = False
        db_details = {}
        
        if self.preflight_results:
            db_test = next((test for test in self.preflight_results.get('test_results', []) 
                          if test.get('test_name') == 'PostgreSQL Database'), None)
            if db_test and db_test.get('success'):
                db_detected = True
                db_details = db_test.get('details', {})
        
        if db_detected:
            rprint(f"[green]✅ PostgreSQL database validated during pre-flight[/green]")
            rprint(f"[dim]Database: {db_details.get('database')}, Version: {db_details.get('version')}[/dim]")
            
            if RICH_AVAILABLE:
                use_detected = Confirm.ask("Use the validated database configuration", default=True)
            else:
                use_detected = input("Use validated database [Y/n]: ").lower() != 'n'
                
            if use_detected:
                return {
                    "enabled": True,
                    "host": "DETECTED_FROM_ENVIRONMENT",
                    "port": 5432,
                    "database": db_details.get('database', 'tidyllm'),
                    "ssl_mode": "require",
                    "connection_pool_size": 20,
                    "mlflow_ready": db_details.get('mlflow_tables', 0) > 0
                }
        
        # Manual configuration
        if RICH_AVAILABLE:
            configure_db = Confirm.ask("Configure PostgreSQL database", default=True)
        else:
            configure_db = input("Configure PostgreSQL [Y/n]: ").lower() != 'n'
        
        if not configure_db:
            return {"enabled": False}
        
        if RICH_AVAILABLE:
            db_host = Prompt.ask("PostgreSQL host", default="localhost")
            db_port = int(Prompt.ask("PostgreSQL port", default="5432"))
            db_name = Prompt.ask("Database name", default="tidyllm")
            db_ssl_mode = Prompt.ask(
                "SSL mode", 
                choices=["require", "prefer", "allow", "disable"],
                default="require"
            )
        else:
            db_host = input("PostgreSQL host [localhost]: ") or "localhost"
            db_port = int(input("PostgreSQL port [5432]: ") or "5432")
            db_name = input("Database name [tidyllm]: ") or "tidyllm"
            db_ssl_mode = input("SSL mode [require]: ") or "require"
        
        return {
            "enabled": True,
            "host": db_host,
            "port": db_port,
            "database": db_name,
            "ssl_mode": db_ssl_mode,
            "connection_pool_size": 20
        }
    
    def collect_mlflow_configuration(self) -> Dict[str, Any]:
        """Collect MLflow configuration."""
        rprint("\n[bold magenta]📊 MLflow Tracking Configuration[/bold magenta]")
        
        # Check if MLflow was detected in pre-flight
        mlflow_detected = False
        mlflow_details = {}
        
        if self.preflight_results:
            mlflow_test = next((test for test in self.preflight_results.get('test_results', []) 
                              if test.get('test_name') == 'MLflow Tracking'), None)
            if mlflow_test and mlflow_test.get('success'):
                mlflow_detected = True
                mlflow_details = mlflow_test.get('details', {})
        
        if mlflow_detected:
            rprint(f"[green]✅ MLflow server validated during pre-flight[/green]")
            rprint(f"[dim]Experiments: {mlflow_details.get('experiment_count', 0)}[/dim]")
            
            if RICH_AVAILABLE:
                use_detected = Confirm.ask("Use the validated MLflow configuration", default=True)
            else:
                use_detected = input("Use validated MLflow [Y/n]: ").lower() != 'n'
                
            if use_detected:
                return {
                    "enabled": True,
                    "tracking_uri": "DETECTED_FROM_ENVIRONMENT",
                    "experiment_count": mlflow_details.get('experiment_count', 0)
                }
        
        # Manual configuration
        if RICH_AVAILABLE:
            configure_mlflow = Confirm.ask("Configure MLflow tracking server", default=True)
        else:
            configure_mlflow = input("Configure MLflow [Y/n]: ").lower() != 'n'
        
        if not configure_mlflow:
            return {"enabled": False}
        
        if RICH_AVAILABLE:
            tracking_uri = Prompt.ask(
                "MLflow tracking URI", 
                default="http://localhost:5000"
            )
        else:
            tracking_uri = input("MLflow tracking URI [http://localhost:5000]: ") or "http://localhost:5000"
        
        return {
            "enabled": True,
            "tracking_uri": tracking_uri
        }
    
    def collect_model_preferences(self) -> Dict[str, Any]:
        """Collect AI model preferences with Bedrock validation context."""
        rprint("\n[bold green]🤖 AI Model Configuration[/bold green]")
        
        # Check Bedrock validation results
        bedrock_models = []
        if self.preflight_results:
            aws_test = next((test for test in self.preflight_results.get('test_results', []) 
                           if test.get('test_name') == 'AWS Services'), None)
            if aws_test and aws_test.get('service_results', {}).get('bedrock', {}).get('success'):
                bedrock_details = aws_test['service_results']['bedrock'].get('details', {})
                bedrock_models = bedrock_details.get('available_models', [])
        
        if bedrock_models:
            rprint(f"[green]✅ {len(bedrock_models)} Claude models available in your AWS account[/green]")
            print_table(
                [[model] for model in bedrock_models[:5]],
                ["Available Claude Models"]
            )
        
        models = {
            "claude-3-sonnet": "Claude 3 Sonnet (Balanced performance)",
            "claude-3-haiku": "Claude 3 Haiku (Fast responses)",
            "claude-3-opus": "Claude 3 Opus (Highest quality)"
        }
        
        if RICH_AVAILABLE:
            default_model = Prompt.ask(
                "Default model",
                choices=list(models.keys()),
                default="claude-3-sonnet"
            )
            
            max_tokens = int(Prompt.ask("Max tokens per request", default="4000"))
            temperature = float(Prompt.ask("Temperature (0.0-1.0)", default="0.3"))
            
        else:
            default_model = input("Default model [claude-3-sonnet]: ") or "claude-3-sonnet"
            max_tokens = int(input("Max tokens [4000]: ") or "4000")
            temperature = float(input("Temperature [0.3]: ") or "0.3")
            
        return {
            "default_model": default_model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "available_bedrock_models": bedrock_models,
            "model_aliases": {
                "corporate-llm": default_model,
                "quick-assist": "claude-3-haiku",
                "premium-llm": "claude-3-opus",
                "analyst-llm": "claude-3-sonnet"
            }
        }
    
    def generate_enhanced_configuration(self) -> Dict[str, Any]:
        """Generate enhanced configuration with validation results."""
        rprint("\n[bold cyan]⚙️ Generating Enhanced Configuration...[/bold cyan]")
        
        # Combine all collected configuration
        complete_config = {
            **self.config,
            "validation_results": self.preflight_results,
            "generated_timestamp": datetime.now().isoformat(),
            "onboarding_version": "enhanced-v2.0"
        }
        
        # Generate using enhanced template
        final_config = create_template_config(complete_config)
        
        # Add enhanced features based on pre-flight results
        if self.preflight_results:
            # Add environment-specific optimizations
            env_info = {}
            for test in self.preflight_results.get('test_results', []):
                if test.get('test_name') == 'Environment Detection':
                    env_info = test.get('environment_info', {})
                    break
            
            # Configure proxy settings if detected
            if env_info.get('proxy_detected'):
                final_config.setdefault('network', {})['proxy'] = {
                    'enabled': True,
                    'settings_detected': True,
                    'auto_configure': True
                }
            
            # Configure SSO settings if detected
            if env_info.get('sso_configured'):
                final_config.setdefault('security', {})['sso'] = {
                    'enabled': True,
                    'auto_detected': True,
                    'provider': self.config.get('security', {}).get('sso_provider', 'aws-sso')
                }
            
            # Add credential management based on discovery
            cred_info = None
            for test in self.preflight_results.get('test_results', []):
                if test.get('test_name') == 'Credential Discovery':
                    cred_info = test.get('active_credential')
                    break
            
            if cred_info:
                final_config['aws']['credential_management'] = {
                    'source': cred_info.get('source'),
                    'auto_refresh': self.config.get('security', {}).get('temp_credential_refresh', True),
                    'expires': cred_info.get('expires'),
                    'account_id': cred_info.get('account_id')
                }
        
        return final_config
    
    def run_interactive_wizard(self):
        """Run the complete enhanced interactive wizard."""
        try:
            self.welcome_banner()
            
            # Step 1: Run pre-flight tests
            rprint("\n[bold cyan]Step 1: Pre-Flight Validation[/bold cyan]")
            preflight_results = self.run_preflight_tests()
            
            if not preflight_results.get("overall_success", False):
                if RICH_AVAILABLE:
                    continue_anyway = Confirm.ask(
                        "Pre-flight tests failed. Continue with configuration anyway?", 
                        default=False
                    )
                else:
                    continue_anyway = input("Continue anyway? [y/N]: ").lower().startswith('y')
                
                if not continue_anyway:
                    rprint("[yellow]Onboarding cancelled. Please resolve pre-flight issues first.[/yellow]")
                    return
            
            # Step 2: Organization setup
            rprint("\n[bold cyan]Step 2: Organization Configuration[/bold cyan]")
            org_info = self.collect_organization_info()
            self.config.update(org_info)
            
            # Step 3: Enhanced security
            rprint("\n[bold cyan]Step 3: Enhanced Security & Compliance[/bold cyan]")
            security_config = self.collect_enhanced_security_preferences()
            self.config["security"] = security_config
            
            # Step 4: Database configuration
            rprint("\n[bold cyan]Step 4: Database Configuration[/bold cyan]")
            db_config = self.collect_database_configuration()
            self.config["database"] = db_config
            
            # Step 5: MLflow configuration
            rprint("\n[bold cyan]Step 5: MLflow Configuration[/bold cyan]")
            mlflow_config = self.collect_mlflow_configuration()
            self.config["mlflow"] = mlflow_config
            
            # Step 6: Model preferences
            rprint("\n[bold cyan]Step 6: AI Model Configuration[/bold cyan]")
            model_config = self.collect_model_preferences()
            self.config["models"] = model_config
            
            # Step 7: Generate configuration
            rprint("\n[bold cyan]Step 7: Configuration Generation[/bold cyan]")
            final_config = self.generate_enhanced_configuration()
            
            # Step 8: Save and generate artifacts
            output_dir = Path.cwd() / "tidyllm-enhanced-corporate-config"
            output_dir.mkdir(exist_ok=True)
            
            self.save_configuration(final_config, output_dir / "settings.yaml")
            
            # Final summary
            print_panel(f"""
🎉 Enhanced Corporate Onboarding Complete!

📁 Configuration saved to: {output_dir.absolute()}
📋 Review DEPLOYMENT_INSTRUCTIONS.md for next steps
🔧 Edit .env.template with your actual credentials
✅ Pre-flight validation results included in configuration

Deployment ready features:
• Universal pre-flight validation framework
• SSO and temporary credential management  
• Corporate proxy and network compatibility
• PostgreSQL and MLflow integration
• Enhanced security and compliance features

To deploy:
  cd {output_dir.name}
  cp .env.template .env  # Edit with your values
  docker-compose up -d

To re-run validation:
  python enhanced_cli_onboarding.py --validate
            """, "Enhanced Setup Complete!")
            
        except KeyboardInterrupt:
            rprint("\n\n[bold red]Setup cancelled by user[/bold red]")
            sys.exit(1)
        except Exception as e:
            rprint(f"\n[bold red]Setup failed: {e}[/bold red]")
            sys.exit(1)
    
    def save_configuration(self, config: Dict[str, Any], output_path: str):
        """Save configuration and generate deployment artifacts."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
            
        rprint(f"✅ Configuration saved to: {output_file.absolute()}")
        
        # Generate enhanced deployment artifacts
        self.generate_enhanced_deployment_artifacts(config, output_file.parent)
    
    def generate_enhanced_deployment_artifacts(self, config: Dict[str, Any], output_dir: Path):
        """Generate enhanced deployment artifacts."""
        from config_generator import generate_dockerfile, generate_docker_compose, generate_kubernetes_manifests
        
        rprint("\n[bold blue]Generating enhanced deployment artifacts...[/bold blue]")
        
        # Standard artifacts
        dockerfile = generate_dockerfile(config)
        with open(output_dir / "Dockerfile", 'w') as f:
            f.write(dockerfile)
        rprint("📦 Generated: Dockerfile")
        
        compose = generate_docker_compose(config)
        with open(output_dir / "docker-compose.yml", 'w') as f:
            f.write(compose)
        rprint("📦 Generated: docker-compose.yml")
        
        # Enhanced artifacts
        self.generate_preflight_config(output_dir)
        self.generate_enhanced_env_template(config, output_dir)
        self.generate_enhanced_instructions(config, output_dir)
        
        rprint("📦 Generated: Enhanced deployment package")
    
    def generate_preflight_config(self, output_dir: Path):
        """Generate pre-flight test configuration."""
        preflight_config = {
            'aws': {
                'enabled': True,
                'region': self.config.get('aws_region', 'us-east-1'),
                'test_bedrock_models': False,
                's3_bucket': None
            },
            'postgresql': {
                'enabled': self.config.get('database', {}).get('enabled', False),
                'host': None,  # Will use environment variables
                'port': 5432,
                'database': self.config.get('database', {}).get('database', 'tidyllm'),
                'username': None,  # Will use environment variables
                'password': None,  # Will use environment variables
                'ssl_mode': 'require'
            },
            'mlflow': {
                'enabled': self.config.get('mlflow', {}).get('enabled', False),
                'tracking_uri': None  # Will use environment variables
            },
            'corporate': {
                'require_proxy': self.config.get('security', {}).get('proxy_configured', False),
                'require_sso': self.config.get('security', {}).get('sso_enabled', False),
                'require_encryption': True,
                'max_latency_ms': 5000,
                'timeout_seconds': 30
            },
            'tests': {
                'fail_on_warnings': False,
                'detailed_output': True,
                'save_report': True,
                'report_file': 'preflight_report.json'
            }
        }
        
        with open(output_dir / 'preflight_config.yaml', 'w') as f:
            yaml.dump(preflight_config, f, default_flow_style=False, indent=2)
        rprint("📦 Generated: preflight_config.yaml")
    
    def generate_enhanced_env_template(self, config: Dict[str, Any], output_dir: Path):
        """Generate enhanced environment template."""
        org_name = config.get('organization', 'company').lower()
        aws_region = config.get('aws_region', 'us-east-1')
        
        env_content = f"""# TidyLLM Enhanced Corporate Environment Variables
# ================================================
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Organization: {config.get('organization', 'YourOrganization')}
# Onboarding Version: Enhanced v2.0

# =============================================================================
# REQUIRED: AWS CREDENTIALS
# =============================================================================
AWS_ACCESS_KEY_ID=your_aws_access_key_here
AWS_SECRET_ACCESS_KEY=your_aws_secret_key_here
AWS_SESSION_TOKEN=your_aws_session_token_here_if_using_sts
AWS_DEFAULT_REGION={aws_region}

# =============================================================================
# REQUIRED: DATABASE CONFIGURATION
# =============================================================================
TIDYLLM_DB_PASSWORD=your_secure_database_password_here
"""

        # Add database config if enabled
        if config.get('database', {}).get('enabled'):
            env_content += f"""
# PostgreSQL Database Settings
TIDYLLM_DB_HOST={config['database'].get('host', 'localhost')}
TIDYLLM_DB_NAME={config['database'].get('database', 'tidyllm')}
TIDYLLM_DB_USER=tidyllm_service
TIDYLLM_DB_PORT={config['database'].get('port', 5432)}
"""

        # Add MLflow config if enabled
        if config.get('mlflow', {}).get('enabled'):
            env_content += f"""
# =============================================================================
# MLFLOW CONFIGURATION
# =============================================================================
MLFLOW_TRACKING_URI={config['mlflow'].get('tracking_uri', 'http://localhost:5000')}
"""

        # Add SSO config if enabled
        if config.get('security', {}).get('sso_enabled'):
            sso_provider = config['security'].get('sso_provider', 'aws-sso')
            env_content += f"""
# =============================================================================
# SSO CONFIGURATION
# =============================================================================
SSO_PROVIDER={sso_provider}
"""
            
            if sso_provider == 'aws-sso':
                env_content += f"""AWS_SSO_START_URL={config['security'].get('sso_start_url', 'https://your-sso.awsapps.com/start')}
AWS_SSO_REGION={config['security'].get('sso_region', aws_region)}
"""

        env_content += f"""
# =============================================================================
# OPTIONAL: CORPORATE CUSTOMIZATION
# =============================================================================
# S3 Storage (will use defaults if not specified)
TIDYLLM_S3_BUCKET={org_name}-tidyllm-storage
TIDYLLM_AUDIT_BUCKET={org_name}-tidyllm-audit

# KMS Encryption (recommended for corporate)
TIDYLLM_KMS_KEY_ARN=arn:aws:kms:{aws_region}:account:key/your-key-id

# Logging Configuration
TIDYLLM_LOG_LEVEL=INFO
TIDYLLM_LOG_FORMAT=json
TIDYLLM_ENCRYPT_LOGS=true

# Security Settings
TIDYLLM_ENCRYPT_CACHE=true
TIDYLLM_AUDIT_ENABLED=true
TIDYLLM_MASK_SENSITIVE_DATA=true

# Network Configuration (uncomment if using corporate proxy)
# HTTP_PROXY=http://proxy.company.com:8080
# HTTPS_PROXY=http://proxy.company.com:8080
# NO_PROXY=localhost,127.0.0.1,.company.com

# =============================================================================
# PRE-FLIGHT TESTING
# =============================================================================
# Set to 'true' to run pre-flight tests on startup
TIDYLLM_RUN_PREFLIGHT=true
TIDYLLM_PREFLIGHT_CONFIG=preflight_config.yaml
"""

        with open(output_dir / ".env.template", 'w') as f:
            f.write(env_content)
        rprint("📦 Generated: .env.template (enhanced)")
    
    def generate_enhanced_instructions(self, config: Dict[str, Any], output_dir: Path):
        """Generate enhanced deployment instructions."""
        org_name = config.get('organization', 'YourOrganization')
        
        instructions = f"""# TidyLLM Enhanced Corporate Deployment Guide

**Organization:** {org_name}  
**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Onboarding Version:** Enhanced v2.0

## 🚀 Quick Start

1. **Set up environment variables:**
   ```bash
   cp .env.template .env
   # Edit .env with your actual credentials and settings
   ```

2. **Run pre-flight validation:**
   ```bash
   python universal_preflight.py --config preflight_config.yaml
   ```

3. **Deploy using Docker Compose:**
   ```bash
   docker-compose up -d
   ```

4. **Verify deployment:**
   ```bash
   curl http://localhost:8000/health
   ```

## 📋 Pre-Deployment Validation

This enhanced deployment package includes comprehensive validation:

### Universal Pre-Flight Tests
```bash
# Run all tests (recommended)
python universal_preflight.py --config preflight_config.yaml

# Quick AWS-only validation
python universal_preflight.py --quick

# Verbose output
python universal_preflight.py --config preflight_config.yaml --verbose
```

### Enhanced Session Validation
```bash
# Test corporate environment with SSO/proxy support
python enhanced_session_validator.py
```

### Re-run Onboarding Validation
```bash
# Validate existing setup
python enhanced_cli_onboarding.py --validate

# Run only pre-flight tests
python enhanced_cli_onboarding.py --preflight-only
```

## 🔧 Enhanced Features

### SSO Integration
- AWS SSO support with temporary credential refresh
- Corporate identity provider integration
- Automatic credential discovery and validation

### Corporate Network Compatibility
- Corporate proxy auto-detection and configuration
- Custom CA certificate support
- Network latency and connectivity validation

### Database Integration
- PostgreSQL connectivity validation
- MLflow tracking server integration
- Database schema and table verification

### Security & Compliance
- Data encryption at rest and in transit
- Comprehensive audit logging
- Sensitive data masking
- Rate limiting and access controls

## 🏢 Corporate Deployment Options

### Option 1: Kubernetes (Production)
```bash
# Create namespace
kubectl create namespace {org_name.lower()}

# Create secrets from .env file
kubectl create secret generic tidyllm-secrets --from-env-file=.env --namespace={org_name.lower()}

# Deploy application
kubectl apply -f kubernetes/ --namespace={org_name.lower()}

# Verify deployment
kubectl get pods --namespace={org_name.lower()}
```

### Option 2: Docker Compose (Development/Staging)
```bash
# Start services
docker-compose up -d

# View logs
docker-compose logs -f tidyllm

# Stop services
docker-compose down
```

### Option 3: Enhanced Monitoring Setup
```bash
# Deploy with monitoring stack
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Access monitoring dashboards
# Prometheus: http://localhost:9090
# Grafana: http://localhost:3000
```

## 🔍 Validation & Troubleshooting

### Health Checks
```bash
# Basic health check
curl http://localhost:8000/health

# Detailed system status
curl http://localhost:8000/api/v1/status

# Pre-flight validation endpoint
curl http://localhost:8000/api/v1/preflight
```

### Common Issues

**1. SSO Authentication Issues**
```
Error: SSO credentials expired or invalid
```
Solution: Run `aws sso login` to refresh SSO credentials

**2. Corporate Proxy Issues**
```
Error: SSL verification failed
```
Solution: Set up corporate CA certificates in .env file

**3. Database Connection Issues**
```
Error: Could not connect to PostgreSQL
```
Solution: Verify database host, credentials, and network connectivity

**4. Temporary Credentials Expiring**
```
Warning: AWS credentials expire soon
```
Solution: Enable automatic credential refresh in configuration

### Getting Help

- **Logs**: `docker-compose logs tidyllm`
- **Validation**: `python enhanced_cli_onboarding.py --validate`
- **Pre-flight**: `python universal_preflight.py`
- **Support**: Contact your IT administrator

## 📊 Monitoring & Observability

### Log Aggregation
- Structured JSON logging
- Audit trail for compliance
- Sensitive data masking
- Configurable retention policies

### Metrics & Alerting
- Prometheus metrics export
- Custom corporate dashboards
- Performance monitoring
- Error rate tracking

### Security Monitoring
- Authentication events
- API access patterns
- Data access auditing
- Compliance reporting

---

**Generated by TidyLLM Enhanced Corporate Onboarding v2.0**
"""

        with open(output_dir / "DEPLOYMENT_INSTRUCTIONS.md", 'w') as f:
            f.write(instructions)
        rprint("📋 Generated: DEPLOYMENT_INSTRUCTIONS.md (enhanced)")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="TidyLLM Enhanced Corporate Onboarding CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--preflight-only", 
        action="store_true",
        help="Run pre-flight validation tests only"
    )
    parser.add_argument(
        "--validate",
        action="store_true", 
        help="Validate existing configuration and environment"
    )
    parser.add_argument(
        "--config-only",
        action="store_true",
        help="Generate configuration from template without wizard"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (AWS only)"
    )
    parser.add_argument(
        "--output",
        default="tidyllm-enhanced-corporate-config",
        help="Output directory for generated files"
    )
    
    args = parser.parse_args()
    
    onboarding = EnhancedCorporateOnboarding()
    
    if args.preflight_only:
        rprint("[bold blue]Running pre-flight validation only...[/bold blue]")
        results = onboarding.run_preflight_tests(quick_mode=args.quick)
        sys.exit(0 if results.get("overall_success", False) else 1)
        
    elif args.validate:
        rprint("[bold blue]Validating TidyLLM setup...[/bold blue]")
        
        # Run comprehensive validation
        success = run_universal_preflight()
        
        if success:
            rprint("[bold green]✅ Setup validation passed![/bold green]")
        else:
            rprint("[bold red]❌ Setup validation failed[/bold red]")
            
        sys.exit(0 if success else 1)
            
    elif args.config_only:
        rprint("[bold blue]Generating default enhanced corporate configuration...[/bold blue]")
        
        # Load default config
        default_config = {
            "organization": "YourCompany",
            "environment": "production", 
            "aws_region": "us-east-1",
            "deployment_type": "kubernetes",
            "security": {
                "sso_enabled": True,
                "audit_logging": True,
                "encrypt_cache": True,
                "network_security": True
            },
            "database": {"enabled": False},
            "mlflow": {"enabled": False},
            "models": {
                "default_model": "claude-3-sonnet",
                "max_tokens": 4000,
                "temperature": 0.3
            }
        }
        
        onboarding.config = default_config
        final_config = create_template_config(default_config)
        
        output_dir = Path(args.output)
        onboarding.save_configuration(final_config, output_dir / "settings.yaml")
        
        rprint(f"[bold green]✅ Enhanced configuration generated in {output_dir}[/bold green]")
    else:
        # Run interactive wizard
        onboarding.run_interactive_wizard()

if __name__ == "__main__":
    main()