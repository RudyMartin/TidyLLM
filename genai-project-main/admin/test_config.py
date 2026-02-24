#!/usr/bin/env python3
"""
TidyLLM Admin Configuration Test
===============================

Verifies all admin configuration settings and infrastructure connectivity.
Tests AWS credentials, database connections, S3 access, and system components.
"""

import os
import sys
import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class ConfigurationTester:
    """Test all TidyLLM admin configurations and infrastructure."""
    
    def __init__(self):
        self.admin_path = Path(__file__).parent
        self.settings_path = self.admin_path / "settings.yaml"
        self.test_results = []
        self.config = None
        
    def log_test(self, test_name: str, status: str, details: Dict[str, Any]):
        """Log test result with timestamp and details."""
        result = {
            'timestamp': datetime.now().isoformat(),
            'test_name': test_name,
            'status': status,
            'details': details
        }
        self.test_results.append(result)
        
        # Color coding for console output
        status_color = {
            'PASS': '\033[92m',    # Green
            'FAIL': '\033[91m',    # Red
            'WARN': '\033[93m',    # Yellow
            'INFO': '\033[94m'     # Blue
        }
        reset_color = '\033[0m'
        
        color = status_color.get(status, '')
        print(f"{color}[{status}]{reset_color} {test_name}: {details.get('message', 'Test completed')}")
        
        if status == 'FAIL' and 'error' in details:
            print(f"   Error: {details['error']}")
        elif 'value' in details:
            print(f"   Value: {details['value']}")
    
    def test_settings_file(self) -> bool:
        """Test settings.yaml file existence and structure."""
        try:
            if not self.settings_path.exists():
                self.log_test(
                    "Settings File Existence",
                    "FAIL", 
                    {"message": "settings.yaml not found", "path": str(self.settings_path)}
                )
                return False
            
            with open(self.settings_path, 'r') as f:
                self.config = yaml.safe_load(f)
            
            file_size = self.settings_path.stat().st_size
            self.log_test(
                "Settings File Load",
                "PASS",
                {"message": "settings.yaml loaded successfully", "size_bytes": file_size}
            )
            
            # Test required sections
            required_sections = ['postgres', 's3', 'aws', 'integrations']
            missing_sections = [section for section in required_sections if section not in self.config]
            
            if missing_sections:
                self.log_test(
                    "Settings Structure",
                    "WARN",
                    {"message": f"Missing sections: {missing_sections}"}
                )
                return True  # Still usable
            else:
                self.log_test(
                    "Settings Structure",
                    "PASS",
                    {"message": "All required sections present", "sections": required_sections}
                )
                return True
                
        except Exception as e:
            self.log_test(
                "Settings File Load",
                "FAIL",
                {"message": "Failed to load settings.yaml", "error": str(e)}
            )
            return False
    
    def test_postgresql_config(self) -> bool:
        """Test PostgreSQL configuration and connectivity."""
        if not self.config or 'postgres' not in self.config:
            self.log_test(
                "PostgreSQL Config",
                "FAIL",
                {"message": "No PostgreSQL configuration found"}
            )
            return False
        
        pg_config = self.config['postgres']
        required_fields = ['host', 'port', 'db_name', 'db_user', 'db_password']
        missing_fields = [field for field in required_fields if field not in pg_config]
        
        if missing_fields:
            self.log_test(
                "PostgreSQL Config",
                "FAIL",
                {"message": f"Missing PostgreSQL fields: {missing_fields}"}
            )
            return False
        
        # Test connectivity
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=pg_config['host'],
                port=pg_config['port'],
                database=pg_config['db_name'],
                user=pg_config['db_user'],
                password=pg_config['db_password'],
                sslmode=pg_config.get('ssl_mode', 'require'),
                connect_timeout=10
            )
            
            # Test basic query
            cursor = conn.cursor()
            cursor.execute("SELECT version();")
            version = cursor.fetchone()[0]
            cursor.close()
            conn.close()
            
            self.log_test(
                "PostgreSQL Connection",
                "PASS",
                {"message": "Database connection successful", "version": version[:50] + "..."}
            )
            return True
            
        except ImportError:
            self.log_test(
                "PostgreSQL Connection",
                "WARN",
                {"message": "psycopg2 not installed", "fix": "pip install psycopg2-binary"}
            )
            return False
        except Exception as e:
            self.log_test(
                "PostgreSQL Connection",
                "FAIL",
                {"message": "Database connection failed", "error": str(e)}
            )
            return False
    
    def test_s3_config(self) -> bool:
        """Test S3 configuration and connectivity."""
        if not self.config or 's3' not in self.config:
            self.log_test(
                "S3 Config",
                "FAIL",
                {"message": "No S3 configuration found"}
            )
            return False
        
        s3_config = self.config['s3']
        required_fields = ['region', 'bucket']
        missing_fields = [field for field in required_fields if field not in s3_config]
        
        if missing_fields:
            self.log_test(
                "S3 Config",
                "FAIL",
                {"message": f"Missing S3 fields: {missing_fields}"}
            )
            return False
        
        self.log_test(
            "S3 Config",
            "PASS",
            {"message": "S3 configuration valid", "bucket": s3_config['bucket'], "region": s3_config['region']}
        )
        
        # Test AWS connectivity
        try:
            import boto3
            from botocore.exceptions import ClientError, NoCredentialsError
            
            s3_client = boto3.client('s3', region_name=s3_config['region'])
            
            # Try to list buckets (requires credentials)
            try:
                response = s3_client.list_buckets()
                buckets = [bucket['Name'] for bucket in response['Buckets']]
                
                if s3_config['bucket'] in buckets:
                    self.log_test(
                        "S3 Connection",
                        "PASS",
                        {"message": f"S3 access confirmed", "bucket_found": True, "total_buckets": len(buckets)}
                    )
                else:
                    self.log_test(
                        "S3 Connection",
                        "WARN",
                        {"message": f"Bucket '{s3_config['bucket']}' not found in account", "available_buckets": len(buckets)}
                    )
                return True
                
            except NoCredentialsError:
                self.log_test(
                    "S3 Connection",
                    "WARN",
                    {"message": "AWS credentials not configured", "fix": "Set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY"}
                )
                return False
            except ClientError as e:
                self.log_test(
                    "S3 Connection",
                    "FAIL",
                    {"message": "S3 access failed", "error": str(e)}
                )
                return False
                
        except ImportError:
            self.log_test(
                "S3 Connection",
                "WARN",
                {"message": "boto3 not installed", "fix": "pip install boto3"}
            )
            return False
    
    def test_aws_credentials(self) -> bool:
        """Test AWS credentials configuration."""
        # Check environment variables
        aws_access_key = os.environ.get('AWS_ACCESS_KEY_ID')
        aws_secret_key = os.environ.get('AWS_SECRET_ACCESS_KEY')
        aws_region = os.environ.get('AWS_DEFAULT_REGION')
        
        if aws_access_key and aws_secret_key:
            self.log_test(
                "AWS Credentials",
                "PASS",
                {"message": "AWS credentials found in environment", "access_key_prefix": aws_access_key[:8] + "..."}
            )
            return True
        
        # Check AWS config in settings
        if self.config and 'aws' in self.config:
            aws_config = self.config['aws']
            if 'bedrock' in aws_config and 'credentials' in aws_config['bedrock']:
                creds = aws_config['bedrock']['credentials']
                if creds.get('access_key_id') and creds.get('secret_access_key'):
                    self.log_test(
                        "AWS Credentials",
                        "PASS",
                        {"message": "AWS credentials found in settings.yaml"}
                    )
                    return True
        
        # Check admin credential files
        cred_script = self.admin_path / "set_aws_credentials.py"
        if cred_script.exists():
            self.log_test(
                "AWS Credentials",
                "INFO",
                {"message": "AWS credential script found", "file": str(cred_script)}
            )
        
        self.log_test(
            "AWS Credentials",
            "WARN",
            {"message": "No AWS credentials configured", "fix": "Set environment variables or run set_aws_credentials.py"}
        )
        return False
    
    def test_embedding_database(self) -> bool:
        """Test embedding database tables and data."""
        if not self.test_postgresql_config():
            return False
        
        try:
            import psycopg2
            pg_config = self.config['postgres']
            conn = psycopg2.connect(
                host=pg_config['host'],
                port=pg_config['port'],
                database=pg_config['db_name'],
                user=pg_config['db_user'],
                password=pg_config['db_password'],
                sslmode=pg_config.get('ssl_mode', 'require')
            )
            
            cursor = conn.cursor()
            
            # Check embedding tables
            embedding_tables = ['document_chunks', 'chunk_embeddings', 'paper_embeddings']
            table_stats = {}
            
            for table in embedding_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table};")
                    count = cursor.fetchone()[0]
                    table_stats[table] = count
                except Exception as e:
                    table_stats[table] = f"ERROR: {str(e)}"
            
            # Check embeddings status
            try:
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_chunks,
                        COUNT(*) FILTER (WHERE embedding IS NOT NULL) as with_embeddings
                    FROM document_chunks;
                """)
                result = cursor.fetchone()
                total_chunks, with_embeddings = result
                embedding_percentage = (with_embeddings / max(total_chunks, 1)) * 100
                
                cursor.close()
                conn.close()
                
                self.log_test(
                    "Embedding Database",
                    "PASS",
                    {
                        "message": "Embedding database accessible",
                        "table_counts": table_stats,
                        "embedding_status": f"{with_embeddings}/{total_chunks} ({embedding_percentage:.1f}%)"
                    }
                )
                
                if embedding_percentage < 50:
                    self.log_test(
                        "Embedding Status",
                        "WARN",
                        {"message": f"Low embedding completion rate: {embedding_percentage:.1f}%"}
                    )
                else:
                    self.log_test(
                        "Embedding Status",
                        "PASS",
                        {"message": f"Good embedding completion rate: {embedding_percentage:.1f}%"}
                    )
                
                return True
                
            except Exception as e:
                cursor.close()
                conn.close()
                self.log_test(
                    "Embedding Database",
                    "FAIL",
                    {"message": "Error querying embedding tables", "error": str(e)}
                )
                return False
                
        except Exception as e:
            self.log_test(
                "Embedding Database",
                "FAIL",
                {"message": "Failed to test embedding database", "error": str(e)}
            )
            return False
    
    def test_mlflow_integration(self) -> bool:
        """Test MLflow integration configuration."""
        if not self.config or 'integrations' not in self.config:
            self.log_test(
                "MLflow Integration",
                "WARN",
                {"message": "No integrations configuration found"}
            )
            return False
        
        integrations = self.config['integrations']
        if 'mlflow' not in integrations:
            self.log_test(
                "MLflow Integration",
                "WARN",
                {"message": "MLflow not configured"}
            )
            return False
        
        mlflow_config = integrations['mlflow']
        if not mlflow_config.get('enabled', False):
            self.log_test(
                "MLflow Integration",
                "INFO",
                {"message": "MLflow integration disabled"}
            )
            return True
        
        # Test MLflow connectivity
        try:
            import mlflow
            tracking_uri = mlflow_config.get('tracking_uri')
            if tracking_uri:
                mlflow.set_tracking_uri(tracking_uri)
                self.log_test(
                    "MLflow Integration",
                    "PASS",
                    {"message": "MLflow configured", "tracking_uri": tracking_uri[:50] + "..."}
                )
                return True
            else:
                self.log_test(
                    "MLflow Integration",
                    "WARN",
                    {"message": "MLflow enabled but no tracking_uri"}
                )
                return False
                
        except ImportError:
            self.log_test(
                "MLflow Integration",
                "WARN",
                {"message": "MLflow not installed", "fix": "pip install mlflow"}
            )
            return False
        except Exception as e:
            self.log_test(
                "MLflow Integration",
                "FAIL",
                {"message": "MLflow configuration error", "error": str(e)}
            )
            return False
    
    def test_drop_zone_systems(self) -> bool:
        """Test drop zone systems availability."""
        drop_zone_files = [
            "scripts/production_tracking_drop_zones.py",
            "scripts/FINAL_real_dropzones.py", 
            "drop_zones/working_s3_dropzones.py"
        ]
        
        available_systems = []
        for drop_zone_file in drop_zone_files:
            file_path = project_root / drop_zone_file
            if file_path.exists():
                available_systems.append(drop_zone_file)
        
        if available_systems:
            self.log_test(
                "Drop Zone Systems",
                "PASS",
                {"message": f"Found {len(available_systems)} drop zone systems", "systems": available_systems}
            )
        else:
            self.log_test(
                "Drop Zone Systems",
                "FAIL",
                {"message": "No drop zone systems found"}
            )
        
        # Test drop zone directories
        drop_zone_dirs = [
            "drop_zones/input",
            "boss_demo_evidence/input_zone"
        ]
        
        existing_dirs = []
        for drop_dir in drop_zone_dirs:
            dir_path = project_root / drop_dir
            if dir_path.exists():
                existing_dirs.append(drop_dir)
        
        self.log_test(
            "Drop Zone Directories",
            "PASS" if existing_dirs else "WARN",
            {"message": f"Found {len(existing_dirs)} drop zone directories", "directories": existing_dirs}
        )
        
        return len(available_systems) > 0
    
    def test_monitoring_tools(self) -> bool:
        """Test monitoring and analysis tools."""
        monitoring_tools = [
            "rudy_test_embeddings.py",
            "RUDY_EMBEDDING_USAGE.md"
        ]
        
        available_tools = []
        for tool in monitoring_tools:
            tool_path = project_root / tool
            if tool_path.exists():
                available_tools.append(tool)
        
        if available_tools:
            self.log_test(
                "Monitoring Tools",
                "PASS",
                {"message": f"Found {len(available_tools)} monitoring tools", "tools": available_tools}
            )
            return True
        else:
            self.log_test(
                "Monitoring Tools", 
                "WARN",
                {"message": "No monitoring tools found"}
            )
            return False
    
    def run_all_tests(self) -> bool:
        """Run all configuration tests and return overall status."""
        print("TidyLLM ADMIN CONFIGURATION TEST")
        print("=" * 50)
        print(f"Testing configuration at: {self.admin_path}")
        print(f"Timestamp: {datetime.now().isoformat()}")
        print()
        
        # Run all tests
        tests_passed = 0
        total_tests = 8
        
        if self.test_settings_file():
            tests_passed += 1
        
        if self.test_postgresql_config():
            tests_passed += 1
            
        if self.test_s3_config():
            tests_passed += 1
            
        if self.test_aws_credentials():
            tests_passed += 1
            
        if self.test_embedding_database():
            tests_passed += 1
            
        if self.test_mlflow_integration():
            tests_passed += 1
            
        if self.test_drop_zone_systems():
            tests_passed += 1
            
        if self.test_monitoring_tools():
            tests_passed += 1
        
        # Summary
        print()
        print("=" * 50)
        print("TEST SUMMARY")
        print("=" * 50)
        
        passed_tests = len([r for r in self.test_results if r['status'] == 'PASS'])
        failed_tests = len([r for r in self.test_results if r['status'] == 'FAIL'])
        warned_tests = len([r for r in self.test_results if r['status'] == 'WARN'])
        
        print(f"Tests Run: {len(self.test_results)}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")  
        print(f"Warnings: {warned_tests}")
        print(f"Success Rate: {(passed_tests/len(self.test_results)*100):.1f}%")
        
        overall_status = "OPERATIONAL" if failed_tests == 0 else "DEGRADED" if passed_tests > failed_tests else "CRITICAL"
        
        status_colors = {
            "OPERATIONAL": '\033[92m',  # Green
            "DEGRADED": '\033[93m',     # Yellow
            "CRITICAL": '\033[91m'      # Red
        }
        
        color = status_colors.get(overall_status, '')
        reset = '\033[0m'
        
        print(f"\nOVERALL STATUS: {color}{overall_status}{reset}")
        
        if overall_status == "OPERATIONAL":
            print("✅ TidyLLM admin configuration is fully operational")
        elif overall_status == "DEGRADED":
            print("⚠️  TidyLLM has some configuration issues but core functionality works")
        else:
            print("❌ TidyLLM has critical configuration issues")
        
        # Save results
        self.save_test_results()
        
        return overall_status == "OPERATIONAL"
    
    def save_test_results(self):
        """Save test results to JSON file."""
        results_file = self.admin_path / f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        summary = {
            'timestamp': datetime.now().isoformat(),
            'admin_path': str(self.admin_path),
            'settings_file': str(self.settings_path),
            'test_results': self.test_results,
            'summary': {
                'total_tests': len(self.test_results),
                'passed': len([r for r in self.test_results if r['status'] == 'PASS']),
                'failed': len([r for r in self.test_results if r['status'] == 'FAIL']),
                'warnings': len([r for r in self.test_results if r['status'] == 'WARN'])
            }
        }
        
        try:
            with open(results_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nTest results saved: {results_file}")
        except Exception as e:
            print(f"\nWarning: Could not save test results: {e}")


def main():
    """Main entry point."""
    tester = ConfigurationTester()
    success = tester.run_all_tests()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())