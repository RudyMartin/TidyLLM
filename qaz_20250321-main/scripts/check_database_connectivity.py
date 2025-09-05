#!/usr/bin/env python3
"""
Database Connectivity Checker

Checks database connectivity and verifies if the filtered database strings
are working properly. This helps identify any issues caused by the
pre-flight filtering process.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess

class DatabaseConnectivityChecker:
    """Check database connectivity and configuration"""
    
    def __init__(self):
        self.project_root = Path.cwd()
        self.results = {
            "database_configs": {},
            "connectivity_tests": {},
            "affected_files": [],
            "recommendations": []
        }
    
    def check_database_connectivity(self) -> bool:
        """Run comprehensive database connectivity checks"""
        print("🔗 DATABASE CONNECTIVITY CHECK")
        print("=" * 60)
        
        # Check database configurations
        self._check_database_configs()
        
        # Check affected files
        self._check_affected_files()
        
        # Test connectivity
        self._test_connectivity()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Print summary
        self._print_summary()
        
        return self._is_overall_success()
    
    def _check_database_configs(self):
        """Check database configuration files"""
        print("⚙️ Checking Database Configurations...")
        
        # Check environment variables
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            print(f"  ✅ DATABASE_URL environment variable: {db_url.split('@')[1] if '@' in db_url else 'Configured'}")
            self.results["database_configs"]["env_var"] = {
                "status": "found",
                "value": db_url.split('@')[1] if '@' in db_url else "configured"
            }
        else:
            print("  ⚠️  DATABASE_URL environment variable not set")
            self.results["database_configs"]["env_var"] = {
                "status": "missing",
                "value": None
            }
        
        # Check for database configuration files
        config_files = [
            "environ_settings/.env.local",
            "environ_settings/config.local.yaml",
            "src/config/settings.py"
        ]
        
        for config_file in config_files:
            config_path = self.project_root / config_file
            if config_path.exists():
                print(f"  ✅ {config_file}: Found")
                self.results["database_configs"][config_file] = {
                    "status": "found",
                    "path": str(config_path)
                }
            else:
                print(f"  ❌ {config_file}: Not found")
                self.results["database_configs"][config_file] = {
                    "status": "missing",
                    "path": None
                }
    
    def _check_affected_files(self):
        """Check files that might have been affected by filtering"""
        print("\n📁 Checking Affected Files...")
        
        # Files that typically contain database configurations
        db_files = [
            "notebooks/01_database_exploration.py",
            "notebooks/05_point_in_time_demo.py",
            "notebooks/06_model_risk_governance_workflow.py",
            "notebooks/07_mcp_implementation_demo.py",
            "notebooks/08_mcp_backend_sequence_demo.py",
            "scripts/create_migration_bundle.py",
            "scripts/pre_flight_cleanup.py"
        ]
        
        for file_path in db_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                # Check if file contains database connection strings
                try:
                    with open(full_path, 'r') as f:
                        content = f.read()
                    
                    if 'postgresql://' in content:
                        print(f"  ⚠️  {file_path}: Contains database connection strings")
                        self.results["affected_files"].append({
                            "file": file_path,
                            "status": "contains_db_strings",
                            "lines": self._find_db_lines(content)
                        })
                    else:
                        print(f"  ✅ {file_path}: No database strings found")
                        self.results["affected_files"].append({
                            "file": file_path,
                            "status": "clean",
                            "lines": []
                        })
                        
                except Exception as e:
                    print(f"  ❌ {file_path}: Error reading file - {e}")
                    self.results["affected_files"].append({
                        "file": file_path,
                        "status": "error",
                        "error": str(e)
                    })
            else:
                print(f"  ❌ {file_path}: File not found")
                self.results["affected_files"].append({
                    "file": file_path,
                    "status": "missing"
                })
    
    def _find_db_lines(self, content: str) -> List[int]:
        """Find line numbers containing database connection strings"""
        lines = content.split('\n')
        db_lines = []
        
        for i, line in enumerate(lines, 1):
            if 'postgresql://' in line:
                db_lines.append(i)
        
        return db_lines
    
    def _test_connectivity(self):
        """Test actual database connectivity"""
        print("\n🔗 Testing Database Connectivity...")
        
        # Check if psql is available
        try:
            result = subprocess.run(['psql', '--version'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print(f"  ✅ PostgreSQL client available: {result.stdout.strip()}")
                self.results["connectivity_tests"]["psql_client"] = {
                    "status": "available",
                    "version": result.stdout.strip()
                }
            else:
                print("  ❌ PostgreSQL client not available")
                self.results["connectivity_tests"]["psql_client"] = {
                    "status": "not_available"
                }
        except FileNotFoundError:
            print("  ❌ PostgreSQL client not found in PATH")
            self.results["connectivity_tests"]["psql_client"] = {
                "status": "not_found"
            }
        
        # Test connection with environment variable
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            print(f"  🔍 Testing connection with DATABASE_URL...")
            
            # Extract connection details
            try:
                # Parse postgresql://user:password@host:port/database
                if db_url.startswith('postgresql://'):
                    parts = db_url.replace('postgresql://', '').split('@')
                    if len(parts) == 2:
                        user_pass = parts[0]
                        host_db = parts[1]
                        
                        if ':' in user_pass and '/' in host_db:
                            user, password = user_pass.split(':', 1)
                            host_port, database = host_db.split('/', 1)
                            
                            if ':' in host_port:
                                host, port = host_port.split(':', 1)
                            else:
                                host, port = host_port, '5432'
                            
                            print(f"    Host: {host}")
                            print(f"    Port: {port}")
                            print(f"    Database: {database}")
                            print(f"    User: {user}")
                            
                            # Test connection
                            self._test_postgres_connection(host, port, database, user, password)
                        else:
                            print("    ❌ Invalid DATABASE_URL format")
                    else:
                        print("    ❌ Invalid DATABASE_URL format")
                else:
                    print("    ❌ DATABASE_URL is not a PostgreSQL connection string")
                    
            except Exception as e:
                print(f"    ❌ Error parsing DATABASE_URL: {e}")
        else:
            print("  ⚠️  No DATABASE_URL to test")
    
    def _test_postgres_connection(self, host: str, port: str, database: str, user: str, password: str):
        """Test PostgreSQL connection"""
        try:
            # Use psql to test connection
            env = os.environ.copy()
            env['PGPASSWORD'] = password
            
            cmd = [
                'psql', 
                '-h', host, 
                '-p', port, 
                '-U', user, 
                '-d', database,
                '-c', 'SELECT version();'
            ]
            
            result = subprocess.run(cmd, 
                                  env=env,
                                  capture_output=True, 
                                  text=True,
                                  timeout=10)
            
            if result.returncode == 0:
                print("    ✅ Database connection successful!")
                print(f"    📊 {result.stdout.strip()}")
                self.results["connectivity_tests"]["database_connection"] = {
                    "status": "success",
                    "host": host,
                    "port": port,
                    "database": database,
                    "user": user
                }
            else:
                print(f"    ❌ Database connection failed: {result.stderr.strip()}")
                self.results["connectivity_tests"]["database_connection"] = {
                    "status": "failed",
                    "error": result.stderr.strip(),
                    "host": host,
                    "port": port,
                    "database": database,
                    "user": user
                }
                
        except subprocess.TimeoutExpired:
            print("    ❌ Database connection timeout")
            self.results["connectivity_tests"]["database_connection"] = {
                "status": "timeout",
                "host": host,
                "port": port,
                "database": database,
                "user": user
            }
        except Exception as e:
            print(f"    ❌ Database connection error: {e}")
            self.results["connectivity_tests"]["database_connection"] = {
                "status": "error",
                "error": str(e),
                "host": host,
                "port": port,
                "database": database,
                "user": user
            }
    
    def _generate_recommendations(self):
        """Generate recommendations based on findings"""
        print("\n💡 Generating Recommendations...")
        
        # Check if database URL is missing
        if not os.getenv('DATABASE_URL'):
            self.results["recommendations"].append(
                "Set DATABASE_URL environment variable for database connectivity"
            )
            print("  • Set DATABASE_URL environment variable")
        
        # Check if files contain hardcoded database strings
        affected_files = [f for f in self.results["affected_files"] 
                         if f.get("status") == "contains_db_strings"]
        
        if affected_files:
            self.results["recommendations"].append(
                f"Review {len(affected_files)} files with database connection strings"
            )
            print(f"  • Review {len(affected_files)} files with database strings")
        
        # Check if PostgreSQL client is missing
        if self.results["connectivity_tests"].get("psql_client", {}).get("status") != "available":
            self.results["recommendations"].append(
                "Install PostgreSQL client for database connectivity testing"
            )
            print("  • Install PostgreSQL client")
        
        # Check if database connection failed
        db_conn = self.results["connectivity_tests"].get("database_connection", {})
        if db_conn.get("status") == "failed":
            self.results["recommendations"].append(
                f"Fix database connection: {db_conn.get('error', 'Unknown error')}"
            )
            print(f"  • Fix database connection: {db_conn.get('error', 'Unknown error')}")
    
    def _print_summary(self):
        """Print summary of findings"""
        print("\n" + "=" * 60)
        print("📊 DATABASE CONNECTIVITY SUMMARY")
        print("=" * 60)
        
        # Configuration status
        config_status = "✅" if os.getenv('DATABASE_URL') else "❌"
        print(f"Database Configuration: {config_status}")
        
        # Connectivity status
        conn_status = self.results["connectivity_tests"].get("database_connection", {}).get("status", "unknown")
        if conn_status == "success":
            print(f"Database Connectivity: ✅")
        elif conn_status == "failed":
            print(f"Database Connectivity: ❌")
        else:
            print(f"Database Connectivity: ⚠️  (Not tested)")
        
        # Affected files
        affected_count = len([f for f in self.results["affected_files"] 
                            if f.get("status") == "contains_db_strings"])
        print(f"Files with DB Strings: {affected_count}")
        
        # Recommendations
        if self.results["recommendations"]:
            print(f"\n💡 Recommendations ({len(self.results['recommendations'])}):")
            for rec in self.results["recommendations"]:
                print(f"  • {rec}")
    
    def _is_overall_success(self) -> bool:
        """Determine if overall check was successful"""
        # Check if database URL is configured
        has_config = bool(os.getenv('DATABASE_URL'))
        
        # Check if connection was successful
        conn_status = self.results["connectivity_tests"].get("database_connection", {}).get("status")
        has_connection = conn_status == "success"
        
        return has_config and has_connection
    
    def get_results(self) -> Dict[str, Any]:
        """Get detailed results"""
        return self.results
    
    def save_results(self, output_file: str = "database_connectivity_results.json"):
        """Save results to file"""
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"📄 Results saved to: {output_file}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Connectivity Checker")
    parser.add_argument("--save-results", help="Save results to file")
    
    args = parser.parse_args()
    
    checker = DatabaseConnectivityChecker()
    success = checker.check_database_connectivity()
    
    if args.save_results:
        checker.save_results(args.save_results)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
