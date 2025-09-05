#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connection Profiles System
Separates connection configurations from DataMart, allowing DataMart to select appropriate profiles
"""

import os
import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class ConnectionType(Enum):
    """Types of data source connections"""
    LOCAL = "local"
    S3 = "s3"
    KINESIS = "kinesis"
    POSTGRESQL = "postgresql"
    REDIS = "redis"
    HTTP = "http"
    DATAMART_LIVE = "datamart_live"
    DATAMART_STREAM = "datamart_stream"


class EnvironmentType(Enum):
    """Environment types for connection profiles"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    DEMO = "demo"
    TESTING = "testing"


@dataclass
class ConnectionProfile:
    """Individual connection profile configuration"""
    name: str
    type: ConnectionType
    environment: EnvironmentType
    connection_string: str
    timeout: int = 30
    retry_attempts: int = 3
    fallback_source: Optional[str] = None
    authentication: Optional[Dict[str, Any]] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 1
    description: str = ""


class ConnectionProfileManager:
    """Manages connection profiles for DataMart selection"""
    
    def __init__(self, profiles_path: str = "config/connection_profiles.yaml"):
        self.profiles_path = Path(profiles_path)
        self.profiles = {}
        self.active_environment = self._detect_environment()
        self._load_profiles()
    
    def _detect_environment(self) -> EnvironmentType:
        """Detect current environment"""
        env = os.getenv('ENVIRONMENT', 'development').lower()
        
        if env == 'production':
            return EnvironmentType.PRODUCTION
        elif env == 'staging':
            return EnvironmentType.STAGING
        elif env == 'demo':
            return EnvironmentType.DEMO
        elif env == 'testing':
            return EnvironmentType.TESTING
        else:
            return EnvironmentType.DEVELOPMENT
    
    def _load_profiles(self):
        """Load connection profiles from configuration"""
        try:
            if self.profiles_path.exists():
                with open(self.profiles_path, 'r') as f:
                    config = yaml.safe_load(f)
                    self._parse_profiles(config)
            else:
                self._create_default_profiles()
        except Exception as e:
            print(f"Warning: Could not load connection profiles: {e}")
            self._create_default_profiles()
    
    def _parse_profiles(self, config: Dict[str, Any]):
        """Parse profiles from configuration"""
        for profile_data in config.get('profiles', []):
            try:
                profile = ConnectionProfile(
                    name=profile_data['name'],
                    type=ConnectionType(profile_data['type']),
                    environment=EnvironmentType(profile_data['environment']),
                    connection_string=profile_data['connection_string'],
                    timeout=profile_data.get('timeout', 30),
                    retry_attempts=profile_data.get('retry_attempts', 3),
                    fallback_source=profile_data.get('fallback_source'),
                    authentication=profile_data.get('authentication'),
                    parameters=profile_data.get('parameters', {}),
                    enabled=profile_data.get('enabled', True),
                    priority=profile_data.get('priority', 1),
                    description=profile_data.get('description', '')
                )
                self.profiles[profile.name] = profile
            except Exception as e:
                print(f"Warning: Could not parse profile {profile_data.get('name', 'unknown')}: {e}")
    
    def _create_default_profiles(self):
        """Create default connection profiles"""
        default_profiles = [
                                # PostgreSQL Database Profile
                    ConnectionProfile(
                        name="vectorqa_postgresql",
                        type=ConnectionType.POSTGRESQL,
                        environment=EnvironmentType.DEVELOPMENT,  # Changed to match current environment
                        connection_string="postgresql://vectorqa_user:REMOVED_PASSWORD@vectorqa-cluster.cluster-cu562e4m02nq.us-east-1.rds.amazonaws.com:5432/vectorqa",
                        timeout=30,
                        retry_attempts=3,
                        enabled=True,
                        priority=1,
                        description="VectorQA PostgreSQL database for DataMart operations"
                    ),
            
            # Local Development Profile
            ConnectionProfile(
                name="local_development",
                type=ConnectionType.LOCAL,
                environment=EnvironmentType.DEVELOPMENT,
                connection_string="local://dev_configs/",
                timeout=5,
                retry_attempts=1,
                enabled=True,
                priority=2,
                description="Local development environment for testing"
            ),
            
            # S3 Profile (when credentials available)
            ConnectionProfile(
                name="vectorqa_s3",
                type=ConnectionType.S3,
                environment=EnvironmentType.PRODUCTION,
                connection_string="s3://vectorqa-datamart/",
                timeout=60,
                retry_attempts=3,
                enabled=False,  # Disabled until credentials configured
                priority=3,
                description="VectorQA S3 bucket for DataMart storage"
            ),
            
                                # Redis Cache Profile
                    ConnectionProfile(
                        name="vectorqa_redis",
                        type=ConnectionType.REDIS,
                        environment=EnvironmentType.DEVELOPMENT,  # Changed to match current environment
                        connection_string="redis://localhost:6379/0",
                        timeout=10,
                        retry_attempts=2,
                        enabled=False,  # Disabled until Redis available
                        priority=4,
                        description="Redis cache for DataMart performance"
                    ),
                    
                    # HTTP API Profile
                    ConnectionProfile(
                        name="vectorqa_http_api",
                        type=ConnectionType.HTTP,
                        environment=EnvironmentType.DEVELOPMENT,
                        connection_string="https://api.vectorqa.com/v1/",
                        timeout=30,
                        retry_attempts=3,
                        enabled=True,
                        priority=5,
                        description="VectorQA HTTP API for external data access"
                    ),
                    
                    # DataMart Live Profile
                    ConnectionProfile(
                        name="datamart_live_production",
                        type=ConnectionType.DATAMART_LIVE,
                        environment=EnvironmentType.DEVELOPMENT,
                        connection_string="datamart://live/vectorqa/",
                        timeout=60,
                        retry_attempts=3,
                        enabled=True,
                        priority=6,
                        description="Live DataMart connection for real-time data processing"
                    ),
                    
                    # DataMart Stream Profile
                    ConnectionProfile(
                        name="datamart_stream_processing",
                        type=ConnectionType.DATAMART_STREAM,
                        environment=EnvironmentType.DEVELOPMENT,
                        connection_string="datamart://stream/vectorqa/",
                        timeout=120,
                        retry_attempts=5,
                        enabled=True,
                        priority=7,
                        description="Streaming DataMart connection for continuous data processing"
                    )
        ]
        
        for profile in default_profiles:
            self.profiles[profile.name] = profile
    
    def get_profiles_for_environment(self, env_type: Optional[EnvironmentType] = None) -> List[ConnectionProfile]:
        """Get profiles for specific environment"""
        if env_type is None:
            env_type = self.active_environment
        
        return [
            profile for profile in self.profiles.values()
            if profile.environment == env_type and profile.enabled
        ]
    
    def get_profiles_by_type(self, connection_type: ConnectionType) -> List[ConnectionProfile]:
        """Get profiles by connection type"""
        return [
            profile for profile in self.profiles.values()
            if profile.type == connection_type and profile.enabled
        ]
    
    def get_best_profile(self, connection_type: ConnectionType, env_type: Optional[EnvironmentType] = None) -> Optional[ConnectionProfile]:
        """Get the best (highest priority) profile for a connection type"""
        if env_type is None:
            env_type = self.active_environment
        
        available_profiles = [
            profile for profile in self.profiles.values()
            if (profile.type == connection_type and 
                profile.environment == env_type and 
                profile.enabled)
        ]
        
        if not available_profiles:
            return None
        
        # Return highest priority profile
        return max(available_profiles, key=lambda p: p.priority)
    
    def get_all_profiles(self) -> Dict[str, ConnectionProfile]:
        """Get all profiles"""
        return self.profiles.copy()
    
    def add_profile(self, profile: ConnectionProfile):
        """Add a new connection profile"""
        self.profiles[profile.name] = profile
    
    def remove_profile(self, profile_name: str):
        """Remove a connection profile"""
        if profile_name in self.profiles:
            del self.profiles[profile_name]
    
    def enable_profile(self, profile_name: str):
        """Enable a connection profile"""
        if profile_name in self.profiles:
            self.profiles[profile_name].enabled = True
    
    def disable_profile(self, profile_name: str):
        """Disable a connection profile"""
        if profile_name in self.profiles:
            self.profiles[profile_name].enabled = False
    
    def test_profile(self, profile_name: str) -> Dict[str, Any]:
        """Test a connection profile"""
        if profile_name not in self.profiles:
            return {"success": False, "error": "Profile not found"}
        
        profile = self.profiles[profile_name]
        
        try:
            if profile.type == ConnectionType.POSTGRESQL:
                return self._test_postgresql_connection(profile)
            elif profile.type == ConnectionType.S3:
                return self._test_s3_connection(profile)
            elif profile.type == ConnectionType.REDIS:
                return self._test_redis_connection(profile)
            elif profile.type == ConnectionType.LOCAL:
                return self._test_local_connection(profile)
            else:
                return {"success": False, "error": f"Unsupported connection type: {profile.type}"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _test_postgresql_connection(self, profile: ConnectionProfile) -> Dict[str, Any]:
        """Test PostgreSQL connection"""
        try:
            import psycopg2
            conn = psycopg2.connect(profile.connection_string)
            conn.close()
            return {"success": True, "message": "PostgreSQL connection successful"}
        except Exception as e:
            return {"success": False, "error": f"PostgreSQL connection failed: {e}"}
    
    def _test_s3_connection(self, profile: ConnectionProfile) -> Dict[str, Any]:
        """Test S3 connection"""
        try:
            import boto3
            s3 = boto3.client('s3')
            # Try to list buckets to test connection
            s3.list_buckets()
            return {"success": True, "message": "S3 connection successful"}
        except Exception as e:
            return {"success": False, "error": f"S3 connection failed: {e}"}
    
    def _test_redis_connection(self, profile: ConnectionProfile) -> Dict[str, Any]:
        """Test Redis connection"""
        try:
            import redis
            r = redis.from_url(profile.connection_string)
            r.ping()
            return {"success": True, "message": "Redis connection successful"}
        except Exception as e:
            return {"success": False, "error": f"Redis connection failed: {e}"}
    
    def _test_local_connection(self, profile: ConnectionProfile) -> Dict[str, Any]:
        """Test local connection"""
        try:
            path = Path(profile.connection_string.replace("local://", ""))
            if path.exists():
                return {"success": True, "message": "Local path exists"}
            else:
                return {"success": False, "error": "Local path does not exist"}
        except Exception as e:
            return {"success": False, "error": f"Local connection failed: {e}"}
    
    def get_connection_summary(self) -> Dict[str, Any]:
        """Get summary of all connections"""
        summary = {
            "active_environment": self.active_environment.value,
            "total_profiles": len(self.profiles),
            "enabled_profiles": len([p for p in self.profiles.values() if p.enabled]),
            "profiles_by_type": {},
            "profiles_by_environment": {}
        }
        
        # Group by type
        for profile in self.profiles.values():
            if profile.type.value not in summary["profiles_by_type"]:
                summary["profiles_by_type"][profile.type.value] = []
            summary["profiles_by_type"][profile.type.value].append({
                "name": profile.name,
                "enabled": profile.enabled,
                "priority": profile.priority
            })
        
        # Group by environment
        for profile in self.profiles.values():
            if profile.environment.value not in summary["profiles_by_environment"]:
                summary["profiles_by_environment"][profile.environment.value] = []
            summary["profiles_by_environment"][profile.environment.value].append({
                "name": profile.name,
                "type": profile.type.value,
                "enabled": profile.enabled
            })
        
        return summary


# Convenience function for DataMart to use
def get_connection_profile_manager() -> ConnectionProfileManager:
    """Get connection profile manager instance"""
    return ConnectionProfileManager()


def get_best_connection_profile(connection_type: ConnectionType) -> Optional[ConnectionProfile]:
    """Get best connection profile for DataMart"""
    manager = get_connection_profile_manager()
    return manager.get_best_profile(connection_type)


if __name__ == "__main__":
    # Test the connection profile system
    manager = ConnectionProfileManager()
    
    print("🔧 Connection Profile Manager Test")
    print("==================================")
    print(f"Active Environment: {manager.active_environment.value}")
    print(f"Total Profiles: {len(manager.profiles)}")
    
    # Test PostgreSQL profile
    postgres_profile = manager.get_best_profile(ConnectionType.POSTGRESQL)
    if postgres_profile:
        print(f"\n✅ Best PostgreSQL Profile: {postgres_profile.name}")
        test_result = manager.test_profile(postgres_profile.name)
        print(f"   Test Result: {test_result}")
    
    # Get summary
    summary = manager.get_connection_summary()
    print(f"\n📊 Connection Summary:")
    print(f"   Enabled Profiles: {summary['enabled_profiles']}")
    print(f"   Profiles by Type: {list(summary['profiles_by_type'].keys())}")
