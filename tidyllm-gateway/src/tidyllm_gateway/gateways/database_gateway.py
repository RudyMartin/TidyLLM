#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database Gateway - Corporate-Controlled Database Access

Enterprise database gateway that routes all database requests through 
corporate IT infrastructure with comprehensive governance, security, 
and compliance controls.

Key Features:
- Zero direct database connections from applications
- IT-controlled connection strings and credentials
- Query validation and SQL injection prevention
- Data classification and access controls
- Connection pooling and failover
- Full audit trails for compliance
"""

from typing import Dict, Any, Optional, List, Union, Tuple
import json
import re
from datetime import datetime
from dataclasses import dataclass
from ..core.base_gateway import BaseGateway, GatewayConfig, GatewayResponse

# Database drivers (optional - loaded dynamically)
try:
    import psycopg2
    import psycopg2.pool
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False

try:
    import pyodbc
    SQLSERVER_AVAILABLE = True
except ImportError:
    SQLSERVER_AVAILABLE = False


@dataclass
class DatabaseConnection:
    """Database connection configuration"""
    name: str                           # IT-managed connection name
    connection_type: str                # postgres, sqlserver, oracle, mysql
    connection_string: str              # IT-managed connection string
    max_connections: int = 10
    connection_timeout: int = 30
    query_timeout: int = 300
    read_only: bool = False
    data_classification: str = "internal"  # public, internal, confidential, restricted


class DatabaseGatewayConfig(GatewayConfig):
    """Configuration specific to Database Gateway"""
    
    # Available connections (IT managed)
    available_connections: Dict[str, DatabaseConnection] = None
    
    # Query controls
    max_query_length: int = 10000
    max_result_rows: int = 100000
    allowed_operations: List[str] = None  # SELECT, INSERT, UPDATE, DELETE
    blocked_patterns: List[str] = None   # SQL patterns to block
    
    # Data protection
    require_where_clause: bool = True    # Prevent accidental full table scans
    sanitize_results: bool = True        # Remove PII from results
    log_query_plans: bool = False        # Log execution plans for optimization
    
    def __post_init__(self):
        if self.available_connections is None:
            self.available_connections = {}
        if self.allowed_operations is None:
            self.allowed_operations = ["SELECT"]  # Default to read-only
        if self.blocked_patterns is None:
            self.blocked_patterns = [
                r";\s*(DROP|ALTER|CREATE|TRUNCATE)",  # DDL operations
                r"xp_cmdshell",                       # SQL Server command execution
                r"sp_execute",                        # Dynamic SQL execution
                r"UNION.*SELECT.*FROM.*information_schema",  # Schema enumeration
            ]


class DatabaseGateway(BaseGateway):
    """
    Corporate Database Gateway - Centralized database access
    
    Routes all database requests through corporate IT infrastructure with:
    - Connection controls (what databases are available)
    - Query controls (SQL validation, operation restrictions)
    - Data controls (classification, PII protection)
    - Audit controls (full query logging)
    - Security controls (no direct database access)
    
    Architecture:
    Application → Database Gateway → Corporate IT → Database Servers
    """
    
    def __init__(self, config: DatabaseGatewayConfig):
        super().__init__(config)
        self.db_config = config
        
        # Connection pools by connection name
        self.connection_pools: Dict[str, Any] = {}
        
        # Query statistics
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "avg_execution_time_ms": 0.0
        }
        
        # Initialize connection pools
        self._initialize_connection_pools()
        
        logger.info("🗄️ Database Gateway initialized")
        logger.info(f"   Available connections: {list(config.available_connections.keys())}")
    
    def _initialize_connection_pools(self):
        """Initialize connection pools for available connections"""
        
        for conn_name, conn_config in self.db_config.available_connections.items():
            try:
                if conn_config.connection_type == "postgres" and POSTGRES_AVAILABLE:
                    pool = psycopg2.pool.SimpleConnectionPool(
                        1, conn_config.max_connections,
                        conn_config.connection_string
                    )
                    self.connection_pools[conn_name] = pool
                    logger.info(f"✅ PostgreSQL pool created for {conn_name}")
                
                # Add other database types as needed
                elif conn_config.connection_type == "sqlserver" and SQLSERVER_AVAILABLE:
                    # SQL Server connection pool would be implemented here
                    logger.info(f"⚠️ SQL Server pool not implemented for {conn_name}")
                
                else:
                    logger.warning(f"⚠️ Unsupported database type for {conn_name}: {conn_config.connection_type}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to create connection pool for {conn_name}: {e}")
    
    def _execute_request(self, endpoint: str, data: Dict[str, Any], **kwargs) -> Any:
        """Execute database request through corporate gateway"""
        
        # Parse request parameters
        connection_name = data.get("connection_name")
        query = data.get("query", "")
        parameters = data.get("parameters", [])
        operation_type = data.get("operation_type", "query")  # query, execute, procedure
        
        # Validate request
        self._validate_database_request(connection_name, query, parameters, operation_type)
        
        # Execute query
        if operation_type == "query":
            return self._execute_query(connection_name, query, parameters)
        elif operation_type == "execute":
            return self._execute_command(connection_name, query, parameters)
        elif operation_type == "procedure":
            return self._execute_procedure(connection_name, query, parameters)
        else:
            raise ValueError(f"Unsupported operation type: {operation_type}")
    
    def _validate_database_request(
        self, 
        connection_name: str, 
        query: str, 
        parameters: List[Any], 
        operation_type: str
    ):
        """Validate database request against corporate policies"""
        
        # Check connection availability
        if not connection_name:
            raise ValueError("Connection name is required")
        
        if connection_name not in self.db_config.available_connections:
            available = list(self.db_config.available_connections.keys())
            raise ValueError(f"Connection '{connection_name}' not available. IT approved: {available}")
        
        # Check query length
        if len(query) > self.db_config.max_query_length:
            raise ValueError(f"Query length {len(query)} exceeds limit: {self.db_config.max_query_length}")
        
        # Check allowed operations
        query_upper = query.upper().strip()
        operation = self._extract_sql_operation(query_upper)
        
        if operation not in self.db_config.allowed_operations:
            raise ValueError(f"Operation '{operation}' not allowed. Permitted: {self.db_config.allowed_operations}")
        
        # Check for blocked patterns
        for pattern in self.db_config.blocked_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                raise ValueError(f"Query contains blocked pattern: {pattern}")
        
        # Check WHERE clause requirement for SELECT statements
        if self.db_config.require_where_clause and operation == "SELECT":
            if not re.search(r'\bWHERE\b', query, re.IGNORECASE):
                # Allow certain exceptions (aggregation queries, system tables, etc.)
                if not self._is_where_clause_exception(query):
                    raise ValueError("WHERE clause required for SELECT statements to prevent full table scans")
        
        # Validate parameters to prevent SQL injection
        self._validate_query_parameters(parameters)
    
    def _extract_sql_operation(self, query: str) -> str:
        """Extract the primary SQL operation from query"""
        
        # Common SQL operations
        operations = ["SELECT", "INSERT", "UPDATE", "DELETE", "CALL", "EXEC", "EXECUTE"]
        
        for op in operations:
            if query.startswith(op):
                return op
        
        return "UNKNOWN"
    
    def _is_where_clause_exception(self, query: str) -> bool:
        """Check if query is exempt from WHERE clause requirement"""
        
        query_upper = query.upper()
        
        # Aggregation queries
        if any(func in query_upper for func in ["COUNT(*)", "SUM(", "AVG(", "MIN(", "MAX("]):
            return True
        
        # System/metadata queries
        if any(table in query_upper for table in ["INFORMATION_SCHEMA", "SYS.", "PG_"]):
            return True
        
        # LIMIT/TOP clauses
        if re.search(r'\b(LIMIT|TOP)\s+\d+\b', query_upper):
            return True
        
        return False
    
    def _validate_query_parameters(self, parameters: List[Any]):
        """Validate query parameters to prevent injection"""
        
        for i, param in enumerate(parameters):
            if isinstance(param, str):
                # Check for SQL injection patterns
                dangerous_patterns = [
                    r"'.*OR.*'", r"'.*UNION.*'", r"'.*--", r"'.*;",
                    r"xp_cmdshell", r"sp_execute"
                ]
                
                for pattern in dangerous_patterns:
                    if re.search(pattern, param, re.IGNORECASE):
                        raise ValueError(f"Parameter {i} contains potentially dangerous pattern")
    
    def _execute_query(self, connection_name: str, query: str, parameters: List[Any]) -> Dict[str, Any]:
        """Execute SELECT query and return results"""
        
        connection_config = self.db_config.available_connections[connection_name]
        start_time = datetime.utcnow()
        
        try:
            # Get connection from pool
            connection = self._get_connection(connection_name)
            
            # Execute query
            cursor = connection.cursor()
            
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            # Fetch results
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            rows = cursor.fetchall()
            
            # Check result size limits
            if len(rows) > self.db_config.max_result_rows:
                logger.warning(f"Query returned {len(rows)} rows, limiting to {self.db_config.max_result_rows}")
                rows = rows[:self.db_config.max_result_rows]
            
            cursor.close()
            self._return_connection(connection_name, connection)
            
            # Process results
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            
            # Convert to list of dictionaries
            results = [dict(zip(columns, row)) for row in rows]
            
            # Sanitize results if needed
            if self.db_config.sanitize_results:
                results = self._sanitize_query_results(results, connection_config.data_classification)
            
            # Update statistics
            self._update_query_stats(True, execution_time)
            
            return {
                "success": True,
                "connection_name": connection_name,
                "query_type": "SELECT",
                "columns": columns,
                "rows": results,
                "row_count": len(results),
                "execution_time_ms": execution_time,
                "data_classification": connection_config.data_classification,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._update_query_stats(False, 0)
            raise e
    
    def _execute_command(self, connection_name: str, query: str, parameters: List[Any]) -> Dict[str, Any]:
        """Execute INSERT/UPDATE/DELETE command"""
        
        connection_config = self.db_config.available_connections[connection_name]
        
        # Check read-only restriction
        if connection_config.read_only:
            raise ValueError(f"Connection '{connection_name}' is read-only")
        
        start_time = datetime.utcnow()
        
        try:
            # Get connection from pool
            connection = self._get_connection(connection_name)
            
            # Execute command
            cursor = connection.cursor()
            
            if parameters:
                cursor.execute(query, parameters)
            else:
                cursor.execute(query)
            
            # Get affected rows
            affected_rows = cursor.rowcount
            
            # Commit transaction
            connection.commit()
            
            cursor.close()
            self._return_connection(connection_name, connection)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_query_stats(True, execution_time)
            
            return {
                "success": True,
                "connection_name": connection_name,
                "query_type": self._extract_sql_operation(query.upper()),
                "affected_rows": affected_rows,
                "execution_time_ms": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._update_query_stats(False, 0)
            # Rollback on error
            try:
                connection.rollback()
                self._return_connection(connection_name, connection)
            except:
                pass
            raise e
    
    def _execute_procedure(self, connection_name: str, procedure_name: str, parameters: List[Any]) -> Dict[str, Any]:
        """Execute stored procedure"""
        
        connection_config = self.db_config.available_connections[connection_name]
        start_time = datetime.utcnow()
        
        try:
            # Get connection from pool
            connection = self._get_connection(connection_name)
            
            # Execute procedure
            cursor = connection.cursor()
            cursor.callproc(procedure_name, parameters)
            
            # Fetch results if any
            results = []
            if cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                results = [dict(zip(columns, row)) for row in rows]
            
            cursor.close()
            self._return_connection(connection_name, connection)
            
            execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_query_stats(True, execution_time)
            
            return {
                "success": True,
                "connection_name": connection_name,
                "query_type": "PROCEDURE",
                "procedure_name": procedure_name,
                "results": results,
                "execution_time_ms": execution_time,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            self._update_query_stats(False, 0)
            raise e
    
    def _get_connection(self, connection_name: str):
        """Get connection from pool"""
        
        if connection_name in self.connection_pools:
            pool = self.connection_pools[connection_name]
            return pool.getconn()
        else:
            # Fallback for connections without pools
            conn_config = self.db_config.available_connections[connection_name]
            
            if conn_config.connection_type == "postgres" and POSTGRES_AVAILABLE:
                import psycopg2
                return psycopg2.connect(conn_config.connection_string)
            else:
                raise ValueError(f"No connection pool or driver available for {connection_name}")
    
    def _return_connection(self, connection_name: str, connection):
        """Return connection to pool"""
        
        if connection_name in self.connection_pools:
            pool = self.connection_pools[connection_name]
            pool.putconn(connection)
        else:
            # Close direct connection
            connection.close()
    
    def _sanitize_query_results(self, results: List[Dict[str, Any]], data_classification: str) -> List[Dict[str, Any]]:
        """Sanitize query results based on data classification"""
        
        if data_classification in ["public", "internal"]:
            return results  # No sanitization needed
        
        # For confidential/restricted data, apply PII protection
        sanitized_results = []
        
        for row in results:
            sanitized_row = {}
            for column, value in row.items():
                
                # Check if column contains PII
                if self._is_pii_column(column):
                    sanitized_row[column] = self._mask_pii_value(value)
                else:
                    sanitized_row[column] = value
            
            sanitized_results.append(sanitized_row)
        
        return sanitized_results
    
    def _is_pii_column(self, column_name: str) -> bool:
        """Check if column name indicates PII"""
        
        pii_indicators = [
            'ssn', 'social_security', 'tax_id', 'credit_card', 'card_number',
            'phone', 'email', 'address', 'dob', 'birth_date', 'driver_license'
        ]
        
        column_lower = column_name.lower()
        return any(indicator in column_lower for indicator in pii_indicators)
    
    def _mask_pii_value(self, value: Any) -> str:
        """Mask PII value for protection"""
        
        if value is None:
            return None
        
        str_value = str(value)
        
        # Mask all but last 4 characters for long values
        if len(str_value) > 4:
            return "***" + str_value[-4:]
        else:
            return "***"
    
    def _update_query_stats(self, success: bool, execution_time_ms: float):
        """Update query execution statistics"""
        
        self.query_stats["total_queries"] += 1
        
        if success:
            self.query_stats["successful_queries"] += 1
        else:
            self.query_stats["failed_queries"] += 1
        
        # Update average execution time
        if success and execution_time_ms > 0:
            total_successful = self.query_stats["successful_queries"]
            current_avg = self.query_stats["avg_execution_time_ms"]
            
            self.query_stats["avg_execution_time_ms"] = (
                (current_avg * (total_successful - 1) + execution_time_ms) / total_successful
            )
    
    def execute_query(
        self,
        connection_name: str,
        query: str,
        parameters: Optional[List[Any]] = None,
        user_id: str = None,
        audit_reason: Optional[str] = None,
        **kwargs
    ) -> GatewayResponse:
        """
        Execute database query through corporate gateway
        
        Args:
            connection_name: IT-managed connection name
            query: SQL query to execute
            parameters: Query parameters (for prepared statements)
            user_id: User making request (for audit)
            audit_reason: Reason for request (compliance)
            
        Returns:
            Standardized gateway response
        """
        
        # Prepare request data
        request_data = {
            "connection_name": connection_name,
            "query": query,
            "parameters": parameters or [],
            "operation_type": "query",
            **kwargs
        }
        
        # Execute through base gateway (includes audit logging, rate limiting, etc.)
        return self.execute(
            endpoint="query",
            data=request_data,
            user_id=user_id or "system",
            audit_reason=audit_reason
        )
    
    def get_available_connections(self) -> List[Dict[str, Any]]:
        """Get IT-approved available database connections"""
        
        connections = []
        for name, config in self.db_config.available_connections.items():
            connections.append({
                "name": name,
                "type": config.connection_type,
                "read_only": config.read_only,
                "data_classification": config.data_classification,
                "max_connections": config.max_connections,
                "pool_available": name in self.connection_pools
            })
        
        return connections
    
    def get_query_statistics(self) -> Dict[str, Any]:
        """Get database query statistics"""
        
        stats = self.query_stats.copy()
        
        # Calculate success rate
        total = stats["total_queries"]
        if total > 0:
            stats["success_rate"] = stats["successful_queries"] / total * 100
        else:
            stats["success_rate"] = 0
        
        stats["timestamp"] = datetime.utcnow().isoformat()
        
        return stats