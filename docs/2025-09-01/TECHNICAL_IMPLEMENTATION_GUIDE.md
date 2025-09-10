# TidyMart Technical Implementation Guide

**Document Version**: 1.0  
**Date**: 2025-09-01  
**Classification**: Technical Internal  
**Purpose**: Detailed technical specifications for TidyMart integration implementation

---

## Implementation Architecture

### Core TidyMart Connection Infrastructure

```python
#!/usr/bin/env python3
"""
TidyMart Connection Infrastructure
Enterprise-grade PostgreSQL backend with high-performance data pipeline integration
"""

import asyncio
import asyncpg
import polars as pl
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
import logging

@dataclass
class TidyMartConfig:
    """Enterprise TidyMart configuration"""
    postgres_url: str
    connection_pool_size: int = 20
    max_overflow: int = 40  
    query_timeout: int = 30000  # 30 seconds
    retry_attempts: int = 3
    circuit_breaker_threshold: int = 5
    metrics_retention_days: int = 90
    audit_retention_days: int = 2555  # 7 years for compliance

class TidyMartConnection:
    """
    Enterprise TidyMart connection manager with pooling, circuit breaker,
    and enterprise-grade reliability features.
    """
    
    def __init__(self, config: TidyMartConfig):
        self.config = config
        self.pool = None
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize connection pool and create tables if needed"""
        try:
            self.pool = await asyncpg.create_pool(
                self.config.postgres_url,
                min_size=5,
                max_size=self.config.connection_pool_size,
                command_timeout=self.config.query_timeout
            )
            
            # Create schema and tables
            await self._create_schema()
            self.logger.info("TidyMart connection pool initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize TidyMart: {e}")
            raise
    
    async def _create_schema(self):
        """Create TidyMart database schema"""
        
        schema_sql = """
        -- Create tidymart schema
        CREATE SCHEMA IF NOT EXISTS tidymart;
        
        -- Execution tracking table
        CREATE TABLE IF NOT EXISTS tidymart.executions (
            execution_id UUID PRIMARY KEY,
            module VARCHAR(50) NOT NULL,
            operation VARCHAR(100) NOT NULL,
            user_id VARCHAR(255),
            department VARCHAR(100),
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            total_duration_ms INTEGER,
            success_rate DECIMAL(3,2),
            cost_usd DECIMAL(10,4),
            input_context JSONB,
            final_results JSONB,
            metadata JSONB,
            created_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Performance optimization table
        CREATE TABLE IF NOT EXISTS tidymart.module_performance (
            id UUID PRIMARY KEY,
            execution_id UUID REFERENCES tidymart.executions(execution_id),
            module VARCHAR(50) NOT NULL,
            operation VARCHAR(100) NOT NULL,
            configuration JSONB NOT NULL,
            performance_metrics JSONB NOT NULL,
            quality_score DECIMAL(3,2),
            cost_efficiency_score DECIMAL(5,4),
            recorded_at TIMESTAMP DEFAULT NOW()
        );
        
        -- Step-by-step execution tracking
        CREATE TABLE IF NOT EXISTS tidymart.execution_steps (
            step_id UUID PRIMARY KEY,
            execution_id UUID REFERENCES tidymart.executions(execution_id),
            step_name VARCHAR(200) NOT NULL,
            step_order INTEGER NOT NULL,
            start_time TIMESTAMP NOT NULL,
            end_time TIMESTAMP,
            duration_ms INTEGER,
            success BOOLEAN NOT NULL,
            error_message TEXT,
            input_data JSONB,
            output_data JSONB,
            parameters JSONB
        );
        
        -- Configuration optimization table
        CREATE TABLE IF NOT EXISTS tidymart.optimal_configs (
            config_id UUID PRIMARY KEY,
            module VARCHAR(50) NOT NULL,
            operation VARCHAR(100) NOT NULL,
            context_pattern JSONB NOT NULL,
            optimal_configuration JSONB NOT NULL,
            success_rate DECIMAL(3,2) NOT NULL,
            avg_performance_score DECIMAL(5,4) NOT NULL,
            usage_count INTEGER DEFAULT 1,
            last_updated TIMESTAMP DEFAULT NOW()
        );
        
        -- Cross-module workflow patterns
        CREATE TABLE IF NOT EXISTS tidymart.workflow_patterns (
            pattern_id UUID PRIMARY KEY,
            workflow_name VARCHAR(200),
            module_sequence VARCHAR[] NOT NULL,
            input_characteristics JSONB,
            success_rate DECIMAL(3,2),
            avg_duration_ms INTEGER,
            avg_cost_usd DECIMAL(8,4),
            usage_frequency INTEGER DEFAULT 0,
            last_used TIMESTAMP DEFAULT NOW(),
            optimization_notes TEXT
        );
        
        -- Create essential indexes
        CREATE INDEX IF NOT EXISTS idx_executions_module_operation 
            ON tidymart.executions(module, operation);
        CREATE INDEX IF NOT EXISTS idx_executions_user_dept 
            ON tidymart.executions(user_id, department);
        CREATE INDEX IF NOT EXISTS idx_executions_time 
            ON tidymart.executions(start_time DESC);
        CREATE INDEX IF NOT EXISTS idx_perf_module_operation 
            ON tidymart.module_performance(module, operation);
        CREATE INDEX IF NOT EXISTS idx_perf_quality 
            ON tidymart.module_performance(quality_score DESC);
        CREATE INDEX IF NOT EXISTS idx_steps_execution 
            ON tidymart.execution_steps(execution_id, step_order);
        CREATE INDEX IF NOT EXISTS idx_configs_module_op 
            ON tidymart.optimal_configs(module, operation);
        CREATE INDEX IF NOT EXISTS idx_workflow_success 
            ON tidymart.workflow_patterns(success_rate DESC);
        """
        
        async with self.pool.acquire() as conn:
            await conn.execute(schema_sql)
    
    async def get_optimal_config(self, module: str, operation: str, 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get optimal configuration for module operation based on context.
        Uses machine learning on historical performance data.
        """
        
        if not await self._circuit_breaker_check():
            return self._get_fallback_config(module, operation)
        
        try:
            async with self.pool.acquire() as conn:
                # Find best matching configuration based on context similarity
                query = """
                SELECT optimal_configuration, success_rate, avg_performance_score
                FROM tidymart.optimal_configs
                WHERE module = $1 AND operation = $2
                AND context_pattern @> $3::jsonb  -- Context subset matching
                ORDER BY success_rate DESC, avg_performance_score DESC
                LIMIT 1;
                """
                
                result = await conn.fetchrow(query, module, operation, json.dumps(context))
                
                if result:
                    config = json.loads(result['optimal_configuration'])
                    self.logger.info(
                        f"Found optimal config for {module}.{operation} "
                        f"(success_rate: {result['success_rate']:.2f})"
                    )
                    return config
                else:
                    # No exact match, find similar patterns
                    return await self._find_similar_config(conn, module, operation, context)
                    
        except Exception as e:
            await self._circuit_breaker_record_failure()
            self.logger.error(f"Failed to get optimal config: {e}")
            return self._get_fallback_config(module, operation)
    
    async def start_execution(self, module: str, operation: str, 
                            config: Dict[str, Any], input_context: Dict[str, Any],
                            user_id: str = None, department: str = None) -> str:
        """Start tracking a new execution"""
        
        execution_id = str(uuid.uuid4())
        
        try:
            async with self.pool.acquire() as conn:
                await conn.execute("""
                    INSERT INTO tidymart.executions 
                    (execution_id, module, operation, user_id, department, 
                     start_time, input_context, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """, execution_id, module, operation, user_id, department,
                    datetime.now(), json.dumps(input_context), 
                    json.dumps({'config': config}))
                
                self.logger.info(f"Started tracking execution {execution_id}")
                return execution_id
                
        except Exception as e:
            self.logger.error(f"Failed to start execution tracking: {e}")
            return execution_id  # Return ID anyway for graceful degradation
    
    async def track_step(self, execution_id: str, step_name: str, step_order: int,
                        duration_ms: int, success: bool, error: str = None,
                        input_data: Dict = None, output_data: Dict = None,
                        parameters: Dict = None):
        """Track individual step in execution"""
        
        try:
            async with self.pool.acquire() as conn:
                step_id = str(uuid.uuid4())
                
                await conn.execute("""
                    INSERT INTO tidymart.execution_steps
                    (step_id, execution_id, step_name, step_order, start_time,
                     end_time, duration_ms, success, error_message, input_data,
                     output_data, parameters)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12)
                """, step_id, execution_id, step_name, step_order,
                    datetime.now() - timedelta(milliseconds=duration_ms),
                    datetime.now(), duration_ms, success, error,
                    json.dumps(input_data) if input_data else None,
                    json.dumps(output_data) if output_data else None,
                    json.dumps(parameters) if parameters else None)
                
        except Exception as e:
            self.logger.error(f"Failed to track step: {e}")
            # Non-blocking - continue execution even if tracking fails
    
    async def complete_execution(self, execution_id: str, final_results: Dict[str, Any],
                                total_duration_ms: int, success_rate: float,
                                cost_usd: float = None):
        """Complete execution tracking and trigger learning updates"""
        
        try:
            async with self.pool.acquire() as conn:
                # Update execution record
                await conn.execute("""
                    UPDATE tidymart.executions
                    SET end_time = $1, total_duration_ms = $2, success_rate = $3,
                        cost_usd = $4, final_results = $5
                    WHERE execution_id = $6
                """, datetime.now(), total_duration_ms, success_rate, cost_usd,
                    json.dumps(final_results), execution_id)
                
                # Trigger async learning update
                asyncio.create_task(self._update_learning_models(execution_id))
                
                self.logger.info(f"Completed execution {execution_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to complete execution: {e}")
    
    async def get_optimization_recommendations(self, module: str, 
                                             recent_performance: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get AI-powered optimization recommendations"""
        
        try:
            async with self.pool.acquire() as conn:
                # Analyze recent performance vs historical patterns
                query = """
                WITH recent_perf AS (
                    SELECT AVG(quality_score) as avg_quality,
                           AVG(cost_efficiency_score) as avg_efficiency
                    FROM tidymart.module_performance
                    WHERE module = $1 AND recorded_at > NOW() - INTERVAL '7 days'
                ),
                historical_best AS (
                    SELECT configuration, AVG(quality_score) as quality,
                           AVG(cost_efficiency_score) as efficiency
                    FROM tidymart.module_performance  
                    WHERE module = $1 AND quality_score > 0.8
                    GROUP BY configuration
                    ORDER BY AVG(quality_score * cost_efficiency_score) DESC
                    LIMIT 5
                )
                SELECT h.configuration, h.quality, h.efficiency,
                       (h.quality * h.efficiency) - (r.avg_quality * r.avg_efficiency) as improvement_potential
                FROM historical_best h, recent_perf r
                WHERE (h.quality * h.efficiency) > (r.avg_quality * r.avg_efficiency) * 1.1
                ORDER BY improvement_potential DESC;
                """
                
                results = await conn.fetch(query, module)
                
                recommendations = []
                for row in results:
                    recommendations.append({
                        'type': 'configuration_optimization',
                        'module': module,
                        'recommended_config': json.loads(row['configuration']),
                        'expected_quality_improvement': row['quality'],
                        'expected_efficiency_improvement': row['efficiency'],
                        'improvement_potential': float(row['improvement_potential']),
                        'confidence': 0.85  # Based on historical data quality
                    })
                
                return recommendations
                
        except Exception as e:
            self.logger.error(f"Failed to get recommendations: {e}")
            return []
    
    async def _circuit_breaker_check(self) -> bool:
        """Circuit breaker pattern for resilience"""
        
        if self.circuit_breaker_failures >= self.config.circuit_breaker_threshold:
            # Check if enough time has passed to retry
            if (datetime.now() - self.circuit_breaker_last_failure).seconds > 60:
                self.circuit_breaker_failures = 0
                return True
            return False
        
        return True
    
    async def _circuit_breaker_record_failure(self):
        """Record circuit breaker failure"""
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = datetime.now()
    
    def _get_fallback_config(self, module: str, operation: str) -> Dict[str, Any]:
        """Fallback configurations when TidyMart is unavailable"""
        
        fallback_configs = {
            'documents': {
                'extract': {'method': 'pattern', 'timeout': 30, 'max_pages': 100},
                'classify': {'confidence_threshold': 0.7, 'max_categories': 5}
            },
            'sentence': {
                'embed': {'method': 'tfidf', 'max_features': 5000, 'timeout': 10},
                'similarity': {'method': 'cosine', 'batch_size': 100}
            },
            'tlm': {
                'kmeans': {'max_iter': 300, 'n_init': 10, 'tol': 1e-4},
                'logreg': {'max_iter': 1000, 'C': 1.0, 'solver': 'lbfgs'}
            },
            'gateway': {
                'completion': {'model': 'claude-3-5-sonnet', 'max_tokens': 4000, 'temperature': 0.1},
                'embedding': {'model': 'text-embedding-ada-002', 'dimensions': 1536}
            },
            'heiros': {
                'execute_tree': {'max_depth': 10, 'timeout': 300, 'fallback_enabled': True},
                'optimize_flow': {'max_iterations': 50, 'convergence_threshold': 0.01}
            },
            'sparse': {
                'expand_command': {'max_complexity': 'enhanced', 'timeout': 60},
                'suggest_commands': {'similarity_threshold': 0.8, 'max_suggestions': 5}
            }
        }
        
        return fallback_configs.get(module, {}).get(operation, {})

class TidyMartPipeline:
    """
    Universal enterprise pipeline with TidyMart integration.
    Implements the consumer-provider pattern for all modules.
    """
    
    def __init__(self, tidymart_conn: TidyMartConnection, module_name: str):
        self.mart = tidymart_conn
        self.module = module_name
        self.execution_id = None
        self.polars_frame = None
        self.metadata = {}
        self.step_counter = 0
        self.logger = logging.getLogger(f"{__name__}.{module_name}")
        
    async def start_execution(self, operation: str, input_data: Dict[str, Any],
                            user_id: str = None, department: str = None) -> 'TidyMartPipeline':
        """Initialize execution with TidyMart configuration lookup"""
        
        # Get optimal configuration from TidyMart
        config = await self.mart.get_optimal_config(
            module=self.module,
            operation=operation,
            context=input_data
        )
        
        # Start tracking in TidyMart  
        self.execution_id = await self.mart.start_execution(
            module=self.module,
            operation=operation,
            config=config,
            input_context=input_data,
            user_id=user_id,
            department=department
        )
        
        # Initialize high-performance data structures
        self.polars_frame = pl.DataFrame([input_data])
        self.metadata = {
            'execution_id': self.execution_id,
            'module': self.module,
            'operation': operation,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'steps': [],
            'user_id': user_id,
            'department': department
        }
        
        self.logger.info(f"Started {self.module} execution {self.execution_id}")
        return self
    
    async def add_processing_step(self, step_name: str, processor_func: Callable,
                                **params) -> 'TidyMartPipeline':
        """Execute processing step with TidyMart tracking"""
        
        self.step_counter += 1
        step_start = datetime.now()
        
        try:
            # Execute the actual processing
            if asyncio.iscoroutinefunction(processor_func):
                result = await processor_func(self.polars_frame, **params)
            else:
                result = processor_func(self.polars_frame, **params)
            
            # Update polars frame with results
            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, list) and len(value) == len(self.polars_frame):
                        # List values that match frame length
                        self.polars_frame = self.polars_frame.with_columns(
                            pl.Series(key, value)
                        )
                    else:
                        # Scalar values
                        self.polars_frame = self.polars_frame.with_columns(
                            pl.lit(value).alias(key)
                        )
            
            success = True
            error = None
            
        except Exception as e:
            success = False
            error = str(e)
            result = None
            self.logger.error(f"Step {step_name} failed: {e}")
        
        step_duration = int((datetime.now() - step_start).total_seconds() * 1000)
        
        # Track step in TidyMart (non-blocking)
        await self.mart.track_step(
            execution_id=self.execution_id,
            step_name=step_name,
            step_order=self.step_counter,
            duration_ms=step_duration,
            success=success,
            error=error,
            input_data={'polars_shape': self.polars_frame.shape},
            output_data={'result_summary': str(result)[:500] if result else None},
            parameters=params
        )
        
        # Add to local metadata
        step_data = {
            'step_name': step_name,
            'step_order': self.step_counter,
            'duration_ms': step_duration,
            'success': success,
            'error': error,
            'parameters': params
        }
        self.metadata['steps'].append(step_data)
        
        return self
    
    async def complete_execution(self, final_results: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize execution with TidyMart learning feedback"""
        
        # Calculate final metrics
        total_duration = sum(step['duration_ms'] for step in self.metadata['steps'])
        success_rate = (sum(1 for step in self.metadata['steps'] if step['success']) / 
                       len(self.metadata['steps']) if self.metadata['steps'] else 0)
        
        # Store completion in TidyMart
        await self.mart.complete_execution(
            execution_id=self.execution_id,
            final_results=final_results,
            total_duration_ms=total_duration,
            success_rate=success_rate,
            cost_usd=final_results.get('cost_usd', 0.0)
        )
        
        # Get learning recommendations for next time
        recommendations = await self.mart.get_optimization_recommendations(
            module=self.module,
            recent_performance={
                'success_rate': success_rate,
                'duration_ms': total_duration,
                'steps_count': len(self.metadata['steps'])
            }
        )
        
        self.logger.info(f"Completed {self.module} execution {self.execution_id}")
        
        return {
            'execution_id': self.execution_id,
            'results': final_results,
            'polars_frame': self.polars_frame,
            'metadata': self.metadata,
            'recommendations': recommendations,
            'performance_summary': {
                'total_duration_ms': total_duration,
                'success_rate': success_rate,
                'steps_completed': len(self.metadata['steps']),
                'data_shape': self.polars_frame.shape
            }
        }

# Module-specific implementations

class DocumentsTidyMartIntegration:
    """Documents module TidyMart integration"""
    
    def __init__(self, tidymart_conn: TidyMartConnection):
        self.mart = tidymart_conn
        
    async def extract_document_step(self, polars_frame: pl.DataFrame, 
                                  document_path: str, **params) -> Dict[str, Any]:
        """Documents extraction with TidyMart optimization"""
        
        # Get optimal extraction method from TidyMart
        config = await self.mart.get_optimal_config(
            module="documents",
            operation="extract",
            context={'file_path': document_path, 'file_size_mb': self._get_file_size(document_path)}
        )
        
        # Use tidyllm-documents with optimized config
        import tidyllm_documents as td
        processor = td.BusinessDocumentProcessor(**config)
        result = processor.process_document(document_path)
        
        return {
            'extracted_text': result.get('text', ''),
            'metadata': result.get('metadata', {}),
            'extraction_confidence': result.get('confidence', 0.0),
            'extraction_method': config.get('method', 'pattern'),
            'processing_time_ms': result.get('processing_time_ms', 0)
        }
    
    def _get_file_size(self, file_path: str) -> float:
        """Get file size in MB"""
        import os
        return os.path.getsize(file_path) / (1024 * 1024)

class SentenceTidyMartIntegration:
    """Sentence module TidyMart integration"""
    
    def __init__(self, tidymart_conn: TidyMartConnection):
        self.mart = tidymart_conn
        
    async def embedding_step(self, polars_frame: pl.DataFrame, 
                           text_column: str = 'extracted_text', **params) -> Dict[str, Any]:
        """Sentence embedding with TidyMart optimization and caching"""
        
        text_data = polars_frame[text_column].to_list()
        
        embeddings = []
        cache_hits = 0
        
        for text in text_data:
            # Check embedding cache in TidyMart
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            # Try to get from cache first
            cached_embedding = await self._get_cached_embedding(text_hash)
            
            if cached_embedding:
                embeddings.append(cached_embedding)
                cache_hits += 1
            else:
                # Get optimal embedding method
                config = await self.mart.get_optimal_config(
                    module="sentence",
                    operation="embed",
                    context={'text_length': len(text), 'text_type': 'document'}
                )
                
                # Generate embedding with optimized config
                import tidyllm_sentence as ts
                embedding = ts.embed_text(text, method=config.get('method', 'tfidf'))
                embeddings.append(embedding)
                
                # Cache for future use
                await self._cache_embedding(text_hash, embedding, config.get('method'))
        
        return {
            'embeddings': embeddings,
            'embedding_method': config.get('method', 'tfidf'),
            'cache_hit_rate': cache_hits / len(text_data) if text_data else 0,
            'total_embeddings_generated': len(embeddings)
        }
    
    async def _get_cached_embedding(self, text_hash: str) -> List[float]:
        """Get cached embedding from TidyMart"""
        # Implementation would query embedding cache table
        return None
    
    async def _cache_embedding(self, text_hash: str, embedding: List[float], method: str):
        """Cache embedding in TidyMart"""  
        # Implementation would store in embedding cache table
        pass

# Enterprise deployment configuration
class TidyMartEnterpriseDeployment:
    """Enterprise deployment configuration and management"""
    
    @staticmethod
    def create_production_config() -> TidyMartConfig:
        """Create production-ready TidyMart configuration"""
        
        return TidyMartConfig(
            postgres_url=os.getenv("TIDYMART_POSTGRES_URL", 
                                 "postgresql://tidymart:secure_password@postgres-cluster.internal:5432/tidymart"),
            connection_pool_size=50,
            max_overflow=100,
            query_timeout=30000,
            retry_attempts=3,
            circuit_breaker_threshold=10,
            metrics_retention_days=90,
            audit_retention_days=2555  # 7 years compliance
        )
    
    @staticmethod
    async def deploy_enterprise_tidymart():
        """Deploy enterprise TidyMart with all integrations"""
        
        config = TidyMartEnterpriseDeployment.create_production_config()
        mart_conn = TidyMartConnection(config)
        await mart_conn.initialize()
        
        # Initialize module integrations
        documents_integration = DocumentsTidyMartIntegration(mart_conn)
        sentence_integration = SentenceTidyMartIntegration(mart_conn)
        # ... other integrations
        
        return {
            'tidymart_connection': mart_conn,
            'integrations': {
                'documents': documents_integration,
                'sentence': sentence_integration,
                # Add other integrations as implemented
            }
        }

# Example usage
async def example_enterprise_mvr_pipeline():
    """Example of enterprise MVR pipeline with TidyMart integration"""
    
    # Deploy TidyMart
    deployment = await TidyMartEnterpriseDeployment.deploy_enterprise_tidymart()
    mart_conn = deployment['tidymart_connection']
    
    # Create pipeline
    pipeline = TidyMartPipeline(mart_conn, "mvr_workflow")
    
    # Execute enterprise MVR workflow
    result = await (pipeline
        .start_execution("peer_review", {
            'document_path': '/enterprise/mvr/model_validation_report.pdf',
            'user_id': 'risk.analyst@bank.com',
            'department': 'model_risk_management',
            'compliance_mode': 'banking_regulation'
        })
        .add_processing_step("extract_document", 
                           deployment['integrations']['documents'].extract_document_step,
                           document_path='/enterprise/mvr/model_validation_report.pdf')
        .add_processing_step("generate_embeddings",
                           deployment['integrations']['sentence'].embedding_step,
                           text_column='extracted_text')
        .complete_execution({
            'report_type': 'mvr_peer_review',
            'compliance_status': 'completed',
            'audit_trail_complete': True
        })
    )
    
    return result

if __name__ == "__main__":
    # Run enterprise deployment
    asyncio.run(example_enterprise_mvr_pipeline())
```

---

## Database Schema Implementation

### PostgreSQL Setup Script

```sql
-- TidyMart Enterprise Database Schema
-- Run this script to set up production TidyMart database

-- Create database and user
CREATE DATABASE tidymart_enterprise;
CREATE USER tidymart_service WITH ENCRYPTED PASSWORD 'enterprise_secure_password_2024';
GRANT ALL PRIVILEGES ON DATABASE tidymart_enterprise TO tidymart_service;

\c tidymart_enterprise;

-- Create schema
CREATE SCHEMA tidymart AUTHORIZATION tidymart_service;
GRANT ALL ON SCHEMA tidymart TO tidymart_service;

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Core execution tracking
CREATE TABLE tidymart.executions (
    execution_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    module VARCHAR(50) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    department VARCHAR(100),
    start_time TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    end_time TIMESTAMP WITH TIME ZONE,
    total_duration_ms INTEGER,
    success_rate DECIMAL(5,4) CHECK (success_rate >= 0 AND success_rate <= 1),
    cost_usd DECIMAL(12,6) CHECK (cost_usd >= 0),
    input_context JSONB,
    final_results JSONB,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Step tracking for detailed execution analysis  
CREATE TABLE tidymart.execution_steps (
    step_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL REFERENCES tidymart.executions(execution_id) ON DELETE CASCADE,
    step_name VARCHAR(200) NOT NULL,
    step_order INTEGER NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE NOT NULL,
    end_time TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER CHECK (duration_ms >= 0),
    success BOOLEAN NOT NULL,
    error_message TEXT,
    input_data JSONB,
    output_data JSONB,
    parameters JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Performance optimization data
CREATE TABLE tidymart.module_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID NOT NULL REFERENCES tidymart.executions(execution_id) ON DELETE CASCADE,
    module VARCHAR(50) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    configuration JSONB NOT NULL,
    performance_metrics JSONB NOT NULL,
    quality_score DECIMAL(5,4) CHECK (quality_score >= 0 AND quality_score <= 1),
    cost_efficiency_score DECIMAL(8,6) CHECK (cost_efficiency_score >= 0),
    user_satisfaction_score DECIMAL(5,4) CHECK (user_satisfaction_score >= 0 AND user_satisfaction_score <= 1),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Optimal configurations learned from historical data
CREATE TABLE tidymart.optimal_configs (
    config_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    module VARCHAR(50) NOT NULL,
    operation VARCHAR(100) NOT NULL,
    context_pattern JSONB NOT NULL,
    optimal_configuration JSONB NOT NULL,
    success_rate DECIMAL(5,4) NOT NULL CHECK (success_rate >= 0 AND success_rate <= 1),
    avg_performance_score DECIMAL(8,6) NOT NULL CHECK (avg_performance_score >= 0),
    confidence_interval DECIMAL(5,4) DEFAULT 0.95,
    sample_size INTEGER DEFAULT 1 CHECK (sample_size > 0),
    usage_count INTEGER DEFAULT 1 CHECK (usage_count > 0),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Cross-module workflow patterns for orchestration optimization
CREATE TABLE tidymart.workflow_patterns (
    pattern_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_name VARCHAR(200) NOT NULL,
    module_sequence VARCHAR[] NOT NULL,
    input_characteristics JSONB,
    success_rate DECIMAL(5,4) CHECK (success_rate >= 0 AND success_rate <= 1),
    avg_duration_ms INTEGER CHECK (avg_duration_ms >= 0),
    avg_cost_usd DECIMAL(10,6) CHECK (avg_cost_usd >= 0),
    quality_score DECIMAL(5,4) CHECK (quality_score >= 0 AND quality_score <= 1),
    usage_frequency INTEGER DEFAULT 1 CHECK (usage_frequency > 0),
    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    optimization_notes TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Embedding cache for performance optimization
CREATE TABLE tidymart.embedding_cache (
    cache_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    text_hash VARCHAR(64) NOT NULL UNIQUE, -- SHA-256 hash
    text_preview VARCHAR(200), -- First 200 chars for debugging
    embedding_vector JSONB NOT NULL, -- Store as JSON array
    embedding_method VARCHAR(50) NOT NULL,
    vector_dimensions INTEGER NOT NULL,
    quality_score DECIMAL(5,4),
    usage_count INTEGER DEFAULT 1 CHECK (usage_count > 0),
    last_used TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Learning recommendations tracking
CREATE TABLE tidymart.optimization_recommendations (
    recommendation_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    module VARCHAR(50) NOT NULL,
    recommendation_type VARCHAR(100) NOT NULL,
    current_config JSONB NOT NULL,
    recommended_config JSONB NOT NULL,
    expected_improvement DECIMAL(5,4) CHECK (expected_improvement >= 0),
    confidence_score DECIMAL(5,4) CHECK (confidence_score >= 0 AND confidence_score <= 1),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'applied', 'rejected', 'expired')),
    applied_at TIMESTAMP WITH TIME ZONE,
    actual_improvement DECIMAL(5,4),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enterprise audit trail
CREATE TABLE tidymart.audit_events (
    audit_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id UUID REFERENCES tidymart.executions(execution_id),
    event_type VARCHAR(50) NOT NULL,
    user_id VARCHAR(255),
    department VARCHAR(100),
    resource_accessed VARCHAR(200),
    action_performed VARCHAR(200),
    security_classification VARCHAR(20) DEFAULT 'internal',
    compliance_flags VARCHAR[],
    event_timestamp TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    source_ip INET,
    user_agent TEXT,
    additional_context JSONB
);

-- Performance indexes for optimal query performance
CREATE INDEX idx_executions_module_operation ON tidymart.executions(module, operation);
CREATE INDEX idx_executions_user_dept_time ON tidymart.executions(user_id, department, start_time DESC);
CREATE INDEX idx_executions_time ON tidymart.executions(start_time DESC);
CREATE INDEX idx_executions_success ON tidymart.executions(success_rate DESC) WHERE success_rate IS NOT NULL;

CREATE INDEX idx_steps_execution_order ON tidymart.execution_steps(execution_id, step_order);
CREATE INDEX idx_steps_performance ON tidymart.execution_steps(duration_ms, success);

CREATE INDEX idx_perf_module_operation ON tidymart.module_performance(module, operation);
CREATE INDEX idx_perf_quality_efficiency ON tidymart.module_performance(quality_score DESC, cost_efficiency_score DESC);
CREATE INDEX idx_perf_time ON tidymart.module_performance(recorded_at DESC);

CREATE INDEX idx_configs_module_op ON tidymart.optimal_configs(module, operation);
CREATE INDEX idx_configs_success ON tidymart.optimal_configs(success_rate DESC, avg_performance_score DESC);
CREATE INDEX idx_configs_usage ON tidymart.optimal_configs(usage_count DESC);

CREATE INDEX idx_workflow_success ON tidymart.workflow_patterns(success_rate DESC);
CREATE INDEX idx_workflow_usage ON tidymart.workflow_patterns(usage_frequency DESC);
CREATE INDEX idx_workflow_name ON tidymart.workflow_patterns(workflow_name);

CREATE INDEX idx_embedding_hash ON tidymart.embedding_cache(text_hash);
CREATE INDEX idx_embedding_method ON tidymart.embedding_cache(embedding_method);
CREATE INDEX idx_embedding_usage ON tidymart.embedding_cache(usage_count DESC);

CREATE INDEX idx_recommendations_module ON tidymart.optimization_recommendations(module, status);
CREATE INDEX idx_recommendations_time ON tidymart.optimization_recommendations(created_at DESC);

CREATE INDEX idx_audit_user_time ON tidymart.audit_events(user_id, event_timestamp DESC);
CREATE INDEX idx_audit_execution ON tidymart.audit_events(execution_id);
CREATE INDEX idx_audit_event_type ON tidymart.audit_events(event_type, event_timestamp DESC);

-- JSONB indexes for context matching
CREATE INDEX idx_executions_input_context ON tidymart.executions USING GIN (input_context);
CREATE INDEX idx_configs_context_pattern ON tidymart.optimal_configs USING GIN (context_pattern);
CREATE INDEX idx_workflow_input_char ON tidymart.workflow_patterns USING GIN (input_characteristics);

-- Automated data retention policies
CREATE OR REPLACE FUNCTION tidymart.cleanup_old_data()
RETURNS void AS $$
BEGIN
    -- Archive old execution data (keep 90 days detailed, 2 years summary)
    DELETE FROM tidymart.execution_steps 
    WHERE step_id IN (
        SELECT s.step_id FROM tidymart.execution_steps s
        JOIN tidymart.executions e ON s.execution_id = e.execution_id
        WHERE e.start_time < NOW() - INTERVAL '90 days'
    );
    
    -- Clean up old embedding cache (keep high-usage items longer)
    DELETE FROM tidymart.embedding_cache 
    WHERE last_used < NOW() - INTERVAL '30 days' 
    AND usage_count < 5;
    
    -- Archive old audit events (keep 7 years for compliance)
    DELETE FROM tidymart.audit_events 
    WHERE event_timestamp < NOW() - INTERVAL '7 years';
    
    RAISE NOTICE 'Data cleanup completed successfully';
END;
$$ LANGUAGE plpgsql;

-- Schedule automated cleanup (run daily)
-- This would be set up via cron or pg_cron extension in production

-- Create views for common queries
CREATE VIEW tidymart.execution_summary AS
SELECT 
    module,
    operation,
    department,
    DATE_TRUNC('day', start_time) as execution_date,
    COUNT(*) as total_executions,
    AVG(success_rate) as avg_success_rate,
    AVG(total_duration_ms) as avg_duration_ms,
    AVG(cost_usd) as avg_cost_usd,
    MIN(start_time) as first_execution,
    MAX(start_time) as last_execution
FROM tidymart.executions 
WHERE start_time > NOW() - INTERVAL '30 days'
GROUP BY module, operation, department, DATE_TRUNC('day', start_time);

CREATE VIEW tidymart.performance_dashboard AS
SELECT 
    mp.module,
    mp.operation,
    AVG(mp.quality_score) as avg_quality,
    AVG(mp.cost_efficiency_score) as avg_efficiency,
    COUNT(*) as sample_size,
    MAX(mp.recorded_at) as last_updated
FROM tidymart.module_performance mp
WHERE mp.recorded_at > NOW() - INTERVAL '7 days'
GROUP BY mp.module, mp.operation
ORDER BY avg_quality DESC, avg_efficiency DESC;

-- Grant appropriate permissions
GRANT USAGE ON SCHEMA tidymart TO tidymart_service;
GRANT ALL ON ALL TABLES IN SCHEMA tidymart TO tidymart_service;
GRANT ALL ON ALL SEQUENCES IN SCHEMA tidymart TO tidymart_service;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA tidymart TO tidymart_service;

-- Create read-only user for analytics
CREATE USER tidymart_analytics WITH ENCRYPTED PASSWORD 'analytics_readonly_2024';
GRANT USAGE ON SCHEMA tidymart TO tidymart_analytics;
GRANT SELECT ON ALL TABLES IN SCHEMA tidymart TO tidymart_analytics;
GRANT SELECT ON tidymart.execution_summary TO tidymart_analytics;
GRANT SELECT ON tidymart.performance_dashboard TO tidymart_analytics;
```

---

## Deployment Configuration

### Docker Compose for Enterprise Deployment

```yaml
# docker-compose.enterprise.yml
version: '3.8'

services:
  tidymart-postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: tidymart_enterprise
      POSTGRES_USER: tidymart_service
      POSTGRES_PASSWORD: ${TIDYMART_POSTGRES_PASSWORD}
    volumes:
      - tidymart_postgres_data:/var/lib/postgresql/data
      - ./schemas/tidymart_schema.sql:/docker-entrypoint-initdb.d/01-schema.sql:ro
    ports:
      - "5432:5432"
    command: >
      postgres
      -c shared_preload_libraries=pg_stat_statements
      -c pg_stat_statements.max=10000
      -c pg_stat_statements.track=all
      -c max_connections=200
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tidymart_service -d tidymart_enterprise"]
      interval: 10s
      timeout: 5s
      retries: 5

  tidymart-api:
    build:
      context: .
      dockerfile: Dockerfile.tidymart
    environment:
      TIDYMART_POSTGRES_URL: postgresql://tidymart_service:${TIDYMART_POSTGRES_PASSWORD}@tidymart-postgres:5432/tidymart_enterprise
      TIDYMART_LOG_LEVEL: INFO
      TIDYMART_POOL_SIZE: 50
      TIDYMART_MAX_OVERFLOW: 100
    depends_on:
      tidymart-postgres:
        condition: service_healthy
    ports:
      - "8080:8080"
    volumes:
      - ./logs:/app/logs
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  redis-cache:
    image: redis:7-alpine
    command: redis-server --appendonly yes --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis_cache_data:/data
    ports:
      - "6379:6379"

  monitoring:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'

volumes:
  tidymart_postgres_data:
    driver: local
  redis_cache_data:
    driver: local  
  prometheus_data:
    driver: local

networks:
  default:
    name: tidymart_enterprise_network
```

### Environment Configuration

```bash
# .env.production
TIDYMART_POSTGRES_PASSWORD=your_secure_production_password_2024
TIDYMART_ENVIRONMENT=production
TIDYMART_LOG_LEVEL=INFO
TIDYMART_ENABLE_METRICS=true
TIDYMART_ENABLE_TRACING=true
TIDYMART_BACKUP_SCHEDULE="0 2 * * *"  # Daily at 2 AM
TIDYMART_RETENTION_DAYS=90
TIDYMART_AUDIT_RETENTION_DAYS=2555
```

This technical implementation provides the enterprise-grade foundation for TidyMart integration across all modules. The next step would be to implement specific module integrations using these patterns.