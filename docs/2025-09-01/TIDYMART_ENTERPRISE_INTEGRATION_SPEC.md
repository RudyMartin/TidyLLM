# TidyMart Enterprise Integration Specification

**Document Version**: 1.0  
**Date**: 2025-09-01  
**Classification**: Corporate Internal  
**Authors**: TidyLLM Architecture Team  

---

## Executive Summary

TidyMart serves as the **universal data backbone** and **learning engine** for the entire TidyLLM ecosystem. This specification defines the enterprise-grade integration patterns that enable all modules (Documents, Sentence, TLM, Gateway, Heiros, SPARSE) to operate as **data consumers** of TidyMart services while feeding back performance and learning data.

**Key Principle**: TidyMart is the **PostgreSQL-based data hub** that transforms individual modules into a **learning, self-optimizing enterprise AI platform**.

---

## Architecture Overview

### TidyMart as Universal Data Backbone

```
┌─────────────────────────────────────────────────────────────────┐
│                         TidyMart Core                           │
│                    (PostgreSQL + Analytics)                     │
├─────────────────────────────────────────────────────────────────┤
│  Configuration    │  Performance    │  Learning      │  Audit   │
│  Provider        │  Tracker        │  Engine        │  Logger  │
└─────────────────────────────────────────────────────────────────┘
            │                    │                    │
    ┌───────┴────────┐  ┌────────┴────────┐  ┌───────┴────────┐
    │   Data In      │  │   Processing    │  │   Data Out     │
    │   (Consumers)  │  │   Pipeline      │  │   (Learning)   │
    └────────────────┘  └─────────────────┘  └────────────────┘
            │                    │                    │
┌───────────┼────────────────────┼────────────────────┼───────────┐
│           ▼                    ▼                    ▼           │
│ Documents  │  Sentence  │  TLM  │  Gateway  │  Heiros  │ SPARSE │
│ Module     │  Module    │ Module│  Module   │ Module   │ Module │
└────────────┴────────────┴───────┴───────────┴──────────┴────────┘
```

### Consumer-Provider Relationship Model

**TidyMart as Provider**:
- Configuration management and optimization recommendations
- Historical performance data and pattern analysis  
- Cross-module learning and intelligence synthesis
- Audit trails and compliance reporting

**Modules as Consumers**:
- Query TidyMart for optimal configurations before execution
- Stream execution data to TidyMart during processing
- Receive performance feedback and improvement recommendations
- Participate in cross-module learning and optimization

---

## Core Integration Patterns

### Pattern 1: Configuration Consumer

**Purpose**: Modules query TidyMart for optimal configurations based on historical performance.

**Interface**:
```python
class TidyMartConfigProvider:
    def get_optimal_config(self, module: str, operation: str, context: dict) -> dict
    def get_fallback_config(self, module: str, operation: str) -> dict
    def register_config_usage(self, module: str, config: dict, performance: dict)
```

**Implementation Contract**:
- Every module **MUST** query TidyMart before major operations
- TidyMart **MUST** provide configuration within 100ms SLA
- Fallback configurations **MUST** be available for offline scenarios

### Pattern 2: Performance Data Producer

**Purpose**: Modules stream execution metrics to TidyMart for learning and optimization.

**Interface**:
```python
class TidyMartPerformanceTracker:
    def start_execution(self, module: str, operation: str, context: dict) -> str
    def track_step(self, execution_id: str, step_data: dict) -> bool
    def complete_execution(self, execution_id: str, results: dict) -> bool
```

**Implementation Contract**:
- Every operation **MUST** be tracked with unique execution_id
- Performance data **MUST** include timing, cost, quality, and success metrics
- Data streaming **MUST** be non-blocking and resilient to TidyMart downtime

### Pattern 3: Learning Intelligence Consumer

**Purpose**: Modules receive cross-module insights and optimization recommendations.

**Interface**:
```python
class TidyMartLearningEngine:
    def get_optimization_recommendations(self, module: str) -> List[dict]
    def get_cross_module_insights(self, workflow: str) -> dict
    def report_improvement_results(self, module: str, before: dict, after: dict)
```

**Implementation Contract**:
- Learning recommendations **SHOULD** be applied during low-traffic periods
- Modules **MUST** report back results of applied optimizations
- Cross-module insights **SHOULD** influence workflow orchestration decisions

---

## Module-Specific Integration Requirements

### Documents Module Integration

**Data Consumer Requirements**:
```sql
-- Configuration queries Documents module needs
SELECT optimal_extraction_method, confidence_threshold, processing_timeout
FROM tidymart.document_configs 
WHERE document_type = ? AND file_size_mb BETWEEN ? AND ?
ORDER BY success_rate DESC, avg_processing_time ASC
LIMIT 1;

-- Pattern library Documents module needs  
SELECT pattern_regex, field_name, confidence_score
FROM tidymart.extraction_patterns
WHERE document_type = ? AND success_rate > 0.8
ORDER BY usage_count DESC;
```

**Data Producer Requirements**:
```python
# What Documents module must track in TidyMart
{
    'execution_id': uuid,
    'document_type': 'invoice|contract|report|etc',
    'file_size_mb': float,
    'pages_count': int,
    'extraction_method': 'pattern|ml|hybrid',
    'processing_time_ms': int,
    'fields_extracted': int,
    'extraction_confidence': float,
    'validation_passed': bool,
    'user_corrections': int,
    'cost_usd': float
}
```

### Sentence Module Integration

**Data Consumer Requirements**:
```sql
-- Optimal embedding method selection
SELECT embedding_method, parameters, avg_similarity_score
FROM tidymart.embedding_performance
WHERE text_type = ? AND text_length BETWEEN ? AND ?
ORDER BY quality_score DESC, processing_time_ms ASC;

-- Embedding cache lookup
SELECT embedding_vector FROM tidymart.embedding_cache 
WHERE text_hash = ? AND embedding_method = ?;
```

**Data Producer Requirements**:
```python
# What Sentence module must track in TidyMart
{
    'execution_id': uuid,
    'text_hash': str,  # SHA-256 for deduplication
    'text_length': int,
    'text_type': 'document|query|classification',
    'embedding_method': 'tfidf|word_avg|sentence_bert',
    'embedding_vector': List[float],
    'generation_time_ms': int,
    'cache_hit': bool,
    'similarity_comparisons': int,
    'downstream_accuracy': float  # From classification results
}
```

### TLM Module Integration

**Data Consumer Requirements**:
```sql
-- Algorithm selection optimization
SELECT algorithm, hyperparameters, avg_accuracy, avg_runtime_ms
FROM tidymart.ml_algorithm_performance
WHERE data_shape = ? AND problem_type = ?
ORDER BY accuracy DESC, runtime_ms ASC;

-- Convergence prediction
SELECT estimated_iterations FROM tidymart.convergence_patterns
WHERE algorithm = ? AND data_characteristics = ?;
```

**Data Producer Requirements**:
```python
# What TLM module must track in TidyMart
{
    'execution_id': uuid,
    'algorithm': 'kmeans|logreg|pca|svm',
    'data_shape': [int, int],  # rows, columns
    'data_sparsity': float,
    'hyperparameters': dict,
    'iterations_to_convergence': int,
    'final_accuracy': float,
    'training_time_ms': int,
    'memory_usage_mb': float,
    'stability_score': float  # How consistent results are
}
```

### Gateway Module Integration

**Data Consumer Requirements**:
```sql
-- Model selection optimization
SELECT model_name, provider, avg_quality, avg_cost, avg_latency
FROM tidymart.llm_performance
WHERE use_case = ? AND department = ?
ORDER BY (quality_score / cost_usd) DESC;

-- Budget and quota management
SELECT current_spend, budget_remaining, quota_usage
FROM tidymart.spend_tracking
WHERE department = ? AND time_period = 'current_month';
```

**Data Producer Requirements**:
```python
# What Gateway module must track in TidyMart
{
    'execution_id': uuid,
    'user_id': str,
    'department': str,
    'model': str,
    'provider': str,
    'input_tokens': int,
    'output_tokens': int,
    'cost_usd': float,
    'latency_ms': int,
    'quality_score': float,  # From downstream validation
    'audit_reason': str,
    'compliance_flags': List[str],
    'success': bool,
    'error_type': str  # If failed
}
```

### Heiros Module Integration

**Data Consumer Requirements**:
```sql
-- Workflow optimization patterns
SELECT workflow_config, avg_success_rate, avg_cost, avg_duration
FROM tidymart.workflow_performance
WHERE workflow_type = ? AND input_characteristics = ?
ORDER BY success_rate DESC, cost_usd ASC;

-- Decision tree optimization
SELECT node_conditions, success_paths, failure_patterns
FROM tidymart.decision_tree_analysis
WHERE workflow_id = ?;
```

**Data Producer Requirements**:
```python
# What Heiros module must track in TidyMart
{
    'execution_id': uuid,
    'workflow_name': str,
    'strategy_version': str,
    'decision_path': List[str],  # Which nodes were executed
    'input_context': dict,
    'step_results': List[dict],
    'total_duration_ms': int,
    'total_cost_usd': float,
    'quality_score': float,
    'user_satisfaction': float,
    'failure_points': List[str],  # Where problems occurred
    'recovery_actions': List[str]  # How failures were handled
}
```

### SPARSE Module Integration

**Data Consumer Requirements**:
```sql
-- Command usage optimization
SELECT command_name, optimal_strategy, success_rate, user_satisfaction
FROM tidymart.sparse_command_performance
WHERE user_role = ? AND use_case_similarity > 0.8
ORDER BY user_satisfaction DESC;

-- Auto-command suggestions
SELECT suggested_command, workflow_pattern, frequency
FROM tidymart.workflow_pattern_analysis
WHERE user_patterns SIMILAR TO current_user_pattern;
```

**Data Producer Requirements**:
```python
# What SPARSE module must track in TidyMart
{
    'execution_id': uuid,
    'command_name': str,
    'user_id': str,
    'user_role': str,
    'expanded_strategy': dict,
    'execution_context': dict,
    'command_success': bool,
    'user_satisfaction_score': float,
    'time_to_completion': int,
    'command_modifications': List[str],  # User customizations
    'follow_up_commands': List[str]  # What user did next
}
```

---

## Enterprise Data Pipeline Architecture

### Universal Pipeline Interface

**Every module operation follows this pattern**:

```python
class TidyMartPipeline:
    """Universal enterprise pipeline with TidyMart integration"""
    
    def __init__(self, postgres_conn: str, module_name: str):
        self.mart = TidyMartConnection(postgres_conn)
        self.module = module_name
        self.execution_id = None
        self.polars_frame = None  # High-performance data processing
        self.metadata = {}        # Growing JSON structure
        
    def start_execution(self, operation: str, input_data: dict) -> 'TidyMartPipeline':
        """Initialize execution with TidyMart configuration lookup"""
        
        # Get optimal configuration from TidyMart
        config = self.mart.get_optimal_config(
            module=self.module,
            operation=operation, 
            context=input_data
        )
        
        # Start tracking in TidyMart
        self.execution_id = self.mart.start_execution(
            module=self.module,
            operation=operation,
            config=config,
            input_context=input_data
        )
        
        # Initialize data structures
        self.polars_frame = pl.DataFrame([input_data])
        self.metadata = {
            'execution_id': self.execution_id,
            'module': self.module,
            'operation': operation,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'steps': []
        }
        
        return self
    
    def add_processing_step(self, step_name: str, processor_func, **params):
        """Execute processing step with TidyMart tracking"""
        
        step_start = datetime.now()
        
        try:
            # Execute the actual processing
            result = processor_func(self.polars_frame, **params)
            
            # Update data frame with results
            if isinstance(result, dict):
                for key, value in result.items():
                    self.polars_frame = self.polars_frame.with_columns(
                        pl.lit(value).alias(key)
                    )
            
            success = True
            error = None
            
        except Exception as e:
            success = False
            error = str(e)
            result = None
        
        step_duration = (datetime.now() - step_start).total_seconds() * 1000
        
        # Track step in TidyMart
        step_data = {
            'step_name': step_name,
            'duration_ms': step_duration,
            'success': success,
            'error': error,
            'parameters': params,
            'result_summary': str(result)[:500] if result else None
        }
        
        self.metadata['steps'].append(step_data)
        
        self.mart.track_step(
            execution_id=self.execution_id,
            step_data=step_data
        )
        
        return self
    
    def complete_execution(self, final_results: dict):
        """Finalize execution with TidyMart learning feedback"""
        
        # Calculate final metrics
        total_duration = sum(step['duration_ms'] for step in self.metadata['steps'])
        success_rate = sum(1 for step in self.metadata['steps'] if step['success']) / len(self.metadata['steps'])
        
        # Store completion in TidyMart
        completion_data = {
            'total_duration_ms': total_duration,
            'success_rate': success_rate,
            'final_results': final_results,
            'data_shape': self.polars_frame.shape,
            'metadata': self.metadata
        }
        
        self.mart.complete_execution(
            execution_id=self.execution_id,
            completion_data=completion_data
        )
        
        # Get learning recommendations for next time
        recommendations = self.mart.get_optimization_recommendations(
            module=self.module,
            execution_data=completion_data
        )
        
        return {
            'execution_id': self.execution_id,
            'results': final_results,
            'polars_frame': self.polars_frame,
            'metadata': self.metadata,
            'recommendations': recommendations
        }
```

### MVR Implementation Example

**How the existing MVR workflow integrates**:

```python
def enterprise_mvr_pipeline(document_path: str, user_context: dict):
    """Enterprise MVR pipeline with full TidyMart integration"""
    
    # Initialize pipeline
    pipeline = TidyMartPipeline(postgres_conn, "mvr_workflow")
    
    return (pipeline
        .start_execution("peer_review", {
            'document_path': document_path,
            'user_id': user_context['user_id'],
            'department': user_context['department'],
            'compliance_mode': 'banking_regulation'
        })
        .add_processing_step("extract_document", documents_processor, 
                           document_path=document_path)
        .add_processing_step("generate_embeddings", sentence_processor,
                           text_column='extracted_text')  
        .add_processing_step("classify_sections", tlm_processor,
                           embeddings_column='embeddings')
        .add_processing_step("llm_analysis", gateway_processor,
                           user_id=user_context['user_id'],
                           audit_reason="MVR peer review analysis")
        .add_processing_step("orchestrate_review", heiros_processor,
                           workflow='comprehensive_peer_review')
        .add_processing_step("format_output", sparse_processor,
                           command='[Compliance Report]')
        .complete_execution({
            'report_type': 'mvr_peer_review',
            'compliance_status': 'completed',
            'audit_trail_complete': True
        })
    )
```

---

## Data Schema Specifications

### Core Tables

**Execution Tracking**:
```sql
CREATE TABLE tidymart.executions (
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

CREATE INDEX idx_executions_module_operation ON tidymart.executions(module, operation);
CREATE INDEX idx_executions_user_dept ON tidymart.executions(user_id, department);
CREATE INDEX idx_executions_time ON tidymart.executions(start_time DESC);
```

**Performance Optimization**:
```sql
CREATE TABLE tidymart.module_performance (
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

CREATE INDEX idx_perf_module_operation ON tidymart.module_performance(module, operation);
CREATE INDEX idx_perf_quality ON tidymart.module_performance(quality_score DESC);
CREATE INDEX idx_perf_efficiency ON tidymart.module_performance(cost_efficiency_score DESC);
```

**Cross-Module Learning**:
```sql
CREATE TABLE tidymart.workflow_patterns (
    pattern_id UUID PRIMARY KEY,
    workflow_name VARCHAR(200),
    module_sequence VARCHAR[],
    input_characteristics JSONB,
    success_rate DECIMAL(3,2),
    avg_duration_ms INTEGER,
    avg_cost_usd DECIMAL(8,4),
    usage_frequency INTEGER DEFAULT 0,
    last_used TIMESTAMP DEFAULT NOW(),
    optimization_notes TEXT
);

CREATE INDEX idx_workflow_success ON tidymart.workflow_patterns(success_rate DESC);
CREATE INDEX idx_workflow_usage ON tidymart.workflow_patterns(usage_frequency DESC);
```

---

## Enterprise Integration Requirements

### Security and Compliance

**Data Classification**:
- All TidyMart data marked as "Corporate Internal" minimum
- PII detection and masking for input/output content  
- Encryption at rest for all performance and audit data
- Role-based access control for different data views

**Audit Requirements**:
- 7-year retention for financial services compliance
- Complete execution lineage for regulatory inquiries
- Real-time alerting for anomalous patterns or failures
- Automated compliance reporting for SOX/Basel requirements

### Operational Excellence  

**Performance SLAs**:
- Configuration queries: <100ms p95
- Step tracking: <50ms p95, non-blocking
- Completion tracking: <200ms p95
- Learning recommendations: <500ms p95

**High Availability**:
- 99.9% uptime SLA for configuration services
- Graceful degradation when TidyMart unavailable
- Automatic failover for PostgreSQL backend
- Circuit breaker pattern for downstream resilience

### Cost Management

**Resource Optimization**:
- Automated data retention and archival policies
- Query optimization for large-scale pattern analysis
- Predictive scaling based on usage patterns
- Cost allocation tracking by department/project

---

## Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
- [ ] Deploy PostgreSQL backend with enterprise security
- [ ] Implement core TidyMartPipeline interface
- [ ] Create universal configuration provider
- [ ] Set up basic performance tracking

### Phase 2: Module Integration (Weeks 3-6)  
- [ ] Integrate Documents module as first consumer
- [ ] Add Sentence module with embedding cache
- [ ] Connect TLM module for algorithm optimization
- [ ] Integrate Gateway module for cost tracking
- [ ] Connect Heiros for workflow optimization
- [ ] Add SPARSE command optimization

### Phase 3: Learning Engine (Weeks 7-8)
- [ ] Implement cross-module pattern discovery
- [ ] Build optimization recommendation engine
- [ ] Create automated configuration updates
- [ ] Deploy A/B testing framework for improvements

### Phase 4: Enterprise Features (Weeks 9-10)
- [ ] Add enterprise security controls
- [ ] Implement compliance reporting
- [ ] Deploy monitoring and alerting
- [ ] Create operational dashboards

---

## Success Metrics

### Technical Metrics
- **Configuration Query Performance**: <100ms p95
- **Data Pipeline Throughput**: 1000+ executions/minute
- **Learning Engine Accuracy**: 85%+ optimization success
- **System Availability**: 99.9% uptime

### Business Metrics  
- **Module Performance Improvement**: 40%+ average optimization
- **Cross-Module Synergy**: 25%+ workflow efficiency gains
- **Cost Optimization**: 30%+ reduction in compute costs
- **Developer Productivity**: 60%+ faster integration time

### Enterprise Metrics
- **Audit Completeness**: 100% execution traceability
- **Compliance Score**: 95%+ regulatory requirement coverage
- **Security Posture**: Zero data breaches or compliance violations
- **Operational Excellence**: <1 hour mean time to recovery

---

## Conclusion

This TidyMart Enterprise Integration Specification transforms the TidyLLM ecosystem from a collection of independent modules into a **unified, learning, self-optimizing enterprise AI platform**. 

By treating TidyMart as the **universal data backbone** and all modules as **intelligent consumers**, we create:

1. **Interoperability**: All modules share common integration patterns
2. **Intelligence**: Cross-module learning improves the entire system
3. **Enterprise-readiness**: Full audit, compliance, and governance controls
4. **Performance**: Data-driven optimization across all components  
5. **Scalability**: Proven patterns for adding new modules and capabilities

This architecture serves as the **foundation for enterprise AI deployment** in regulated industries while maintaining the simplicity and transparency that makes TidyLLM unique.

---

**Next Steps**: Begin Phase 1 implementation with PostgreSQL backend deployment and TidyMartPipeline interface development.