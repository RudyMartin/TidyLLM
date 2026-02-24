Here's the hexagonal architecture view for the PostgreSQL database with resilient connection pooling:

  HEXAGONAL ARCHITECTURE - POSTGRESQL WITH RESILIENT CONNECTION POOLING
  =====================================================================

           PRIMARY PORTS (Application Side)                    SECONDARY PORTS (Database Side)
      ┌────────────────────────────────────┐                ┌──────────────────────────────┐
      │  Application Services              │                │  IDatabase Interface         │
      │  • MLflow (port 5000)             │                │  • execute_query()           │
      │  • RAG Systems (5 variants)       │                │  • begin_transaction()       │
      │  • Vector Search                  │                │  • commit()                  │
      │  • Workflow Management            │                │  • rollback()                │
      │  (Database Consumers)              │                │  • get_connection()          │
      └──────────┬─────────────────────────┘                └─────────▲────────────────────┘
                 │                                                      │
                 ▼                                                      │
      ┌─────────────────────────────────────────────────────────────────────────────────────┐
      │                         DOMAIN CORE (Database Access Layer)                         │
      │  ┌───────────────────────────────────────────────────────────────────────────────┐ │
      │  │                     ResilientPoolManager (Core Logic)                         │ │
      │  │  ┌─────────────────────────────────────────────────────────────────────────┐ │ │
      │  │  │  Resilience Strategies:                                                 │ │ │
      │  │  │  • Primary/Backup/Failover pool architecture                           │ │ │
      │  │  │  • Automatic failover on pool hang (3-pool strategy)                   │ │ │
      │  │  │  • Health monitoring (30-second intervals)                              │ │ │
      │  │  │  • Load balancing between healthy pools                                │ │ │
      │  │  │  • Circuit breaker pattern (3 failures = failover)                     │ │ │
      │  │  │  • Connection retry logic (max 3 retries)                              │ │ │
      │  │  │  • Timeout management (10-second threshold)                            │ │ │
      │  │  └─────────────────────────────────────────────────────────────────────────┘ │ │
      │  │                                                                               │ │
      │  │  ┌─────────────────────────────────────────────────────────────────────────┐ │ │
      │  │  │  Pool Metrics & Monitoring:                                             │ │ │
      │  │  │  • PoolMetrics dataclass (per pool)                                     │ │ │
      │  │  │  • active_connections tracking                                          │ │ │
      │  │  │  • total_requests counting                                              │ │ │
      │  │  │  • failed_requests monitoring                                           │ │ │
      │  │  │  • avg_response_time calculation                                        │ │ │
      │  │  │  • health_status (healthy/degraded/unhealthy)                          │ │ │
      │  │  └─────────────────────────────────────────────────────────────────────────┘ │ │
      │  └───────────────────────────────────────────────────────────────────────────────┘ │
      └──────────────────────────────────────┬──────────────────────────────────────────────┘
                                             │
                            ┌────────────────┴────────────────┐
                            ▼                                  ▼
      ┌─────────────────────────────────────────────────────────────────────────────────────┐
      │                           CONNECTION POOL LAYER                                      │
      │                                                                                      │
      │  ┌─────────────────────────┐  ┌─────────────────────────┐  ┌───────────────────┐  │
      │  │   PRIMARY POOL          │  │   BACKUP POOL           │  │  FAILOVER POOL    │  │
      │  │  ┌─────────────────┐   │  │  ┌─────────────────┐   │  │ ┌─────────────┐   │  │
      │  │  │ psycopg2 pool   │   │  │  │ psycopg2 pool   │   │  │ │ psycopg2    │   │  │
      │  │  │ • min_conn: 2   │   │  │  │ • min_conn: 1   │   │  │ │ • min: 1    │   │  │
      │  │  │ • max_conn: 20  │   │  │  │ • max_conn: 10  │   │  │ │ • max: 5    │   │  │
      │  │  │ • timeout: 30s  │   │  │  │ • timeout: 30s  │   │  │ │ • timeout:  │   │  │
      │  │  │ • autocommit    │   │  │  │ • autocommit    │   │  │ │   60s       │   │  │
      │  │  └─────────────────┘   │  │  └─────────────────┘   │  │ └─────────────┘   │  │
      │  └─────────────────────────┘  └─────────────────────────┘  └───────────────────┘  │
      │                                                                                      │
      │  ┌─────────────────────────────────────────────────────────────────────────────┐   │
      │  │                    TidyLLMConnectionPool (Base Implementation)              │   │
      │  │  • Connection lifecycle management                                           │   │
      │  │  • Thread-safe connection acquisition                                        │   │
      │  │  • Connection validation and recycling                                       │   │
      │  │  • Query execution with automatic retry                                      │   │
      │  └─────────────────────────────────────────────────────────────────────────────┘   │
      └─────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
      ┌─────────────────────────────────────────────────────────────────────────────────────┐
      │                         CREDENTIAL & CONFIG LAYER                                    │
      │  ┌─────────────────────────────────────────────────────────────────────────────┐   │
      │  │                         CredentialCarrier                                    │   │
      │  │  • Loads from settings.yaml                                                  │   │
      │  │  • Manages multiple database credentials:                                    │   │
      │  │    - vectorqa (primary)                                                      │   │
      │  │    - mlflow_alt_db (MLflow tracking)                                        │   │
      │  │    - postgresql_primary (main app DB)                                       │   │
      │  │  • Environment variable fallback                                             │   │
      │  │  • Secure credential storage                                                 │   │
      │  └─────────────────────────────────────────────────────────────────────────────┘   │
      └─────────────────────────────────────────────────────────────────────────────────────┘
                                             │
                                             ▼
      ┌─────────────────────────────────────────────────────────────────────────────────────┐
      │                          EXTERNAL INFRASTRUCTURE                                     │
      │  ┌─────────────────────────────────────────────────────────────────────────────┐   │
      │  │                      AWS RDS PostgreSQL Instances                            │   │
      │  │  ┌────────────────────────────────────────────────────────────────────┐    │   │
      │  │  │  Primary: vectorqa.czxuk7no9zzp.us-east-1.rds.amazonaws.com       │    │   │
      │  │  │  • PostgreSQL 15.4                                                 │    │   │
      │  │  │  • db.t3.medium (2 vCPU, 4 GB RAM)                                 │    │   │
      │  │  │  • 100 GB SSD storage                                              │    │   │
      │  │  │  • Multi-AZ deployment                                              │    │   │
      │  │  └────────────────────────────────────────────────────────────────────┘    │   │
      │  │                                                                              │   │
      │  │  ┌────────────────────────────────────────────────────────────────────┐    │   │
      │  │  │  Read Replica: vectorqa-read.czxuk7no9zzp.us-east-1.rds...        │    │   │
      │  │  │  • Async replication from primary                                   │    │   │
      │  │  │  • Read-only queries offloading                                    │    │   │
      │  │  └────────────────────────────────────────────────────────────────────┘    │   │
      │  └─────────────────────────────────────────────────────────────────────────────┘   │
      └─────────────────────────────────────────────────────────────────────────────────────┘

  FAILOVER SEQUENCE:
  ════════════════
  1. Normal Operation: PRIMARY pool handles all requests
  2. Primary Degradation: Detected via health checks (response time > threshold)
  3. Automatic Failover: Switch to BACKUP pool
  4. Backup Issues: If backup also fails, switch to FAILOVER pool
  5. Recovery: Background health checks attempt to restore PRIMARY
  6. Rebalancing: Gradual shift back to PRIMARY when healthy

  CONNECTION FLOW:
  ══════════════
  1. Application requests connection via ResilientPoolManager.get_connection()
  2. Manager checks active pool health status
  3. If healthy: Returns connection from active pool
  4. If degraded: Initiates failover sequence
  5. Connection wrapped in context manager for automatic cleanup
  6. Metrics updated for monitoring and alerting

  MONITORING METRICS:
  ═════════════════
  • Connection acquisition time
  • Query execution time
  • Pool utilization percentage
  • Failed connection attempts
  • Failover events count
  • Health check results

  ERROR HANDLING:
  ═════════════
  • PoolTimeoutException: Connection acquisition timeout
  • PoolHungException: Pool appears frozen
  • PoolException: General pool errors
  • Automatic retry with exponential backoff
  • Circuit breaker prevents cascade failures

  Key Hexagonal Principles in PostgreSQL Implementation:

  1. Pool Abstraction: Applications don't know about multiple pools - transparent failover
  2. Health Monitoring: Continuous background health checks without blocking operations
  3. Graceful Degradation: 3-tier failover (primary → backup → failover)
  4. Metrics Collection: All operations tracked for observability
  5. Thread Safety: RLock ensures concurrent access safety
  6. Resource Carrier Pattern: Credentials managed separately from pool logic
  7. Circuit Breaker: Prevents hammering failed resources
  8. Context Management: Automatic connection cleanup via context managers

  Resilience Features:
  - 3-Pool Strategy: Always have a fallback option
  - Automatic Recovery: Background threads attempt to restore failed pools
  - Load Balancing: Can distribute load across healthy pools
  - Timeout Management: Prevents indefinite hangs
  - Health-based Routing: Routes to healthiest pool automatically

  This architecture ensures database availability even when:
  - Primary database is under heavy load
  - Network issues cause intermittent failures
  - Connection pools become exhausted
  - Database maintenance is occurring
  - Unexpected connection drops happen
