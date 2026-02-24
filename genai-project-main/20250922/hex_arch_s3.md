Here's the hexagonal architecture view for the S3 implementation:

  HEXAGONAL ARCHITECTURE - S3 STORAGE SYSTEM
  ==========================================

           PRIMARY PORTS (Application Side)                 SECONDARY PORTS (Infrastructure Side)
      ┌──────────────────────────────────┐                ┌────────────────────────────────┐
      │  Domain Scripts                  │                │  IS3Storage Interface          │
      │  • mvr_analysis_s3.py           │                │  • upload_file()               │
      │  • s3_flow_parser.py            │                │  • download_file()             │
      │  • execute_robots3_workflow.py  │                │  • list_objects()              │
      │  (Application Entry Points)      │                │  • delete_object()             │
      └──────────┬───────────────────────┘                └──────────▲─────────────────────┘
                 │                                                     │
                 ▼                                                     │
      ┌────────────────────────────────────────────────────────────────────────────────────┐
      │                           DOMAIN CORE (S3 Business Logic)                          │
      │  ┌──────────────────────────────────────────────────────────────────────────────┐ │
      │  │                          S3Manager (Knowledge Systems)                        │ │
      │  │  ┌────────────────────────────────────────────────────────────────────────┐ │ │
      │  │  │  Core Business Rules:                                                  │ │ │
      │  │  │  • File versioning logic                                              │ │ │
      │  │  │  • Path generation strategies (build_s3_path)                         │ │ │
      │  │  │  • Metadata management                                                │ │ │
      │  │  │  • Content hashing and deduplication                                  │ │ │
      │  │  │  • Knowledge base organization                                        │ │ │
      │  │  └────────────────────────────────────────────────────────────────────────┘ │ │
      │  │                                                                              │ │
      │  │  ┌────────────────────────────────────────────────────────────────────────┐ │ │
      │  │  │  Domain Models:                                                        │ │ │
      │  │  │  • S3Config (configuration dataclass)                                  │ │ │
      │  │  │  • UploadResult (operation results)                                    │ │ │
      │  │  │  • S3Path (path abstraction)                                          │ │ │
      │  │  └────────────────────────────────────────────────────────────────────────┘ │ │
      │  └──────────────────────────────────────────────────────────────────────────────┘ │
      └─────────────────────────────────────┬──────────────────────────────────────────────┘
                                            │
                                            ▼
      ┌────────────────────────────────────────────────────────────────────────────────────┐
      │                              ADAPTER LAYER                                         │
      │                                                                                    │
      │  ┌──────────────────────────────────────────────────────────────────────────────┐ │
      │  │                     UnifiedSessionManager (Delegation Hub)                    │ │
      │  │  • Manages S3 client lifecycle                                               │ │
      │  │  • Handles credential resolution                                              │ │
      │  │  • Provides connection pooling                                                │ │
      │  │  • Routes to appropriate S3 service                                           │ │
      │  └────────────────────────────────┬────────────────────────────────────────────┘ │
      │                                    │                                               │
      │                                    ▼                                               │
      │  ┌──────────────────────────────────────────────────────────────────────────────┐ │
      │  │                         S3Service (Infrastructure)                            │ │
      │  │  ┌────────────────────────────────────────────────────────────────────────┐ │ │
      │  │  │  AWS S3 Operations:                                                     │ │ │
      │  │  │  • boto3 client management (_client, _resource)                        │ │ │
      │  │  │  • Direct S3 API calls                                                 │ │ │
      │  │  │  • Error handling (ClientError, NoCredentialsError)                    │ │ │
      │  │  │  • Region and bucket management                                        │ │ │
      │  │  └────────────────────────────────────────────────────────────────────────┘ │ │
      │  │                                                                              │ │
      │  │  ┌────────────────────────────────────────────────────────────────────────┐ │ │
      │  │  │  Credential Sources (via credential_loader):                            │ │ │
      │  │  │  • settings.yaml configuration                                         │ │ │
      │  │  │  • Environment variables (AWS_ACCESS_KEY_ID, etc.)                     │ │ │
      │  │  │  • IAM role credentials                                                │ │ │
      │  │  │  • AWS CLI profile                                                     │ │ │
      │  │  └────────────────────────────────────────────────────────────────────────┘ │ │
      │  └──────────────────────────────────────────────────────────────────────────────┘ │
      └────────────────────────────────────────────────────────────────────────────────────┘

      ┌────────────────────────────────────────────────────────────────────────────────────┐
      │                          EXTERNAL INFRASTRUCTURE                                   │
      │  ┌──────────────────────────────────────────────────────────────────────────────┐ │
      │  │                             AWS S3 Service                                    │ │
      │  │  • Buckets: tidyllm-knowledge, vectorqa-artifacts, mlflow-models             │ │
      │  │  • Regions: us-east-1 (primary), us-west-2 (backup)                         │ │
      │  │  • Storage classes: STANDARD, INTELLIGENT_TIERING                           │ │
      │  └──────────────────────────────────────────────────────────────────────────────┘ │
      └────────────────────────────────────────────────────────────────────────────────────┘

  DEPENDENCY FLOW:
  ═══════════════
  1. Domain scripts → S3Manager (never directly to S3Service)
  2. S3Manager → UnifiedSessionManager (delegation pattern)
  3. UnifiedSessionManager → S3Service (adapter)
  4. S3Service → boto3 → AWS S3

  KEY PATTERNS:
  ═══════════
  • DELEGATION: S3Manager delegates infrastructure calls to UnifiedSessionManager
  • ABSTRACTION: Domain uses S3Manager's business methods, not AWS specifics
  • CONFIGURATION: Centralized in settings.yaml, loaded via credential_loader
  • FALLBACK: Graceful degradation when boto3 unavailable

  DATA FLOW EXAMPLE (File Upload):
  ═══════════════════════════════
  1. Script calls: s3_manager.upload_file("data.json")
  2. S3Manager:
     - Generates S3 key using business rules
     - Adds metadata (hash, timestamp, etc.)
  3. UnifiedSessionManager:
     - Gets S3 client with proper credentials
     - Manages connection lifecycle
  4. S3Service:
     - Executes boto3.upload_file()
     - Handles AWS-specific errors
  5. Response flows back with UploadResult

  TESTING BOUNDARIES:
  ═════════════════
  • Domain tests: Mock UnifiedSessionManager
  • Integration tests: Mock S3Service
  • E2E tests: Use LocalStack or test bucket

  Key Hexagonal Principles in S3 Implementation:

  1. Domain Independence: S3Manager contains business logic (path generation, versioning) without AWS dependencies
  2. Delegation Pattern: S3Manager delegates to UnifiedSessionManager instead of direct AWS calls
  3. Port Abstraction: IS3Storage interface defines contract, not implementation
  4. Adapter Isolation: S3Service is the only component with boto3 dependency
  5. Configuration Externalization: All AWS config in settings.yaml, not hardcoded
  6. Graceful Degradation: System works without boto3 (returns mock results)
  7. Clean Boundaries: Domain scripts never import boto3 or know about AWS

  This architecture allows:
  - Swapping S3 for MinIO, GCS, or local storage without changing domain
  - Testing without AWS credentials
  - Managing multiple S3 configurations (dev/staging/prod)
  - Monitoring and logging at adapter boundaries
