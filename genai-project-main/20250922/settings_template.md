configuration:
  architecture_version: v2_4layer_clean
  credential_patterns:
  - '*password*'
  - '*secret*'
  - '*key*'
  - api_keys.*
  - credentials.*
  - postgres.db_password
  - aws.access_key_id
  - aws.secret_access_key
  environment: staging
  environment_overrides:
    development:
      adapter_debug: true
      log_level: DEBUG
      s3_prefix: dev/
    local:
      adapter_development: true
      log_level: INFO
      s3_prefix: local/
    production:
      adapter_optimization: true
      log_level: WARNING
      s3_prefix: prod/
    staging:
      adapter_monitoring: true
      log_level: INFO
      s3_prefix: staging/
  generated_from: v2_4layer_clean_template
  last_updated: '2025-09-15'
  loaded_by: v2_config_loader
  official:
    contains_credentials: true
    is_template_source: true
    location: tidyllm/admin/settings.yaml
    v2_compliant: true
  security:
    adapter_validation: true
    audit_changes: true
    connection_pool_validation: true
    encrypt_at_rest: false
    file_permissions: '0600'
    validate_on_load: true
  template_generation:
    add_placeholders: true
    mask_sensitive: true
    output_template: onboarding/template.settings.yaml
    strip_credentials: true
    v2_annotations: true
  version: 2.0.0
credentials:
  aws_basic:
    access_key_id: <ACCESS_KEY_ID>
    default_region: us-east-1
    profile: null
    secret_access_key: <SECRET_ACCESS_KEY>
    secrets_manager:
      access_control: corporate_managed
      auto_refresh: true
      cache_ttl_seconds: 3600
      enabled: true
      managed_by: corporate_dba
      region: us-east-1
      secret_arn: <ARN_SECRET_STRING>
      secret_string_key: password
      secret_string_value: <SECRET_KEY_STRING_PWD>
    type: aws_credentials
  bedrock_llm:
    adapter_config:
      circuit_breaker: true
      retry_attempts: 3
      timeout: 60
    default_model: anthropic.claude-3-sonnet-20240229-v1:0
    disabled_models: []
    embeddings:
      batch_size: 25
      cache_enabled: true
      dimensions: 1024
      max_chunk_size: 2000
      model_id: cohere.embed-english-v3
      normalize: true
      timeout: 30
      type: bedrock_embeddings
    model_mapping:
      claude-3-5-sonnet: anthropic.claude-3-5-sonnet-20240620-v1:0
      claude-3-haiku: anthropic.claude-3-haiku-20240307-v1:0
      claude-3-opus: anthropic.claude-3-opus-20240229-v1:0
      claude-3-sonnet: anthropic.claude-3-sonnet-20240229-v1:0
      titan-v2-recommended-1024d: amazon.titan-embed-text-v2:0
    region: us-east-1
    service_provider: aws_bedrock
    type: llm_service
  mlflow_alt_db:
    database: mlflow_separate
    engine: postgresql
    host: alternative-mlflow-db.us-east-1.rds.amazonaws.com
    password: mlflow_password_here
    port: 5432
    purpose: alternative_mlflow_backend
    ssl_mode: require
    type: database_credentials
    username: mlflow_user
  mlflow_api:
    artifact_store: s3://<BUCKET>/<PREFIX>/mlflow/
    service: mlflow
    tracking_uri: postgresql://<DB_USER>:<DB_PWD>@<DB_HOST>:5432/<DB_NAME>
    type: api_credentials
  postgresql_primary:
    connection_pool:
      enabled: true
      max_connections: 20
      min_connections: 2
      pool_recycle: 3600
      pool_timeout: 30
    database: <DB_NAME>
    engine: postgresql
    host: <DB_HOST>
    password: <DB_PWD>
    port: 5432
    ssl_mode: require
    type: database_credentials
    username: <DB_USER>
  sqlite_backup:
    database: ./data/backup.db
    engine: sqlite
    host: ./data/backup.db
    password: YOUR_BACKUP_PASSWORD_HERE
    security:
      encryption: false
      file_permissions: '0600'
    type: database_credentials
    username: backup_user
databases:
  backup:
    auto_create: true
    backup_rotation:
      enabled: true
      location: ./data/backups/
      retention_days: 7
    connection_pool_size: 1
    credential_ref: sqlite_backup
    db_path: ./data/backup.db
    engine: sqlite
    max_retries: 3
    retry_delay: 2
    security:
      encryption: false
      file_permissions: '0600'
      secure_delete: true
      wal_mode: true
    type: v2_sqlite_adapter
  primary:
    connection_pool_size: 5
    credential_ref: postgresql_primary
    engine: postgresql
    features:
      connection_validation: true
      health_monitoring: true
      performance_metrics: true
      query_logging: true
    max_retries: 3
    pool_client_name: primary_adapter
    retry_delay: 2
    ssl_mode: require
    type: v2_postgres_adapter
    use_connection_pool: true
  secondary:
    connection_pool_size: 3
    credential_ref: secondary_credentials
    database: backup_db
    engine: postgresql
    host: localhost
    max_retries: 3
    password: ''
    port: 5432
    retry_delay: 2
    ssl_mode: prefer
    type: v2_postgres_adapter
    username: backup_user
deployment:
  adapter_registry:
    auto_discovery: true
    enabled: true
    health_monitoring: true
    validation: true
  architecture: v2_4layer_clean
  backup:
    enabled: true
    format: settings_backup_{timestamp}.yaml
    location: ./admin/backups/
    retention_days: 30
    v2_metadata: true
  connection_pool:
    enabled: true
    global_pool: true
    health_checks: true
    monitoring: true
    statistics: true
  mode: staging
  search_paths:
  - description: V2 Official admin config with credentials
    path: tidyllm/admin/settings.yaml
    permissions: '0600'
    v2_compliant: true
  - description: V2 Local development config
    path: ./settings.yaml
    permissions: '0600'
    v2_fallback: true
  - description: V2 Repository-level config
    path: ../settings.yaml
    permissions: '0600'
    v2_template: true
  - description: V2 User profile config
    path: ~/.tidyllm/settings.yaml
    permissions: '0600'
    v2_personal: true
features:
  gateways:
    ai_processing:
      adapter_type: AIProcessingAdapter
      batch_processing: true
      enabled: true
      retry_attempts: 2
      timeout: 45
    corporate_llm:
      adapter_type: CorporateLLMAdapter
      circuit_breaker: true
      enabled: true
      retry_attempts: 3
      timeout: 30
    knowledge_mcp:
      adapter_type: KnowledgeMCPAdapter
      enabled: false
      retry_attempts: 3
      timeout: 30
      vector_search: false
    workflow_optimizer:
      adapter_type: WorkflowOptimizerAdapter
      dag_support: true
      enabled: true
      retry_attempts: 2
      timeout: 60
  security:
    access:
      auth_adapter: V2AuthAdapter
      rate_limit:
        adapter_type: RateLimitAdapter
        enabled: true
        requests_per_hour: 1000
        requests_per_minute: 100
      require_auth: false
    audit:
      audit_adapter: V2AuditAdapter
      audit_all_requests: false
      audit_retention_days: 7
      enabled: true
    data:
      cache_retention_days: 1
      encrypt_cache: false
      encrypt_logs: false
      encryption_adapter: V2EncryptionAdapter
      log_retention_days: 7
      mask_sensitive_data: true
  workflow_optimizer:
    adapter_config:
      circuit_breaker: true
      enable_metrics: false
      use_hexagonal_ports: true
    audit_trail: false
    compliance_mode: false
    enable_auto_optimization: false
    enable_dag_manager: false
    enable_flow_agreements: true
    max_workflow_depth: 10
    optimization_level: 0
    performance_threshold: 0.8
    timeout: 60
integrations:
  ldap:
    adapter_type: V2LDAPAdapter
    connection_pool: true
    enabled: false
    type: ldap_integration
  mlflow:
    adapter_type: V2MLflowAdapter
    artifact_store: s3://<BUCKET>/<PREFIX>/mlflow/
    backend_options:
      alternative: mlflow_alt_db
      fallback: file://./mlflow_data
      primary: postgresql_shared_pool
      test_mode: null
    backend_store_uri: auto_select
    credential_ref: mlflow_api
    enabled: true
    integration_config:
      backend_fallback_enabled: true
      circuit_breaker: true
      health_monitoring: false
      metrics_collection: false
      pool_client_name: mlflow_integration
      test_backend_on_startup: true
      use_shared_pool: true
    mlflow_gateway_uri: http://localhost:5000
    server:
      host: 0.0.0.0
      port: 5000
    tracking_uri: postgresql://<DB_USER>:<DB_PWD>@<DB_HOST>:5432/<DB_NAME>
    type: tracking_integration
  observability:
    datadog:
      adapter_type: V2DatadogAdapter
      enabled: false
    newrelic:
      adapter_type: V2NewRelicAdapter
      enabled: false
    type: observability_integration
  sso:
    adapter_type: V2SSOAdapter
    enabled: false
    token_validation: true
    type: sso_integration
llm_models:
  bedrock:
    enable_fallback: true
    max_tokens: 4096
    model_id: anthropic.claude-3-haiku-20240307-v1:0
    region: us-east-1
    stream: true
    temperature: 0.7
onboarding:
  allow_updates: true
  architecture_version: v2
  auto_detect_root_path: true
  backup_before_update: true
  enable_health_checks: true
  health_checks:
    connection_pool: false
    database_adapters: false
    external_adapters: false
    integration_adapters: false
    service_adapters: false
  max_retry_attempts: 3
  refresh_interval: 5
  show_debug_info: false
  template_source: true
  v2_features:
    adapter_validation: true
    architecture_compliance: true
    architecture_verification: true
    connection_pool_init: true
    port_binding_check: true
operations:
  logging:
    adapter_type: V2LoggingAdapter
    backup_count: 5
    file_rotation: true
    format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers:
      file:
        backup_count: 3
        enabled: true
        max_size_mb: 50
        path: ./logs/tidyllm.log
    include_request_id: true
    include_session_id: true
    include_user_id: false
    level: INFO
    log_correlation: true
    max_file_size: 10MB
    structured_logging: true
  monitoring:
    adapter_type: V2MonitoringAdapter
    alerts:
      alert_adapter: V2AlertAdapter
      enabled: false
    enabled: true
    health_check:
      adapter_validation: true
      enabled: true
      endpoint: /health
      interval_seconds: 120
    metrics:
      adapter_metrics: true
      connection_pool_metrics: true
      enabled: true
      export_to_prometheus: false
  network:
    adapter_type: V2NetworkAdapter
    proxy:
      enabled: false
      proxy_adapter: V2ProxyAdapter
    ssl:
      cert_validation: true
      verify: true
    timeouts:
      adapter_timeout: 30
      connect_timeout: 15
      read_timeout: 60
      total_timeout: 120
paths:
  cache_folder: infrastructure/cache
  config_folder: infrastructure/admin
  data_folder: infrastructure/data
  domain_folder: domain
  logs_folder: infrastructure/logs
  onboarding_folder: infrastructure/onboarding
  path_separator: \
  portals_folder: portals
  root_path: C:\Users\XXXX\compliance-qa
  temp_folder: infrastructure/temp
  templates_folder: domain/templates
  workflows_folder: domain/workflows
search_paths:
- description: V2 Official admin config with credentials
  path: tidyllm/admin/settings.yaml
  permissions: '0600'
  v2_compliant: true
services:
  aws_infrastructure:
    credential_ref: aws_basic
    region: us-east-1
    secrets_manager:
      access_control: corporate_managed
      adapter_config:
        cache_enabled: true
        cache_ttl_seconds: 3600
        circuit_breaker: true
        retry_attempts: 3
        timeout: 30
      adapter_type: V2SecretsManagerAdapter
      enabled: true
      health_check_interval: 300
      managed_by: corporate_dba
      region: us-east-1
      secret_arn: <ARN_SECRET_STRING>
      secret_string_value: <DB_PWD>
    type: aws_service
  data_tracking:
    adapter_config:
      health_check: true
      pool_client_name: mlflow_service
      use_connection_pool: true
    artifact_store: s3://<BUCKET>/<PREFIX>/mlflow/
    backend_options:
      alternative: mlflow_alt_db
      fallback: file://./mlflow_data
      primary: postgresql_shared_pool
      s3_artifacts_only: true
    backend_store_uri: auto_select
    credential_ref: mlflow_api
    enabled: true
    mlflow_gateway_uri: http://localhost:5000
    server:
      host: 0.0.0.0
      port: 5000
    tracking_uri: postgresql://<DB_USER>:<DB_PWD>@<DB_HOST>:5432/<DB_NAME>
    type: tracking_service
  database_service:
    adapter_type: V2PostgresAdapter
    credential_ref: postgresql_primary
    health_check_interval: 60
    query_timeout: 30
    type: database_service
    use_shared_pool: true
  llm_processing:
    credential_ref: bedrock_llm
    enabled: true
    type: llm_service
  s3:
    adapter_config:
      chunk_size: 8388608
      max_concurrency: 10
      multipart_threshold: 67108864
    bucket: <BUCKET>
    connection_timeout: 30
    max_retries: 3
    prefix: staging/new/
    region: us-east-1
    test_marker: test_marker_v2
    type: s3_service
system:
  architecture:
    enable_adapters: true
    enable_ports: true
    pattern: 4layer_clean
    strict_boundaries: true
    version: v2
  auto_detect_root_path: true
  corporate_mode: false
  deep_folder_support: true
  deployment_type: development
  environment: staging
  organization: TidyLLM V2
  test_deployment: true
testing:
  environment_variables:
    DB_TIMEOUT: 10
    DISABLE_MLFLOW_TELEMETRY: '1'
    MLFLOW_HTTP_TIMEOUT: 15
    MLFLOW_TELEMETRY_OPT_OUT: '1'
    MLFLOW_TRACKING_TIMEOUT: 15
    POOL_TIMEOUT: 10
    REQUESTS_TIMEOUT: 10
  retry_policies:
    api_calls: 3
    connection_attempts: 3
    database_operations: 3
    mlflow_operations: 3
  standardized_config:
    connection_timeout: 10
    default_timeout: 30
    max_retries: 3
    mlflow_timeout: 15
    no_mocks_policy: true
    use_real_infrastructure: true

