● The hexagonal architecture view of the chat system:

  HEXAGONAL ARCHITECTURE - CHAT SYSTEM
  ====================================

      PORTS (Interface Layer)                      ADAPTERS (Infrastructure)
      ┌─────────────────────────┐                 ┌──────────────────────────┐
      │  chat_test_app.py       │                 │  BedrockService          │
      │  (Streamlit UI Port)    │                 │  (AWS Bedrock Adapter)   │
      │  - Port 8520            │                 │  - Claude models         │
      │  - User interface       │                 │  - Titan models          │
      └──────────┬──────────────┘                 └────────────┬─────────────┘
                 │                                              │
                 ▼                                              ▼
      ┌─────────────────────────┐                 ┌──────────────────────────┐
      │  DOMAIN SERVICE         │                 │  RAG Adapter             │
      │  UnifiedChatManager     │◄────────────────│  (Vector DB Integration) │
      │  ┌──────────────────┐  │                 │  - ChromaDB              │
      │  │ ChatMode Enum:   │  │                 │  - PostgreSQL            │
      │  │ - DIRECT         │  │                 └──────────────────────────┘
      │  │ - RAG            │  │                              ▲
      │  │ - DSPY           │  │                              │
      │  │ - HYBRID         │  │                 ┌──────────────────────────┐
      │  │ - CUSTOM         │  │◄────────────────│  DSPy Adapter            │
      │  └──────────────────┘  │                 │  (Prompt Optimization)   │
      │                         │                 └──────────────────────────┘
      │  Core Business Logic:   │
      │  - Mode routing         │                 ┌──────────────────────────┐
      │  - Response generation  │◄────────────────│  InfraDelegate           │
      │  - Context management   │                 │  (Consolidated Access)   │
      └─────────────────────────┘                 │  - Connection pooling    │
                                                  │  - Credential management │
                                                  └──────────────────────────┘

  FLOW:
  1. UI (Port) → receives user input
  2. UnifiedChatManager (Domain) → processes based on mode
  3. InfraDelegate → routes to appropriate adapter
  4. Adapter (Bedrock/RAG/DSPy) → executes infrastructure call
  5. Response flows back through layers

  The key hexagonal principles:
  - Ports: UI interfaces (Streamlit apps) that users interact with
  - Domain: Business logic (UnifiedChatManager) independent of infrastructure
  - Adapters: Infrastructure implementations (Bedrock, RAG, databases)
  - Dependency direction: Always inward (adapters depend on domain, never reverse)

● The hexagonal architecture view of the chat system:

  HEXAGONAL ARCHITECTURE - CHAT SYSTEM
  ====================================

           PRIMARY PORTS (User Side)                    SECONDARY PORTS (Infrastructure Side)
      ┌─────────────────────────────┐                 ┌──────────────────────────────┐
      │  chat_test_app.py (8520)    │                 │  IBedrock Interface          │
      │  chat_app.py (8502)         │                 │  IRAGSystem Interface        │
      │  (Streamlit UI Ports)       │                 │  IDSPyOptimizer Interface    │
      └──────────┬──────────────────┘                 └────────────▲─────────────────┘
                 │                                                  │
                 ▼                                                  │
      ┌──────────────────────────────────────────────────────────────────────────┐
      │                         DOMAIN CORE (Application Hexagon)                │
      │  ┌────────────────────────────────────────────────────────────────────┐ │
      │  │                    UnifiedChatManager                              │ │
      │  │  ┌──────────────────────────────────────────────────────────┐    │ │
      │  │  │  Business Rules:                                         │    │ │
      │  │  │  • Chat mode selection (DIRECT/RAG/DSPY/HYBRID/CUSTOM)  │    │ │
      │  │  │  • Message processing logic                              │    │ │
      │  │  │  • Context management                                    │    │ │
      │  │  │  • Response orchestration                                │    │ │
      │  │  └──────────────────────────────────────────────────────────┘    │ │
      │  └────────────────────────────────────────────────────────────────────┘ │
      └──────────────────────────────┬───────────────────────────────────────────┘
                                     │
                                     ▼
      ┌──────────────────────────────────────────────────────────────────────────┐
      │                         ADAPTERS (Infrastructure Layer)                   │
      │                                                                           │
      │  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────────┐    │
      │  │ BedrockAdapter    │  │ RAGAdapter       │  │ DSPyAdapter        │    │
      │  │ ─────────────────│  │ ─────────────────│  │ ──────────────────│    │
      │  │ • AWS Bedrock     │  │ • ChromaDB       │  │ • Prompt Library   │    │
      │  │ • Claude models   │  │ • PostgreSQL     │  │ • Optimization     │    │
      │  │ • Titan models    │  │ • Vector search  │  │ • Few-shot learns  │    │
      │  └──────────────────┘  └──────────────────┘  └────────────────────┘    │
      │                                                                           │
      │  ┌────────────────────────────────────────────────────────────────────┐ │
      │  │                    InfraDelegate (Adapter Orchestrator)             │ │
      │  │  • Manages all infrastructure connections                           │ │
      │  │  • Connection pooling (ResilientPoolManager)                       │ │
      │  │  • Credential management (from settings.yaml)                      │ │
      │  └────────────────────────────────────────────────────────────────────┘ │
      └──────────────────────────────────────────────────────────────────────────┘

  DEPENDENCY RULES:
  ═══════════════
  → Dependencies point INWARD only
  → Domain knows nothing about UI or Infrastructure
  → Adapters implement domain interfaces
  → UI depends on domain, never on infrastructure directly

  DATA FLOW:
  ═════════
  1. User Input → UI Port (chat_test_app.py)
  2. UI → Domain Service (UnifiedChatManager.chat())
  3. Domain → Port Interface (e.g., IBedrock)
  4. Port → Adapter Implementation (BedrockAdapter)
  5. Adapter → External Service (AWS Bedrock)
  6. Response flows back through same layers

  Key Hexagonal Principles in this Implementation:

  1. Domain Isolation: UnifiedChatManager contains pure business logic with no infrastructure dependencies
  2. Port Interfaces: Define contracts between domain and adapters
  3. Adapter Pattern: Each external service (Bedrock, RAG, DSPy) has its own adapter
  4. Dependency Inversion: Domain defines interfaces; adapters implement them
  5. Testability: Can swap real adapters with test doubles
  6. Flexibility: Can change infrastructure without touching domain logic
