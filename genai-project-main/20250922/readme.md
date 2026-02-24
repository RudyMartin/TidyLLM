flowchart LR
  %% ===== LAYOUT HINTS =====
  %% Keep Ports on the left, Domain center, Adapters right
  classDef port fill:#eef7ff,stroke:#4a90e2,stroke-width:1px,color:#0b3a75;
  classDef domain fill:#f7fff0,stroke:#69b34c,stroke-width:1px,color:#1e4d1e;
  classDef adapter fill:#fff5ee,stroke:#ef6c00,stroke-width:1px,color:#7a2f00;
  classDef infra fill:#fff5ee,stroke:#ef6c00,stroke-width:1px,color:#7a2f00;
  classDef enum fill:#fbfbfb,stroke:#999,stroke-dasharray: 3 3;

  %% ---------- PORTS ----------
  subgraph PORTS["PORTS (Interface Layer)"]
    direction TB
    P1[chat_test_app.py<br/>(Streamlit UI Port)<br/>• Port 8520 • User interface]
    class P1 port
  end

  %% ---------- DOMAIN ----------
  subgraph DOMAIN["DOMAIN SERVICE"]
    direction TB
    D1[UnifiedChatManager<br/><i>Core Business Logic</i><br/>• Mode routing<br/>• Response generation<br/>• Context management]
    class D1 domain

    D2[ChatMode Enum<br/>• DIRECT<br/>• RAG<br/>• DSPY<br/>• HYBRID<br/>• CUSTOM]
    class D2 enum
  end

  %% ---------- ADAPTERS ----------
  subgraph ADAPTERS["ADAPTERS (Infrastructure)"]
    direction TB

    A1[BedrockService<br/>(AWS Bedrock Adapter)<br/>• Claude models • Titan models]
    class A1 adapter

    A2[RAG Adapter<br/>(Vector DB Integration)<br/>• ChromaDB • PostgreSQL]
    class A2 adapter

    A3[DSPy Adapter<br/>(Prompt Optimization)]
    class A3 adapter

    A4[InfraDelegate<br/>(Consolidated Access)<br/>• Connection pooling<br/>• Credential management]
    class A4 infra
  end

  %% ---------- FLOWS ----------
  P1 --> D1
  D2 -. used by .- D1

  %% Domain uses InfraDelegate (inward dependency rule: adapters depend on domain contracts)
  D1 --> A4
  A4 --> A1
  A4 --> A2
  A4 --> A3

  %% Return path (responses flow back)
  A1 --> A4 --> D1 --> P1
  A2 --> A4 --> D1
  A3 --> A4 --> D1

Flow (high level):

UI Port receives user input

UnifiedChatManager processes by mode (DIRECT/RAG/DSPY/HYBRID/CUSTOM)

InfraDelegate selects the right adapter

Adapter executes infra call (Bedrock/RAG/DSPy)

Response bubbles back through Domain → Port

Hexagonal principles: Ports (UI), Domain (business logic), Adapters (infrastructure); dependencies point inward; adapters depend on domain contracts, never the reverse.


---------------

Hexagonal Architecture – Chat System (Detailed)
flowchart TB
  classDef port fill:#eef7ff,stroke:#4a90e2,stroke-width:1px,color:#0b3a75;
  classDef domain fill:#f7fff0,stroke:#69b34c,stroke-width:1px,color:#1e4d1e;
  classDef adapter fill:#fff5ee,stroke:#ef6c00,stroke-width:1px,color:#7a2f00;
  classDef infra fill:#fff5ee,stroke:#ef6c00,stroke-width:1px,color:#7a2f00;
  classDef iface fill:#fbfbfb,stroke:#999,stroke-dasharray: 4 3,color:#333;

  %% ---------- PRIMARY PORTS ----------
  subgraph PRIMARY_PORTS["PRIMARY PORTS (User Side)"]
    direction TB
    PP1[chat_test_app.py (8520)]
    PP2[chat_app.py (8502)]
    PP3[(Streamlit UI Ports)]
    class PP1,PP2,PP3 port
  end

  %% ---------- SECONDARY PORTS / INTERFACES ----------
  subgraph SECONDARY_PORTS["SECONDARY PORTS (Infrastructure Side)"]
    direction TB
    SP1[IBedrock Interface]
    SP2[IRAGSystem Interface]
    SP3[IDSPyOptimizer Interface]
    class SP1,SP2,SP3 iface
  end

  %% ---------- DOMAIN CORE ----------
  subgraph DOMAIN_CORE["DOMAIN CORE (Application Hexagon)"]
    direction TB
    DC1[UnifiedChatManager<br/><i>Business Rules</i><br/>• Mode selection (DIRECT/RAG/DSPY/HYBRID/CUSTOM)<br/>• Message processing<br/>• Context management<br/>• Response orchestration]
    class DC1 domain
  end

  %% ---------- ADAPTERS / INFRA ----------
  subgraph INFRA_ADAPTERS["ADAPTERS (Infrastructure Layer)"]
    direction LR

    A1[BedrockAdapter<br/>• AWS Bedrock<br/>• Claude/Titan]
    A2[RAGAdapter<br/>• ChromaDB • PostgreSQL • Vector search]
    A3[DSPyAdapter<br/>• Prompt Library • Optimization • Few-shot]
    class A1,A2,A3 adapter

    subgraph ORCH["InfraDelegate (Adapter Orchestrator)"]
      direction TB
      ID1[InfraDelegate<br/>• Manages connections<br/>• ResilientPoolManager<br/>• Credential mgmt (settings.yaml)]
    end
    class ID1 infra
  end

  %% ---------- DEPENDENCIES (INWARD) ----------
  PP1 --> DC1
  PP2 --> DC1
  PP3 --> DC1

  %% Domain talks to interfaces (ports), not concrete impls
  DC1 --> SP1
  DC1 --> SP2
  DC1 --> SP3

  %% Interfaces are implemented by adapters behind InfraDelegate
  SP1 -. through .-> ID1 --> A1
  SP2 -. through .-> ID1 --> A2
  SP3 -. through .-> ID1 --> A3

  %% Return paths (responses)
  A1 --> ID1 --> DC1
  A2 --> ID1 --> DC1
  A3 --> ID1 --> DC1
  DC1 --> PP1
  DC1 --> PP2


Dependency rules:

Dependencies point INWARD only

Domain knows nothing about UI or infra details

Adapters implement domain-defined interfaces (ports)

UI depends on domain, never directly on infra
