 ## Phase 2: Hexagonal Architecture (Ports & Adapters)
        83    
        84 +  ### Why Hexagonal?
        85 +  
        86 +  The YRSN package needs to work with multiple ecosystems (ROS2, CrewAI, REST, CLI) and multiple backends (OpenAI, Anthropic, ChromaDB, Redis).
           +  Hexagonal architecture provides:
        87 +  
        88 +  1. **Core isolation** - Business logic doesn't know about infrastructure
        89 +  2. **Testability** - Swap real adapters for mocks
        90 +  3. **Flexibility** - Add new LLM providers, storage backends without touching core
        91 +  4. **Plugin-friendly** - Community adapters via entry points
        92 +  
        93 +  ### Architecture
        94 +  
        95 +  ```
        96 +                      ┌─────────────────────────────────────────┐
        97 +                      │            INBOUND ADAPTERS             │
        98 +                      │  REST │ CLI │ ROS2 │ CrewAI │ MCP      │
        99 +                      └───────────────────┬─────────────────────┘
       100 +                                          │
       101 +                                          ▼
       102 +                      ┌─────────────────────────────────────────┐
       103 +                      │               PORTS (IN)                │
       104 +                      │  IObservationPort │ ICommandPort        │
       105 +                      └───────────────────┬─────────────────────┘
       106 +                                          │
       107 +                      ┌───────────────────▼─────────────────────┐
       108 +                      │                                         │
       109 +                      │              CORE DOMAIN                │
       110 +                      │                                         │
       111 +                      │   contracts/  - Typed schemas           │
       112 +                      │   core/       - Decomposition, τ, memory│
       113 +                      │   nodes/      - Lifecycle components    │
       114 +                      │   detect/     - OOD algorithms          │
       115 +                      │   reason/     - Reasoners, constraints  │
       116 +                      │   quality/    - Consistency checking    │
       117 +                      │                                         │
       118 +                      └───────────────────┬─────────────────────┘
       119 +                                          │
       120 +                      ┌───────────────────▼─────────────────────┐
       121 +                      │              PORTS (OUT)                │
       122 +                      │  ILLMPort │ IStoragePort │ IEventBus    │
       123 +                      └───────────────────┬─────────────────────┘
       124 +                                          │
       125 +                                          ▼
       126 +                      ┌─────────────────────────────────────────┐
       127 +                      │           OUTBOUND ADAPTERS             │
       128 +                      │  OpenAI │ Chroma │ Redis │ ROS2 Topics  │
       129 +                      └─────────────────────────────────────────┘
       130 +  ```
       131 +  
       132 +  ### Ports (Abstract Interfaces)
       133 +  
       134 +  ```python
       135 +  # yrsn/ports/llm.py
       136 +  from typing import Protocol, List
       137 +  from yrsn.contracts import Observation, YrsnState
       138 +  
       139 +  class ILLMClient(Protocol):
       140 +      """Port for LLM providers."""
       141 +      def complete(self, prompt: str, temperature: float = 0.7) -> str: ...
       142 +      def embed(self, text: str) -> List[float]: ...
       143 +  
       144 +  class IYRSNEnhancedLLM(Protocol):
       145 +      """Port for YRSN-enhanced LLM."""
       146 +      def complete_with_quality(self, prompt: str, min_alpha: float = 0.5) -> tuple[str, YrsnState]: ...
       147 +  ```
       148 +  
       149 +  ```python
       150 +  # yrsn/ports/storage.py
       151 +  from typing import Protocol, List, Optional, Any
       152 +  
       153 +  class IStorage(Protocol):
       154 +      """Port for key-value storage."""
       155 +      def get(self, key: str) -> Optional[Any]: ...
       156 +      def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None: ...
       157 +      def delete(self, key: str) -> None: ...
       158 +  
       159 +  class IVectorDB(Protocol):
       160 +      """Port for vector databases."""
       161 +      def add(self, id: str, embedding: List[float], metadata: dict) -> None: ...
       162 +      def query(self, embedding: List[float], k: int = 10) -> List[dict]: ...
       163 +      def delete(self, id: str) -> None: ...
       164 +  ```
       165 +  
       166 +  ```python
       167 +  # yrsn/ports/messaging.py
       168 +  from typing import Protocol, Callable, Any
       169 +  
       170 +  class IEventBus(Protocol):
       171 +      """Port for pub/sub messaging."""
       172 +      def publish(self, topic: str, message: Any) -> None: ...
       173 +      def subscribe(self, topic: str, callback: Callable) -> None: ...
       174 +      def unsubscribe(self, topic: str, callback: Callable) -> None: ...
       175 +  ```
       176 +  
       177 +  ---
       178 +  
       179 +  ## Phase 2b: Functional Organization
       180 +  
       181    ### What the Code Actually Does
       182    
       183    The merged `yrsn` package provides **13 functional domains**:

⏺ Now let me update the package structure to consolidate adapters:

⏺ Update(~/GitHub/yrsn-tools/MERGE_PLAN.md)
  ⎿  Updated ../../../yrsn-tools/MERGE_PLAN.md with 17 additions and 1 removal             
       260    
       261    ---
       262    
       263 -  ## Phase 2b: Package Structure (Comprehensive - 20 Modules)
       263 +  ## Phase 2c: Package Structure (Hexagonal - 18 Modules)
       264    
       265    ```
       266    yrsn/
     ...
       268    │   └── yrsn/
       269    │       ├── __init__.py              # Public API
       270    │       │
       271 +  │       │ ══════════════════════════════════════════════════════
       272 +  │       │                    PORTS (Interfaces)
       273 +  │       │ ══════════════════════════════════════════════════════
       274 +  │       │
       275 +  │       ├── ports/                   # Abstract interfaces (protocols)
       276 +  │       │   ├── __init__.py
       277 +  │       │   ├── llm.py               # ILLMClient, IYRSNEnhancedLLM
       278 +  │       │   ├── storage.py           # IStorage, IVectorDB, IExperienceMemory
       279 +  │       │   ├── retriever.py         # IRetriever, IReranker
       280 +  │       │   ├── messaging.py         # IEventBus, IMessageQueue
       281 +  │       │   └── hardware.py          # IMemristorArray, ICrossbar
       282 +  │       │
       283 +  │       │ ══════════════════════════════════════════════════════
       284 +  │       │                    CORE DOMAIN
       285 +  │       │ ══════════════════════════════════════════════════════
       286 +  │       │
       287    │       ├── contracts/               # Message schemas (pydantic)
       288    │       │   ├── __init__.py
       289    │       │   ├── observation.py       # Input contract

⏺ Update(~/GitHub/yrsn-tools/MERGE_PLAN.md)
  ⎿  Updated ../../../yrsn-tools/MERGE_PLAN.md with 68 additions and 50 removals              
       472    │       │       ├── __init__.py
       473    │       │       └── calibrator.py
       474    │       │
       475 -  │       ├── hardware/                # Neuromorphic computing
       475 +  │       ├── hardware/                # Neuromorphic computing (core algorithms)
       476    │       │   ├── __init__.py
       477    │       │   ├── memristor.py         # VirtualMemristor, YRSNMemristorProjection
       478 -  │       │   ├── crosssim.py          # CrossSim integration
       479 -  │       │   ├── adaptive_calibration.py  # Adaptive calibration
       478    │       │   └── snn.py               # Spiking neural network support
       479    │       │
       480    │       ├── tools/                   # Pre-built tools (from yrsn-tools)
     ...
       491    │       │   ├── multi_source_fusion.py     # Multi-source merging
       492    │       │   └── adaptive_router.py   # Temperature-based routing
       493    │       │
       494 -  │       ├── serve/                   # API + integrations
       495 -  │       │   ├── __init__.py
       496 -  │       │   ├── app.py               # YRSNApp, AppMode
       497 -  │       │   ├── tool_registry.py     # @YRSNTool decorator, registry
       498 -  │       │   ├── schema/              # Schema generation
       499 -  │       │   │   ├── __init__.py
       500 -  │       │   │   ├── openapi.py       # OpenAPI schema
       501 -  │       │   │   ├── mcp.py           # MCP manifest
       502 -  │       │   │   └── crewai.py        # CrewAI tools schema
       503 -  │       │   ├── crewai/              # CrewAI integration
       504 -  │       │   │   ├── __init__.py
       505 -  │       │   │   ├── tools.py         # YRSN_TOOLS
       506 -  │       │   │   ├── agents.py        # YRSN_AGENTS
       507 -  │       │   │   └── crews.py         # Pre-built crews
       508 -  │       │   └── langchain/           # LangChain integration
       509 -  │       │       └── __init__.py
       510 -  │       │
       511 -  │       ├── infrastructure/          # LLM clients & adapters
       512 -  │       │   ├── __init__.py
       513 -  │       │   ├── llm_adapters.py      # OpenAI, Anthropic, Bedrock, Ollama
       514 -  │       │   ├── model_adapters.py    # Model wrappers
       515 -  │       │   ├── retriever_adapters.py  # Retriever adapters
       516 -  │       │   └── storage/             # Storage backends
       517 -  │       │       ├── __init__.py
       518 -  │       │       └── chroma.py        # ChromaDB integration
       519 -  │       │
       520 -  │       ├── ros/                     # ROS2 adapter (thin layer)
       521 -  │       │   ├── __init__.py
       522 -  │       │   ├── adapters/            # msg ↔ dict ↔ contracts
       523 -  │       │   ├── nodes/               # ROS2 LifecycleNode wrappers
       524 -  │       │   ├── msgs/                # YrsnState.msg, YrsnEvent.msg
       525 -  │       │   ├── srvs/                # ScoreObservation.srv, ResetBaseline.srv
       526 -  │       │   └── launch/              # yrsn_stack.launch.py
       527 -  │       │
       528 -  │       ├── contrib/                 # Community extensions
       529 -  │       │   ├── __init__.py
       530 -  │       │   └── templates/
       531 -  │       │
       494    │       ├── apps/                    # Pre-built applications
       495    │       │   ├── __init__.py
       496    │       │   ├── quality.py           # quality_app
       497    │       │   ├── temperature.py       # temperature_app
       498    │       │   ├── retrieval.py         # retrieval_app
       499    │       │   ├── full.py              # full_app
       500 -  │       │   ├── crewai_ready.py      # crewai_app
       500    │       │   └── sudoku/              # Sudoku solver app
       501    │       │
       502    │       ├── data/                    # Datasets + benchmarks
     ...
       547    │       ├── visuals/                 # Visualization
       548    │       │   ├── __init__.py
       549    │       │   ├── visualizer.py        # Main visualizer
       550 -  │       │   ├── cli.py               # CLI for visuals
       550    │       │   └── panels/              # Visualization panels
       551 -  │       │       ├── __init__.py
       552 -  │       │       ├── base.py
       553 -  │       │       ├── board.py
       554 -  │       │       ├── timeline.py
       555 -  │       │       └── routing.py
       551    │       │
       552    │       ├── utils/                   # Utility functions
       553    │       │   └── __init__.py
       554    │       │
       555 -  │       ├── cli.py                   # Command-line interface
       556 -  │       └── training_config.py       # Training configuration
       555 +  │       ├── training_config.py       # Training configuration
       556 +  │       │
       557 +  │       │ ══════════════════════════════════════════════════════
       558 +  │       │                    ADAPTERS (Implementations)
       559 +  │       │ ══════════════════════════════════════════════════════
       560 +  │       │
       561 +  │       ├── adapters/                # Hexagonal adapters
       562 +  │       │   ├── __init__.py
       563 +  │       │   │
       564 +  │       │   ├── inbound/             # How things come IN to YRSN
       565 +  │       │   │   ├── __init__.py
       566 +  │       │   │   ├── rest.py          # FastAPI adapter (YRSNApp)
       567 +  │       │   │   ├── cli.py           # Command-line adapter
       568 +  │       │   │   ├── mcp.py           # MCP protocol adapter
       569 +  │       │   │   ├── crewai/          # CrewAI integration
       570 +  │       │   │   │   ├── __init__.py
       571 +  │       │   │   │   ├── tools.py     # YRSN_TOOLS
       572 +  │       │   │   │   ├── agents.py    # YRSN_AGENTS
       573 +  │       │   │   │   └── crews.py     # Pre-built crews
       574 +  │       │   │   ├── langchain/       # LangChain integration
       575 +  │       │   │   │   └── __init__.py
       576 +  │       │   │   └── ros2/            # ROS2 adapter
       577 +  │       │   │       ├── __init__.py
       578 +  │       │   │       ├── nodes.py     # ROS2 LifecycleNode wrappers
       579 +  │       │   │       ├── converters.py # msg ↔ contracts
       580 +  │       │   │       ├── msgs/        # YrsnState.msg, YrsnEvent.msg
       581 +  │       │   │       ├── srvs/        # ScoreObservation.srv
       582 +  │       │   │       └── launch/      # yrsn_stack.launch.py
       583 +  │       │   │
       584 +  │       │   └── outbound/            # How YRSN talks OUT
       585 +  │       │       ├── __init__.py
       586 +  │       │       ├── llm/             # LLM providers
       587 +  │       │       │   ├── __init__.py
       588 +  │       │       │   ├── openai.py    # OpenAI adapter
       589 +  │       │       │   ├── anthropic.py # Anthropic adapter
       590 +  │       │       │   ├── bedrock.py   # AWS Bedrock adapter
       591 +  │       │       │   ├── ollama.py    # Ollama adapter
       592 +  │       │       │   └── huggingface.py # HuggingFace adapter
       593 +  │       │       ├── storage/         # Storage backends
       594 +  │       │       │   ├── __init__.py
       595 +  │       │       │   ├── memory.py    # In-memory storage
       596 +  │       │       │   ├── chroma.py    # ChromaDB adapter
       597 +  │       │       │   ├── redis.py     # Redis adapter
       598 +  │       │       │   └── postgres.py  # PostgreSQL adapter
       599 +  │       │       ├── messaging/       # Event bus implementations
       600 +  │       │       │   ├── __init__.py
       601 +  │       │       │   ├── local.py     # In-process EventBus
       602 +  │       │       │   ├── redis.py     # Redis pub/sub
       603 +  │       │       │   └── ros2.py      # ROS2 topics
       604 +  │       │       └── hardware/        # Hardware backends
       605 +  │       │           ├── __init__.py
       606 +  │       │           ├── crosssim.py  # CrossSim memristor simulation
       607 +  │       │           └── virtual.py   # Virtual memristor (default)
       608 +  │       │
       609 +  │       │ ══════════════════════════════════════════════════════
       610 +  │       │                    SCHEMA & REGISTRATION
       611 +  │       │ ══════════════════════════════════════════════════════
       612 +  │       │
       613 +  │       ├── schema/                  # Schema generation
       614 +  │       │   ├── __init__.py
       615 +  │       │   ├── openapi.py           # OpenAPI schema
       616 +  │       │   ├── mcp.py               # MCP manifest
       617 +  │       │   └── crewai.py            # CrewAI tools schema
       618 +  │       │
       619 +  │       ├── contrib/                 # Community extensions (entry points)
       620 +  │       │   ├── __init__.py
       621 +  │       │   └── templates/
       622    │
       623    ├── tests/
       624    ├── examples/

