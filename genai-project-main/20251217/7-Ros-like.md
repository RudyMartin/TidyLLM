⏺ This is excellent analysis. The ROS patterns give you architectural wins even without ROS. Let me reshape the module plan:

  Revised Structure: Contract-First + Lifecycle-Aware

  yrsn/
  ├── __init__.py              # Public API
  │
  ├── contracts/               # 🆕 Message schemas (pydantic)
  │   ├── __init__.py
  │   ├── observation.py       # Input contract
  │   ├── state.py             # YrsnState (α, ω, τ, collapse, tags)
  │   ├── event.py             # YrsnEvent (typed events + severity)
  │   └── diagnostic.py        # DiagnosticArray equivalent
  │
  ├── runtime/                 # 🆕 Lifecycle + bus
  │   ├── __init__.py
  │   ├── lifecycle.py         # configure() → activate() → tick() → shutdown()
  │   ├── bus.py               # In-process pub/sub (upgradeable to NATS/Redis)
  │   └── registry.py          # Plugin discovery (entry points)
  │
  ├── core/                    # Engine (ROS-agnostic)
  │   ├── decomposition/       # R/S/N
  │   ├── temperature/         # τ mapping
  │   └── memory/              # Baseline, replay buffer
  │
  ├── nodes/                   # 🆕 Lifecycle components (like ROS nodes)
  │   ├── __init__.py
  │   ├── sensor_filter.py     # Cleaned signals + per-sensor R/S/N
  │   ├── quality_monitor.py   # Computes α, aggregates sources
  │   ├── anomaly_detector.py  # Computes ω, collapse flags
  │   └── safety_supervisor.py # Policy: slow/replan/stop
  │
  ├── detect/                  # OOD algorithms (pluggable)
  ├── reason/                  # Reasoners (pluggable)
  ├── serve/                   # API, MCP, tools
  │
  ├── ros/                     # 🆕 ROS2 adapter (thin layer)
  │   ├── adapters/            # msg ↔ dict ↔ contracts
  │   ├── nodes/               # ROS2 LifecycleNode wrappers
  │   ├── msgs/                # YrsnState.msg, YrsnEvent.msg
  │   ├── srvs/                # ScoreObservation.srv, ResetBaseline.srv
  │   └── launch/              # yrsn_stack.launch.py
  │
  ├── contrib/                 # Community extensions
  └── apps/                    # Pre-built apps (sudoku, etc.)

  ---
  Key Changes from Original Plan

  | Aspect        | Before          | After (ROS-informed)               |
  |---------------|-----------------|------------------------------------|
  | Contracts     | Implicit        | Explicit contracts/ module         |
  | Lifecycle     | None            | runtime/lifecycle.py               |
  | Components    | 20 flat modules | nodes/ with 4 core components      |
  | Observability | Ad-hoc          | contracts/diagnostic.py            |
  | Plugins       | contrib/ only   | runtime/registry.py + entry points |
  | ROS           | Not considered  | ros/ as thin adapter               |

  ---
  The 4-Node Architecture (Matches Your 10 Items)

  ┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
  │ sensor_filter   │────▶│ quality_monitor  │────▶│ anomaly_detector  │
  │ (R/S/N per src) │     │ (α, τ)           │     │ (ω, collapse)     │
  └─────────────────┘     └──────────────────┘     └───────────────────┘
                                                            │
                                                            ▼
                                                   ┌───────────────────┐
                                                   │ safety_supervisor │
                                                   │ (policy decisions)│
                                                   └───────────────────┘

  ---
  Contracts (Non-ROS, but ROS-compatible)

  # yrsn/contracts/state.py
  from pydantic import BaseModel
  from typing import List, Optional
  from datetime import datetime

  class YrsnState(BaseModel):
      """Core telemetry - maps to YrsnState.msg"""
      timestamp: datetime
      alpha: float          # Quality (0-1)
      omega: float          # OOD/reliability (0-1)  
      tau: float            # Temperature
      collapse: bool        # Collapse flag
      mode: str             # NORMAL / CAUTION / RECOVERY
      tags: List[str] = []  # Source attribution

  class YrsnEvent(BaseModel):
      """Discrete events - maps to YrsnEvent.msg"""
      timestamp: datetime
      event_type: str       # COLLAPSE_RISE, OOD_SPIKE, ALPHA_DROP...
      severity: float       # 0-1
      source_id: Optional[str] = None
      details: str = ""

  ---
  Lifecycle (Non-ROS, but ROS-compatible)

  # yrsn/runtime/lifecycle.py
  from enum import Enum
  from abc import ABC, abstractmethod

  class State(Enum):
      UNCONFIGURED = "unconfigured"  # No baseline
      INACTIVE = "inactive"          # Baseline built, not enforcing
      ACTIVE = "active"              # Enforcing thresholds
      ERROR = "error"                # Fallback policy

  class LifecycleNode(ABC):
      """Base class for all YRSN components"""

      def __init__(self):
          self._state = State.UNCONFIGURED

      @property
      def state(self) -> State:
          return self._state

      @abstractmethod
      def on_configure(self, config: dict) -> bool:
          """Build baseline, load models"""
          pass

      @abstractmethod  
      def on_activate(self) -> bool:
          """Start enforcing"""
          pass

      @abstractmethod
      def on_tick(self, observation) -> "YrsnState":
          """Process one observation"""
          pass

      @abstractmethod
      def on_shutdown(self) -> None:
          """Cleanup"""
          pass

      def configure(self, config: dict) -> bool:
          if self.on_configure(config):
              self._state = State.INACTIVE
              return True
          return False

      def activate(self) -> bool:
          if self._state == State.INACTIVE and self.on_activate():
              self._state = State.ACTIVE
              return True
          return False

  ---
  Module Count: 9 (down from 20)

  | Module     | Purpose                    |
  |------------|----------------------------|
  | contracts/ | Input/output schemas       |
  | runtime/   | Lifecycle, bus, registry   |
  | core/      | Engine internals           |
  | nodes/     | 4 lifecycle components     |
  | detect/    | OOD algorithms (pluggable) |
  | reason/    | Reasoners (pluggable)      |
  | serve/     | API/MCP/tools              |
  | ros/       | ROS2 thin adapter          |
  | contrib/   | Extensions                 |

  Plus apps/ for pre-built applications.

  ---
  Benefits Summary

  | Benefit          | Without ROS                | With ROS               |
  |------------------|----------------------------|------------------------|
  | Stable contracts | ✅ Pydantic models         | ✅ Direct .msg mapping |
  | Lifecycle        | ✅ State machine           | ✅ LifecycleNode       |
  | Observability    | ✅ Structured logs/metrics | ✅ DiagnosticArray     |
  | Replay           | ✅ JSONL/Parquet           | ✅ rosbag              |
  | Distributed      | ✅ NATS/Redis/HTTP         | ✅ ROS2 DDS            |
  | Plugins          | ✅ Entry points            | ✅ pluginlib           |

  Should I update MERGE_PLAN.md with this structure?

> yes  and will this also work with crewai? 

⏺ Yes! The contract-first architecture works perfectly with CrewAI. Let me update the plan:

