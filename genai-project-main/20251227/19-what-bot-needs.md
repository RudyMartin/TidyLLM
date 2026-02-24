  What Real Robot NLP Needs

  ┌─────────────────────────────────────────────────────────────────┐
  │  Voice: "Hey robot, pick up the red cup but be careful"        │
  │                              ↓                                  │
  │  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐    │
  │  │  ASR/TTS    │ →  │  UnifoLM /   │ →  │  YRSN Quality   │    │
  │  │  (BenBen)   │    │  External LLM│    │  Assessment     │    │
  │  └─────────────┘    └──────────────┘    └────────┬────────┘    │
  │                                                   ↓             │
  │                     ┌─────────────────────────────────────┐    │
  │                     │  Command-Level R/S/N Decomposition  │    │
  │                     │                                      │    │
  │                     │  R: Intent clarity (pick, careful)   │    │
  │                     │  S: Grounding (red cup → object_id)  │    │
  │                     │  N: Ambiguity (which cup? how careful?)│   │
  │                     │                                      │    │
  │                     │  α = intent_confidence × grounding   │    │
  │                     └────────────────┬────────────────────┘    │
  │                                      ↓                          │
  │                     ┌─────────────────────────────────────┐    │
  │                     │  Safety Gate (α > threshold?)       │    │
  │                     └────────────────┬────────────────────┘    │
  │                                      ↓                          │
  │                     ┌─────────────────────────────────────┐    │
  │                     │  Unitree SDK → Actuation            │    │
  │                     └─────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────────────┘
