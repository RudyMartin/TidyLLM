ow the Stanford lineage is properly emphasized. Here's the narrative:

  ---
  The Stanford Data Programming Story

  ┌─────────────────────────────────────────────────────────────────┐
  │                 STANFORD DATA PROGRAMMING LINEAGE               │
  ├─────────────────────────────────────────────────────────────────┤
  │                                                                 │
  │  2016: Data Programming (Ratner, Ré)                           │
  │        └─▶ "Labels have quality. Model it."                    │
  │                                                                 │
  │  2017: Snorkel (Ratner, Bach, Ré)                              │
  │        └─▶ "Labeling functions are noisy. Denoise them."       │
  │                                                                 │
  │  2019: Snorkel DryBell (Google deployment)                     │
  │        └─▶ "This scales to industry."                          │
  │                                                                 │
  │  2023: DSPy (Khattab, Potts, Zaharia)                          │
  │        └─▶ "Prompts are programs. Optimize them."              │
  │                                                                 │
  │  2024: YRSN (This work)                                        │
  │        └─▶ "Information has quality. Decompose it."            │
  │                                                                 │
  │  Pattern: Treat quality as latent → Model it → Optimize        │
  └─────────────────────────────────────────────────────────────────┘

  The key insight: Snorkel showed that you don't need perfect labels—you can learn their quality. YRSN shows
  you don't need perfect information—you can decompose its quality into R, S, and N.

  Our implementation: YRSN + DSPy = Quality-aware programmatic LLM pipelines

  This positions the work as a natural evolution of established, impactful research rather than an isolated
  contribution.
