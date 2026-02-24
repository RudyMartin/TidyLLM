Continuing with the synthesis of the formula-by-formula prior art analysis. Let me consolidate all the research findings with the yrsn-context repo documentation.

  Formula-by-Formula Prior Art Report: YRSN 16 Collapse Types

  Quality Domain (Types 1-4)

  | #   | Type        | Formula               | Prior Art                                                                                            | Novelty Level                                         | Required Citations                  |
  |-----|-------------|-----------------------|------------------------------------------------------------------------------------------------------|-------------------------------------------------------|-------------------------------------|
  | 1   | POISONING   | N > 0.30              | STANDARD - Data poisoning (Biggio 2012) established. Threshold 0.30 is heuristic.                    | LOW - Detection concept standard; threshold arbitrary | Biggio et al. 2012                  |
  | 2   | DISTRACTION | S > 0.40              | NOVEL FRAMING - "Information overload" (Eppler 2004) exists but S-component framing is YRSN-specific | MEDIUM - S decomposition is novel                     | Eppler & Mengis 2004 (concept only) |
  | 3   | CONFUSION   | N > 0.20 AND S > 0.25 | NOVEL - Compound detection unique to YRSN                                                            | HIGH - No prior compound N+S detection                | None required (novel)               |
  | 4   | CLASH       | var(S_sources) > 0.15 | STANDARD CONCEPT - Source disagreement in multi-doc QA exists                                        | MEDIUM - Variance on S is novel                       | Multi-doc QA literature             |

  Reliability Domain (Types 5-7)

  | #   | Type          | Formula                         | Prior Art                                                 | Novelty Level                            | Required Citations                    |
  |-----|---------------|---------------------------------|-----------------------------------------------------------|------------------------------------------|---------------------------------------|
  | 5   | HALLUCINATION | alpha > 0.70 AND omega < 0.50   | STANDARD - Hallucination detection well-studied (Ji 2023) | LOW - alpha×omega interaction adds value | Ji et al. 2023                        |
  | 6   | O_POISONING   | R > 0.50 AND omega < 0.50       | NOVEL - OOD+signal interaction unique                     | HIGH - R×omega coupling novel            | Hendrycks & Gimpel 2017 (OOD concept) |
  | 7   | DRIFT         | omega < 0.35 OR d(omega)/dt < 0 | STANDARD - Dataset shift well-studied                     | LOW - omega as drift proxy adds value    | Quinonero-Candela et al. 2009         |

  Representation Domain (Types 8-10)

  | #   | Type               | Formula          | Prior Art                                 | Novelty Level             | Required Citations |
  |-----|--------------------|------------------|-------------------------------------------|---------------------------|--------------------|
  | 8   | POSTERIOR_COLLAPSE | Var(z) < 0.01    | STANDARD - VAE collapse (Bowman 2016)     | LOW - Threshold arbitrary | Bowman et al. 2016 |
  | 9   | MODE_COLLAPSE      | diversity < 0.20 | STANDARD - GAN collapse (Goodfellow 2016) | LOW - Threshold arbitrary | Goodfellow 2016    |
  | 10  | RSN_COLLAPSE       | `                | R-S                                       | < 0.15 AND                | S-N                |

  Uncertainty Domain (Types 11-13)

  | #   | Type                | Formula                                         | Prior Art                                                 | Novelty Level                                | Required Citations                     |
  |-----|---------------------|-------------------------------------------------|-----------------------------------------------------------|----------------------------------------------|----------------------------------------|
  | 11  | OVERCONFIDENCE      | epistemic < 0.10 AND alpha < 0.40               | STANDARD - Calibration (Guo 2017)                         | LOW - epistemic threshold standard           | Guo et al. 2017                        |
  | 12  | EPISTEMIC_SPIKE     | epistemic > 0.50 OR d(epistemic)/dt > threshold | STANDARD - MC Dropout uncertainty (Kendall & Gal 2017)    | MEDIUM - Rate-of-change detection adds value | Kendall & Gal 2017                     |
  | 13  | ALEATORIC_DOMINANCE | aleatoric > 2.0 × epistemic                     | STANDARD - Uncertainty decomposition (Nix & Weigend 1994) | LOW - Ratio threshold standard               | Nix & Weigend 1994; Kendall & Gal 2017 |

  Distributional Domain (Types 14-16)

  | #   | Type                 | Formula                              | Prior Art                            | Novelty Level                               | Required Citations            |
  |-----|----------------------|--------------------------------------|--------------------------------------|---------------------------------------------|-------------------------------|
  | 14  | DISTRIBUTIONAL_SHIFT | KS p-value < 0.05                    | STANDARD - KS test (Kolmogorov 1933) | LOW - Classic statistical test              | Kolmogorov 1933; Smirnov 1948 |
  | 15  | GRADUAL_DRIFT        | Wasserstein increasing monotonically | STANDARD - Concept drift (Gama 2014) | MEDIUM - Wasserstein for RSN novel          | Gama et al. 2014              |
  | 16  | CONFIRMED_DRIFT      | heuristic_trigger AND KS_confirms    | NOVEL - Two-stage detection unique   | HIGH - Heuristic→statistical pipeline novel | None (cite KS for component)  |

  ---
  Summary: YRSN Novelty Assessment

  Genuinely Novel Contributions (Claim these)

  1. RSN_COLLAPSE - No prior work detects R≈S≈N indistinguishability
  2. CONFUSION - Compound N+S threshold detection is unique
  3. O_POISONING - R×omega interaction for OOD-as-signal detection
  4. CONFIRMED_DRIFT - Two-stage heuristic→statistical pipeline
  5. S (Superfluous) Component - Three-way decomposition vs binary signal/noise

  Novel Applications of Standard Methods

  - Quality domain thresholds applied to R/S/N (novel decomposition, standard detection)
  - Temperature coupling τ = 1/α_ω (novel formula, standard adaptive LR concept)
  - Wasserstein on R/S/N distributions (standard metric, novel application)

  Standard Methods (Cite properly)

  - POISONING, HALLUCINATION, DRIFT - Standard concepts with YRSN framing
  - POSTERIOR_COLLAPSE, MODE_COLLAPSE - Direct from VAE/GAN literature
  - OVERCONFIDENCE, EPISTEMIC_SPIKE, ALEATORIC_DOMINANCE - Standard uncertainty quantification
  - KS test, Wasserstein distance - Classic statistical methods

  ---
  Alignment with yrsn-context Documentation

  The mathematical_formulations_extracted.md confirms the Y=R+S+N decomposition draws from:
  - Signal-noise separation (2404.04870) - but Y=R+S+N adds the S component
  - Low-rank matrix factorization (2409.19550) - conceptual alignment
  - Noisy Tucker decomposition (2208.08287) - tensor structure inspiration

  The YRSN_research_whitepaper.md properly cites Martin 2024 and Bagherian 2024 for foundation concepts.

  Verdict: The existing attribution in YRSN_Collapse_Types.md is accurate and complete. The 5 genuinely novel contributions above should be emphasized as YRSN's key innovations.

──────
