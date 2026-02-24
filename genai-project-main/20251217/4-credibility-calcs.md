Yes — you can add a **sample-size credibility** function as a *separate, reusable “trust weight”* that plugs into **quality / reliability / representation** without changing your core structure.

## 1) New function: sample-size credibility

The Bühlmann-style credibility weight is already perfect as a generic “how much should I trust this estimate?” knob:

[
c(n;K) ;=; \frac{n}{n+K}
]

* **n** = effective sample size behind the estimate (counts, tokens, examples, eval cases, etc.)
* **K** = “how many samples until I’m ~50% confident?” (the half-saturation point)

Properties: monotone, bounded ([0,1]), interpretable.

**Practical defaults**

* If you want *strong skepticism* until you have decent evidence, pick a larger (K).
* If (n) is noisy/weighted, use **effective sample size** (e.g., (n_\text{eff}=1/\sum w_i^2) for normalized weights).

## 2) Use it to make one more universal weight

Right now you have:
[
\text{posterior} = \omega \cdot \alpha + (1-\omega)\cdot prior
]
where (\omega) is “distribution alignment → reliability”.

Add credibility as a *second independent gate*:

### Option A (simple and clean): multiply gates

[
\omega_{\text{eff}} ;=; \omega_{\text{OOD}} \cdot c(n;K)
]
[
\text{posterior} ;=; \omega_{\text{eff}}\cdot \alpha + (1-\omega_{\text{eff}})\cdot prior
]

Interpretation:

* if **OOD is bad**, trust drops
* if **sample size is tiny**, trust drops
* you only trust the estimate when **both** are good

### Option B (more forgiving): noisy-OR combine

[
\omega_{\text{eff}} ;=; 1 - (1-\omega_{\text{OOD}})(1-c(n;K))
]
This says “either strong alignment or lots of data can rescue trust.” (Use when you don’t want small-n to completely kneecap in-distribution estimates.)

## 3) Where it plugs into “quality / reliability / representation”

You asked for “one more function we can use for quality, reliability, or representation.” Here are three clean placements (pick one pattern and reuse it everywhere):

### (i) Reliability weight (recommended)

Define reliability as **two-factor trust**:
[
R ;=; \omega_{\text{OOD}}\cdot c(n;K)
]
Then your existing blend is unchanged; you just compute (R) consistently.

### (ii) Quality shrinkage

If (\alpha) is a quality score estimated from data:
[
\alpha^* ;=; c(n;K)\cdot \alpha + (1-c(n;K))\cdot \alpha_{prior}
]
Then feed (\alpha^*) into your normal pipeline.

### (iii) Representation penalty (coverage-aware)

If “representation” is about how well your sample covers the domain, credibility can be your “coverage proxy”:
[
Rep^* ;=; Rep \cdot c(n;K)
]
(Useful when Rep is computed from embeddings/clusters but is unstable at low n.)

## 4) Mini table to match your style

|                    | Bühlmann                  | Ours (add credibility)                                |
| ------------------ | ------------------------- | ----------------------------------------------------- |
| Base blend         | (Z\bar{X}+(1-Z)\mu)       | (\omega\alpha+(1-\omega)prior)                        |
| Credibility weight | (Z=\frac{n}{n+K})         | (c(n;K)=\frac{n}{n+K})                                |
| Combined trust     | *(implicit)*              | (\omega_{\text{eff}}=\omega_{\text{OOD}}\cdot c(n;K)) |
| Meaning            | sample size → credibility | alignment × evidence → reliability                    |

If you want a single “drop-in” function name for the library, call it something like:

**`credibility_weight(n, K)`** → returns (c(n;K))
**`effective_reliability(ood_weight, n, K, mode="product"|"noisy_or")`** → returns (\omega_\text{eff})

If you tell me what you currently store as “OOD score” (is it higher=more OOD or higher=more aligned?), I’ll map it cleanly to (\omega_{\text{OOD}}\in[0,1]) so the semantics stay consistent everywhere.
