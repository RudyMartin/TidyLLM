

## 1. Which Sudoku-Bench leaderboard models are also on Bedrock?

From Table 1 in the paper: O3 Mini High, Gemini 2.5 Pro, Qwen 3.* (A22B/A3B/32B), DeepSeek R1, Grok 3 Mini, Qwen QwQ 32B, Qwen 3 32B, Claude 3.7 Sonnet (Thinking), GPT-4.1, Gemini 2.0 Flash, Gemma 3 27B IT, Llama 4 Maverick. 

**Direct or near-direct matches in Bedrock:**

1. **Claude 3.7 Sonnet (Thinking)**

   * Anthropic Claude 3.7 Sonnet is offered as a managed Bedrock FM (including the “Thinking” variant).

2. **DeepSeek R1**

   * DeepSeek-R1 is now a fully managed, serverless Bedrock model (plus distilled variants). ([Amazon Web Services, Inc.][1])

3. **Qwen 3.* family (Qwen 3.235B A22B, Qwen 3.30B A3B, Qwen 3 32B)**

   * Bedrock now offers **Qwen 3 foundation models** as managed FMs; the blog explicitly calls out Qwen 3 (MoE + dense). These line up with the A22B/A3B family used in Sudoku-Bench. ([Serverless Land][2])

4. **Qwen QwQ 32B**

   * Not a default Bedrock FM, but **Bedrock Custom Model Import** supports Qwen architectures, and AWS docs/community posts explicitly mention importing Qwen weights. So you can realistically run QwQ 32B *through* Bedrock infra even if it’s BYO weights. ([Reddit][3])

5. **Llama 4 Maverick**

   * Meta’s **Llama 4 Maverick 17B** is available as a fully-managed, serverless Bedrock model. ([Amazon Web Services, Inc.][4])

**Not on Bedrock as managed FMs (as of the current docs):**

* **O3 Mini High, GPT-4.1** – OpenAI’s *open-weight* GPT-OSS models (20B, 120B) *are* on Bedrock, but the closed O3/GPT-4.1 models are not. ([TechRadar][5])
* **Gemini 2.5 Pro, Gemini 2.0 Flash, Gemma 3 27B IT** – Google’s Gemini/Gemma live on Vertex & friends, not Bedrock.
* **Grok 3 Mini** – xAI stack; separate infra. I don’t see any indication it’s exposed via Bedrock.

So for your **Bedrock-based experiments** you can line up:

* Claude 3.7 Sonnet (Thinking)
* DeepSeek-R1
* One or more **Qwen 3** variants (A22B/A3B scale analogues)
* Llama 4 Maverick 17B
* Plus your existing **gpt-oss-20B** from OpenAI (not in the paper, but nice for cross-family comparison) ([TechRadar][5])

That gives you a very clean overlap between the paper’s leaderboard and your Bedrock environment.

---

## 2. How to make *your* PhD evaluation more robust than the paper

The Sudoku-Bench paper is already quite careful, but it has some limitations you can explicitly “fix” in your PhD design.

### 2.1. Multiple runs + confidence intervals (the paper doesn’t)

They explicitly say they run **a single evaluation per model per puzzle**, then average across 100 puzzles. 

For robustness:

* Run **K independent trials** per (model, puzzle) with different RNG seeds / temperature noise.
* Report **mean ± 95% CI** for:

  * Solve rate (per size & overall)
  * Correct placements per puzzle
* Use **paired tests** (McNemar / Wilcoxon signed-rank) for:

  * Vanilla vs YRSN *within the same architecture and puzzle set*.

That lets you say: *“YRSN improves Sudoku-Bench performance by X±Y percentage points (p < 0.01) across N puzzles”* instead of just eyeballing table rows.

### 2.2. Richer metrics directly targeted at your hypothesis

They track:

* Multi-step & single-shot solve rates
* Average correct placements


You can keep these but add **YRSN-specific metrics**:

1. **Break-in success rate**

   * Define a binary label: “model correctly identifies the first human-annotated logical breakthrough within T steps” on puzzles like *Ascension* and *Sumthings*.
   * Compare vanilla vs YRSN: does YRSN move more quickly to the human-like break-in, instead of brute-force search?

2. **Reasoning efficiency**

   * Tokens generated per solved puzzle
   * Number of multi-step turns until solution
   * “Cost per solve” (approx Bedrock $ or FLOPs)

   A good story for the committee: *YRSN not only improves accuracy, it makes reasoning more efficient, especially on larger grids.*

3. **Failure-mode distribution**
   The paper categorizes failures into Incorrect Solution, Surrender, Missing Information, Claimed Contradiction, No Reasoning Trace and uses Claude-3.5-Haiku to auto-label them. 

   You can re-use that taxonomy but show:

   * Baseline vs YRSN: proportion of each error type
   * Hypothesis: YRSN should **reduce Missing Information / bogus contradictions** (better internal consistency) and **reduce Surrender** at similar or lower hallucination rates.

### 2.3. Stronger experimental design around architectures

The paper just compares **whatever APIs are available**, so architecture and training details are wildly heterogeneous. You’re explicitly doing:

* CTM (recurrent, neuron-level) and HRM (hierarchical RNN)
* Vanilla Transformer
* YRSN-CTM, YRSN-HRM, YRSN-Transformer
* Bedrock FMs (Claude, Qwen, DeepSeek, Llama, gpt-oss)

Ways to make this **defense-tight**:

1. **Within-architecture ablations**

   For each of CTM, HRM, Transformer:

   * Vanilla model
   * +YRSN, **same** parameter budget, training data, and decoding settings
   * Report:

     * ΔSolve rate
     * ΔCorrect placements
     * ΔBreak-in success
     * ΔTokens / puzzle

   That isolates YRSN as the *only* difference.

2. **Cross-architecture generalization**

   Your key claim is “YRSN is architecture-agnostic.”

   * Show consistent gains on:

     * Recurrent family (CTM/HRM)
     * Self-attention (Transformer)
     * At least one Bedrock FM family (e.g., Qwen 3 or Llama 4) via a **YRSN adapter** (reasoning-layer on top of the FM).

   If gains appear **in both local models and hosted FMs**, that’s a strong story.

3. **Controlled “reasoning budget”**

   The paper allows models to run until first wrong digit in multi-step; it doesn’t normalize the “cost” across models. 

   You can:

   * Fix a **max token budget or max turns** per puzzle per model.
   * Compare: at equal budget, YRSN models get higher solve rates and more correct placements.

   That’s exactly the kind of rigor a committee likes.

### 2.4. Better use of the Sudoku-Bench ecosystem

The paper mentions:

* **challenge_100** (main benchmark)
* **nikoli_100** (creative vanilla Sudokus)
* **ctc** (≈2,565 Cracking the Cryptic puzzles + rich human traces) 

And a large **transcript + SudokuPad-actions dataset** from Cracking the Cryptic. 

You can stay “within scope” but do more:

1. **Train/validation/test split by subset**

   * Use part of **ctc** for *tuning* YRSN (e.g., prompt templates, hyperparameters).
   * Reserve **challenge_100** strictly as a **test set only**.
   * Possibly use **nikoli_100** as an out-of-distribution *generalization* set.

2. **Human-trace-aware analysis (lightweight)**

   Without fully training on the transcripts, you can:

   * Align your models’ step-by-step traces with the human logs on a small puzzle subset.
   * Score overlap in:

     * Constraint references (e.g., “arrow”, “anti-knight”, “parity”)
     * Structural operations (case splits, contradiction search).
   * Compare vanilla vs YRSN: does YRSN produce *more human-like reasoning structure*?

   This can be a **small chapter or appendix**, not a whole new project.

### 2.5. Reproducibility & reporting

The paper already ships:

* The dataset on Hugging Face and GitHub. 

You can layer on:

1. **Config-first evaluation harness**

   * All runs (local + Bedrock) driven by JSON/YAML configs:

     * model_id / model_key
     * decoding params
     * puzzle subset
     * seeds / trials
     * “reasoning budget” (max tokens, max rounds)

2. **Open-sourced post-processing notebook → `.py` script**

   * You already started this; extend it to:

     * Auto-compute CIs and significance tests
     * Produce all tables + figures for the chapter
     * Emit a “Sudoku-Bench-YRSN leaderboard” CSV that mirrors Table 1.

3. **Bedrock-specific logging**

   * For each Bedrock FM: log **model ARN, version, region, and date** of evaluation.
   * That makes replication realistic despite Bedrock’s evolving catalog.

---

## 3. How to frame this in the dissertation

If I’m sitting on your committee, the **high-level narrative** I’d want:

1. *Baseline:* Reproduce something close to Table 1 with a subset of public models (including Bedrock FMs where they overlap).
2. *Core contribution:* Carefully controlled comparisons of **with vs without YRSN** across:

   * CTM, HRM, Transformer
   * At least one Bedrock FM family (e.g., Qwen or Llama 4) via adapter.
3. *Robustness:* Multiple trials, statistical tests, failure-mode shifts, and cost/efficiency metrics.
4. *Insight:* Show *why* YRSN helps: better break-in detection, fewer contradictory / “missing info” failures, more human-like reasoning traces (at least qualitatively).

If you want, next step we can sketch the **exact experiment grid** (models × puzzle subsets × trials) and the **tables/figures** that would go into the “Sudoku-Bench + YRSN” chapter.

[1]: https://aws.amazon.com/bedrock/deepseek/?utm_source=chatgpt.com "DeepSeek - Models in Amazon Bedrock – AWS"
[2]: https://serverlessland.com/blog/qwen-models-are-now-available-in-amazon-bedrock---aws-news-blog?utm_source=chatgpt.com "Qwen models are now available in Amazon Bedrock"
[3]: https://www.reddit.com/r/gpt5/comments/1lalo6n/amazon_introduces_model_import_for_qwen_on/?utm_source=chatgpt.com "Amazon Introduces Model Import for Qwen on Bedrock ..."
[4]: https://aws.amazon.com/blogs/aws/llama-4-models-from-meta-now-available-in-amazon-bedrock-serverless/?utm_source=chatgpt.com "Llama 4 models from Meta now available in ..."
[5]: https://www.techradar.com/pro/openai-has-new-smaller-open-models-to-take-on-deepseek-and-theyll-be-available-on-aws?utm_source=chatgpt.com "OpenAI has new, smaller open models to take on DeepSeek - and they'll be available on AWS for the first time"
