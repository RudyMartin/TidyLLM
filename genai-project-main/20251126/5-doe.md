Nice—let’s lock this in so it’s “drop-in ready” for code + thesis.

I’ll give you four concrete artifacts:

1. A DOE table for the **Sudoku-Bench × Bedrock models × YRSN** experiments
2. A **config YAML** for your Bedrock models
3. Standardized **prompt templates** (single-shot + multi-step)
4. A **README / appendix text block** describing the setup

You can paste any of these straight into your repo / thesis.

---

## 1️⃣ DOE Table – Sudoku-Bench × Bedrock Transformers × YRSN

Assume core subsets:

* **challenge_100** (main)
* **nikoli_100** (classical)

And two protocols:

* **SS** = Single-Shot (full grid in one response)
* **MS** = Multi-Step (iterative cell or move proposals)

### 1.1 Model Set

We’ll treat **YRSN** as an *external reasoning enhancement* (prompting/agent layer) wrapped around each API model.

| ID            | Provider / Model ID                      | Size Class | Used As                        |
| ------------- | ---------------------------------------- | ---------- | ------------------------------ |
| M1            | anthropic.claude.haiku-4-5-20251001-v1:0 | Small      | Fast/cheap baseline            |
| M2            | meta.llama3-1-8b-instruct-v1:0           | Small–Mid  | Open-ish baseline              |
| M3            | openai.gpt-oss-20b1:0                    | Mid        | Canonical transformer baseline |
| M4            | anthropic.sonnet-4-5-20250929-v1:0       | Mid–Large  | Strong reasoning baseline      |
| M5            | anthropic.opus-4-5-20251101-v1:0         | Large      | Upper bound sanity check       |
| M6 (optional) | deepseek.v3-v1:0                         | Mid        | Alternate inductive bias       |

You can treat CTM/HRM as a **separate family block** in the DOE (you already conceptually have those).

### 1.2 Experimental Grid (per model)

For each model Mi:

| Model | YRSN        | Protocol | Subset        | Repeats | Metrics                                         |
| ----- | ----------- | -------- | ------------- | ------- | ----------------------------------------------- |
| Mi    | ✗ (vanilla) | SS       | challenge_100 | N seeds | Solve rate, first error depth, valid board %    |
| Mi    | ✗           | MS       | challenge_100 | N seeds | Solve rate, steps to solve/fail, contradictions |
| Mi    | ✓ (YRSN)    | SS       | challenge_100 | N seeds | Same metrics                                    |
| Mi    | ✓           | MS       | challenge_100 | N seeds | Same metrics                                    |
| Mi    | ✗           | SS       | nikoli_100    | N seeds | Solve rate, etc.                                |
| Mi    | ✓           | SS       | nikoli_100    | N seeds | Solve rate, etc.                                |

**N seeds**: e.g. 3–5 runs per puzzle if you want robustness to sampling.

High-priority runs if budget is tight:

* All models on **challenge_100, SS + MS**
* Haiku/Llama/OSS-20B/Sonnet on **nikoli_100 (SS)**
* Opus only on **challenge_100 (SS + MS)** as reference

This is already a strong chapter-level DOE.

---

## 2️⃣ Bedrock Config – models.yaml

Here’s a clean YAML config you can use in your framework (`config_helper`, `config_manager`, etc.):

```yaml
bedrock_models:
  anthropic_claude_haiku_4_5:
    provider: anthropic
    model_id: anthropic.claude.haiku-4-5-20251001-v1:0
    family: transformer
    size_class: small
    context_window_tokens: 200000      # adjust from docs if needed
    max_output_tokens: 4096
    enabled: true
    enable_yrsn_wrapper: true
    default_temperature: 0.2
    default_top_p: 0.9
    cost_tier: low

  anthropic_claude_sonnet_4_5:
    provider: anthropic
    model_id: anthropic.sonnet-4-5-20250929-v1:0
    family: transformer
    size_class: medium_large
    context_window_tokens: 200000
    max_output_tokens: 4096
    enabled: true
    enable_yrsn_wrapper: true
    default_temperature: 0.2
    default_top_p: 0.9
    cost_tier: medium

  anthropic_claude_opus_4_5:
    provider: anthropic
    model_id: anthropic.opus-4-5-20251101-v1:0
    family: transformer
    size_class: large
    context_window_tokens: 200000
    max_output_tokens: 4096
    enabled: true
    enable_yrsn_wrapper: true
    default_temperature: 0.15
    default_top_p: 0.9
    cost_tier: high

  openai_gpt_oss_20b1:
    provider: openai
    model_id: openai.gpt-oss-20b1:0
    family: transformer
    size_class: medium
    context_window_tokens: 128000
    max_output_tokens: 4096
    enabled: true
    enable_yrsn_wrapper: true
    default_temperature: 0.2
    default_top_p: 0.9
    cost_tier: medium

  deepseek_v3:
    provider: deepseek
    model_id: deepseek.v3-v1:0
    family: transformer
    size_class: medium
    context_window_tokens: 128000
    max_output_tokens: 4096
    enabled: true
    enable_yrsn_wrapper: true
    default_temperature: 0.2
    default_top_p: 0.9
    cost_tier: medium

  meta_llama3_1_8b_instruct:
    provider: meta
    model_id: meta.llama3-1-8b-instruct-v1:0
    family: transformer
    size_class: small_medium
    context_window_tokens: 8192       # update from docs
    max_output_tokens: 2048
    enabled: true
    enable_yrsn_wrapper: true
    default_temperature: 0.2
    default_top_p: 0.9
    cost_tier: low

  meta_llama3_2_11b_instruct:
    provider: meta
    model_id: meta.llama3-2-11b-instruct-v1:0
    family: transformer
    size_class: medium
    context_window_tokens: 8192       # update from docs
    max_output_tokens: 2048
    enabled: true
    enable_yrsn_wrapper: true
    default_temperature: 0.2
    default_top_p: 0.9
    cost_tier: low_medium
```

Tie this into your existing `get_model_details(model_key)` pattern and you’re set.

---

## 3️⃣ Standard Sudoku-Bench Prompt Templates

You want **one set of templates** used for *all* transformers to keep the DOE clean.

### 3.1 Single-Shot Prompt (Full Grid Output)

```text
You are a Sudoku variant solving assistant.

You will be given:
- The type of Sudoku variant and its rules.
- The initial grid.
- Any additional constraints (e.g., thermometers, killer cages, arrows, diagonals, etc.).

Your task:
1. Reason step by step to find a valid solution that satisfies ALL rules.
2. At the end, output ONLY the final solved grid in a strict machine-readable format.

FORMAT REQUIREMENTS:
- Represent the grid as 9 lines (for 9x9) or N lines for other sizes.
- Each line contains digits only, no spaces, no commas.
- Do NOT include any explanatory text after the final grid.
- You may use reasoning before the final answer, but clearly separate it with a line containing exactly:
FINAL_GRID:

Example final output:
<your reasoning...>

FINAL_GRID:
123456789
456789123
...

Now solve the following puzzle.

Variant: {variant_name}
Rules:
{rules_text}

Grid size: {grid_size}  # e.g., 9x9, 6x6, 4x4

Initial grid:
{initial_grid_text}

Additional constraints:
{constraints_text}
```

You’ll programmatically fill in `{variant_name}`, `{rules_text}`, `{initial_grid_text}`, `{constraints_text}` from Sudoku-Bench JSON.

### 3.2 Multi-Step Prompt (Iterative Reasoning)

```text
You are a Sudoku variant solving assistant working in INTERACTIVE MODE.

You will be given:
- The variant rules.
- The current state of the grid.
- Any additional constraints.

At each step:
1. Analyze the current grid.
2. Propose 1 to 5 logically justified placements.
3. For each placement, explain the reasoning briefly.
4. Output the moves in a strict machine-readable format so the caller can update the grid.

Move format (per step):
MOVES:
r{row}c{col}={digit}
r{row}c{col}={digit}
END_MOVES

Where row and col are 1-based indices.

Important:
- Only propose moves that you are confident DO NOT violate any rule.
- If you detect a contradiction (i.e., the grid is unsolvable under the rules), say so.
- If the puzzle is completely solved, output:
SOLVED:
<final grid lines as in single-shot format>

Here are the puzzle details:

Variant: {variant_name}
Rules:
{rules_text}

Grid size: {grid_size}

Current grid:
{current_grid_text}

Additional constraints:
{constraints_text}
```

Your driver code will:

* Call the model with this prompt + current grid
* Parse the `MOVES:` block
* Apply them
* Repeat until solved / contradiction / max steps

### 3.3 YRSN-Enhanced Wrapper (Conceptual Prompt Hook)

The YRSN wrapper sits “around” these prompts. At its simplest, it can add a **meta-instruction + structured scratchpad**, e.g.:

```text
[BEGIN YRSN REASONING PROTOCOL]

You must structure your reasoning in the following way:
1. YRSN_STATE: Summarize the global state of the puzzle in 3–5 short bullet points.
2. YRSN_CONSTRAINTS: List the critical constraints that are currently most binding.
3. YRSN_PLAN: Outline a 2–4 step plan of logical deductions you will attempt.
4. YRSN_EXECUTION: Carry out the reasoning steps, noting any eliminations or forced placements.
5. YRSN_CHECK: Verify that your proposed digits do not violate any rule.

Only AFTER completing these steps, produce the required output format (FINAL_GRID or MOVES).

[END YRSN REASONING PROTOCOL]
```

You can prepend/insert this block to the base prompt when `enable_yrsn_wrapper=True`.

---

## 4️⃣ README / Appendix Text (Drop-In)

Here’s a concise block you can paste into your `README.md` or thesis appendix.

```markdown
### Evaluation on Sudoku-Bench with Bedrock Models

We evaluate our YRSN reasoning enhancement on the Sudoku-Bench dataset using a diverse set of transformer-based models hosted on Amazon Bedrock. The goal is to measure whether YRSN improves multi-step logical reasoning in a way that generalizes across model families, scales, and providers.

#### Models

We use the following Bedrock model IDs:

- `anthropic.claude.haiku-4-5-20251001-v1:0` (Claude 4.5 Haiku, small)
- `anthropic.sonnet-4-5-20250929-v1:0` (Claude 4.5 Sonnet, medium/large)
- `anthropic.opus-4-5-20251101-v1:0` (Claude 4.5 Opus, large)
- `openai.gpt-oss-20b1:0` (OpenAI GPT-style 20B model)
- `meta.llama3-1-8b-instruct-v1:0` (Llama 3.1 8B Instruct)
- `meta.llama3-2-11b-instruct-v1:0` (Llama 3.2 11B Instruct)
- `deepseek.v3-v1:0` (DeepSeek V3)

All models are treated as black-box sequence models accessed via the Bedrock API. YRSN is implemented as an external reasoning protocol (prompt wrapper and step controller) that can be applied to any of these models without modifying their internal weights.

#### Datasets and Protocols

We focus on two Sudoku-Bench subsets:

- `challenge_100`: 100 mixed-difficulty puzzles, including modern variants.
- `nikoli_100`: 100 classical Sudoku puzzles from Nikoli.

For each model, we evaluate both:

1. **Single-shot (SS)**: The model receives the full puzzle description and must output a complete solved grid in one response.
2. **Multi-step (MS)**: The model iteratively proposes moves (cell assignments), and the environment updates the grid after each step until the puzzle is solved or a contradiction is reached.

#### YRSN vs. Vanilla

For each model and protocol, we run two conditions:

- **Vanilla**: Base prompt only, no YRSN protocol.
- **YRSN**: Base prompt augmented with the YRSN reasoning structure (state summary, constraint focus, plan, execution, and consistency check).

This yields a 2×2 grid per model:
- (Vanilla, SS), (YRSN, SS), (Vanilla, MS), (YRSN, MS).

#### Metrics

We report:

- **Solve rate**: Percentage of puzzles correctly solved.
- **First-error depth** (MS): Average number of steps before the first invalid move.
- **Average steps to solution** (MS): For solved puzzles.
- **Constraint violation rate**: Fraction of proposed moves that violate Sudoku or variant rules.
- **Valid board fraction**: Percentage of intermediate boards that remain fully consistent with all constraints.

These metrics allow us to quantify not only whether YRSN improves final success rates but also how it affects the stability and robustness of the reasoning process across architectures.
```

---

If you want to go one level deeper next, we can:

* Turn this into actual **Python stubs** (`run_bedrock_sudoku_experiment.py`)
* Or design the **results tables** and plot specs you’ll need for the thesis chapter.
