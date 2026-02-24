Got you. Let’s make this concrete and plug-and-play.

I’ll give you:

1. A clean `rnn_adapter.py` sketch that makes **CTM/HRM look like the Bedrock models** (same interface).
2. Ready-to-use **results table templates** (Markdown + LaTeX) for CTM/HRM vs transformers, Vanilla vs YRSN, SS vs MS.

---

## 1️⃣ `rnn_adapter.py` – unify CTM/HRM with the LLM interface

### Design goals

* Treat CTM + HRM like “local LLMs” with a simple API:

  * `generate_single_shot(puzzle)` → final grid or failure
  * `generate_multi_step(puzzle)` → move stream until done
* Make it easy to drop into the same experiment runner you’ll use for Bedrock.

Below is a **skeletal but realistic** `rnn_adapter.py`. You’ll just fill in the `TODO` spots where you call your actual CTM/HRM code.

```python
# rnn_adapter.py

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Protocol, Literal, Dict, Any
import abc


# ---------- Core data structures ----------

GridSize = Literal[4, 6, 9]


@dataclass
class SudokuPuzzle:
    puzzle_id: str
    variant_name: str
    rules_text: str
    grid_size: GridSize
    initial_grid_text: str        # text block used in prompts (e.g. "." or "0" for blanks)
    constraints_text: str         # thermo, killer cages, etc.


@dataclass
class SudokuMove:
    row: int   # 1-based
    col: int   # 1-based
    digit: int


@dataclass
class SudokuResult:
    puzzle_id: str
    model_key: str
    mode: Literal["single_shot", "multi_step"]
    yrns_enabled: bool
    solved: bool
    final_grid_text: Optional[str] = None
    moves: Optional[List[SudokuMove]] = None
    steps_taken: Optional[int] = None
    first_error_step: Optional[int] = None
    error_message: Optional[str] = None
    extra_metrics: Optional[Dict[str, Any]] = None


# ---------- Generic interface for "local models" ----------

class LocalReasoningModel(Protocol):
    """
    Protocol for any local model (CTM, HRM, etc.) that can be used in Sudoku experiments.
    """

    model_key: str

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> str:
        """
        Low-level "LLM-like" interface: takes a text prompt, returns a text completion.
        For CTM/HRM, this may adapt to an internal tensor API under the hood.
        """
        ...


# ---------- Base adapter with shared behavior ----------

class BaseRNNAdapter(abc.ABC):
    """
    Base adapter that wraps a local RNN model (CTM or HRM) and exposes
    Sudoku-Bench-specific helpers for single-shot and multi-step evaluation.
    """

    def __init__(
        self,
        model: LocalReasoningModel,
        yrns_enabled: bool = False,
    ) -> None:
        self.model = model
        self.model_key = model.model_key
        self.yrns_enabled = yrns_enabled

    # ---- Prompt builders (reuse the same structure as Bedrock) ----

    def build_single_shot_prompt(self, puzzle: SudokuPuzzle) -> str:
        yrns_block = ""
        if self.yrns_enabled:
            yrns_block = (
                "[BEGIN YRSN REASONING PROTOCOL]\n"
                "You must structure your reasoning in the following way:\n"
                "1. YRSN_STATE: Summarize the global state of the puzzle.\n"
                "2. YRSN_CONSTRAINTS: List the most binding constraints.\n"
                "3. YRSN_PLAN: Outline a short plan of deductions.\n"
                "4. YRSN_EXECUTION: Carry out the plan step by step.\n"
                "5. YRSN_CHECK: Verify that your solution satisfies all rules.\n"
                "Only AFTER completing these steps, produce the final grid.\n"
                "[END YRSN REASONING PROTOCOL]\n\n"
            )

        prompt = f"""
You are a Sudoku variant solving assistant.

{yrns_block}You will be given:
- The type of Sudoku variant and its rules.
- The initial grid.
- Any additional constraints.

Your task:
1. Reason step by step to find a valid solution.
2. At the end, output ONLY the final solved grid in a strict machine-readable format.

FORMAT REQUIREMENTS:
- Represent the grid as {puzzle.grid_size} lines.
- Each line contains digits only, no spaces, no commas.
- Do NOT include any explanatory text after the final grid.
- Separate your reasoning and final grid with a line containing exactly:
FINAL_GRID:

Now solve the following puzzle.

Variant: {puzzle.variant_name}
Rules:
{puzzle.rules_text}

Grid size: {puzzle.grid_size}x{puzzle.grid_size}

Initial grid:
{puzzle.initial_grid_text}

Additional constraints:
{puzzle.constraints_text}
""".strip()
        return prompt

    def build_multi_step_prompt(self, puzzle: SudokuPuzzle, current_grid_text: str) -> str:
        yrns_block = ""
        if self.yrns_enabled:
            yrns_block = (
                "[BEGIN YRSN REASONING PROTOCOL]\n"
                "At each step, follow this structure:\n"
                "1. YRSN_STATE: Brief state summary.\n"
                "2. YRSN_CONSTRAINTS: Active constraints.\n"
                "3. YRSN_PLAN: Planned deductions for this step.\n"
                "4. YRSN_EXECUTION: Derive a small set of safe moves.\n"
                "5. YRSN_CHECK: Ensure moves do not violate any rule.\n"
                "[END YRSN REASONING PROTOCOL]\n\n"
            )

        prompt = f"""
You are a Sudoku variant solving assistant working in INTERACTIVE MODE.

{yrns_block}You will be given:
- The rules and current grid.
- You must propose a small number of safe moves at each step.

Move format:
MOVES:
r{{row}}c{{col}}={{digit}}
...
END_MOVES

If the puzzle is completely solved, output:
SOLVED:
<final grid in {puzzle.grid_size} lines of digits>

Variant: {puzzle.variant_name}
Rules:
{puzzle.rules_text}

Grid size: {puzzle.grid_size}x{puzzle.grid_size}

Current grid:
{current_grid_text}

Additional constraints:
{puzzle.constraints_text}
""".strip()
        return prompt

    # ---------- Public evaluation methods ----------

    def run_single_shot(
        self,
        puzzle: SudokuPuzzle,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> SudokuResult:
        prompt = self.build_single_shot_prompt(puzzle)
        raw_output = self.model.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        final_grid = self._extract_final_grid(raw_output, puzzle.grid_size)

        solved = final_grid is not None
        return SudokuResult(
            puzzle_id=puzzle.puzzle_id,
            model_key=self.model_key,
            mode="single_shot",
            yrns_enabled=self.yrns_enabled,
            solved=solved,
            final_grid_text=final_grid,
            extra_metrics={
                "raw_output": raw_output if not solved else None,
            },
        )

    def run_multi_step(
        self,
        puzzle: SudokuPuzzle,
        max_steps: int = 64,
        max_tokens_per_step: int = 512,
        temperature: float = 0.0,
    ) -> SudokuResult:
        current_grid_text = puzzle.initial_grid_text
        moves: List[SudokuMove] = []
        first_error_step: Optional[int] = None
        error_message: Optional[str] = None

        for step in range(1, max_steps + 1):
            prompt = self.build_multi_step_prompt(puzzle, current_grid_text)
            raw_output = self.model.generate(
                prompt=prompt,
                max_tokens=max_tokens_per_step,
                temperature=temperature,
            )

            if "SOLVED:" in raw_output:
                final_grid = self._extract_final_grid(raw_output, puzzle.grid_size)
                if final_grid is None:
                    first_error_step = first_error_step or step
                    error_message = "SOLVED marker but could not parse final grid."
                    break
                # Let the evaluator validate final_grid correctness separately
                return SudokuResult(
                    puzzle_id=puzzle.puzzle_id,
                    model_key=self.model_key,
                    mode="multi_step",
                    yrns_enabled=self.yrns_enabled,
                    solved=True,
                    final_grid_text=final_grid,
                    moves=moves,
                    steps_taken=step,
                )

            step_moves = self._parse_moves_block(raw_output)
            if not step_moves:
                first_error_step = first_error_step or step
                error_message = "No valid MOVES block parsed."
                break

            # TODO: Here you update current_grid_text by applying step_moves.
            #       You may call into your Sudoku environment / checker.
            # new_grid_text, is_valid = apply_moves_and_validate(current_grid_text, step_moves, puzzle)
            # if not is_valid:
            #     first_error_step = first_error_step or step
            #     error_message = "Invalid move detected by environment."
            #     break
            # current_grid_text = new_grid_text

            moves.extend(step_moves)

        # If we exit loop without SOLVED:
        return SudokuResult(
            puzzle_id=puzzle.puzzle_id,
            model_key=self.model_key,
            mode="multi_step",
            yrns_enabled=self.yrns_enabled,
            solved=False,
            moves=moves,
            steps_taken=len(moves),
            first_error_step=first_error_step,
            error_message=error_message or "Max steps reached without SOLVED.",
        )

    # ---------- Parsing helpers ----------

    def _extract_final_grid(self, raw_output: str, grid_size: int) -> Optional[str]:
        """
        Extract the lines after 'FINAL_GRID:' or 'SOLVED:' and
        validate that there are exactly grid_size lines of digits.
        """
        marker_idx = raw_output.find("FINAL_GRID:")
        if marker_idx == -1:
            marker_idx = raw_output.find("SOLVED:")
            if marker_idx == -1:
                return None

        lines = raw_output[marker_idx:].splitlines()
        # drop the marker line itself
        lines = lines[1:]
        grid_lines = [ln.strip() for ln in lines if ln.strip()]

        if len(grid_lines) < grid_size:
            return None

        grid_lines = grid_lines[:grid_size]
        if any((len(ln) != grid_size or not ln.isdigit()) for ln in grid_lines):
            return None

        return "\n".join(grid_lines)

    def _parse_moves_block(self, raw_output: str) -> List[SudokuMove]:
        """
        Parse moves in the format:

        MOVES:
        r3c5=7
        r1c9=4
        END_MOVES
        """
        moves: List[SudokuMove] = []
        if "MOVES:" not in raw_output:
            return moves

        after = raw_output.split("MOVES:", 1)[1]
        if "END_MOVES" in after:
            after = after.split("END_MOVES", 1)[0]

        for line in after.splitlines():
            line = line.strip()
            if not line or "=" not in line:
                continue
            try:
                lhs, rhs = line.split("=", 1)
                digit = int(rhs.strip())
                lhs = lhs.strip()
                # format r{row}c{col}
                if lhs[0] != "r" or "c" not in lhs:
                    continue
                r_str, c_str = lhs[1:].split("c", 1)
                row = int(r_str)
                col = int(c_str)
                moves.append(SudokuMove(row=row, col=col, digit=digit))
            except Exception:
                continue

        return moves


# ---------- Concrete adapters for CTM and HRM ----------

class CTMAdapter(BaseRNNAdapter):
    """
    Wraps a CTM implementation and exposes the LocalReasoningModel interface.
    """

    def __init__(self, ctm_engine, yrns_enabled: bool = False) -> None:
        """
        ctm_engine: your existing CTM object or function bundle.
        It should expose something like:
            ctm_engine.generate_text(prompt: str, max_tokens: int, temperature: float) -> str
        """
        class _CTMWrapper:
            def __init__(self, engine):
                self.engine = engine
                self.model_key = "ctm"

            def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
                # TODO: adapt this to your real CTM API
                return self.engine.generate_text(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

        super().__init__(model=_CTMWrapper(ctm_engine), yrns_enabled=yrns_enabled)


class HRMAdapter(BaseRNNAdapter):
    """
    Wraps an HRM implementation and exposes the LocalReasoningModel interface.
    """

    def __init__(self, hrm_engine, yrns_enabled: bool = False) -> None:
        """
        hrm_engine: your existing HRM object or function bundle.
        It should expose something like:
            hrm_engine.generate_text(prompt: str, max_tokens: int, temperature: float) -> str
        """
        class _HRMWrapper:
            def __init__(self, engine):
                self.engine = engine
                self.model_key = "hrm"

            def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
                # TODO: adapt this to your real HRM API
                return self.engine.generate_text(prompt=prompt, max_tokens=max_tokens, temperature=temperature)

        super().__init__(model=_HRMWrapper(hrm_engine), yrns_enabled=yrns_enabled)
```

You can now have your **experiment runner** treat CTM/HRM just like Bedrock models:

```python
result = ctm_adapter.run_single_shot(puzzle)
result = hrm_adapter.run_multi_step(puzzle)
```

Same `SudokuResult` object as for transformers.

---

## 2️⃣ Results table templates (CTM/HRM vs Transformers)

### 2.1 Markdown table – high-level summary (challenge_100)

**Single-Shot (SS) – `challenge_100`**

```markdown
### Solve Rate and Error Metrics – Single-Shot (challenge_100)

| Model                         | Arch    | YRSN | Solve Rate (%) | Avg First Error Depth* | Notes                |
|------------------------------|---------|------|----------------|------------------------|----------------------|
| CTM                          | RNN     | ✗    |                | N/A                    | Local implementation |
| YRSN-CTM                     | RNN     | ✓    |                | N/A                    |                      |
| HRM                          | RNN     | ✗    |                | N/A                    |                      |
| YRSN-HRM                     | RNN     | ✓    |                | N/A                    |                      |
| Llama3-1-8B Instruct         | Trf     | ✗    |                |                        | Bedrock              |
| YRSN-Llama3-1-8B             | Trf     | ✓    |                |                        |                      |
| GPT-OSS-20B1                 | Trf     | ✗    |                |                        | Bedrock              |
| YRSN-GPT-OSS-20B1            | Trf     | ✓    |                |                        |                      |
| Claude 4.5 Haiku             | Trf     | ✗    |                |                        | Bedrock              |
| YRSN-Claude 4.5 Haiku        | Trf     | ✓    |                |                        |                      |
| Claude 4.5 Sonnet            | Trf     | ✗    |                |                        | Bedrock              |
| YRSN-Claude 4.5 Sonnet       | Trf     | ✓    |                |                        |                      |
| Claude 4.5 Opus              | Trf     | ✗    |                |                        | Upper bound          |
| YRSN-Claude 4.5 Opus         | Trf     | ✓    |                |                        | (optional)           |

\*First error depth is more meaningful in MS, can be left blank here or defined as “first invalid token / contradiction if any”.
```

**Multi-Step (MS) – `challenge_100`**

```markdown
### Solve Rate and Stability – Multi-Step (challenge_100)

| Model                  | Arch | YRSN | Solve Rate (%) | Avg Steps (Solved) | Avg First Error Step | Contradiction Rate (%) |
|------------------------|------|------|----------------|--------------------|----------------------|------------------------|
| CTM                    | RNN  | ✗    |                |                    |                      |                        |
| YRSN-CTM               | RNN  | ✓    |                |                    |                      |                        |
| HRM                    | RNN  | ✗    |                |                    |                      |                        |
| YRSN-HRM               | RNN  | ✓    |                |                    |                      |                        |
| Llama3-1-8B            | Trf  | ✗    |                |                    |                      |                        |
| YRSN-Llama3-1-8B       | Trf  | ✓    |                |                    |                      |                        |
| GPT-OSS-20B1           | Trf  | ✗    |                |                    |                      |                        |
| YRSN-GPT-OSS-20B1      | Trf  | ✓    |                |                    |                      |                        |
| Claude 4.5 Haiku       | Trf  | ✗    |                |                    |                      |                        |
| YRSN-Claude 4.5 Haiku  | Trf  | ✓    |                |                    |                      |                        |
| Claude 4.5 Sonnet      | Trf  | ✗    |                |                    |                      |                        |
| YRSN-Claude 4.5 Sonnet | Trf  | ✓    |                |                    |                      |                        |
| Claude 4.5 Opus        | Trf  | ✗    |                |                    |                      |                        |
| YRSN-Claude 4.5 Opus   | Trf  | ✓    |                |                    |                      |                        |
```

You can mirror the same pattern for `nikoli_100` if you want a second table.

---

### 2.2 LaTeX table – thesis-ready (single-shot example)

```latex
\begin{table}[ht]
\centering
\caption{Single-shot performance on \texttt{challenge\_100} (Sudoku-Bench). YRSN denotes our reasoning enhancement wrapper.}
\begin{tabular}{lcccc}
\hline
Model & Architecture & YRSN & Solve rate (\%) & Notes \\
\hline
CTM                         & RNN       & No  &        & Local recurrent model \\
YRSN-CTM                    & RNN       & Yes &        &                        \\
HRM                         & RNN       & No  &        & Hierarchical recurrent \\
YRSN-HRM                    & RNN       & Yes &        &                        \\
Llama3-1-8B Instruct        & Transformer & No  &        & Bedrock               \\
YRSN-Llama3-1-8B            & Transformer & Yes &        &                        \\
GPT-OSS-20B1                & Transformer & No  &        & Bedrock               \\
YRSN-GPT-OSS-20B1           & Transformer & Yes &        &                        \\
Claude 4.5 Haiku            & Transformer & No  &        & Bedrock               \\
YRSN-Claude 4.5 Haiku       & Transformer & Yes &        &                        \\
Claude 4.5 Sonnet           & Transformer & No  &        & Bedrock               \\
YRSN-Claude 4.5 Sonnet      & Transformer & Yes &        &                        \\
Claude 4.5 Opus             & Transformer & No  &        & Upper bound           \\
YRSN-Claude 4.5 Opus        & Transformer & Yes &        & (optional)            \\
\hline
\end{tabular}
\label{tab:sudoku_single_shot}
\end{table}
```

You can create a similar LaTeX table for multi-step including extra columns: `Avg steps`, `First error step`, `Contradiction rate`.

---

If you want, next step I can sketch a minimal `run_experiments.py` that:

* loads a Sudoku-Bench JSON file,
* builds `SudokuPuzzle` objects,
* loops over `{CTM, HRM, Bedrock models} × {Vanilla, YRSN} × {SS, MS}`,
* writes results as a CSV ready to plug into those tables.
