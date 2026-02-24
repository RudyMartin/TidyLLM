Alright, let’s wire this up so CTM/HRM + Bedrock models all run through the **same experiment harness**.

I’ll give you:

1. A minimal `run_experiments.py` skeleton that:

   * Loads Sudoku-Bench puzzles
   * Uses CTM/HRM via `rnn_adapter.py`
   * Leaves clear hooks for Bedrock models
   * Writes a CSV compatible with your tables
2. A quick note on how to plug in Bedrock.

---

## 1️⃣ `run_experiments.py` – unified experiment driver

This assumes you already have `rnn_adapter.py` (from earlier) in your repo with `SudokuPuzzle`, `SudokuResult`, `CTMAdapter`, `HRMAdapter`.

```python
# run_experiments.py

from __future__ import annotations
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Iterable, Literal, Any, Tuple

from rnn_adapter import (
    SudokuPuzzle,
    SudokuResult,
    CTMAdapter,
    HRMAdapter,
)

Mode = Literal["single_shot", "multi_step"]


# ---------- 1. Loading Sudoku-Bench puzzles ----------

def load_sudoku_puzzles_from_json(
    path: Path,
    subset_name: str,
) -> List[SudokuPuzzle]:
    """
    Load Sudoku-Bench puzzles from a JSON file and convert to SudokuPuzzle objects.

    NOTE: You will need to adapt this based on the exact Sudoku-Bench JSON schema
    you are using (Hugging Face / GitHub release).
    """
    puzzles: List[SudokuPuzzle] = []

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Example expected structure: { "puzzles": [ { ... }, ... ] }
    # Adjust this to your actual schema.
    for item in data["puzzles"]:
        # TODO: map these fields from your actual JSON keys
        puzzle_id = item.get("id") or item.get("puzzle_id")
        variant_name = item.get("variant_name", "Unknown")
        rules_text = item.get("rules_text", "")
        grid_size = item.get("grid_size", 9)

        # These should be the text formats you already decided
        # (e.g., '.' or '0' for blanks, one row per line).
        initial_grid_text = item["initial_grid_text"]
        constraints_text = item.get("constraints_text", "")

        puzzles.append(
            SudokuPuzzle(
                puzzle_id=f"{subset_name}:{puzzle_id}",
                variant_name=variant_name,
                rules_text=rules_text,
                grid_size=grid_size,
                initial_grid_text=initial_grid_text,
                constraints_text=constraints_text,
            )
        )

    return puzzles


# ---------- 2. Small metadata struct for model runners ----------

class ModelRunnerWrapper:
    """
    Wraps any adapter (CTMAdapter, HRMAdapter, BedrockAdapter, etc.)
    together with metadata like architecture type and YRSN flag.
    """

    def __init__(
        self,
        name: str,
        architecture: str,
        yrns_enabled: bool,
        adapter: Any,  # has run_single_shot and run_multi_step
    ) -> None:
        self.name = name
        self.architecture = architecture
        self.yrns_enabled = yrns_enabled
        self.adapter = adapter

    def run(self, mode: Mode, puzzle: SudokuPuzzle) -> SudokuResult:
        if mode == "single_shot":
            return self.adapter.run_single_shot(puzzle)
        elif mode == "multi_step":
            return self.adapter.run_multi_step(puzzle)
        else:
            raise ValueError(f"Unknown mode: {mode}")


# ---------- 3. Build CTM/HRM and Bedrock runners ----------

def build_model_runners() -> List[ModelRunnerWrapper]:
    """
    Instantiate all model runners (local CTM/HRM and Bedrock models).

    You will need to:
    - Plug in your real CTM/HRM engines.
    - Implement a BedrockAdapter similar to BaseRNNAdapter, or reuse the same pattern.
    """

    runners: List[ModelRunnerWrapper] = []

    # ---- 3.1 Local RNNs: CTM & HRM ----

    # TODO: replace with your actual CTM / HRM objects
    ctm_engine = ...  # e.g., CTMEngine(config=...)
    hrm_engine = ...  # e.g., HRMEngine(config=...)

    # Vanilla CTM
    ctm_vanilla = CTMAdapter(ctm_engine, yrns_enabled=False)
    runners.append(
        ModelRunnerWrapper(
            name="ctm",
            architecture="rnn",
            yrns_enabled=False,
            adapter=ctm_vanilla,
        )
    )

    # YRSN-CTM
    ctm_yrsn = CTMAdapter(ctm_engine, yrns_enabled=True)
    runners.append(
        ModelRunnerWrapper(
            name="yrsn_ctm",
            architecture="rnn",
            yrns_enabled=True,
            adapter=ctm_yrsn,
        )
    )

    # Vanilla HRM
    hrm_vanilla = HRMAdapter(hrm_engine, yrns_enabled=False)
    runners.append(
        ModelRunnerWrapper(
            name="hrm",
            architecture="rnn",
            yrns_enabled=False,
            adapter=hrm_vanilla,
        )
    )

    # YRSN-HRM
    hrm_yrsn = HRMAdapter(hrm_engine, yrns_enabled=True)
    runners.append(
        ModelRunnerWrapper(
            name="yrsn_hrm",
            architecture="rnn",
            yrns_enabled=True,
            adapter=hrm_yrsn,
        )
    )

    # ---- 3.2 Bedrock models (outline, you wire the actual adapter) ----
    # You'll implement BedrockAdapter with same interface as CTMAdapter/HRMAdapter:
    #   bedrock_adapter = BedrockAdapter(model_id="anthropic.claude.haiku-4-5-20251001-v1:0", yrns_enabled=False)
    # Then wrap it in ModelRunnerWrapper.

    # Example placeholders (commented out until you implement BedrockAdapter):

    # from bedrock_adapter import BedrockAdapter

    # haiku = BedrockAdapter(
    #     model_id="anthropic.claude.haiku-4-5-20251001-v1:0",
    #     yrns_enabled=False,
    # )
    # runners.append(
    #     ModelRunnerWrapper(
    #         name="haiku",
    #         architecture="transformer",
    #         yrns_enabled=False,
    #         adapter=haiku,
    #     )
    # )
    #
    # haiku_yrsn = BedrockAdapter(
    #     model_id="anthropic.claude.haiku-4-5-20251001-v1:0",
    #     yrns_enabled=True,
    # )
    # runners.append(
    #     ModelRunnerWrapper(
    #         name="yrsn_haiku",
    #         architecture="transformer",
    #         yrns_enabled=True,
    #         adapter=haiku_yrsn,
    #     )
    # )

    return runners


# ---------- 4. Experiment loop ----------

def run_experiments(
    puzzles: Iterable[SudokuPuzzle],
    model_runners: List[ModelRunnerWrapper],
    modes: List[Mode],
    csv_path: Path,
    subset_name: str,
) -> None:
    """
    Main experiment loop.

    Writes one row per (puzzle, model, mode) with enough info for post-hoc
    aggregation into solve rates, etc.
    """

    fieldnames = [
        "subset",
        "puzzle_id",
        "variant_name",
        "grid_size",
        "model_name",
        "architecture",
        "yrsn_enabled",
        "mode",
        "solved",
        "steps_taken",
        "first_error_step",
        "error_message",
        # You can add more later (timing, token counts, etc.)
    ]

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for puzzle in puzzles:
            for runner in model_runners:
                for mode in modes:
                    print(f"Running {subset_name} | {puzzle.puzzle_id} | {runner.name} | {mode}")
                    result = runner.run(mode=mode, puzzle=puzzle)

                    row = {
                        "subset": subset_name,
                        "puzzle_id": puzzle.puzzle_id,
                        "variant_name": puzzle.variant_name,
                        "grid_size": puzzle.grid_size,
                        "model_name": runner.name,
                        "architecture": runner.architecture,
                        "yrsn_enabled": runner.yrns_enabled,
                        "mode": result.mode,
                        "solved": int(result.solved),
                        "steps_taken": result.steps_taken or 0,
                        "first_error_step": result.first_error_step or 0,
                        "error_message": (result.error_message or "")[:512],
                    }
                    writer.writerow(row)


# ---------- 5. CLI entrypoint ----------

def main():
    # Paths / subsets – adapt to where you store Sudoku-Bench JSON
    base_dir = Path("data/sudoku_bench")
    challenge_path = base_dir / "challenge_100.json"
    nikoli_path = base_dir / "nikoli_100.json"

    # Load puzzles
    challenge_puzzles = load_sudoku_puzzles_from_json(challenge_path, subset_name="challenge_100")
    nikoli_puzzles = load_sudoku_puzzles_from_json(nikoli_path, subset_name="nikoli_100")

    # Build model runners (CTM, HRM, Bedrock)
    model_runners = build_model_runners()

    # Run modes
    modes: List[Mode] = ["single_shot", "multi_step"]

    # Output locations
    out_dir = Path("results")
    run_experiments(
        puzzles=challenge_puzzles,
        model_runners=model_runners,
        modes=modes,
        csv_path=out_dir / "challenge_100_results.csv",
        subset_name="challenge_100",
    )

    run_experiments(
        puzzles=nikoli_puzzles,
        model_runners=model_runners,
        modes=modes,
        csv_path=out_dir / "nikoli_100_results.csv",
        subset_name="nikoli_100",
    )


if __name__ == "__main__":
    main()
```

---

## 2️⃣ How to plug in Bedrock cleanly

You only need **one more adapter** mirroring CTM/HRM’s pattern.

Very quick sketch:

```python
# bedrock_adapter.py  (sketch)

from rnn_adapter import BaseRNNAdapter  # reuse the same base class
import boto3


class BedrockLLMWrapper:
    def __init__(self, model_id: str, region: str = "us-east-1"):
        self.model_id = model_id
        self.model_key = model_id
        self.client = boto3.client("bedrock-runtime", region_name=region)

    def generate(self, prompt: str, max_tokens: int = 1024, temperature: float = 0.0) -> str:
        # TODO: adapt to the specific Bedrock model's request/response schema
        body = {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokens": max_tokens,
                "temperature": temperature,
            },
        }
        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
        )
        out = json.loads(response["body"].read())
        # TODO: adapt "out" to extract the text completion
        return out["outputText"][0]["text"]


class BedrockAdapter(BaseRNNAdapter):
    def __init__(self, model_id: str, yrns_enabled: bool = False, region: str = "us-east-1"):
        wrapper = BedrockLLMWrapper(model_id=model_id, region=region)
        super().__init__(model=wrapper, yrns_enabled=yrns_enabled)
```

Then in `build_model_runners()` you just:

```python
from bedrock_adapter import BedrockAdapter

haiku = BedrockAdapter("anthropic.claude.haiku-4-5-20251001-v1:0", yrns_enabled=False)
haiku_yrsn = BedrockAdapter("anthropic.claude.haiku-4-5-20251001-v1:0", yrns_enabled=True)
# same for Sonnet, Opus, Llama, GPT-OSS-20B1, DeepSeek
```

Now CTM, HRM, and all Bedrock models all share:

* `run_single_shot(puzzle)`
* `run_multi_step(puzzle)`
* `SudokuResult` output
* CSV that plugs straight into your tables.

---

If you want, next step I can:

* Draft a **post-processing notebook outline**: read `challenge_100_results.csv`, compute solve rates, plot per-model bar charts, and fill the table templates automatically.
