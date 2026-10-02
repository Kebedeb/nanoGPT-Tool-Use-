# Learned Calculator Agent — Project Log

## Project goal

Build a small, model-driven calculator agent. The trained model should choose whether to call a calculator or retrieve a saved calculation from the memory table of contents (ToC). Python executes the chosen calculator call and manages memory. The project should remain extensible toward document retrieval ideas from the STAIR paper.

This is a narrow research prototype, not a general-purpose conversational assistant.

## Current design

- `generate_data.py` creates synthetic examples for calculator tool calls (`+`, `-`, `*`, `/`) and calculation-leaf selection.
- `train_agent.py` fine-tunes pretrained DistilGPT-2 on those examples. Training uses CUDA when PyTorch detects an NVIDIA GPU.
- `agent.py` asks the trained model to emit either `<tool:calc>…</tool:calc>` or a leaf ID such as `leaf calculation_2`.
- `calculator.py` safely evaluates arithmetic expressions using a restricted Python AST parser.
- `stair_memory.py` keeps calculations in a ToC-like list for the current interactive session.
- `evaluate_retrieval.py` runs a fixed held-out evaluation and saves individual predictions and summary metrics as timestamped JSON.

The model chooses the action and produces its arguments. Python performs the arithmetic or retrieves the selected memory record. The current agent has no hard-coded phrase shortcuts.

## Prototype status: complete

The calculator-agent scope for this repo is **done as a prototype**:

- Four arithmetic operations with balanced operand widths (1–4 digits)
- Learned routing between calculator calls and ToC leaf retrieval
- Reproducible train → eval workflow and interactive `agent.py`
- Checkpoint backups under `stair_agent_model_before_4digit` from earlier runs

What is **not** in scope for v1: persistent memory, W&B, constrained leaf decoding, document corpora, or full STAIR paper reproduction.

## Latest completed evaluation

Report: `evaluation_results_20261002T035141Z.json` after 1500 optimizer steps on 16,000 examples (10,000 tool + 6,000 retrieval).

| Metric | Result |
|---|---:|
| Overall tool result exact match | 73.8% |
| Overall `+` | 71.2% |
| Overall `-` | 68.8% |
| Overall `*` | 77.5% |
| Overall `/` | 77.5% |
| ToC leaf, varied memory layouts | 100% |
| ToC leaf, held-out retrieval wording | 94% |

**Familiar training phrasing** is strong (often 85–100% per op and digit band). **Held-out paraphrases** in `evaluate_retrieval.py` remain the main weakness (roughly 35–65% on several `-`, `*`, `/`, and some `+` cells). That is expected for a small DistilGPT-2 fine-tune on synthetic templates; the eval separates familiar vs held-out wording on purpose.

Earlier checkpoint: `evaluation_results_20261002T032730Z.json` (before paraphrase expansion; subtraction new-phrasing was worse).

## Known limitations

1. Synthetic templates only; free-form user language is not guaranteed.
2. Residual digit and operator errors on unseen phrasing and some four-digit cases.
3. Session-only memory; history clears when `agent.py` exits.
4. `AGENT_RESUME=1` loads weights but not optimizer state.
5. STAIR-inspired ToC/leaf interface only — not hierarchical document retrieval.

## How to run

```powershell
cd stair-agent
python .\generate_data.py
python .\train_agent.py
python .\evaluate_retrieval.py
python .\agent.py
```

Resume fine-tuning:

```powershell
$env:AGENT_RESUME = "1"
$env:AGENT_STEPS = "1500"
python .\train_agent.py
```

## Optional next work (post-v1)

- Persist calculation history to disk
- Constrained decoding for valid leaf IDs (STAIR direction)
- Richer hierarchical ToC and document retrieval benchmarks
- Larger model or more diverse paraphrase / adversarial eval data
