# Learned Calculator Agent

This project trains a language model to choose between two actions:

- call the calculator with an expression
- select a saved calculation from the current table of contents

Python executes the selected tool and stores successful calculations for the current session. The model chooses the action and, for retrieval, the leaf ID. There are no input-pattern shortcuts in the interactive agent.

## Setup and run

Open a terminal in this folder. Use the Python environment that already has PyTorch, Transformers, and `tiktoken` installed. Model weight files (`*.safetensors`) are gitignored because they exceed GitHub's size limit; run training locally to produce `stair_agent_model/`.

```powershell
python .\generate_data.py
python .\train_agent.py
python .\evaluate_retrieval.py
python .\agent.py
```

Training starts from pretrained DistilGPT-2 weights, then fine-tunes on synthetic arithmetic and calculation-history examples covering `+`, `-`, `*`, and `/`. The first run downloads the base model if it is not already cached. Training defaults to CUDA when PyTorch can see an NVIDIA GPU. It prints loss every 20 optimizer steps and saves model checkpoints every 100 steps. Resume loads the latest saved model weights and restarts the optimizer. The interactive agent will only load a model after training completes.

Use `generate_data.py`, `train_agent.py`, `evaluate_retrieval.py`, and `agent.py` for the current workflow. Training balances operators and one- through four-digit operands. The evaluator scores each operator separately under familiar versus new wording and shorter versus four-digit operands, then saves predictions to a timestamped JSON file. The old mock loop and binary-token scripts are legacy experiments.

Useful PowerShell settings before training:

```powershell
$env:AGENT_STEPS = "1200"
$env:AGENT_RESUME = "1"  # continue from the most recently saved model checkpoint
python .\train_agent.py
```

Remove `AGENT_RESUME` or set it to `0` to start again from the pretrained GPT-2 base. The default training batch is one example with eight-step gradient accumulation for a 6 GB laptop GPU. `transformer_model.py` is retained for reference; running it no longer starts the old random-initialized training loop.

## Extending toward STAIR

The current memory is a small calculation history. The ToC/leaf action format gives the learned router a clean seam for later work: hierarchical memory, richer leaf content, constrained leaf decoding, and document retrieval can be added without changing the calculator tool interface.
