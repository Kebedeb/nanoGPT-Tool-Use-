"""Fine-tune GPT-2 on the generated calculator-agent routing examples."""

import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL = "distilgpt2"
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "train_examples.npz"
OUTPUT_DIR = ROOT / "stair_agent_model"
SPECIAL_TOKENS = [
    "<tool:calc>",
    "</tool:calc>",
    "<lookup:calculations>",
    "[TABLE OF CONTENTS]",
    "[END TABLE OF CONTENTS]",
    "[CALCULATIONS CONTENT]:",
]


def main():
    if not DATA_PATH.exists():
        raise SystemExit("Training data is missing. Run: python generate_data.py")

    seed = int(os.environ.get("AGENT_SEED", "1729"))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Training on: {device}"
        + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""),
        flush=True,
    )

    dataset = np.load(DATA_PATH)
    train_x, train_y = dataset["x"], dataset["y"]
    if len(train_x) == 0:
        raise SystemExit("The training dataset is empty. Regenerate it first.")

    resume = os.environ.get("AGENT_RESUME", "0").lower() not in {"0", "false", "no"}
    tokenizer_source = (
        str(OUTPUT_DIR)
        if resume and (OUTPUT_DIR / "tokenizer.json").exists()
        else BASE_MODEL
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.add_special_tokens({"additional_special_tokens": SPECIAL_TOKENS})

    special_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    expected_ids = list(range(50257, 50257 + len(SPECIAL_TOKENS)))
    if special_ids != expected_ids:
        raise SystemExit(
            f"Tokenizer special-token IDs {special_ids} do not match training data IDs "
            f"{expected_ids}. Regenerate data with the matching GPT-2 tokenizer."
        )

    model_source = str(OUTPUT_DIR) if resume and (OUTPUT_DIR / "config.json").exists() else BASE_MODEL
    model = AutoModelForCausalLM.from_pretrained(model_source)
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.to(device)

    batch_size = int(os.environ.get("AGENT_BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("AGENT_GRAD_ACCUM", "8"))
    steps = int(os.environ.get("AGENT_STEPS", "1200"))
    log_interval = int(os.environ.get("AGENT_LOG_INTERVAL", "20"))
    checkpoint_interval = int(os.environ.get("AGENT_CHECKPOINT_INTERVAL", "100"))
    learning_rate = float(os.environ.get("AGENT_LR", "2e-5"))
    if batch_size < 1 or grad_accum < 1 or steps < 1 or log_interval < 1 or checkpoint_interval < 1:
        raise SystemExit("Batch size, accumulation, steps, and logging intervals must be positive.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    OUTPUT_DIR.mkdir(exist_ok=True)
    completion_marker = OUTPUT_DIR / "training_complete.txt"
    if completion_marker.exists():
        completion_marker.unlink()

    order = np.arange(len(train_x))
    cursor = len(order)
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    print(
        f"Examples: {len(train_x)} | batch: {batch_size} | gradient accumulation: "
        f"{grad_accum} | optimizer steps: {steps}",
        flush=True,
    )

    for step in range(1, steps + 1):
        for _microstep in range(grad_accum):
            if cursor + batch_size > len(order):
                np.random.shuffle(order)
                cursor = 0
            indices = order[cursor : cursor + batch_size]
            cursor += batch_size
            x = torch.from_numpy(train_x[indices]).to(device)
            targets = torch.from_numpy(train_y[indices]).to(device)
            attention_mask = (x != 50256).long()
            seq_len = max(1, int(attention_mask.sum(dim=1).max().item()))
            x = x[:, :seq_len]
            targets = targets[:, :seq_len]
            attention_mask = attention_mask[:, :seq_len]

            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                logits = model(input_ids=x, attention_mask=attention_mask).logits
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    ignore_index=-100,
                ) / grad_accum

            scaler.scale(loss).backward()
            running_loss += float(loss.detach())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if step % log_interval == 0 or step == 1:
            average_loss = running_loss / (log_interval if step % log_interval == 0 else step)
            print(f"Step {step}/{steps} | loss: {average_loss:.4f}", flush=True)
            running_loss = 0.0

        if step % checkpoint_interval == 0:
            model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
            tokenizer.save_pretrained(OUTPUT_DIR)
            print(f"Checkpoint saved at step {step} to {OUTPUT_DIR}.", flush=True)

    model.config.use_cache = True
    model.save_pretrained(OUTPUT_DIR, safe_serialization=True)
    tokenizer.save_pretrained(OUTPUT_DIR)
    completion_marker.write_text(f"Completed {steps} optimizer steps.\n", encoding="utf-8")
    print(f"Training complete. Model saved to {OUTPUT_DIR}.", flush=True)


if __name__ == "__main__":
    main()
