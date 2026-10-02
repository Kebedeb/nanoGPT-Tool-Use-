"""Interactive learned tool-using calculator agent."""

from pathlib import Path
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import calculator
from stair_memory import MemoryStore


MODEL_DIR = Path(__file__).resolve().parent / "stair_agent_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(
    f"Using device: {DEVICE}"
    + (f" ({torch.cuda.get_device_name(0)})" if DEVICE == "cuda" else "")
)

if not (MODEL_DIR / "training_complete.txt").exists():
    raise SystemExit(
        f"No completed trained agent found at {MODEL_DIR}.\n"
        "From this folder, run: python generate_data.py, then python train_agent.py"
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
model.eval()

memory = MemoryStore()
INSTRUCTIONS = (
    "You are a calculator agent. For an arithmetic request, respond with only "
    "<tool:calc>expression</tool:calc>. For a question about a saved calculation, "
    "respond with only its ToC leaf ID, in the form leaf calculation_2."
)


def generate_action(user_input: str) -> str:
    prompt = (
        f"{memory.get_toc()}\n{INSTRUCTIONS}\n"
        f"User: {user_input}\nModel: "
    )
    encoded = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        output = model.generate(
            **encoded,
            max_new_tokens=24,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        output[0, encoded["input_ids"].shape[1] :], skip_special_tokens=False
    ).split(tokenizer.eos_token, 1)[0].strip()


def run_turn(user_input: str) -> None:
    print(f"\nUser: {user_input}")
    action = generate_action(user_input)
    print(f"[Model action]: {action}")

    calc_match = re.fullmatch(r"<tool:calc>(.*?)</tool:calc>", action, re.DOTALL)
    if calc_match:
        expression = calc_match.group(1).strip()
        try:
            result = calculator.evaluate(expression)
        except (ValueError, ZeroDivisionError, OverflowError) as exc:
            print(f"[Assistant]: I couldn't evaluate that expression: {exc}")
            return
        memory.add_calculation(expression, result)
        print(f"[Assistant]: {expression} = {result}")
        return

    leaf_match = re.fullmatch(r"leaf\s+calculation_(\d+)", action, re.IGNORECASE)
    if leaf_match:
        leaf_number = int(leaf_match.group(1))
        calculation = memory.get_calculation(leaf_number - 1)
        if calculation is None:
            print("[Assistant]: The model selected a leaf that is not in memory.")
        else:
            print(f"[Assistant]: Calculation {leaf_number} was {calculation}.")
        return

    print("[Assistant]: I couldn't produce a valid tool call or memory selection.")


if __name__ == "__main__":
    print("Calculator agent ready. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if user_input:
            run_turn(user_input)
