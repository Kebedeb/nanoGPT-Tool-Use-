"""Run a reproducible held-out evaluation of tool routing and leaf selection."""

from datetime import datetime, timezone
import json
import operator
from pathlib import Path
import random
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import calculator
from stair_memory import MemoryStore


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "stair_agent_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INSTRUCTIONS = (
    "You are a calculator agent. For an arithmetic request, respond with only "
    "<tool:calc>expression</tool:calc>. For a question about a saved calculation, "
    "respond with only its ToC leaf ID, in the form leaf calculation_2."
)
ORDINALS = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]

# Familiar templates overlap training. "new" templates are held out from generate_data.py.
OPERATIONS = {
    "+": {
        "name": "add",
        "compute": operator.add,
        "familiar": [
            "What is {a} plus {b}?",
            "Calculate {a} + {b}.",
            "Add {a} and {b}.",
            "What's the sum of {a} and {b}?",
            "How much is {a} plus {b}?",
            "Please compute {a}+{b}.",
        ],
        "new": [
            "Could you add together {a} and {b}?",
            "I need {a} increased by {b}.",
            "Give me the total of {a} with {b}.",
            "Combine {a} and {b} numerically.",
            "What's {a} added onto {b}?",
        ],
    },
    "-": {
        "name": "sub",
        "compute": operator.sub,
        "familiar": [
            "What is {a} minus {b}?",
            "Calculate {a} - {b}.",
            "Subtract {b} from {a}.",
            "How much is {a} minus {b}?",
            "Please compute {a}-{b}.",
            "Please subtract {b} from {a}.",
        ],
        "new": [
            "Reduce {a} by {b}.",
            "How much remains after {b} is removed from {a}?",
            "Decrease {a} using {b}.",
            "What is {a} less {b}?",
            "How much less is {b} than {a}?",
        ],
    },
    "*": {
        "name": "mul",
        "compute": operator.mul,
        "familiar": [
            "What is {a} times {b}?",
            "Calculate {a} * {b}.",
            "Multiply {a} by {b}.",
            "What's the product of {a} and {b}?",
            "How much is {a} times {b}?",
            "Please compute {a}*{b}.",
        ],
        "new": [
            "Scale {a} by a factor of {b}.",
            "Give me {a} x {b}.",
            "What is {a} multiplied with {b}?",
            "Compute {a} lots of {b}.",
            "Raise {a} through multiplication by {b}.",
        ],
    },
    "/": {
        "name": "div",
        "compute": operator.truediv,
        "familiar": [
            "What is {a} divided by {b}?",
            "Calculate {a} / {b}.",
            "Divide {a} by {b}.",
            "What's {a} over {b}?",
            "How much is {a} divided by {b}?",
            "Please compute {a}/{b}.",
        ],
        "new": [
            "How many times does {b} go into {a}?",
            "Compute {a} ÷ {b}.",
            "What is the result of splitting {a} by {b}?",
            "Give me {a} per {b}.",
            "Perform division of {a} using {b}.",
        ],
    },
}


def load_model():
    if not (MODEL_DIR / "training_complete.txt").exists():
        raise SystemExit("Train the agent first by running: python train_agent.py")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, memory, query):
    prompt = f"{memory.get_toc()}\n{INSTRUCTIONS}\nUser: {query}\nModel: "
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


def sample_pair(rng, low, high, op_symbol):
    a, b = rng.randint(low, high), rng.randint(low, high)
    if op_symbol == "/":
        b = max(1, b)
    return a, b


def results_match(predicted, expected):
    if predicted is None:
        return False
    if isinstance(expected, float) or isinstance(predicted, float):
        return abs(float(predicted) - float(expected)) < 1e-9
    return predicted == expected


def build_cases():
    rng = random.Random(918273)
    tool_cases = []
    digit_bands = [
        ("1_to_3_digits", 1, 999),
        ("4_digits", 1000, 9999),
    ]
    cases_per_cell = 20
    for op_symbol, op in OPERATIONS.items():
        for phrasing, templates in (("familiar", op["familiar"]), ("new", op["new"])):
            for band_name, low, high in digit_bands:
                group = f"{op['name']}_{phrasing}_phrasing_{band_name}"
                for _ in range(cases_per_cell):
                    a, b = sample_pair(rng, low, high, op_symbol)
                    query = rng.choice(templates).format(a=a, b=b)
                    tool_cases.append(
                        {
                            "group": group,
                            "op": op_symbol,
                            "query": query,
                            "a": a,
                            "b": b,
                            "expected_expression": f"{a}{op_symbol}{b}",
                            "expected_result": op["compute"](a, b),
                        }
                    )

    retrieval_cases = []
    familiar_retrieval_templates = [
        "What was the result of my {ordinal} calculation?",
        "What did I get for the {ordinal} calculation?",
        "Retrieve my {ordinal} calculation.",
        "What was calculation number {number}?",
    ]
    for _ in range(50):
        count = rng.randint(2, 8)
        selected = rng.randrange(count)
        ordinal = "last" if selected == count - 1 and rng.random() < 0.25 else ORDINALS[selected]
        query = rng.choice(familiar_retrieval_templates).format(
            ordinal=ordinal, number=selected + 1
        )
        retrieval_cases.append(
            {"group": "new_memory_layouts", "query": query, "count": count, "selected": selected}
        )

    new_retrieval_templates = [
        "Pull up my {ordinal} saved calculation.",
        "Show calculation index {number}.",
        "Which leaf matches my {ordinal} result?",
        "Fetch calculations entry {number}.",
    ]
    for _ in range(50):
        count = rng.randint(2, 8)
        selected = rng.randrange(count)
        query = rng.choice(new_retrieval_templates).format(
            ordinal=ORDINALS[selected], number=selected + 1
        )
        retrieval_cases.append(
            {"group": "new_phrasings", "query": query, "count": count, "selected": selected}
        )
    return tool_cases, retrieval_cases, rng


def evaluate(model, tokenizer):
    tool_cases, retrieval_cases, rng = build_cases()
    outcomes = []

    for case in tool_cases:
        action = generate(model, tokenizer, MemoryStore(), case["query"])
        match = re.fullmatch(r"<tool:calc>(.*?)</tool:calc>", action, re.DOTALL)
        expression = match.group(1).strip() if match else None
        expression_correct = (
            re.sub(r"\s+", "", expression) == case["expected_expression"]
            if expression is not None
            else False
        )
        try:
            result = calculator.evaluate(expression) if expression is not None else None
        except (ValueError, ZeroDivisionError, OverflowError):
            result = None
        result_correct = results_match(result, case["expected_result"])
        outcomes.append(
            {
                "task": "tool",
                "group": case["group"],
                "op": case["op"],
                "query": case["query"],
                "action": action,
                "tool_format_valid": match is not None,
                "expression_correct": expression_correct,
                "result_correct": result_correct,
            }
        )

    for case in retrieval_cases:
        memory = MemoryStore()
        for _ in range(case["count"]):
            op_symbol = rng.choice(list(OPERATIONS))
            left, right = sample_pair(rng, 1, 999, op_symbol)
            expression = f"{left}{op_symbol}{right}"
            result = OPERATIONS[op_symbol]["compute"](left, right)
            memory.add_calculation(expression, result)
        action = generate(model, tokenizer, memory, case["query"])
        match = re.fullmatch(r"leaf\s+calculation_(\d+)", action, re.IGNORECASE)
        predicted = int(match.group(1)) - 1 if match else None
        correct = predicted == case["selected"]
        outcomes.append(
            {
                "task": "retrieval",
                "group": case["group"],
                "query": case["query"],
                "action": action,
                "expected_leaf": case["selected"] + 1,
                "predicted_leaf": predicted + 1 if predicted is not None else None,
                "correct": correct,
            }
        )

    return outcomes


def summarize(outcomes):
    summary = {}
    groups = sorted({(item["task"], item["group"]) for item in outcomes})
    for task, group in groups:
        rows = [item for item in outcomes if item["task"] == task and item["group"] == group]
        if task == "tool":
            summary[f"{group}_tool_format_rate"] = sum(
                item["tool_format_valid"] for item in rows
            ) / len(rows)
            summary[f"{group}_expression_exact_match"] = sum(
                item["expression_correct"] for item in rows
            ) / len(rows)
            summary[f"{group}_result_exact_match"] = sum(
                item["result_correct"] for item in rows
            ) / len(rows)
        else:
            summary[f"{group}_leaf_exact_match"] = sum(item["correct"] for item in rows) / len(rows)

    tool_rows = [item for item in outcomes if item["task"] == "tool"]
    for op_symbol, op in OPERATIONS.items():
        rows = [item for item in tool_rows if item["op"] == op_symbol]
        if rows:
            summary[f"overall_{op['name']}_result_exact_match"] = sum(
                item["result_correct"] for item in rows
            ) / len(rows)
    if tool_rows:
        summary["overall_tool_result_exact_match"] = sum(
            item["result_correct"] for item in tool_rows
        ) / len(tool_rows)
    return summary


def main():
    model, tokenizer = load_model()
    outcomes = evaluate(model, tokenizer)
    summary = summarize(outcomes)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_path = ROOT / f"evaluation_results_{timestamp}.json"
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model": str(MODEL_DIR),
        "device": DEVICE,
        "case_count": len(outcomes),
        "metrics": summary,
        "cases": outcomes,
    }
    results_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Evaluated {len(outcomes)} held-out cases on {DEVICE}.")
    for metric, value in summary.items():
        print(f"{metric}: {value:.1%}")
    print(f"Detailed predictions saved to {results_path.name}.")


if __name__ == "__main__":
    main()
