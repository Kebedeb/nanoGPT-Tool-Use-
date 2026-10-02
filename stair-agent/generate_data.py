"""Generate supervised examples for model-learned tool routing and retrieval."""

import operator
import random
from pathlib import Path

import numpy as np
import tiktoken


BLOCK_SIZE = 256
PAD_ID = 50256
N_TOOL_EXAMPLES = 10000
N_RETRIEVAL_EXAMPLES = 6000
SEED = 1729
DATA_PATH = Path(__file__).resolve().parent / "train_examples.npz"

base_enc = tiktoken.get_encoding("gpt2")
special_tokens = {
    "<tool:calc>": 50257,
    "</tool:calc>": 50258,
    "<lookup:calculations>": 50259,
    "[TABLE OF CONTENTS]": 50260,
    "[END TABLE OF CONTENTS]": 50261,
    "[CALCULATIONS CONTENT]:": 50262,
}
enc = tiktoken.Encoding(
    name="gpt2_calculator_agent",
    pat_str=base_enc._pat_str,
    mergeable_ranks=base_enc._mergeable_ranks,
    special_tokens={**base_enc._special_tokens, **special_tokens},
)

INSTRUCTIONS = (
    "You are a calculator agent. For an arithmetic request, respond with only "
    "<tool:calc>expression</tool:calc>. For a question about a saved calculation, "
    "respond with only its ToC leaf ID, in the form leaf calculation_2."
)

# Train on a wide paraphrase net, including order-sensitive wording.
# Held-out eval templates live only in evaluate_retrieval.py.
OPERATIONS = {
    "+": {
        "symbol": "+",
        "compute": operator.add,
        "templates": [
            "What is {a} plus {b}?",
            "Calculate {a} + {b}.",
            "Add {a} and {b}.",
            "What's the sum of {a} and {b}?",
            "How much is {a} plus {b}?",
            "Please compute {a}+{b}.",
            "Can you total {a} and {b}?",
            "Please add {a} to {b}.",
            "What do {a} and {b} add up to?",
            "Find the sum of {a} and {b}.",
            "Tell me {a} plus {b}.",
            "Sum {a} and {b}.",
        ],
    },
    "-": {
        "symbol": "-",
        "compute": operator.sub,
        "templates": [
            "What is {a} minus {b}?",
            "Calculate {a} - {b}.",
            "Subtract {b} from {a}.",
            "What's the difference between {a} and {b}?",
            "How much is {a} minus {b}?",
            "Please compute {a}-{b}.",
            "Please subtract {b} from {a}.",
            "Can you take {b} away from {a}?",
            "Find the difference of {a} and {b}.",
            "Tell me {a} minus {b}.",
            "What is left if {a} loses {b}?",
            "Take {b} from {a}.",
            "Remove {b} from {a}.",
            "{a} minus {b} equals what?",
        ],
    },
    "*": {
        "symbol": "*",
        "compute": operator.mul,
        "templates": [
            "What is {a} times {b}?",
            "Calculate {a} * {b}.",
            "Multiply {a} by {b}.",
            "What's the product of {a} and {b}?",
            "How much is {a} times {b}?",
            "Please compute {a}*{b}.",
            "Can you multiply {a} with {b}?",
            "Please find {a} multiplied by {b}.",
            "What do {a} and {b} multiply to?",
            "Find the product of {a} and {b}.",
            "Tell me {a} times {b}.",
            "{a} multiplied by {b} equals what?",
            "Compute the product of {a} and {b}.",
            "Times {a} by {b}.",
        ],
    },
    "/": {
        "symbol": "/",
        "compute": operator.truediv,
        "templates": [
            "What is {a} divided by {b}?",
            "Calculate {a} / {b}.",
            "Divide {a} by {b}.",
            "What's {a} over {b}?",
            "How much is {a} divided by {b}?",
            "Please compute {a}/{b}.",
            "Can you divide {a} by {b}?",
            "Please find {a} divided by {b}.",
            "Find the quotient of {a} and {b}.",
            "Tell me {a} over {b}.",
            "What is {a} split into {b} equal parts?",
            "{a} divided by {b} equals what?",
            "Compute {a} divided by {b}.",
            "Share {a} among {b}.",
        ],
    },
}


def make_toc(entries):
    lines = [
        "[TABLE OF CONTENTS]",
        " - variables (0 items)",
        f" - calculations ({len(entries)} items)",
    ]
    lines.extend(
        f"   leaf calculation_{index}: {expression}"
        for index, (expression, _result) in enumerate(entries, start=1)
    )
    lines.extend([" - notes (0 items)", "[END TABLE OF CONTENTS]"])
    return "\n".join(lines)


def encode_example(prompt, answer):
    """Return next-token training arrays with loss restricted to the answer."""
    prompt_ids = enc.encode(prompt, allowed_special="all")
    answer_ids = enc.encode(answer, allowed_special="all") + [enc.eot_token]
    token_ids = prompt_ids + answer_ids
    if len(token_ids) - 1 > BLOCK_SIZE:
        raise ValueError(f"Example has {len(token_ids) - 1} tokens; limit is {BLOCK_SIZE}.")

    x = np.full(BLOCK_SIZE, PAD_ID, dtype=np.int64)
    y = np.full(BLOCK_SIZE, -100, dtype=np.int64)
    n = len(token_ids) - 1
    x[:n] = token_ids[:-1]
    y[:n] = token_ids[1:]
    y[: len(prompt_ids) - 1] = -100
    return x, y


def sample_operand(rng):
    """Balance digit lengths so the model practices copying each width."""
    low, high = rng.choice([(1, 9), (10, 99), (100, 999), (1000, 9999)])
    return rng.randint(low, high)


def sample_operands(rng, op_symbol):
    """Sample a,b for an op. Division always uses a non-zero divisor."""
    left, right = sample_operand(rng), sample_operand(rng)
    if op_symbol == "/":
        while right == 0:
            right = sample_operand(rng)
    return left, right


def format_result(value):
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def sample_calculation(rng, max_digits=3):
    """Build one saved calculation for the retrieval ToC."""
    op_symbol = rng.choice(list(OPERATIONS))
    low, high = (1, 9) if max_digits == 1 else (1, 10**max_digits - 1)
    left, right = rng.randint(low, high), rng.randint(low, high)
    if op_symbol == "/":
        right = max(1, right)
    expression = f"{left}{op_symbol}{right}"
    result = format_result(OPERATIONS[op_symbol]["compute"](left, right))
    return expression, result


def main():
    rng = random.Random(SEED)
    examples = []
    op_names = list(OPERATIONS)
    per_op = N_TOOL_EXAMPLES // len(op_names)

    for op_symbol in op_names:
        op = OPERATIONS[op_symbol]
        for _ in range(per_op):
            left, right = sample_operands(rng, op_symbol)
            query = rng.choice(op["templates"]).format(a=left, b=right)
            expression = f"{left}{op_symbol}{right}"
            prompt = f"{make_toc([])}\n{INSTRUCTIONS}\nUser: {query}\nModel: "
            examples.append(encode_example(prompt, f"<tool:calc>{expression}</tool:calc>"))

    ordinal_words = ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth"]
    retrieval_templates = [
        "What was the result of my {ordinal} calculation?",
        "What did I get for the {ordinal} calculation?",
        "Retrieve my {ordinal} calculation.",
        "What was calculation number {number}?",
        "Which saved calculation was my {ordinal} one?",
        "Return the result from calculation #{number}.",
        "Can you remind me what my {ordinal} result was?",
        "Look up entry number {number} in my calculations.",
    ]
    for _ in range(N_RETRIEVAL_EXAMPLES):
        count = rng.randint(2, len(ordinal_words))
        entries = [sample_calculation(rng, max_digits=3) for _entry in range(count)]

        selected = rng.randrange(count)
        ordinal = "last" if selected == count - 1 and rng.random() < 0.35 else ordinal_words[selected]
        query_template = rng.choice(retrieval_templates)
        if ordinal == "last":
            query_template = rng.choice(
                [
                    "What was the result of my {ordinal} calculation?",
                    "Retrieve my {ordinal} calculation.",
                ]
            )
        query = query_template.format(ordinal=ordinal, number=selected + 1)
        prompt = f"{make_toc(entries)}\n{INSTRUCTIONS}\nUser: {query}\nModel: "
        examples.append(encode_example(prompt, f"leaf calculation_{selected + 1}"))

    rng.shuffle(examples)
    train_x = np.stack([row[0] for row in examples])
    train_y = np.stack([row[1] for row in examples])
    np.savez_compressed(DATA_PATH, x=train_x, y=train_y)
    print(
        f"Saved {len(examples)} examples to {DATA_PATH.name} "
        f"({per_op * len(op_names)} tool calls across {', '.join(op_names)}, "
        f"{N_RETRIEVAL_EXAMPLES} leaf selections)."
    )


if __name__ == "__main__":
    main()
