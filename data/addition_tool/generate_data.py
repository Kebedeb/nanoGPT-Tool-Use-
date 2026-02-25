import random
import os

def generate_example():
    # TRAIN ONLY ON SMALL NUMBERS
    a = random.randint(0, 200)
    b = random.randint(0, 200)
    result = a + b

    return f"{a}+{b}=\n<CALL>add(a={a},b={b})</CALL>\n<RESULT>{result}</RESULT>\n\n"

def generate_file(filename, n):
    with open(filename, "w") as f:
        for _ in range(n):
            f.write(generate_example())

if __name__ == "__main__":
    os.makedirs("data/addition_tool", exist_ok=True)
    generate_file("data/addition_tool/train.txt", 50000)
    generate_file("data/addition_tool/val.txt", 5000)