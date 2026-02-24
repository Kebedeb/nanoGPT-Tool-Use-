"""
Generate training data: 32+23=[TOOL]32+23[/TOOL]
No answers. Just syntax.
"""

import random
import os


def generate_example(num1, num2):
    """Generate: 32+23=[TOOL]32+23[/TOOL]\n"""
    return f"{num1}+{num2}=[TOOL]{num1}+{num2}[/TOOL]\n"


def main():
    print("Generating training data...")
    
    # 10000 training examples
    train_data = []
    for _ in range(10000):
        num1 = random.randint(10, 99)
        num2 = random.randint(10, 99)
        train_data.append(generate_example(num1, num2))
    
    # 1000 validation examples
    val_data = []
    for _ in range(1000):
        num1 = random.randint(10, 99)
        num2 = random.randint(10, 99)
        val_data.append(generate_example(num1, num2))
    
    # Save
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open(os.path.join(script_dir, 'input.txt'), 'w') as f:
        f.writelines(train_data)
    
    with open(os.path.join(script_dir, 'input_val.txt'), 'w') as f:
        f.writelines(val_data)
    
    print(f"✓ Saved {len(train_data)} training examples")
    print(f"✓ Saved {len(val_data)} validation examples")
    print("\nSample:")
    for i in range(3):
        print(f"  {train_data[i].strip()}")
    print("✅ Done")


if __name__ == "__main__":
    main()