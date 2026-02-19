"""
Generate training data for tool-use experiments.

Creates addition problems with tool-call annotations that teach
the model how to use a calculator for multi-digit addition.
"""

import random
import os


def generate_addition_example(num1, num2):
    """
    Generate a single addition problem with tool annotations.
    
    For example, 23+45 becomes:
    23+45=[TOOL]3+5[/TOOL]8[TOOL]2+4[/TOOL]6 68
    
    Args:
        num1 (int): First 2-digit number (10-99)
        num2 (int): Second 2-digit number (10-99)
    
    Returns:
        str: Formatted training example with newline
    """
    # Extract individual digits
    # For 23: tens_digit=2, ones_digit=3
    num1_tens = num1 // 10
    num1_ones = num1 % 10
    num2_tens = num2 // 10
    num2_ones = num2 % 10
    
    # Add ones place
    ones_sum = num1_ones + num2_ones
    ones_carry = 1 if ones_sum >= 10 else 0
    ones_result = ones_sum % 10
    
    # Add tens place (including carry)
    tens_sum = num1_tens + num2_tens + ones_carry
    tens_result = tens_sum % 10
    hundreds_digit = tens_sum // 10
    
    # Build the training example
    problem = f"{num1}+{num2}="
    
    # Ones place tool call
    solution = f"[TOOL]{num1_ones}+{num2_ones}[/TOOL]{ones_sum}"
    
    # Tens place tool call (include carry if needed)
    if ones_carry:
        solution += f"[TOOL]{num1_tens}+{num2_tens}+1[/TOOL]{tens_sum}"
    else:
        solution += f"[TOOL]{num1_tens}+{num2_tens}[/TOOL]{tens_sum}"
    
    # Final answer
    final_answer = num1 + num2
    solution += f" {final_answer}\n"
    
    return problem + solution


def generate_dataset(num_train=10000, num_val=1000):
    """
    Generate complete training and validation datasets.
    
    Args:
        num_train (int): Number of training examples
        num_val (int): Number of validation examples
    
    Returns:
        tuple: (train_examples, val_examples) as lists of strings
    """
    print(f"Generating {num_train} training examples...")
    train_data = []
    
    for i in range(num_train):
        if i % 2000 == 0:
            print(f"  Progress: {i}/{num_train}")
        
        # Generate random 2-digit numbers
        num1 = random.randint(10, 99)
        num2 = random.randint(10, 99)
        
        # Create training example
        example = generate_addition_example(num1, num2)
        train_data.append(example)
    
    print(f"Generating {num_val} validation examples...")
    val_data = []
    
    for i in range(num_val):
        num1 = random.randint(10, 99)
        num2 = random.randint(10, 99)
        example = generate_addition_example(num1, num2)
        val_data.append(example)
    
    return train_data, val_data


def main():
    """Generate and save the datasets."""
    print("=" * 60)
    print("Tool-Use Training Data Generator")
    print("=" * 60)
    print()
    
    # Generate data
    train_data, val_data = generate_dataset(num_train=10000, num_val=1000)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Save training data
    train_file = os.path.join(script_dir, 'input.txt')
    with open(train_file, 'w', encoding='utf-8') as f:
        f.writelines(train_data)
    
    print(f"\n✓ Saved {len(train_data)} training examples to: {train_file}")
    
    # Save validation data
    val_file = os.path.join(script_dir, 'input_val.txt')
    with open(val_file, 'w', encoding='utf-8') as f:
        f.writelines(val_data)
    
    print(f"✓ Saved {len(val_data)} validation examples to: {val_file}")
    
    # Show a sample
    print("\n" + "=" * 60)
    print("Sample training examples:")
    print("=" * 60)
    for i in range(3):
        print(train_data[i].strip())
    
    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)
    print("\nNext step: Convert to binary format for training")


if __name__ == "__main__":
    main()