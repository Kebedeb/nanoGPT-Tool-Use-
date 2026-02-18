"""
Generate training data for tool-use experiments.

This script creates addition problems with tool-call annotations
that teach the GPT how to use the calculator tool.
"""

import random
import os


def generate_basic_format(num1, num2):
    """
    Generate addition problem in basic format (no CoT, no scratchpad).
    
    Example: "23+45=[TOOL]3+5[/TOOL]8[TOOL]2+4[/TOOL]6 68"
    
    Args:
        num1 (int): First 2-digit number (10-99)
        num2 (int): Second 2-digit number (10-99)
    
    Returns:
        str: Formatted training example
    """
    # Break numbers into digits
    a_tens, a_ones = divmod(num1, 10)
    b_tens, b_ones = divmod(num2, 10)
    
    # Calculate ones place
    ones_sum = a_ones + b_ones
    carry = 1 if ones_sum >= 10 else 0
    ones_digit = ones_sum % 10
    
    # Calculate tens place
    tens_sum = a_tens + b_tens + carry
    tens_digit = tens_sum % 10
    hundreds_digit = tens_sum // 10
    
    # Build the formatted string
    problem = f"{num1}+{num2}="
    
    # Tool call for ones place
    solution = f"[TOOL]{a_ones}+{b_ones}[/TOOL]{ones_sum}"
    
    # Tool call for tens place
    if carry:
        solution += f"[TOOL]{a_tens}+{b_tens}+1[/TOOL]{tens_sum}"
    else:
        solution += f"[TOOL]{a_tens}+{b_tens}[/TOOL]{tens_sum}"
    
    # Final answer
    answer = num1 + num2
    solution += f" {answer}"
    
    return problem + solution + "\n"


def generate_cot_format(num1, num2):
    """
    Generate addition problem with Chain-of-Thought reasoning.
    
    Example: "23+45=Let me think. Ones: [TOOL]3+5[/TOOL]8. Tens: [TOOL]2+4[/TOOL]6. So 68"
    
    Args:
        num1 (int): First 2-digit number (10-99)
        num2 (int): Second 2-digit number (10-99)
    
    Returns:
        str: Formatted training example with CoT
    """
    # Break numbers into digits
    a_tens, a_ones = divmod(num1, 10)
    b_tens, b_ones = divmod(num2, 10)
    
    # Calculate
    ones_sum = a_ones + b_ones
    carry = 1 if ones_sum >= 10 else 0
    ones_digit = ones_sum % 10
    
    tens_sum = a_tens + b_tens + carry
    answer = num1 + num2
    
    # Build with explicit reasoning
    problem = f"{num1}+{num2}="
    solution = "Let me solve step by step. "
    
    # Ones place reasoning
    solution += f"Ones place: {a_ones}+{b_ones}=[TOOL]{a_ones}+{b_ones}[/TOOL]{ones_sum}"
    if carry:
        solution += f", carry 1. "
    else:
        solution += ". "
    
    # Tens place reasoning
    if carry:
        solution += f"Tens place: {a_tens}+{b_tens}+1=[TOOL]{a_tens}+{b_tens}+1[/TOOL]{tens_sum}. "
    else:
        solution += f"Tens place: {a_tens}+{b_tens}=[TOOL]{a_tens}+{b_tens}[/TOOL]{tens_sum}. "
    
    # Final answer
    solution += f"Answer: {answer}"
    
    return problem + solution + "\n"


def generate_scratchpad_format(num1, num2):
    """
    Generate addition problem with scratchpad workspace.
    
    Example: "23+45=<SCRATCH>ones [TOOL]3+5[/TOOL]8 tens [TOOL]2+4[/TOOL]6</SCRATCH> 68"
    
    Args:
        num1 (int): First 2-digit number (10-99)
        num2 (int): Second 2-digit number (10-99)
    
    Returns:
        str: Formatted training example with scratchpad
    """
    # Break numbers into digits
    a_tens, a_ones = divmod(num1, 10)
    b_tens, b_ones = divmod(num2, 10)
    
    # Calculate
    ones_sum = a_ones + b_ones
    carry = 1 if ones_sum >= 10 else 0
    
    tens_sum = a_tens + b_tens + carry
    answer = num1 + num2
    
    # Build with scratchpad
    problem = f"{num1}+{num2}="
    solution = "<SCRATCH>"
    
    # Scratchpad work
    solution += f"ones:[TOOL]{a_ones}+{b_ones}[/TOOL]{ones_sum} "
    if carry:
        solution += f"tens:[TOOL]{a_tens}+{b_tens}+1[/TOOL]{tens_sum}"
    else:
        solution += f"tens:[TOOL]{a_tens}+{b_tens}[/TOOL]{tens_sum}"
    
    solution += "</SCRATCH> "
    
    # Final answer (what user sees)
    solution += f"{answer}"
    
    return problem + solution + "\n"


def generate_dataset(n_train=10000, n_val=1000, format_type="basic"):
    """
    Generate complete training and validation datasets.
    
    Args:
        n_train (int): Number of training examples
        n_val (int): Number of validation examples
        format_type (str): "basic", "cot", or "scratchpad"
    
    Returns:
        tuple: (train_data, val_data) as lists of strings
    """
    # Choose the format function
    if format_type == "basic":
        format_fn = generate_basic_format
    elif format_type == "cot":
        format_fn = generate_cot_format
    elif format_type == "scratchpad":
        format_fn = generate_scratchpad_format
    else:
        raise ValueError(f"Unknown format: {format_type}")
    
    print(f"Generating {n_train} training examples ({format_type} format)...")
    train_data = []
    for i in range(n_train):
        if i % 1000 == 0:
            print(f"  Progress: {i}/{n_train}")
        
        # Generate random 2-digit numbers
        num1 = random.randint(10, 99)
        num2 = random.randint(10, 99)
        
        # Format and add to dataset
        train_data.append(format_fn(num1, num2))
    
    print(f"Generating {n_val} validation examples...")
    val_data = []
    for i in range(n_val):
        num1 = random.randint(10, 99)
        num2 = random.randint(10, 99)
        val_data.append(format_fn(num1, num2))
    
    return train_data, val_data


def main():
    """
    Main function to generate all dataset variants.
    """
    print("=" * 60)
    print("nanoGPT Tool-Use Data Generator")
    print("=" * 60)
    
    # Create output directory
    output_dir = os.path.dirname(__file__)
    if not output_dir:
        output_dir = "."
    
    # Generate each format
    formats = ["basic", "cot", "scratchpad"]
    
    for fmt in formats:
        print(f"\nGenerating {fmt} format...")
        print("-" * 60)
        
        # Generate data
        train_data, val_data = generate_dataset(
            n_train=10000,
            n_val=1000,
            format_type=fmt
        )
        
        # Save training data
        train_file = os.path.join(output_dir, f"train_{fmt}.txt")
        with open(train_file, 'w') as f:
            f.writelines(train_data)
        print(f"✓ Saved {len(train_data)} training examples to {train_file}")
        
        # Save validation data
        val_file = os.path.join(output_dir, f"val_{fmt}.txt")
        with open(val_file, 'w') as f:
            f.writelines(val_data)
        print(f"✓ Saved {len(val_data)} validation examples to {val_file}")
        
        # Show sample
        print("\nSample training example:")
        print(train_data[0].strip())
    
    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    print("=" * 60)
    print("\nGenerated files:")
    for fmt in formats:
        print(f"  - train_{fmt}.txt")
        print(f"  - val_{fmt}.txt")
    print("\nNext step: Train a model using one of these datasets")


if __name__ == "__main__":
    main()