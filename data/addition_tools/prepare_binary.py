"""
Prepare character-level data for nanoGPT training.

Converts text data to binary format that nanoGPT can read efficiently.
"""

import os
import pickle
import numpy as np


def prepare_data_file(input_file, output_prefix):
    """
    Convert text file to binary format for nanoGPT.
    
    Args:
        input_file (str): Path to text file
        output_prefix (str): Prefix for output files (e.g., 'train' or 'val')
    """
    print(f"\nProcessing: {input_file}")
    print("-" * 60)
    
    # Read the text
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Dataset length: {len(data):,} characters")
    
    # Get unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Vocabulary: {repr(''.join(chars))}")
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}  # string to int
    itos = {i: ch for i, ch in enumerate(chars)}  # int to string
    
    # Encode the entire dataset
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    # Convert to numpy array
    data_ids = np.array(encode(data), dtype=np.uint16)
    
    # Save binary file
    bin_file = f'{output_prefix}.bin'
    data_ids.tofile(bin_file)
    print(f"✓ Saved {len(data_ids):,} tokens to: {bin_file}")
    
    # Save metadata (vocabulary mappings)
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    meta_file = f'{output_prefix}_meta.pkl'
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"✓ Saved metadata to: {meta_file}")
    
    return vocab_size


def main():
    """Prepare both training and validation data."""
    print("=" * 60)
    print("Converting Data to Binary Format for nanoGPT")
    print("=" * 60)
    
    # Get script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Prepare training data
    train_input = os.path.join(script_dir, 'input.txt')
    train_output = os.path.join(script_dir, 'train')
    vocab_size = prepare_data_file(train_input, train_output)
    
    # Prepare validation data
    val_input = os.path.join(script_dir, 'input_val.txt')
    val_output = os.path.join(script_dir, 'val')
    prepare_data_file(val_input, val_output)
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)
    print(f"\nVocabulary size: {vocab_size}")
    print("\nGenerated files:")
    print("  - train.bin (training data)")
    print("  - train_meta.pkl (vocabulary)")
    print("  - val.bin (validation data)")
    print("  - val_meta.pkl (vocabulary)")
    print("\nNext step: Create training config and train the model")


if __name__ == "__main__":
    main()