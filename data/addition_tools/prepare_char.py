"""
Prepare character-level data for nanoGPT training.

This converts our text data into the format nanoGPT expects.
"""

import os
import pickle
import numpy as np


def prepare_data(input_file, output_prefix):
    """
    Convert text file to character-level token IDs.
    
    Args:
        input_file (str): Path to input text file
        output_prefix (str): Prefix for output files (train/val)
    """
    # Read the text data
    with open(input_file, 'r') as f:
        data = f.read()
    
    print(f"Length of dataset: {len(data):,} characters")
    
    # Get all unique characters
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"Vocabulary size: {vocab_size} unique characters")
    print(f"Vocabulary: {''.join(chars)}")
    
    # Create character <-> integer mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode the entire dataset
    def encode(s):
        return [stoi[c] for c in s]
    
    def decode(l):
        return ''.join([itos[i] for i in l])
    
    # Convert to numpy array
    ids = np.array(encode(data), dtype=np.uint16)
    
    # Save to binary file
    ids_file = f'{output_prefix}.bin'
    ids.tofile(ids_file)
    print(f"Saved {len(ids):,} tokens to {ids_file}")
    
    # Save metadata
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    meta_file = f'{output_prefix}_meta.pkl'
    with open(meta_file, 'wb') as f:
        pickle.dump(meta, f)
    print(f"Saved metadata to {meta_file}")
    
    return vocab_size


def main():
    """
    Prepare all three dataset formats.
    """
    data_dir = os.path.dirname(__file__)
    if not data_dir:
        data_dir = "."
    
    formats = ["basic", "cot", "scratchpad"]
    
    for fmt in formats:
        print("\n" + "=" * 60)
        print(f"Preparing {fmt} format")
        print("=" * 60)
        
        # Prepare training data
        train_file = os.path.join(data_dir, f"train_{fmt}.txt")
        train_prefix = os.path.join(data_dir, f"train_{fmt}")
        print(f"\nProcessing {train_file}...")
        prepare_data(train_file, train_prefix)
        
        # Prepare validation data
        val_file = os.path.join(data_dir, f"val_{fmt}.txt")
        val_prefix = os.path.join(data_dir, f"val_{fmt}")
        print(f"\nProcessing {val_file}...")
        prepare_data(val_file, val_prefix)
    
    print("\n" + "=" * 60)
    print("✅ Data preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()