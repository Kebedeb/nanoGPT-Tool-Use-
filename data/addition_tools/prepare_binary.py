"""
Convert text data to binary for nanoGPT.
"""

import os
import pickle
import numpy as np


def prepare(input_file, output_prefix):
    """Convert text to binary."""
    with open(input_file, 'r', encoding='utf-8') as f:
        data = f.read()
    
    print(f"Processing {input_file}")
    print(f"  Length: {len(data):,} characters")
    
    # Get vocabulary
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print(f"  Vocab size: {vocab_size}")
    print(f"  Vocab: {repr(''.join(chars))}")
    
    # Create mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    
    # Encode
    data_ids = np.array([stoi[c] for c in data], dtype=np.uint16)
    
    # Save binary
    data_ids.tofile(f'{output_prefix}.bin')
    print(f"  ✓ Saved {output_prefix}.bin")
    
    # Save meta
    meta = {'vocab_size': vocab_size, 'itos': itos, 'stoi': stoi}
    with open(f'{output_prefix}_meta.pkl', 'wb') as f:
        pickle.dump(meta, f)
    print(f"  ✓ Saved {output_prefix}_meta.pkl")
    
    return vocab_size


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Train
    train_input = os.path.join(script_dir, 'input.txt')
    train_output = os.path.join(script_dir, 'train')
    vocab_size = prepare(train_input, train_output)
    
    print()
    
    # Val
    val_input = os.path.join(script_dir, 'input_val.txt')
    val_output = os.path.join(script_dir, 'val')
    prepare(val_input, val_output)
    
    print(f"\n✅ Done! Vocab size: {vocab_size}")


if __name__ == "__main__":
    main()