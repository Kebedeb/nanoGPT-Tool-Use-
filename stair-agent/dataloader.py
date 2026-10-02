import torch
import numpy as np

# 1. Load the binary array we generated
data = np.fromfile('train.bin', dtype=np.uint16)

def get_batch(batch_size=4, block_size=64):
    """
    block_size: the maximum context length for predictions
    batch_size: how many independent sequences to process in parallel
    """
    # Grab random starting indices for the batch
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # X is the input sequence
    x = torch.stack([torch.from_numpy((data[i : i+block_size]).astype(np.int64)) for i in ix])
    
    # Y is the target sequence (shifted right by exactly one token)
    y = torch.stack([torch.from_numpy((data[i+1 : i+1+block_size]).astype(np.int64)) for i in ix])
    
    return x, y

if __name__ == "__main__":
    X, Y = get_batch()
    print("Inputs (X) shape:", X.shape)
    print("Targets (Y) shape:", Y.shape)
    
    # Show what the model actually sees in the first sequence
    print("\nFirst input sequence (X[0]):", X[0].tolist())
    print("First target sequence (Y[0]):", Y[0].tolist())