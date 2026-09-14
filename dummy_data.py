import os
import numpy as np

# Create the directory train.py is looking for
os.makedirs('data/openwebtext', exist_ok=True)

# Generate an array of random token IDs (between 0 and 50000)
train_data = np.random.randint(0, 50000, size=(100000,), dtype=np.uint16)
val_data = np.random.randint(0, 50000, size=(10000,), dtype=np.uint16)

# Save them as binary files exactly how nanoGPT expects
train_data.tofile('data/openwebtext/train.bin')
val_data.tofile('data/openwebtext/val.bin')

print("Dummy data successfully created!")