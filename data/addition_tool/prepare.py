import os
import pickle
import numpy as np

input_file_path = os.path.join(os.path.dirname(__file__), 'train.txt')
with open(input_file_path, 'r') as f:
    data = f.read()

# get all unique characters
chars = sorted(list(set(data)))
vocab_size = len(chars)

print("Vocab size:", vocab_size)

# create mappings
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# save meta
meta = {
    'vocab_size': vocab_size,
    'stoi': stoi,
    'itos': itos,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)

# process train
train_ids = encode(open(os.path.join(os.path.dirname(__file__), 'train.txt')).read())
val_ids = encode(open(os.path.join(os.path.dirname(__file__), 'val.txt')).read())

train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)

train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print("Done.")