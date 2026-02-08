"""Quick verification that the generated dataset loads correctly."""
import numpy as np
import json

d = np.load("data/raw/dataset.npz")
print(f"Batches shape: {d['batches'].shape}")
print(f"Labels shape:  {d['labels'].shape}")
print(f"Strong: {(d['labels']==1).sum()}, Weak: {(d['labels']==0).sum()}")
print(f"Unique values in batches: {np.unique(d['batches'])}")

with open("data/raw/metadata.json") as f:
    m = json.load(f)
print(f"Generator types used: {len(set(m['generator_names']))}")
print(f"Strong gens: {m['strong_generators']}")
print(f"Weak gens:   {m['weak_generators']}")
print("\nDataset verified OK.")
