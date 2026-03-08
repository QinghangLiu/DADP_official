#!/usr/bin/env python3
"""
Script to load and compare embedding data from .npz files
"""

import numpy as np
import os

# File paths
file1 = "/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/transformer/exp_walker_28(2)/RandomWalker2d_28dynamics-v0/transformer/20260227_181256/embeddings_data.npz"
file2 = "/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/exp/RandomWalker2d_28dynamics-v9/transformer/20251201_160026/embeddings_data.npz"

print("=" * 80)
print("Loading embeddings from .npz files")
print("=" * 80)

# Check if files exist
for fpath in [file1, file2]:
    if not os.path.exists(fpath):
        print(f"Warning: File not found: {fpath}")
    else:
        print(f"✓ Found: {fpath}")

print()

# Load file 1
print("Loading File 1:")
print("-" * 80)
try:
    data1 = np.load(file1, allow_pickle=True)
    print(f"Keys in file: {list(data1.keys())}")
    for key in data1.keys():
        arr = data1[key]
        if isinstance(arr, np.ndarray):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.size <= 20:
                print(f"    values: {arr}")
        else:
            print(f"  {key}: {type(arr)} = {arr}")
    print()
except Exception as e:
    print(f"Error loading file 1: {e}")
    print()

# Load file 2
print("Loading File 2:")
print("-" * 80)
try:
    data2 = np.load(file2, allow_pickle=True)
    print(f"Keys in file: {list(data2.keys())}")
    for key in data2.keys():
        arr = data2[key]
        if isinstance(arr, np.ndarray):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
            if arr.size <= 20:
                print(f"    values: {arr}")
        else:
            print(f"  {key}: {type(arr)} = {arr}")
    print()
except Exception as e:
    print(f"Error loading file 2: {e}")
    print()

# Compare if both loaded successfully
print("Comparison:")
print("-" * 80)
try:
    if 'data1' in locals() and 'data2' in locals():
        keys1 = set(data1.keys())
        keys2 = set(data2.keys())
        
        print(f"File 1 keys: {keys1}")
        print(f"File 2 keys: {keys2}")
        print(f"Common keys: {keys1.intersection(keys2)}")
        print(f"Unique to File 1: {keys1 - keys2}")
        print(f"Unique to File 2: {keys2 - keys1}")
        print()
        
        # Compare shared keys
        for key in keys1.intersection(keys2):
            arr1 = data1[key]
            arr2 = data2[key]
            
            if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
                print(f"Key '{key}':")
                print(f"  File 1 shape: {arr1.shape}, dtype: {arr1.dtype}")
                print(f"  File 2 shape: {arr2.shape}, dtype: {arr2.dtype}")
                
                if arr1.shape == arr2.shape:
                    diff = np.abs(arr1 - arr2).max()
                    print(f"  Max difference: {diff}")
                    if diff < 1e-5:
                        print(f"  → Shapes match and values are similar!")
                    else:
                        print(f"  → Shapes match but values differ")
                else:
                    print(f"  → Shape mismatch")
                print()
except Exception as e:
    print(f"Error during comparison: {e}")

print("=" * 80)
