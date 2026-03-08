#!/usr/bin/env python3
"""Simple loader for NPZ files."""

import argparse
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and summarize an NPZ file")
    parser.add_argument(
        "--path",
        default=(
            "/home/pengcheng/DomainAdaptiveDiffusionPolicy/dadp/embedding/logs/"
            "transformer/exp_walker_28(2)/RandomWalker2d_28dynamics-v0/transformer/"
            "20260227_181256/embeddings_data.npz"
        ),
        help="Path to the .npz file",
    )
    args = parser.parse_args()

    data = np.load(args.path, allow_pickle=True)
    print(f"Loaded: {args.path}")
    print("Keys:", list(data.files))
    for key in data.files:
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"- {key}: shape={arr.shape}, dtype={arr.dtype}")
        else:
            print(f"- {key}: type={type(arr)}")


if __name__ == "__main__":
    main()
