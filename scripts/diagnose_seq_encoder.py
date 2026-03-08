#!/usr/bin/env python3
"""Print system info relevant to determinism.

Example:
    python scripts/diagnose_seq_encoder.py
"""

from __future__ import annotations

import argparse
import torch


def _print_env_info() -> None:
    print("=== System ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"CuDNN version: {torch.backends.cudnn.version()}")
        try:
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            try:
                print(f"Driver: {torch._C._cuda_getDriverVersion()}")
            except Exception as exc:
                print(f"Driver: <error: {exc}>")
        except Exception as exc:
            print(f"GPU: <error: {exc}>")
    print("=== Backends ===")
    print(f"cudnn.deterministic: {torch.backends.cudnn.deterministic}")
    print(f"cudnn.benchmark: {torch.backends.cudnn.benchmark}")
    print(f"cuda.matmul.allow_tf32: {torch.backends.cuda.matmul.allow_tf32}")
    print(f"cudnn.allow_tf32: {torch.backends.cudnn.allow_tf32}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Print determinism-related system info")
    parser.parse_args()
    _print_env_info()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
