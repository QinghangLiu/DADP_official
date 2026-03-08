#!/usr/bin/env python3
"""Compare two Minari datasets for exact equality."""

import argparse
import sys
import json
import numpy as np
import minari


def space_repr(space):
    try:
        return repr(space)
    except Exception:
        return str(space)


def compare_arrays(a, b, label):
    if a is None and b is None:
        return True
    if (a is None) != (b is None):
        print(f"Mismatch: {label} is None in one dataset")
        return False
    if a.shape != b.shape:
        print(f"Mismatch: {label} shape {a.shape} != {b.shape}")
        return False
    if a.dtype != b.dtype:
        print(f"Mismatch: {label} dtype {a.dtype} != {b.dtype}")
        return False
    if not np.array_equal(a, b):
        print(f"Mismatch: {label} values differ")
        return False
    return True


def compare_episode(ep_a, ep_b, idx):
    for field in ["observations", "actions", "rewards", "terminations", "truncations"]:
        a = getattr(ep_a, field, None)
        b = getattr(ep_b, field, None)
        if not compare_arrays(a, b, f"episode[{idx}].{field}"):
            return False

    infos_a = getattr(ep_a, "infos", None)
    infos_b = getattr(ep_b, "infos", None)
    if infos_a is None and infos_b is None:
        return True
    if (infos_a is None) != (infos_b is None):
        print(f"Mismatch: episode[{idx}].infos is None in one dataset")
        return False
    if isinstance(infos_a, dict) and isinstance(infos_b, dict):
        if infos_a.keys() != infos_b.keys():
            print(f"Mismatch: episode[{idx}].infos keys differ")
            return False
        for key in infos_a:
            a = infos_a[key]
            b = infos_b[key]
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                if not compare_arrays(np.asarray(a), np.asarray(b), f"episode[{idx}].infos[{key}]"):
                    return False
            else:
                if a != b:
                    print(f"Mismatch: episode[{idx}].infos[{key}] values differ")
                    return False
        return True

    if infos_a != infos_b:
        print(f"Mismatch: episode[{idx}].infos values differ")
        return False
    return True


def compare_metadata(ds_a, ds_b):
    meta_a = getattr(ds_a, "metadata", None)
    meta_b = getattr(ds_b, "metadata", None)
    if meta_a is None and meta_b is None:
        return True
    if (meta_a is None) != (meta_b is None):
        print("Mismatch: metadata is None in one dataset")
        return False
    if meta_a.keys() != meta_b.keys():
        print("Mismatch: metadata keys differ")
        return False
    for key in meta_a:
        a = meta_a[key]
        b = meta_b[key]
        if isinstance(a, dict) and isinstance(b, dict):
            if json.dumps(a, sort_keys=True) != json.dumps(b, sort_keys=True):
                print(f"Mismatch: metadata[{key}] differs")
                return False
        else:
            if a != b:
                print(f"Mismatch: metadata[{key}] differs")
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Compare two Minari datasets")
    parser.add_argument("dataset_a", type=str, help="First dataset id")
    parser.add_argument("dataset_b", type=str, help="Second dataset id")
    args = parser.parse_args()

    ds_a = minari.load_dataset(args.dataset_a)
    ds_b = minari.load_dataset(args.dataset_b)

    if len(ds_a) != len(ds_b):
        print(f"Mismatch: number of episodes {len(ds_a)} != {len(ds_b)}")
        sys.exit(1)

    if space_repr(ds_a.observation_space) != space_repr(ds_b.observation_space):
        print("Mismatch: observation_space differs")
        print(space_repr(ds_a.observation_space))
        print(space_repr(ds_b.observation_space))
        sys.exit(1)

    if space_repr(ds_a.action_space) != space_repr(ds_b.action_space):
        print("Mismatch: action_space differs")
        print(space_repr(ds_a.action_space))
        print(space_repr(ds_b.action_space))
        sys.exit(1)

    if not compare_metadata(ds_a, ds_b):
        sys.exit(1)

    for idx, (ep_a, ep_b) in enumerate(zip(ds_a, ds_b)):
        if not compare_episode(ep_a, ep_b, idx):
            sys.exit(1)

    print("Datasets are identical.")


if __name__ == "__main__":
    main()
