#!/usr/bin/env python3
"""Extract a task slice from a Minari dataset and save it as a new dataset."""

import argparse
from typing import Dict, Optional

import minari
import gymnasium as gym
from minari.dataset.minari_dataset import EpisodeBuffer
import numpy as np

DEFAULT_SOURCE = "RandomWalker2d/28dynamics-v9"
DEFAULT_TARGET = "RandomWalker2d/28dynamics-v0"


def _as_dict(infos):
    """Return a representative info dict or None."""
    if infos is None:
        return None
    if isinstance(infos, dict):
        return infos
    if isinstance(infos, (list, tuple)) and infos and isinstance(infos[0], dict):
        return infos[0]
    return None


def infer_task_id(infos, task_registry: Dict[tuple, int]) -> Optional[int]:
    """Infer integer task id from info payload or task vector."""
    info_dict = _as_dict(infos)
    if info_dict is None:
        return None

    for key in ("task_id", "task_index"):
        if key in info_dict:
            try:
                return int(np.asarray(info_dict[key]).flatten()[0])
            except Exception:
                pass

    task_val = info_dict.get("task")
    if task_val is not None:
        key = tuple(np.asarray(task_val).flatten().tolist())
        if key not in task_registry:
            task_registry[key] = len(task_registry)
        return task_registry[key]

    return None


def parse_tasks(selector: str) -> list[int]:
    """Parse task selectors while preserving the user's order and uniqueness."""
    if not selector:
        return []
    parts = selector.split(",")
    ordered_ids = []
    seen = set()
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            start_str, end_str = part.split(":", 1)
            start = int(start_str)
            end = int(end_str)
            if start >= end:
                raise ValueError(f"Invalid range {part}: start must be < end")
            for tid in range(start, end):
                if tid not in seen:
                    ordered_ids.append(tid)
                    seen.add(tid)
        else:
            tid = int(part)
            if tid not in seen:
                ordered_ids.append(tid)
                seen.add(tid)
    return ordered_ids


def main():
    parser = argparse.ArgumentParser(description="Slice Minari dataset by task id selection")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="Source Minari dataset id")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="New Minari dataset id to create")
    parser.add_argument(
        "--tasks",
        default="0:28",
        help="Task selector. Use Python-style ranges (e.g., 3:28 keeps 3-27) or comma lists (e.g., 3,4,5,27).",
    )
    parser.add_argument("--author", default=None, help="Author name metadata override")
    parser.add_argument("--author_email", default=None, help="Author email metadata override")
    args = parser.parse_args()

    task_ids = parse_tasks(args.tasks)
    if not task_ids:
        raise ValueError("No tasks selected. Provide --tasks like '3:28' or '1,2,5'.")

    source_ds = minari.load_dataset(args.source)
    env_spec = source_ds.env_spec
    obs_space = source_ds.observation_space
    act_space = source_ds.action_space

    class _SpecEnv(gym.Env):
        def __init__(self, spec, obs_space, act_space):
            super().__init__()
            self.spec = spec
            self.observation_space = obs_space
            self.action_space = act_space

        def reset(self, *, seed=None, options=None):  # pragma: no cover - not used
            raise NotImplementedError("Stub env; not for execution")

        def step(self, action):  # pragma: no cover - not used
            raise NotImplementedError("Stub env; not for execution")

    env = _SpecEnv(env_spec, obs_space, act_space)

    task_registry: Dict[tuple, int] = {}
    buckets: Dict[int, list[EpisodeBuffer]] = {tid: [] for tid in task_ids}
    for ep_idx in range(source_ds.total_episodes):
        episode = source_ds[ep_idx]
        task_id = infer_task_id(episode.infos, task_registry)
        if task_id is None:
            raise ValueError(f"Could not infer task id for episode {ep_idx}")
        if task_id in buckets:
            buckets[task_id].append(
                EpisodeBuffer(
                    id=None,
                    observations=episode.observations,
                    actions=episode.actions,
                    rewards=episode.rewards,
                    terminations=episode.terminations,
                    truncations=episode.truncations,
                    infos=episode.infos,
                )
            )

    kept_buffers: list[EpisodeBuffer] = []
    next_id = 0
    for tid in task_ids:
        for buf in buckets.get(tid, []):
            kept_buffers.append(
                EpisodeBuffer(
                    id=next_id,
                    observations=buf.observations,
                    actions=buf.actions,
                    rewards=buf.rewards,
                    terminations=buf.terminations,
                    truncations=buf.truncations,
                    infos=buf.infos,
                )
            )
            next_id += 1

    if not kept_buffers:
        raise RuntimeError("No episodes matched the requested task range")

    target_path = minari.storage.local.get_dataset_path(args.target)
    if target_path.exists():
        print(f"Dataset {args.target} already exists. Deleting...")
        minari.delete_dataset(args.target)

    metadata = getattr(source_ds, "metadata", {}) or {}
    algorithm_name = metadata.get("algorithm_name", f"subset-of-{args.source}")
    author = args.author or metadata.get("author", "Unknown")
    author_email = args.author_email or metadata.get("author_email", "unknown@example.com")
    code_permalink = metadata.get("code_permalink", "")
    description = metadata.get("description")
    data_format = metadata.get("data_format")
    requirements = metadata.get("requirements")
    ref_min_score = metadata.get("ref_min_score")
    ref_max_score = metadata.get("ref_max_score")
    num_episodes_average_score = metadata.get("num_episodes_average_score", 100)

    minari.create_dataset_from_buffers(
        dataset_id=args.target,
        env=env,
        buffer=kept_buffers,
        algorithm_name=algorithm_name,
        author=author,
        author_email=author_email,
        code_permalink=code_permalink,
        observation_space=obs_space,
        action_space=act_space,
        description=description,
        data_format=data_format,
        requirements=requirements,
        ref_min_score=ref_min_score,
        ref_max_score=ref_max_score,
        num_episodes_average_score=num_episodes_average_score,
    )

    kept_task_ids = [tid for tid in task_ids if buckets.get(tid)]
    print(
        f"Created {args.target} with {len(kept_buffers)} episodes from tasks in order {kept_task_ids}."
    )


if __name__ == "__main__":
    main()
