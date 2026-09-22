#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build one SFT-ready LeRobot dataset from collected rollout shards.

``CollectEpisode`` (``env.train.data_collection`` with ``export_format:
lerobot``) writes one LeRobot dataset per ``rank_<r>/id_<n>/`` shard, and the
OpenPI SFT loader reads a single dataset root. This tool merges every shard
found under the given roots, optionally keeps only successful episodes (the
per-frame ``is_success`` feature written by ``CollectEpisode``), optionally
draws a seeded subsample of N episodes, and writes the result with
``merge_lerobot_datasets``. Episode counts are printed before and after each
selection stage.

Example::

    python toolkits/lerobot/lerobot_select.py \\
        --source-dir /rollouts/task2/collect_rollouts \\
        --only-success --n-episodes 64 --seed 0 \\
        --out /rollouts/task2/sft_data/success64_s0
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from merge_lerobot_datasets import EpisodeRef, merge_lerobot_datasets  # noqa: E402


def episode_is_success(parquet_path: Path) -> bool:
    """Return the episode-level success flag stored per frame by ``CollectEpisode``."""
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path, columns=["is_success"])
    col = table.column("is_success").to_pylist()
    if not col:
        return False
    last = col[-1]
    if isinstance(last, (list, tuple)):
        last = last[0]
    return bool(last)


def build_selector(only_success: bool, n_episodes: int | None, seed: int):
    """Return the ``select_episodes`` hook for ``merge_lerobot_datasets``."""

    def select(episodes: list[EpisodeRef]) -> list[EpisodeRef]:
        print(f"[select] discovered episodes: {len(episodes)}")
        if only_success:
            episodes = [e for e in episodes if episode_is_success(e[2])]
            print(f"[select] after --only-success: {len(episodes)}")
        if n_episodes is not None:
            if n_episodes > len(episodes):
                raise SystemExit(
                    f"--n-episodes {n_episodes} exceeds available {len(episodes)}"
                )
            rng = random.Random(seed)
            picked = sorted(rng.sample(range(len(episodes)), n_episodes))
            episodes = [episodes[i] for i in picked]
            print(
                f"[select] after --n-episodes {n_episodes} --seed {seed}: {len(episodes)}"
            )
        return episodes

    return select


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--source-dir",
        nargs="+",
        required=True,
        help="Shard root(s) written by CollectEpisode.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output dataset root (used as data.train_data_paths); optional with --dry-run.",
    )
    p.add_argument(
        "--only-success",
        action="store_true",
        help="Keep episodes whose is_success flag is set.",
    )
    p.add_argument(
        "--n-episodes",
        type=int,
        default=None,
        help="Seeded subsample size after filtering.",
    )
    p.add_argument(
        "--seed", type=int, default=0, help="Seed for --n-episodes sampling."
    )
    p.add_argument(
        "--dry-run", action="store_true", help="Print counts only; write nothing."
    )
    args = p.parse_args()
    if args.out is None and not args.dry_run:
        p.error("--out is required unless --dry-run is given")
    n = merge_lerobot_datasets(
        source_dirs=args.source_dir,
        output_dir=args.out or "",
        dry_run=args.dry_run,
        select_episodes=build_selector(args.only_success, args.n_episodes, args.seed),
    )
    if n == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
