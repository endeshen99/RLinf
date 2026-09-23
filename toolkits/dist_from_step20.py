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

"""Action-chunk distance of a checkpoint export from step20 on a fixed observation set.

Draws a fixed set of task-8 observations (default 500, seed 0) from the recorded moka rollouts on disk
(``steps.npz`` per episode: 224x224 rotated frames, 8-d state, prompt), runs both policies with the
deterministic sampler and the same flow-noise draw per query, and reports the mean squared difference of the
10x7 unnormalized action chunks. Both models load through the standalone toolkit exactly as in
``libero_eval_draws.py``. Run inside the rlinf-osmesa container on a free GPU::

    python toolkits/dist_from_step20.py --ckpt /ckpts/exports/task2_A_final --out /results/dist/task2_A_final.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import random

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import torch

from toolkits.standalone_eval_scripts.openpi import setup_policy


class _Args:
    pass


def load_policy(path: str, config_name: str, num_steps: int):
    a = _Args()
    a.config_name, a.pretrained_path, a.num_steps = config_name, path, num_steps
    return setup_policy(a)


def sample_queries(episodes_glob: str, n: int, seed: int) -> list[tuple[str, int]]:
    """Return ``n`` (episode_dir, step) pairs drawn without replacement over all recorded steps."""
    pool = []
    for d in sorted(glob.glob(episodes_glob)):
        z = np.load(pathlib.Path(d) / "steps.npz")
        pool.extend((d, i) for i in range(len(z["t"])))
    if n > len(pool):
        raise SystemExit(
            f"requested {n} queries but only {len(pool)} recorded steps match {episodes_glob}"
        )
    picked = random.Random(seed).sample(pool, n)
    return sorted(picked)


def sample_queries_shards(root: str, n: int, seed: int) -> list[tuple[str, int]]:
    """Return ``n`` (parquet_path, row) pairs drawn without replacement over all frames of the LeRobot shards under ``root``."""
    import pyarrow.parquet as pq

    pool = []
    for f in sorted(
        glob.glob(
            os.path.join(root, "rank_*", "id_*", "data", "chunk-*", "episode_*.parquet")
        )
    ):
        pool.extend((f, i) for i in range(pq.read_metadata(f).num_rows))
    if n > len(pool):
        raise SystemExit(
            f"requested {n} queries but only {len(pool)} frames under {root}"
        )
    return sorted(random.Random(seed).sample(pool, n))


def shard_obs(cache: dict, f: str, i: int) -> dict:
    """Build the policy input for row ``i`` of shard parquet ``f`` (256x256 frames as the RLinf env produced them)."""
    import io

    import pyarrow.parquet as pq
    from PIL import Image

    if f not in cache:
        t = pq.read_table(f, columns=["image", "wrist_image", "state", "task_index"])
        root = pathlib.Path(f).parents[2]
        tasks = {
            json.loads(l)["task_index"]: json.loads(l)["task"]
            for l in open(root / "meta" / "tasks.jsonl")
        }
        cache[f] = (t, tasks)
    t, tasks = cache[f]
    r = t.slice(i, 1).to_pylist()[0]
    return {
        "observation/image": np.asarray(Image.open(io.BytesIO(r["image"]["bytes"]))),
        "observation/wrist_image": np.asarray(
            Image.open(io.BytesIO(r["wrist_image"]["bytes"]))
        ),
        "observation/state": np.asarray(r["state"], dtype=np.float32),
        "prompt": tasks[r["task_index"]],
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--ckpt",
        required=True,
        help="Checkpoint export dir (step20 layout) to compare.",
    )
    p.add_argument(
        "--ref",
        default="/ckpts/exports/step20",
        help="Reference export (default step20).",
    )
    p.add_argument("--config_name", default="pi05_libero")
    p.add_argument(
        "--episodes",
        default="/rollouts/libero_10/t08_put_both_moka_pots_on_the_stove/ep*",
        help="Recorded task-8 rollout episodes.",
    )
    p.add_argument(
        "--shards",
        default=None,
        help="LeRobot shard root (CollectEpisode layout) to draw observations from instead of --episodes, e.g. /rollouts/task2b/collect_rollouts for task 64.",
    )
    p.add_argument("--n", type=int, default=500)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the observation draw and the per-query flow noise.",
    )
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--ref_cache",
        default=None,
        help="npz path caching the reference policy's chunks on this query set; created on first use so later runs load only one policy (halves GPU memory).",
    )
    a = p.parse_args()

    queries = (
        sample_queries_shards(a.shards, a.n, a.seed)
        if a.shards
        else sample_queries(a.episodes, a.n, a.seed)
    )
    ref_chunks = None
    if a.ref_cache and os.path.exists(a.ref_cache):
        z = np.load(a.ref_cache)
        if int(z["n"]) == len(queries) and int(z["seed"]) == a.seed:
            ref_chunks = z["chunks"]
    ref = (
        None
        if ref_chunks is not None
        else load_policy(a.ref, a.config_name, a.num_steps)
    )
    ckpt = load_policy(a.ckpt, a.config_name, a.num_steps)
    cache: dict[str, tuple[np.lib.npyio.NpzFile, str]] = {}
    sq, per_dim, new_ref = [], [], []
    for q, (d, i) in enumerate(queries):
        if a.shards:
            obs = shard_obs(cache, d, i)
        else:
            if d not in cache:
                cache[d] = (
                    np.load(pathlib.Path(d) / "steps.npz"),
                    json.load(open(pathlib.Path(d) / "meta.json"))["task"],
                )
            z, task = cache[d]
            obs = {
                "observation/image": z["observation__image"][i],
                "observation/wrist_image": z["observation__wrist_image"][i],
                "observation/state": z["observation__state"][i],
                "prompt": task,
            }
        outs = []
        for pol in (ref, ckpt):
            torch.manual_seed(
                a.seed * 100003 + q
            )  # identical x0 for both models on this query
            if pol is None:
                outs.append(np.asarray(ref_chunks[q], dtype=np.float64))
            else:
                outs.append(np.asarray(pol.infer(obs)["actions"], dtype=np.float64))
        if ref is not None:
            new_ref.append(outs[0])
        diff = (outs[0] - outs[1]) ** 2
        sq.append(diff.mean())
        per_dim.append(diff.mean(0))
    sq = np.asarray(sq)
    if a.ref_cache and ref is not None:
        pathlib.Path(a.ref_cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            a.ref_cache, chunks=np.stack(new_ref), n=len(queries), seed=a.seed
        )
    rep = {
        "ref": a.ref,
        "ckpt": a.ckpt,
        "n_queries": int(len(sq)),
        "seed": a.seed,
        "episodes": a.shards or a.episodes,
        "chunk_mse_mean": float(sq.mean()),
        "chunk_mse_median": float(np.median(sq)),
        "chunk_mse_max": float(sq.max()),
        "chunk_rmse": float(np.sqrt(sq.mean())),
        "per_dim_mse": [float(x) for x in np.stack(per_dim).mean(0)],
    }
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
