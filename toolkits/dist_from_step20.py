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
    p.add_argument("--n", type=int, default=500)
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the observation draw and the per-query flow noise.",
    )
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    queries = sample_queries(a.episodes, a.n, a.seed)
    ref, ckpt = (
        load_policy(a.ref, a.config_name, a.num_steps),
        load_policy(a.ckpt, a.config_name, a.num_steps),
    )
    cache: dict[str, tuple[np.lib.npyio.NpzFile, str]] = {}
    sq, per_dim = [], []
    for q, (d, i) in enumerate(queries):
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
            outs.append(np.asarray(pol.infer(obs)["actions"], dtype=np.float64))
        diff = (outs[0] - outs[1]) ** 2
        sq.append(diff.mean())
        per_dim.append(diff.mean(0))
    sq = np.asarray(sq)
    rep = {
        "ref": a.ref,
        "ckpt": a.ckpt,
        "n_queries": int(len(sq)),
        "seed": a.seed,
        "episodes": a.episodes,
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
