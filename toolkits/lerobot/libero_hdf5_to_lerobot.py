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

"""Convert one LIBERO demo file (hdf5) into the LeRobot layout that
``CollectEpisode`` writes, so human demos and policy rollouts train through the
same OpenPI SFT loader.

Per frame the output carries ``image`` and ``wrist_image`` (uint8 HWC),
``state`` (eef position, eef axis-angle, gripper qpos: 8 floats, the same
concatenation ``LiberoEnv`` builds), ``actions`` (7 floats, the LIBERO action
space), the task instruction from the file's ``problem_info``, and the
``is_success`` / ``done`` flags. Demo frames are rotated by 180 degrees
(``[::-1, ::-1]``) exactly as ``rlinf.envs.sim.libero.utils.get_libero_image``
rotates live observations; the 128x128 demo resolution is kept and left to the
policy's resize transform.

Example::

    python toolkits/lerobot/libero_hdf5_to_lerobot.py \\
        --hdf5 /rollouts/task2/libero_hdf5/KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_demo.hdf5 \\
        --out /rollouts/task2/sft_data/demos_task38 --n-demos 2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

import h5py
import numpy as np


def rotate_like_env(img: np.ndarray) -> np.ndarray:
    """Rotate a raw robosuite frame by 180 degrees, as the live env does."""
    return np.ascontiguousarray(img[::-1, ::-1])


def demo_frames(demo: h5py.Group, task: str) -> list[dict]:
    obs = demo["obs"]
    actions = np.asarray(demo["actions"], dtype=np.float32)
    state = np.concatenate(
        [
            np.asarray(obs["ee_pos"], dtype=np.float32),
            np.asarray(obs["ee_ori"], dtype=np.float32),
            np.asarray(obs["gripper_states"], dtype=np.float32),
        ],
        axis=1,
    )
    main = np.asarray(obs["agentview_rgb"])
    wrist = np.asarray(obs["eye_in_hand_rgb"])
    n = len(actions)
    frames = []
    for i in range(n):
        frames.append(
            {
                "image": rotate_like_env(main[i]),
                "wrist_image": rotate_like_env(wrist[i]),
                "state": state[i],
                "actions": actions[i],
                "task": task,
                "is_success": np.array([True], dtype=bool),
                "done": np.array([i == n - 1], dtype=bool),
                "intervene_flag": np.array([False], dtype=bool),
            }
        )
    return frames


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hdf5", required=True, help="LIBERO demo file for one task.")
    p.add_argument(
        "--out",
        required=True,
        help="Output dataset root (used as data.train_data_paths).",
    )
    p.add_argument(
        "--n-demos",
        type=int,
        default=None,
        help="Seeded subsample of demos; default keeps all.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--fps",
        type=int,
        default=10,
        help="LeRobot fps metadata; LIBERO demos and rollouts are 10 Hz control.",
    )
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from rlinf.data.storage.lerobot import LeRobotDatasetWriter

    out = Path(args.out).resolve()
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    with h5py.File(args.hdf5, "r") as f:
        data = f["data"]
        task = json.loads(data.attrs["problem_info"])["language_instruction"]
        names = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))
        print(f"[demos] {args.hdf5}: {len(names)} demos, task '{task}'")
        if args.n_demos is not None:
            names = sorted(
                random.Random(args.seed).sample(names, args.n_demos),
                key=lambda k: int(k.split("_")[1]),
            )
            print(
                f"[demos] after --n-demos {args.n_demos} --seed {args.seed}: {len(names)} -> {names}"
            )

        writer = LeRobotDatasetWriter()
        first = data[names[0]]["obs"]["agentview_rgb"]
        writer.create(
            repo_id=str(out),
            robot_type="panda",
            fps=args.fps,
            image_shape=tuple(first.shape[1:]),
            state_dim=8,
            action_dim=7,
            has_image=True,
            wrist_image_keys={"wrist_image": tuple(first.shape[1:])},
            has_intervene_flag=True,
            has_segment_id=False,
        )
        n_frames = 0
        for name in names:
            frames = demo_frames(data[name], task)
            writer.add_episode(frames)
            n_frames += len(frames)
        writer.finalize()
    print(f"[demos] wrote {len(names)} episodes / {n_frames} frames -> {out}")


if __name__ == "__main__":
    main()
