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

"""Deterministic LIBERO eval with per-step trajectory dumps for failure diagnosis.

Same protocol and episode order as ``libero_eval_draws.py`` (25 init states, env seed 7, one
seeded flow-noise draw, chunk 5 / 10 steps, ODE sampler), but every episode also writes
``<log_dir>/<exp_name>/ep<trial>.npz`` with the end-effector position, gripper joint positions,
the executed action, and the position of every object the env exposes (``*_pos`` keys), so
stalls, gripper use and object approach can be measured after the fact.
"""

from __future__ import annotations

import collections
import json
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
import torch
from libero.libero import benchmark

from toolkits.standalone_eval_scripts.openpi import setup_logger, setup_policy
from toolkits.standalone_eval_scripts.openpi.libero_eval_draws import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    _get_libero_env,
    _quat2axisangle,
)

MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}


def main(args):
    logger = setup_logger(args.exp_name, args.log_dir)
    np.random.seed(args.seed)
    torch.manual_seed(args.noise_seed)
    torch.cuda.manual_seed_all(args.noise_seed)
    out = pathlib.Path(args.log_dir) / args.exp_name
    out.mkdir(parents=True, exist_ok=True)
    task_suite = benchmark.get_benchmark_dict()[args.task_suite_name]()
    max_steps = MAX_STEPS[args.task_suite_name]
    policy = setup_policy(args)
    logger.info("policy setup done")
    task = task_suite.get_task(args.task_id)
    initial_states = task_suite.get_task_init_states(args.task_id)
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
    n_succ = 0
    for episode_idx in range(args.num_trials_per_task):
        policy.reset()
        env.reset()
        action_plan = collections.deque()
        obs = env.set_init_state(initial_states[episode_idx])
        obj_keys = sorted(
            k for k in obs if k.endswith("_pos") and not k.startswith("robot0")
        )
        rec = {
            "eef_pos": [],
            "gripper_qpos": [],
            "action": [],
            **{k: [] for k in obj_keys},
        }
        done = False
        for t in range(max_steps + args.num_steps_wait):
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                continue
            if not action_plan:
                observation = {
                    "observation/image": np.ascontiguousarray(
                        obs["agentview_image"][::-1, ::-1]
                    ),
                    "observation/wrist_image": np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    ),
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    ),
                    "prompt": str(task_description),
                }
                action_plan.extend(
                    policy.infer(observation)["actions"][: args.action_chunk]
                )
            action = action_plan.popleft()
            rec["eef_pos"].append(np.asarray(obs["robot0_eef_pos"], dtype=np.float32))
            rec["gripper_qpos"].append(
                np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
            )
            rec["action"].append(np.asarray(action, dtype=np.float32))
            for k in obj_keys:
                rec[k].append(np.asarray(obs[k], dtype=np.float32))
            obs, reward, done, info = env.step(action.tolist())
            if done:
                break
        n_succ += bool(done)
        np.savez_compressed(
            out / f"ep{episode_idx:02d}.npz",
            success=bool(done),
            task=str(task_description),
            **{k: np.stack(v) for k, v in rec.items()},
        )
        with open(out / "episodes.jsonl", "a") as f:
            f.write(
                json.dumps(
                    {
                        "task_id": args.task_id,
                        "trial": episode_idx,
                        "success": bool(done),
                        "steps": len(rec["action"]),
                        "noise_seed": args.noise_seed,
                    }
                )
                + "\n"
            )
        logger.info(
            f"episode {episode_idx}: success={bool(done)} steps={len(rec['action'])} ({n_succ}/{episode_idx + 1})"
        )
    env.close()
    logger.info(
        f"Total Success Rate: {n_succ}/{args.num_trials_per_task} = {100 * n_succ / args.num_trials_per_task:.1f}%"
    )


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--log_dir", default="logs")
    p.add_argument("--exp_name", required=True)
    p.add_argument("--config_name", default="pi05_libero")
    p.add_argument("--pretrained_path", required=True)
    p.add_argument("--task_suite_name", default="libero_90")
    p.add_argument("--task_id", type=int, required=True)
    p.add_argument("--num_trials_per_task", type=int, default=25)
    p.add_argument("--action_chunk", type=int, default=5)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--num_steps_wait", type=int, default=10)
    p.add_argument(
        "--noise_seed", type=int, default=1000, help="draw 0 of the 8-draw protocol"
    )
    p.add_argument("--seed", type=int, default=7)
    main(p.parse_args())
