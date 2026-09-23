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

"""Re-render a LIBERO demo file at the env's native resolution from its stored sim states.

The stock demo hdf5 carries 128x128 frames; the RLinf rollouts are rendered at 256x256. This
tool replays every stored MuJoCo state of every demo through ``OffScreenRenderEnv`` at the
requested resolution and writes a new hdf5 with the same layout (``data/demo_N/obs/agentview_rgb``,
``obs/eye_in_hand_rgb``, ``obs/ee_pos``, ``obs/ee_ori``, ``obs/gripper_states``, ``actions`` and
the ``problem_info`` attribute), so ``libero_hdf5_to_lerobot.py`` converts it unchanged. Frames
are stored in the raw render orientation, as in the original file; the converter applies the
same 180-degree rotation the live env applies. Actions and states are copied, not recomputed.
CPU-only (osmesa)::

    python toolkits/lerobot/libero_rerender_demos.py --hdf5 <demo.hdf5> --out <demo_256.hdf5> --resolution 256
"""

from __future__ import annotations

import argparse
import os
import pathlib

os.environ.setdefault("MUJOCO_GL", "osmesa")
import h5py
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--hdf5", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--resolution", type=int, default=256)
    p.add_argument(
        "--n-demos",
        type=int,
        default=None,
        help="Only the first N demos (for smoke tests).",
    )
    a = p.parse_args()

    src = h5py.File(a.hdf5, "r")
    data = src["data"]
    bddl = pathlib.Path(get_libero_path("bddl_files")) / pathlib.Path(
        data.attrs["bddl_file_name"]
    ).relative_to("libero/libero/bddl_files")
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=a.resolution,
        camera_widths=a.resolution,
    )
    env.seed(0)
    env.reset()
    names = sorted(data.keys(), key=lambda k: int(k.split("_")[1]))[: a.n_demos]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(a.out, "w") as dst:
        g = dst.create_group("data")
        for k, v in data.attrs.items():
            g.attrs[k] = v
        g.attrs["rerendered_resolution"] = a.resolution
        total = 0
        for name in names:
            d = data[name]
            states = np.asarray(d["states"])
            agent, wrist = [], []
            for t in range(len(states)):
                obs = env.set_init_state(states[t])
                agent.append(np.asarray(obs["agentview_image"], dtype=np.uint8))
                wrist.append(
                    np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
                )
            gd = g.create_group(name)
            for k, v in d.attrs.items():
                gd.attrs[k] = v
            og = gd.create_group("obs")
            og.create_dataset("agentview_rgb", data=np.stack(agent), compression="gzip")
            og.create_dataset(
                "eye_in_hand_rgb", data=np.stack(wrist), compression="gzip"
            )
            for key in (
                "ee_pos",
                "ee_ori",
                "gripper_states",
                "ee_states",
                "joint_states",
            ):
                if key in d["obs"]:
                    og.create_dataset(key, data=np.asarray(d["obs"][key]))
            for key in ("actions", "dones", "rewards", "states", "robot_states"):
                if key in d:
                    gd.create_dataset(key, data=np.asarray(d[key]))
            total += len(states)
            print(
                f"[rerender] {name}: {len(states)} frames at {a.resolution}x{a.resolution}",
                flush=True,
            )
    env.close()
    print(f"[rerender] wrote {len(names)} demos / {total} frames -> {a.out}")


if __name__ == "__main__":
    main()
