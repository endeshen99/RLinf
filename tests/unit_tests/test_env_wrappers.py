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

"""Tests for the env wrappers under ``rlinf.envs.wrappers``."""

import glob
import os
import pickle

import numpy as np
import torch

from rlinf.envs.wrappers.collect_episode import CollectEpisode


class _StickyDoneEnv:
    """Vectorized fake env without auto-reset.

    Env 0 succeeds at ``done_step`` and, like ``LiberoEnv`` with
    ``auto_reset: False``, keeps reporting ``terminated`` on every later step
    until ``reset`` is called. Env 1 never terminates.
    """

    def __init__(self, done_step: int = 3):
        self.num_envs = 2
        self.done_step = done_step
        self.t = 0

    def _obs(self):
        return {
            "main_images": torch.zeros(self.num_envs, 4, 4, 3, dtype=torch.uint8),
            "states": torch.full((self.num_envs, 2), float(self.t)),
            "task_descriptions": ["task"] * self.num_envs,
        }

    def reset(self, **kwargs):
        self.t = 0
        return self._obs(), {}

    def chunk_step(self, chunk_actions):
        obs_list, infos_list, rewards, terms, truncs = [], [], [], [], []
        for i in range(chunk_actions.shape[1]):
            self.t += 1
            term = torch.tensor([self.t >= self.done_step, False])
            obs_list.append(self._obs())
            infos_list.append({"episode": {"success_once": term.clone()}})
            rewards.append(term.float())
            terms.append(term)
            truncs.append(torch.zeros(self.num_envs, dtype=torch.bool))
        return (
            obs_list,
            torch.stack(rewards, 1),
            torch.stack(terms, 1),
            torch.stack(truncs, 1),
            infos_list,
        )


def test_collect_episode_sticky_done_without_auto_reset(tmp_path):
    env = CollectEpisode(
        _StickyDoneEnv(done_step=3), save_dir=str(tmp_path), num_envs=2
    )
    env.reset()
    actions = torch.zeros(2, 5, 1)
    env.chunk_step(actions)  # env 0 terminates at sub-step 3 and stays terminated
    env.chunk_step(actions)  # must not flush again nor crash on the unseeded buffer
    env.reset()
    env.chunk_step(actions)  # a new episode after reset is recorded normally
    env.close()

    files = sorted(glob.glob(os.path.join(str(tmp_path), "*.pkl")))
    assert len(files) == 2, files
    lengths = []
    for f in files:
        with open(f, "rb") as fh:
            ep = pickle.load(fh)
        assert ep["env_idx"] == 0 and ep["success"] is True
        lengths.append(len(ep["actions"]))
    assert lengths == [3, 3]
    np.testing.assert_array_equal([len(b["actions"]) for b in env._buffers], [0, 5])
