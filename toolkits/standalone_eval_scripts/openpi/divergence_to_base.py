"""KL-to-base proxy: deterministic (ODE) actions of a checkpoint vs the base on recorded observations.
Loads both policies with the standalone toolkit's create_trained_policy; iterates every k-th query of the given rollout
episodes (steps.npz from openpi's client: observation/image, wrist_image, state, prompt); reports mean/max |dA| overall,
per action dim, and split by freeze-window queries (eef displacement < 3 mm per 10 steps sustained >= 30 steps)."""
import argparse, glob, json, os, pathlib
os.environ.setdefault("MUJOCO_GL", "osmesa")
import numpy as np
from toolkits.standalone_eval_scripts.openpi import setup_policy


def stall_mask(z, thr=0.003, min_len=30):
    pos = z["observation__state"][:, :3]; disp = np.linalg.norm(pos[10:] - pos[:-10], axis=1); disp = np.concatenate([disp, np.full(10, disp[-1])]); still = disp < thr
    m = np.zeros(len(pos), bool); i = 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]: j += 1
            if j - i >= min_len: m[i:j] = True
            i = j
        else: i += 1
    return m


class A: pass


def main():
    p = argparse.ArgumentParser(); p.add_argument("--base", required=True); p.add_argument("--ckpt", required=True); p.add_argument("--config_name", default="pi05_libero")
    p.add_argument("--episodes", default="/rollouts/libero_10/t08_put_both_moka_pots_on_the_stove/ep*"); p.add_argument("--every", type=int, default=5); p.add_argument("--num_steps", type=int, default=10); p.add_argument("--out", required=True)
    a = p.parse_args(); pols = {}
    for name, path in (("base", a.base), ("ckpt", a.ckpt)):
        args = A(); args.config_name = a.config_name; args.pretrained_path = path; args.num_steps = a.num_steps; pols[name] = setup_policy(args)
    diffs, stall, dims = [], [], []
    for d in sorted(glob.glob(a.episodes)):
        z = np.load(pathlib.Path(d) / "steps.npz"); meta = json.load(open(pathlib.Path(d) / "meta.json")); sm = stall_mask(z)
        for i in range(0, len(z["t"]), a.every):
            obs = {"observation/image": z["observation__image"][i], "observation/wrist_image": z["observation__wrist_image"][i], "observation/state": z["observation__state"][i], "prompt": meta["task"]}
            ab = np.asarray(pols["base"].infer(obs)["actions"]); ac = np.asarray(pols["ckpt"].infer(obs)["actions"])
            dd = np.abs(ab - ac); diffs.append(dd.mean()); stall.append(bool(sm[i])); dims.append(dd.mean(0))
    diffs = np.array(diffs); stall = np.array(stall); dims = np.stack(dims)
    rep = dict(base=a.base, ckpt=a.ckpt, n_queries=int(len(diffs)), mean_abs_diff=float(diffs.mean()), max_abs_diff=float(diffs.max()), per_dim_mean=[float(x) for x in dims.mean(0)],
               mean_abs_diff_in_stall=float(diffs[stall].mean()) if stall.any() else None, mean_abs_diff_moving=float(diffs[~stall].mean()) if (~stall).any() else None)
    pathlib.Path(a.out).write_text(json.dumps(rep, indent=2)); print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
