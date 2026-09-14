"""Export an RLinf FSDP checkpoint (checkpoints/global_step_N/{actor/,}model_state_dict/full_weights.pt) to an openpi-PyTorch
model dir (model.safetensors + config.json + norm stats copied from the base dir) so the standalone evals and the
divergence tool can load it. Usage: export_rlinf_ckpt.py <ckpt_dir> <base_model_dir> <out_dir>"""
import os, shutil, sys, torch, safetensors.torch as st
ckpt, base, out = sys.argv[1:4]
for cand in (os.path.join(ckpt, "actor", "model_state_dict", "full_weights.pt"), os.path.join(ckpt, "model_state_dict", "full_weights.pt")):
    if os.path.exists(cand): path = cand; break
else: raise SystemExit(f"no full_weights.pt under {ckpt}")
sd = torch.load(path, map_location="cpu"); sd = sd.get("model", sd) if isinstance(sd, dict) and "model" in sd and len(sd) < 5 else sd
sd = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in sd.items() if torch.is_tensor(v)}
os.makedirs(out, exist_ok=True); st.save_file(sd, os.path.join(out, "model.safetensors"))
for f in ("config.json",): shutil.copy(os.path.join(base, f), out)
for d in ("assets", "physical-intelligence"):
    if os.path.isdir(os.path.join(base, d)) and not os.path.exists(os.path.join(out, d)): shutil.copytree(os.path.join(base, d), os.path.join(out, d))
print(f"exported {len(sd)} tensors from {path} -> {out}")
