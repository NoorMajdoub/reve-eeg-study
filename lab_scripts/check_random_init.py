"""Sanity check: does the REVE Optuna random arm really start from random weights,
with exactly the same architecture as the pretrained arm?

Builds both models through reve_optuna_search.build_model (the exact code path
the search uses), on CPU, and compares their state dicts.

Usage (REVE env):
  python check_random_init.py caueeg
"""

import sys

import torch

task = sys.argv[1] if len(sys.argv) > 1 else "caueeg"
sys.argv = [sys.argv[0]]
sys.path.insert(0, "/lustre10/scratch/nourmaj")
import reve_optuna_search as R  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402


def make(arm):
    pp = R.PRETRAINED_HF_ID if arm == "pretrained" else ""
    with initialize_config_dir(config_dir=R.CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_dt", overrides=[
            f"task={task}", "training_mode=ft", f"pretrained_path={pp}",
            f"cache_dir={R.HF_CACHE_DIR}", "trainer.device=cpu",
        ])
    torch.manual_seed(42)
    return R.build_model(cfg, pp).state_dict()


pre = make("pretrained")
rnd = make("random")

same_keys = list(pre) == list(rnd)
same_shapes = same_keys and all(pre[k].shape == rnd[k].shape for k in pre)
n_pre = sum(v.numel() for v in pre.values())
n_rnd = sum(v.numel() for v in rnd.values())
print(f"[arch] same parameter names: {same_keys}, same shapes: {same_shapes}, "
      f"total params pretrained={n_pre:,} random={n_rnd:,}")
if not same_keys:
    print("  only in pretrained:", sorted(set(pre) - set(rnd))[:10])
    print("  only in random:   ", sorted(set(rnd) - set(pre))[:10])

enc = [k for k in pre if (k.startswith("encoder.") or "cls_query_token" in k)
       and k in rnd and pre[k].shape == rnd[k].shape and pre[k].is_floating_point()]
informative = [k for k in enc if pre[k].numel() > 1 and pre[k].float().std() > 0]
identical = [k for k in informative if torch.allclose(pre[k].float(), rnd[k].float())]
print(f"[weights] encoder tensors compared: {len(informative)} "
      f"(constant ones like LayerNorm=1/bias=0 skipped)")
print(f"[weights] identical between random and pretrained: {len(identical)}  <- should be 0")
if identical:
    print("  identical:", identical[:10])
cq = [k for k in pre if "cls_query_token" in k]
for k in cq:
    print(f"[cls_query_token] {k} identical: {torch.allclose(pre[k].float(), rnd[k].float())}  <- should be False")
ex = informative[len(informative) // 2]
print(f"[example] {ex}: pretrained std={pre[ex].float().std():.4f} random std={rnd[ex].float().std():.4f}")
