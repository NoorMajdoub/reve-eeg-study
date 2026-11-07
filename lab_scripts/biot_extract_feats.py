"""STEP 1 of the BIOT linear-probing study: extract frozen embeddings (GPU).

BIOT's own BIOTEncoder.forward() already mean-pools over all tokens internally
(model/biot.py: `emb = self.transformer(emb).mean(dim=1)`), and that exact
(B, 256) vector is what ClassificationHead receives -- so unlike REVE, BIOT's
"head" and "mean" pooling are the SAME tensor. We hook model.biot itself for
that pooled vector, and model.biot.transformer for the pre-pooling per-token
sequence (used for "flat"; near-certain to exceed --max_dim and get skipped,
same situation REVE hit on CHB-MIT).

Usage (eeg-gpu env):
  python biot_extract_feats.py --task isruc --arm pretrained
  python biot_extract_feats.py --task isruc --arm random --seed 2
  python biot_extract_feats.py --task chbmit --arm pretrained --fraction 50
"""
import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

BIOT_DIR = os.environ.get("BIOT_DIR", "/home/nourmaj/BIOT")
sys.path.insert(0, BIOT_DIR)

# BIOT's training scripts import pyhealth only for evaluation metrics, which extraction never
# uses. pyhealth drags in heavy, version-pinned dependencies (pandas<2, rdkit, ...), so instead
# of installing it we register a tiny stand-in module. If the real pyhealth is installed, it's used.
try:
    import pyhealth.metrics  # noqa: F401
except Exception:
    import types
    _ph = types.ModuleType("pyhealth"); _phm = types.ModuleType("pyhealth.metrics")
    def _unused(*args, **kwargs):
        raise RuntimeError("pyhealth metrics are not available in the linear-probing extractor")
    _phm.binary_metrics_fn = _phm.multiclass_metrics_fn = _unused
    _ph.metrics = _phm
    sys.modules["pyhealth"] = _ph; sys.modules["pyhealth.metrics"] = _phm
os.chdir(BIOT_DIR)  # prepare_*_dataloader + the relative checkpoint path assume this cwd

SCRATCH = os.environ.get("SCRATCH", "/scratch/nourmaj")
OUT_DIR_DEFAULT = f"{SCRATCH}/biot_lp_feats"
PRETRAIN_CKPT = "pretrained-models/EEG-SHHS+PREST-18-channels.ckpt"  # relative to BIOT_DIR

from biot_production_finetune import DATASET_CONFIG as BINARY_CFG, build_args  # noqa: E402
from biot_production_finetune_multiclass import DATASET_CONFIG as MULTICLASS_CFG  # noqa: E402
from model import BIOTClassifier  # noqa: E402

ALL_TASKS = {**{k: ("binary", v) for k, v in BINARY_CFG.items()},
             **{k: ("multiclass", v) for k, v in MULTICLASS_CFG.items()}}


def build_loader_args(task, pipeline, cfg, seed, batch_size):
    if pipeline == "binary":
        # build_args() constructs the exact Namespace run_binary_supervised.py's
        # prepare_* functions expect. lr is irrelevant here -- no training happens.
        args = build_args(task, "pretrained", lr=1.0, seed=seed)
        args.batch_size = batch_size
        return args
    # multiclass prepare_* functions only read .seed/.sampling_rate/.batch_size/.num_workers
    return argparse.Namespace(seed=seed, sampling_rate=200, batch_size=batch_size, num_workers=8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", required=True, choices=list(ALL_TASKS))
    p.add_argument("--arm", required=True, choices=["pretrained", "random"])
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fraction", type=int, default=100, choices=[100, 50, 25])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_dim", type=int, default=20_000)
    p.add_argument("--out_dir", default=OUT_DIR_DEFAULT)
    a = p.parse_args()

    pipeline, cfg = ALL_TASKS[a.task]
    data_root = cfg["root"] if a.fraction == 100 else f"{cfg['root']}_{a.fraction}pct"
    if not os.path.isdir(data_root):
        raise SystemExit(f"data folder not found: {data_root}")
    os.environ["BIOT_DATA_ROOT_OVERRIDE"] = data_root

    loader_args = build_loader_args(a.task, pipeline, cfg, a.seed, a.batch_size)
    train_loader, test_loader, val_loader = cfg["loader"](loader_args)  # BIOT's own order: (train, test, val)

    # The project's own train loaders use shuffle=True; linear-probing extraction
    # must save rows in FIXED order (same rule reve_extract_feats.py follows), so
    # rebuild every split as a fresh, non-shuffled loader over the same dataset.
    raw_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    loaders = {
        split: DataLoader(dl.dataset, batch_size=a.batch_size, shuffle=False, num_workers=4)
        for split, dl in raw_loaders.items()
    }

    torch.manual_seed(a.seed)  # random arm: this seed decides which random encoder we get
    model = BIOTClassifier(
        n_classes=cfg["n_classes"], n_channels=cfg["in_channels"],
        n_fft=cfg["token_size"], hop_length=cfg["hop_length"],
    )
    if a.arm == "pretrained":
        ckpt = torch.load(PRETRAIN_CKPT, map_location="cpu")
        model_state = model.biot.state_dict()
        skip_keys = [k for k in ckpt if k in model_state and ckpt[k].shape != model_state[k].shape]
        if skip_keys:
            print(f"[extract] skipping mismatched pretrained keys (channel-count dependent): {skip_keys}")
        filtered_ckpt = {k: v for k, v in ckpt.items() if k not in skip_keys}
        model.biot.load_state_dict(filtered_ckpt, strict=False)
        n_loaded = sum(1 for k in filtered_ckpt if k in model_state)
        print(f"[extract] pretrained: loaded {n_loaded}/{len(model_state)} encoder tensors from {PRETRAIN_CKPT}")
        if n_loaded < len(model_state) - 2:
            raise SystemExit("too few pretrained tensors loaded -- checkpoint/model mismatch, stopping")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    store = {}
    model.biot.register_forward_hook(lambda m, inp, out: store.__setitem__("pooled", out.detach()))
    model.biot.transformer.register_forward_hook(lambda m, inp, out: store.__setitem__("tokens", out.detach()))

    print(f"[extract] {a.task} ({pipeline}) | {a.arm} seed {a.seed} | {a.fraction}% | data {data_root}")

    feats, t0 = {}, time.time()
    flat_ok = True
    for split, dl in loaders.items():
        H, M, Fl, Y = [], [], [], []
        with torch.no_grad():
            for X, y in dl:
                X = X.to(device)
                model(X)
                pooled = store["pooled"].float()   # (B, 256) -- both "head" and "mean" for BIOT
                tokens = store["tokens"].float()    # (B, n_channels*ts, 256) -- pre-mean
                B = pooled.shape[0]
                if split == "train" and not M:
                    flat_dim = tokens.shape[1] * tokens.shape[2]
                    print(f"[extract] pooled shape {tuple(pooled.shape)} | token-seq shape {tuple(tokens.shape)} (flat dim {flat_dim:,})")
                    flat_ok = flat_dim <= a.max_dim
                    if not flat_ok:
                        print(f"[extract] flat dim {flat_dim:,} > {a.max_dim:,} -> 'flat' skipped")
                pooled_np = pooled.cpu().numpy()
                H.append(pooled_np)
                M.append(pooled_np)
                if flat_ok:
                    Fl.append(tokens.reshape(B, -1).cpu().numpy().astype(np.float16))
                y_np = y.numpy() if torch.is_tensor(y) else np.array(y)
                Y.append(y_np)
        feats[f"X_{split}_head"] = np.concatenate(H)
        feats[f"X_{split}_mean"] = np.concatenate(M)
        if flat_ok:
            feats[f"X_{split}_flat"] = np.concatenate(Fl)
        feats[f"y_{split}"] = np.concatenate(Y)
        print(f"[extract] {split}: {len(feats[f'y_{split}']):,} windows | mean {feats[f'X_{split}_mean'].shape[1]}"
              + (f" | flat {feats[f'X_{split}_flat'].shape[1]:,}" if flat_ok else ""))

    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"{a.task}_{a.arm}_seed{a.seed}_{a.fraction}pct.npz")
    np.savez(out, **feats, n_classes=cfg["n_classes"])
    print(f"[extract] saved {out} ({os.path.getsize(out)/1e9:.2f} GB) in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()