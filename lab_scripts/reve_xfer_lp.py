"""REVE cross-dataset transfer: frozen backbone + new linear head on a target dataset.

--init source      backbone = encoder (+ cls_query_token) of a model fine-tuned on a SOURCE
                   dataset (its model_best.pth); source classification head discarded
--init pretrained  backbone = original Hugging Face REVE checkpoint (reference)
--init random      backbone = random initialisation (floor)

In all three cases the encoder is frozen (REVE's own freeze_model, as in dt.py's
linear-probing mode) and only a fresh head for the TARGET task is trained, using the
target task yaml's `linear_probing` settings. The training loop mirrors dt.py's
train_stage (same warmup, scheduler, patience, val-selected test metric).

Result: one JSON per run in --out_dir with best val / test balanced accuracy.

Usage (REVE env):
  python reve_xfer_lp.py --target isruc --init source --source_name clinical \
      --source_ckpt /lustre10/scratch/nourmaj/reve_outputs/clinical_pretrained/model_best.pth --seed 1
  python reve_xfer_lp.py --target isruc --init pretrained --seed 1
"""

import argparse
import json
import os
import random as py_random
import sys
import time

import torch
from torch import nn


def run_lp(cfg, cur, model, loaders, R, get_optimizer, get_lr_scheduler):
    device = cfg.trainer.device
    scaler = torch.amp.GradScaler(device=device,
                                  enabled="cuda" in device and "16" in cfg.get("torch_dtype", "fp32"))
    criterion = nn.CrossEntropyLoss()
    torch_dtype = R.dtype_map.get(cfg.trainer.get("torch_dtype", "fp32"))
    train_loader, val_loader, test_loader = loaders["train"], loaders["val"], loaders["test"]

    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), cur.optimizer)
    scheduler = get_lr_scheduler(optimizer, cur, len(train_loader))
    warmup_epochs = cur.warmup_epochs
    warmup_scheduler = None
    if warmup_epochs > 0:
        total = len(train_loader) * warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda s: (10 ** (s / total) - 1) / 9 if s < total else 1.0)

    best_val, best_test, best_epoch, patience, epoch = 0.0, 0.0, -1, 0, -1
    for epoch in range(cur.n_epochs):
        warm = epoch < warmup_epochs
        loss = R.train_one_epoch_optuna(model, criterion, optimizer, scaler, train_loader,
                                        warmup_scheduler if warm else None, cfg, cur)
        val = R.test_optuna(model, val_loader, device=device, amp_dtype=torch_dtype)
        if val[1] > best_val:
            best_val, best_epoch, patience = val[1], epoch, 0
            best_test = R.test_optuna(model, test_loader, device=device, amp_dtype=torch_dtype)[1]
        else:
            patience += 1
        print(f"Stage lp | epoch {epoch} val {val[1]:.4f} | best val {best_val:.4f} "
              f"best test {best_test:.4f} | train_loss {loss:.4f}", flush=True)
        if epoch > warmup_epochs:
            scheduler.step(val[0])
        if patience > cur.patience:
            break
    return best_val, best_test, best_epoch, epoch + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--init", required=True, choices=["source", "pretrained", "random"])
    p.add_argument("--source_name", default="none")
    p.add_argument("--source_ckpt", default=None)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out_dir", default="/lustre10/scratch/nourmaj/reve_xfer_results")
    a = p.parse_args()
    if a.init == "source" and not (a.source_ckpt and os.path.exists(a.source_ckpt)):
        raise SystemExit(f"--init source needs an existing --source_ckpt (got {a.source_ckpt})")

    sys.argv = [sys.argv[0]]
    sys.path.insert(0, "/lustre10/scratch/nourmaj")
    import reve_optuna_search as R  # noqa: E402  (sets cwd + sys.path to REVE/src)
    from hydra import compose, initialize_config_dir  # noqa: E402
    from downstream_tasks.dataloaders import get_data_loaders  # noqa: E402
    from utils.model_utils import freeze_model  # noqa: E402
    from utils.optim import get_lr_scheduler, get_optimizer  # noqa: E402

    root = R.DATASET_CONFIG[a.target]["root"]
    pp = R.PRETRAINED_HF_ID if a.init in ("source", "pretrained") else ""
    overrides = [f"task={a.target}", "training_mode=lp", f"task.data_loader.dataset.path={root}",
                 f"pretrained_path={pp}", f"cache_dir={R.HF_CACHE_DIR}", f"seed={a.seed}"]
    with initialize_config_dir(config_dir=R.CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_dt", overrides=overrides)

    torch.manual_seed(a.seed)
    py_random.seed(a.seed)
    model = R.build_model(cfg, pp)  # target head; encoder = HF weights or random

    if a.init == "source":
        sd = torch.load(a.source_ckpt, map_location="cpu")
        enc = {k[len("encoder."):]: v for k, v in sd.items() if k.startswith("encoder.")}
        missing, unexpected = model.encoder.load_state_dict(enc, strict=False)
        if missing or unexpected:
            raise SystemExit(f"encoder mismatch: missing {missing[:5]} unexpected {unexpected[:5]}")
        if "cls_query_token" in sd:
            model.cls_query_token.data.copy_(sd["cls_query_token"])
        print(f"[xfer] loaded source encoder from {a.source_ckpt} ({len(enc)} tensors)")

    freeze_model(model)
    n_train = sum(t.numel() for t in model.parameters() if t.requires_grad)
    n_all = sum(t.numel() for t in model.parameters())
    print(f"[xfer] {a.source_name if a.init == 'source' else a.init} -> {a.target} | seed {a.seed} | "
          f"trainable {n_train:,} / {n_all:,} params")

    cur = cfg.task.linear_probing
    loaders = get_data_loaders(cfg.task.data_loader, cfg.loader)
    t0 = time.time()
    best_val, best_test, best_epoch, epochs = run_lp(cfg, cur, model, loaders, R, get_optimizer, get_lr_scheduler)

    os.makedirs(a.out_dir, exist_ok=True)
    src = a.source_name if a.init == "source" else a.init
    out = {"source": src, "init": a.init, "target": a.target, "seed": a.seed,
           "best_val_balanced_acc": best_val, "best_test_balanced_acc": best_test,
           "best_epoch": best_epoch, "epochs_run": epochs, "trainable_params": n_train,
           "source_ckpt": a.source_ckpt if a.init == "source" else None,
           "lp_settings": {"lr": float(cur.optimizer.lr), "n_epochs": int(cur.n_epochs),
                           "patience": int(cur.patience)},
           "minutes": round((time.time() - t0) / 60, 1)}
    path = os.path.join(a.out_dir, f"{src}__{a.target}__seed{a.seed}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[xfer] RESULT {src} -> {a.target} seed {a.seed}: val {best_val:.4f} test {best_test:.4f} -> {path}")


if __name__ == "__main__":
    main()
