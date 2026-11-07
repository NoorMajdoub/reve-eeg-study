"""Optuna hyperparameter search for REVE fine-tuning, mirroring
biot_optuna_search.py's approach and lessons learned:

  - Non-invasive: imports dt.py's train_one_epoch/test/get_optimizer/
    get_lr_scheduler unmodified where possible. Only train_one_epoch and
    train_stage are re-implemented here (as train_one_epoch_optuna /
    train_stage_optuna) because the original versions (a) unconditionally
    apply grad-norm clipping with no way to disable it, and (b) save a
    checkpoint to disk on every new best epoch, which we don't want during
    a 20-trial search.
  - Builds each trial's config in-process via Hydra's compose() API rather
    than shelling out to dt.py as a subprocess per trial.
  - random arm explicitly passes pretrained_path="" (does NOT rely on the
    checkpoint file simply not existing on disk, unlike the existing
    reve_*_random.sbatch files -- confirmed safe today, but explicit is
    safer going forward).
  - No Lightning here (REVE's dt.py is a hand-written training loop), so
    the sanity-check-contamination bug we hit and fixed in
    biot_optuna_search.py cannot recur -- there is no automatic
    pre-training validation pass in this loop at all.
  - Optimizes val_balanced_acc (what dt.py's own model-selection logic
    already tracks) rather than AUROC, since AUROC is never computed here
    (train_stage always calls test(..., binary=False, ...)) and switching
    that was explicitly not requested.
  - optimizer is left exactly as already configured (StableAdamW, per
    config_dt.yaml's defaults) -- not searched.
  - weight_decay fixed at 1e-5 (matches BIOT), not searched.

Usage:
  python reve_optuna_search.py --dataset clinical --arm pretrained --fraction 100 --n_trials 20
"""
import argparse
import json
import os
import random as py_random
import sys
from contextlib import nullcontext

import idr_torch
import optuna
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score
from torch import nn
from tqdm import tqdm

REVE_SRC = "/lustre10/scratch/nourmaj/REVE/src"
CONFIG_DIR = f"{REVE_SRC}/configs"
sys.path.insert(0, REVE_SRC)
os.chdir(REVE_SRC)  # dt.py's own modules assume this (relative HF cache symlinks etc.)

from configs.resolver import register_resolvers  # noqa: E402
from downstream_tasks.dataloaders import get_data_loaders  # noqa: E402
from models.classifier import ReveClassifier  # noqa: E402
from models.encoder import REVE  # noqa: E402
from utils.model_utils import (  # noqa: E402
    freeze_model,  # noqa: F401 -- kept for parity with dt.py's imports, unused (ft only)
    get_flattened_output_dim,
    load_cls_query_token,
    load_encoder_checkpoint,
    unfreeze_model,
)
from utils.optim import get_lr_scheduler, get_optimizer  # noqa: E402

register_resolvers()

dtype_map = {"fp16": torch.float16, "float16": torch.float16, "bf16": torch.bfloat16, "float32": torch.float32}
FREQ = 200
HF_CACHE_DIR = "/lustre10/scratch/nourmaj/hf_cache"
PRETRAINED_HF_ID = "hf:brain-bzh/reve-base"

# Same underlying processed-data roots as BIOT's already-verified
# DATASET_CONFIG (biot_optuna_search.py) -- confirmed matching for
# clinical via configs/task/clinical.yaml's data_loader.dataset.path.
DATASET_CONFIG = {
    "chbmit": dict(root="/lustre10/scratch/nourmaj/chb-mit-processed-80h"),
    "clinical": dict(root="/lustre10/scratch/nourmaj/ds004504_processed"),
    "dream": dict(root="/lustre10/scratch/nourmaj/dream_processed"),
    "karaone": dict(root="/lustre10/scratch/nourmaj/karaone_processed"),
    "yang2025": dict(root="/lustre10/scratch/nourmaj/yang2025_processed"),
    "japaneeg": dict(root="/lustre10/scratch/nourmaj/japaneeg_processed_80h"),
    "isruc": dict(root="/lustre10/scratch/nourmaj/isruc_processed_80h"),
    "thoughtviz": dict(root="/lustre10/scratch/nourmaj/thoughtviz_processed"),
    "stress_study": dict(root="/lustre10/scratch/nourmaj/stress_reve_processed"),
    "things_eeg2": dict(root="/lustre10/scratch/nourmaj/things_eeg2_reve_processed"),
    "bci2020": dict(root="/lustre10/scratch/nourmaj/bci2020_reve_processed"),
    "sleepedf": dict(root="/lustre10/scratch/nourmaj/sleepedf_reve_processed"),
    "caueeg": dict(root="/lustre10/scratch/nourmaj/caueeg_reve_processed"),
}

# lr ranges per arm, per the spec given for this search.
LR_RANGE = {
    "random": (1e-4, 1e-3),
    "pretrained": (1e-5, 2.5e-4),
}
GRAD_CLIP_CHOICES = [None, 0.5, 1.0, 2.0, 5.0]
FIXED_WEIGHT_DECAY = 1e-5

# Per-dataset fine_tuning.n_epochs override -- only set here when we
# deliberately want a different epoch budget than the task yaml's own
# default. japaneeg.yaml defaults to n_epochs=25 (vs. 50 for every other
# 8-dataset-study task); re-running at 50 for consistency with the rest of
# the study, per explicit decision on 2026-09-13. IMPORTANT: this changes
# what's actually being tuned, so any existing study .db/*.json for an
# affected dataset from a prior epoch budget is stale and must not be
# reused (rename/move it aside, don't let load_if_exists=True silently mix
# 25-epoch and 50-epoch trials into the same "n_trials" count).
N_EPOCHS_OVERRIDE = {
    "japaneeg": 50,
    "yang2025": 50,
}


def build_model(cfg: DictConfig, pretrained_path: str):
    """Same construction logic as dt.py's main() (lines ~326-395), extracted
    so it can be called once per trial without going through @hydra.main."""
    cls_query_token = None
    if pretrained_path and "hf:" in pretrained_path:
        model_id = pretrained_path.split("hf:")[-1]
        print(f"Loading encoder from Hugging Face: {model_id}")
        encoder, cls_query_token = REVE.from_pretrained(model_id, cache_dir=cfg.get("cache_dir", ".cache"))
    else:
        from types import SimpleNamespace

        backbone_args = SimpleNamespace(
            embed_dim=cfg.encoder.transformer.embed_dim,
            depth=cfg.encoder.transformer.depth,
            heads=cfg.encoder.transformer.heads,
            head_dim=cfg.encoder.transformer.head_dim,
            mlp_dim_ratio=cfg.encoder.transformer.mlp_dim_ratio,
            use_geglu=cfg.encoder.transformer.use_geglu,
        )
        encoder = REVE(
            args_backbone=backbone_args,
            freqs=cfg.encoder.freqs,
            patch_size=cfg.encoder.patch_size,
            overlap_size=cfg.encoder.patch_overlap,
            noise_ratio=cfg.encoder.noise_ratio,
        )
        if pretrained_path and os.path.exists(pretrained_path):
            load_encoder_checkpoint(encoder, pretrained_path)
        # else: genuinely random init -- expected for the random arm.

    n_chans = cfg.task.n_chans
    n_timepoints = int(cfg.task.duration * FREQ)
    out_shape = None
    if cfg.task.classifier.pooling == "no":
        out_shape = get_flattened_output_dim(cfg, n_timepoints, n_chans)

    model = ReveClassifier(
        encoder=encoder,
        n_classes=cfg.task.classifier.n_classes,
        dropout=cfg.get("dropout", 0.0),
        pooling=cfg.task.classifier.pooling,
        out_shape=out_shape,
    )

    if pretrained_path and "hf:" in pretrained_path:
        if cls_query_token is not None:
            model.cls_query_token.data.copy_(cls_query_token)
    elif pretrained_path and os.path.exists(pretrained_path):
        load_cls_query_token(model, pretrained_path)

    model.to(cfg.trainer.device)
    return model


def train_one_epoch_optuna(model, criterion, optimizer, scaler, train_loader, warmup_scheduler, config, current_cfg):
    """Same as dt.py's train_one_epoch, except grad clipping is skipped
    entirely when config.trainer.clip_grad is None (the original always
    calls clip_grad_norm_, with no way to opt out)."""
    model.train()
    use_mixup = current_cfg.get("mixup", False)
    device = config.trainer.device
    dtype = config.trainer.get("torch_dtype", "fp32")
    torch_dtype = dtype_map.get(dtype, torch.float32)
    ctx = torch.amp.autocast(device_type="cuda", dtype=torch_dtype) if "16" in dtype else nullcontext()

    losses = []
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), disable=not idr_torch.is_master)
    for _, batch_data in pbar:
        optimizer.zero_grad(set_to_none=True)
        with ctx:
            if isinstance(batch_data, dict):
                data, target, pos = batch_data["sample"], batch_data["label"], batch_data["pos"]
            else:
                data, target, pos = batch_data
            data = data.to(device, non_blocking=True)
            target = target.long().to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)

            if use_mixup:
                mm = py_random.random()
                perm = torch.randperm(data.shape[0])
                output = model(mm * data + (1 - mm) * data[perm], pos)
                loss = mm * criterion(output, target) + (1 - mm) * criterion(output, target[perm])
            else:
                output = model(data, pos)
                loss = criterion(output, target)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if config.trainer.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.trainer.clip_grad)
        scaler.step(optimizer)
        scale = scaler.get_scale()
        scaler.update()

        losses.append(loss.item())
        skip_lr_sched = scale != scaler.get_scale()
        if not skip_lr_sched and warmup_scheduler is not None:
            warmup_scheduler.step()

    return sum(losses) / len(losses)


def test_optuna(model, loader, device="cuda", amp_dtype=torch.float16):
    """Same as dt.py's test(binary=False) -- only balanced_acc is used as
    the objective, matching train_stage's own selection criterion."""
    model.eval()
    score, count = 0, 0
    y_decisions, y_targets = [], []
    ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if device == "cuda" else nullcontext()
    with torch.no_grad(), ctx:
        for batch_data in loader:
            if isinstance(batch_data, dict):
                data, target, pos = batch_data["sample"], batch_data["label"], batch_data["pos"]
            else:
                data, target, pos = batch_data
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            pos = pos.to(device, non_blocking=True)
            output = model(data, pos)
            decisions = torch.argmax(output, dim=1)
            score += (decisions == target).int().sum().item()
            count += target.shape[0]
            y_decisions.append(decisions.cpu())
            y_targets.append(target.cpu())

    gt = torch.cat(y_targets).numpy()
    pr = torch.cat(y_decisions).numpy()
    acc = score / count
    balanced_acc = balanced_accuracy_score(gt, pr)
    cohen_kappa = cohen_kappa_score(gt, pr)
    f1 = f1_score(gt, pr, average="weighted")
    return acc, balanced_acc, cohen_kappa, f1


def train_stage_optuna(trial, config, current_cfg, model, train_loader, val_loader):
    scaler = torch.amp.GradScaler(
        device=config.trainer.device,
        enabled="cuda" in config.trainer.device and "16" in config.get("torch_dtype", "fp32"),
    )
    device = config.trainer.device
    if config.task.name == "chbmit":
        class_weights = torch.tensor([0.503, 75.76], dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    dtype_str = config.trainer.get("torch_dtype", "fp32")
    torch_dtype = dtype_map.get(dtype_str)

    optimizer = get_optimizer(filter(lambda p: p.requires_grad, model.parameters()), current_cfg.optimizer)
    n_iter_per_epoch = len(train_loader)
    scheduler = get_lr_scheduler(optimizer, current_cfg, n_iter_per_epoch)

    warmup_epochs = current_cfg.warmup_epochs
    if warmup_epochs > 0:
        total_steps = len(train_loader) * warmup_epochs

        def exponential_warmup_lambda(step):
            if step < total_steps:
                return (10 ** (step / total_steps) - 1) / 9
            return 1.0

        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=exponential_warmup_lambda)
    else:
        warmup_scheduler = None

    model.train()
    best_val = 0.0
    patience = 0
    n_epochs = current_cfg.n_epochs
    patience_limit = current_cfg.patience

    for epoch in range(n_epochs):
        warmup = epoch < warmup_epochs
        train_one_epoch_optuna(
            model, criterion, optimizer, scaler, train_loader,
            warmup_scheduler if warmup else None, config, current_cfg,
        )

        val_acc, val_balanced_acc, _, _ = test_optuna(model, val_loader, device=device, amp_dtype=torch_dtype)

        if val_balanced_acc > best_val:
            best_val = val_balanced_acc
            patience = 0
        else:
            patience += 1

        trial.report(val_balanced_acc, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

        if epoch > warmup_epochs:
            scheduler.step(val_acc)
        if patience > patience_limit:
            break

    return best_val


def objective(trial, dataset_key, arm, fraction):
    cfg_ds = DATASET_CONFIG[dataset_key]
    lo, hi = LR_RANGE[arm]
    lr = trial.suggest_float("lr", lo, hi, log=True)
    grad_clip = trial.suggest_categorical("grad_clip", GRAD_CLIP_CHOICES)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.15)

    data_root = cfg_ds["root"] if fraction == "100" else f"{cfg_ds['root']}_{fraction}pct"
    pretrained_path = PRETRAINED_HF_ID if arm == "pretrained" else ""

    overrides = [
        f"task={dataset_key}",
        "training_mode=ft",
        f"task.data_loader.dataset.path={data_root}",
        f"task.fine_tuning.optimizer.lr={lr}",
        f"task.fine_tuning.optimizer.weight_decay={FIXED_WEIGHT_DECAY}",
        f"pretrained_path={pretrained_path}",
        f"cache_dir={HF_CACHE_DIR}",
        "seed=42",
        f"trainer.clip_grad={'null' if grad_clip is None else grad_clip}",
    ]
    if dataset_key in N_EPOCHS_OVERRIDE:
        overrides.append(f"task.fine_tuning.n_epochs={N_EPOCHS_OVERRIDE[dataset_key]}")

    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(config_name="config_dt", overrides=overrides)

    n_epochs = cfg.task.fine_tuning.n_epochs
    warmup_epochs = round(warmup_ratio * n_epochs)
    OmegaConf.set_struct(cfg, False)
    cfg.task.fine_tuning.warmup_epochs = warmup_epochs

    torch.manual_seed(cfg.seed)
    py_random.seed(cfg.seed)

    model = build_model(cfg, pretrained_path)
    unfreeze_model(model)  # ft mode -- full model trainable, matching dt.py

    data_loaders = get_data_loaders(cfg.task.data_loader, cfg.loader)
    current_cfg = cfg.task.fine_tuning

    best_val = train_stage_optuna(trial, cfg, current_cfg, model, data_loaders["train"], data_loaders["val"])

    del model
    torch.cuda.empty_cache()
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASET_CONFIG))
    parser.add_argument("--arm", required=True, choices=["pretrained", "random"])
    parser.add_argument("--fraction", default="100", choices=["100", "50", "25"])
    parser.add_argument("--n_trials", type=int, default=20)
    cli_args = parser.parse_args()

    out_dir = "/lustre10/scratch/nourmaj/reve_optuna_results"
    os.makedirs(out_dir, exist_ok=True)
    tag = f"{cli_args.dataset}_{cli_args.arm}_{cli_args.fraction}pct"
    db_path = f"{out_dir}/{tag}.db"
    study_name = f"reve_{tag}"

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=3)
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    n_done = len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)])
    n_remaining = max(0, cli_args.n_trials - n_done)
    print(f"=== REVE Optuna search: dataset={cli_args.dataset} arm={cli_args.arm} fraction={cli_args.fraction} ===")
    print(f"{n_done} trials already recorded in {db_path}, running {n_remaining} more")

    if n_remaining > 0:
        study.optimize(
            lambda trial: objective(trial, cli_args.dataset, cli_args.arm, cli_args.fraction),
            n_trials=n_remaining,
        )

    print("=== BEST TRIAL ===")
    print(f"dataset={cli_args.dataset} arm={cli_args.arm} fraction={cli_args.fraction}")
    print(f"best val_balanced_acc = {study.best_value}")
    print(f"best params = {study.best_params}")

    with open(f"{out_dir}/{tag}_best.json", "w") as f:
        json.dump(
            {
                "dataset": cli_args.dataset,
                "arm": cli_args.arm,
                "fraction": cli_args.fraction,
                "metric": "val_balanced_acc",
                "best_val_balanced_acc": study.best_value,
                "best_params": study.best_params,
                "weight_decay": FIXED_WEIGHT_DECAY,
                "optimizer": "as configured in config_dt.yaml (stable_adamw)",
                "n_trials_run": len(study.trials),
            },
            f,
            indent=2,
        )
    print(f"Saved: {out_dir}/{tag}_best.json")


if __name__ == "__main__":
    main()