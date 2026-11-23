"""
Real-world one-epoch benchmark: conservative_cgrad vs standard LIF on CIFAR10-DVS.

Runs one full training epoch (forward + backward + optimizer step) for each
configuration using the actual MaxFormer model, data pipeline, and AMP,
then prints a timing comparison.

Configurations
--------------
  1. Baseline (sigmoid + LIF):  dS_du=sigmoid, du_du=LIF, detach_reset=False
  2. Baseline (Gamma  + LIF):  dS_du=Gamma,   du_du=LIF, detach_reset=False
  3. C-Grad  (sigmoid + cons): dS_du=sigmoid, du_du=conservative_cgrad, detach_reset=False
  4. C-Grad  (Gamma  + cons):  dS_du=Gamma,   du_du=conservative_cgrad, detach_reset=False

Usage
-----
    conda run -n snn-maxformer python benchmark_epoch.py
    conda run -n snn-maxformer python benchmark_epoch.py --data-path /data/rboone/datasets/wg_dvst
    conda run -n snn-maxformer python benchmark_epoch.py --device cuda:1 --batch-size 8
"""

import argparse
import copy
import datetime
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from torch.cuda import amp

# ---------------------------------------------------------------------------
# Ensure local imports work
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import custom_neuron
import max_former  # registers the timm model
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.models import create_model
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
import autoaugment
from torchvision import transforms


# ---------------------------------------------------------------------------
# train.py utilities (re-used for split)
# ---------------------------------------------------------------------------
def split_to_train_test_set(train_ratio, origin_dataset, num_classes):
    label_idx = [[] for _ in range(num_classes)]
    for i, item in enumerate(origin_dataset):
        y = item[1]
        if isinstance(y, (np.ndarray, torch.Tensor)):
            y = y.item()
        label_idx[y].append(i)
    train_idx = []
    for i in range(num_classes):
        pos = math.ceil(len(label_idx[i]) * train_ratio)
        train_idx.extend(label_idx[i][:pos])
    return torch.utils.data.Subset(origin_dataset, train_idx)


# ---------------------------------------------------------------------------
# One-epoch timing harness
# ---------------------------------------------------------------------------
def timed_train_one_epoch(model, criterion, optimizer, data_loader, device, scaler,
                          mixup_fn, aug, trival_aug, print_every=50, max_batches=0):
    """Train for exactly one epoch (or max_batches), returning (total_seconds, num_batches, avg_loss)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    # --- Warm-up: 2 steps with synthetic batch to prime CUDA caches and JIT without dangling worker queues ---
    dummy_x = torch.randn(data_loader.batch_size, 2, 128, 128, device=device)
    dummy_y = torch.zeros(data_loader.batch_size, dtype=torch.long, device=device)
    for _ in range(2):
        optimizer.zero_grad()
        with amp.autocast():
            output = model(dummy_x)
            loss = criterion(output, dummy_y)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        functional.reset_net(model)
    del dummy_x, dummy_y

    # Sync before starting the clock
    if device.type == "cuda":
        torch.cuda.synchronize()

    t_start = time.perf_counter()

    for batch_idx, (image, target) in enumerate(data_loader):
        if max_batches > 0 and num_batches >= max_batches:
            break

        image, target = image.to(device).float(), target.to(device)
        N = image.shape[0]

        if aug is not None:
            image = torch.stack([aug(image[i]) for i in range(N)])
        if trival_aug is not None:
            image = torch.stack([trival_aug(image[i]) for i in range(N)])
        if mixup_fn is not None:
            image, target = mixup_fn(image, target)

        with amp.autocast():
            output = model(image)
            loss = criterion(output, target)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        functional.reset_net(model)

        total_loss += loss.item()
        num_batches += 1

        if (batch_idx + 1) % print_every == 0:
            elapsed = time.perf_counter() - t_start
            print(f"      batch {batch_idx+1:4d}  |  elapsed {elapsed:7.1f}s  |  "
                  f"avg loss {total_loss/num_batches:.4f}")

    if device.type == "cuda":
        torch.cuda.synchronize()

    t_end = time.perf_counter()
    return t_end - t_start, num_batches, total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Build the args namespace that custom_neuron and train.py helpers expect
# ---------------------------------------------------------------------------
def build_args(cli, dS_du, du_du, detach_reset):
    """Return a full args namespace compatible with train.py utilities."""
    return argparse.Namespace(
        # Model / data
        model="max_former",
        dataset="cifar10dvs",
        num_classes=10,
        data_path=cli.data_path,
        device=cli.device,
        batch_size=cli.batch_size,
        workers=cli.workers,
        T=cli.T,
        T_train=None,
        dim=cli.dim,
        distributed=False,
        sync_bn=False,
        amp=cli.amp,
        # Optimizer
        opt="adamw",
        opt_eps=1e-8,
        opt_betas=None,
        weight_decay=0.06,
        momentum=0.9,
        lr=1e-3,
        # Scheduler (only need 1 epoch but timm needs these)
        sched="cosine",
        lr_noise=None,
        lr_noise_pct=0.67,
        lr_noise_std=1.0,
        lr_cycle_mul=1.0,
        lr_cycle_limit=1,
        warmup_lr=1e-5,
        min_lr=2e-5,
        epochs=1,
        epoch_repeats=0.0,
        start_epoch=0,
        decay_epochs=20,
        warmup_epochs=0,
        cooldown_epochs=0,
        patience_epochs=10,
        decay_rate=0.1,
        # Augmentation
        smoothing=0.1,
        mixup=0.5,
        cutmix=0.0,
        cutmix_minmax=None,
        mixup_prob=0.5,
        mixup_switch_prob=0.5,
        mixup_mode="batch",
        mixup_off_epoch=0,
        # Custom neuron / SNNBP
        dS_du=dS_du,
        du_du=du_du,
        detach_reset=detach_reset,
        use_custom_neuron=True,
        snnbp_alpha=2.0,
        snnbp_beta=2.0,
        snnbp_epsilon=0.3,
        snnbp_p=9.5334,
        snnbp_k_dir=1.0,
        snnbp_tau=0.5,
        snnbp_max_ratio=3.0,
        snnbp_decay=0.5,
        snnbp_intervention=0.8,
        surrogate_alpha=4.0,
        gama=1.0,
        # Misc
        seed=42,
        log_wandb=False,
        experiment="",
        output_dir="./logs",
        resume="",
        print_freq=256,
        test_only=False,
        early_stop_patience=-1,
        # Ablation defaults
        ablation_gm=True,
        ablation_gd=True,
        ablation_gmisalign=True,
        ablation_intervention=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-world one-epoch timing benchmark.")
    parser.add_argument("--data-path", default="/data/rboone/datasets/wg_dvst",
                        help="Path to CIFAR10-DVS frames (spikingjelly format)")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--T", type=int, default=16, help="Simulation timesteps")
    parser.add_argument("--dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument("--amp", action="store_true", default=True, help="Use AMP")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-batches", type=int, default=100, help="Max batches to benchmark per config (0 for full epoch)")
    parser.add_argument("--print-every", type=int, default=25, help="Print progress every N batches")
    cli = parser.parse_args()

    device = torch.device(cli.device)

    # ---- Configurations to compare ----
    # (Label, dS_du, du_du, detach_reset, disable_triton)
    configs = [
        ("PyTorch Baseline: Gamma + LIF",           "Gamma", "LIF",               False, "1"),
        ("PyTorch C-Grad:   Gamma + conservative",  "Gamma", "conservative_cgrad", False, "1"),
        ("PyTorch C-Grad:   Gamma + invariant",     "Gamma", "invariant_cgrad",    False, "1"),
        ("Triton Baseline:  Gamma + LIF",           "Gamma", "LIF",               False, "0"),
        ("Triton C-Grad:    Gamma + conservative",  "Gamma", "conservative_cgrad", False, "0"),
        ("Triton C-Grad:    Gamma + invariant",     "Gamma", "invariant_cgrad",    False, "0"),
    ]

    # ---- Load dataset ONCE (expensive, don't repeat) ----
    print("Loading CIFAR10-DVS dataset...")
    t_data = time.perf_counter()
    origin_set = cifar10_dvs.CIFAR10DVS(
        root=cli.data_path, data_type="frame", frames_number=cli.T, split_by="number"
    )
    dataset_train = split_to_train_test_set(0.9, origin_set, 10)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=cli.batch_size,
        shuffle=True,
        num_workers=cli.workers,
        drop_last=True,
        pin_memory=True,
    )
    print(f"Dataset loaded in {time.perf_counter() - t_data:.1f}s  "
          f"({len(dataset_train)} train samples, "
          f"{len(data_loader)} batches @ bs={cli.batch_size})\n")

    # ---- Augmentation (matches train.py) ----
    train_snn_aug = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5)])
    train_trivalaug = autoaugment.SNNAugmentWide()

    # ---- Run each configuration ----
    results = {}

    print(f"{'=' * 72}")
    print(f" One-Epoch Training Benchmark — CIFAR10-DVS")
    print(f" device={device}  T={cli.T}  batch_size={cli.batch_size}  dim={cli.dim}  AMP={cli.amp}")
    print(f"{'=' * 72}\n")

    for label, dS_du, du_du, detach_reset, disable_triton in configs:
        print(f"─── {label} ───")
        os.environ['DISABLE_TRITON'] = disable_triton

        # Build fresh args, set global, build fresh model + optimizer
        args = build_args(cli, dS_du, du_du, detach_reset)
        custom_neuron.set_global_args(args)

        # Deterministic init
        torch.manual_seed(cli.seed)
        torch.cuda.manual_seed_all(cli.seed)
        np.random.seed(cli.seed)

        model = create_model(
            "max_former", in_channels=2, num_classes=10,
            embed_dims=cli.dim, mlp_ratios=1.0, depths=2, T=cli.T,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        criterion = SoftTargetCrossEntropy().to(device)
        optimizer = create_optimizer(args, model)
        scaler = amp.GradScaler() if cli.amp else None

        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.num_classes,
        )

        epoch_secs, num_batches, avg_loss = timed_train_one_epoch(
            model, criterion, optimizer, data_loader, device, scaler,
            mixup_fn, train_snn_aug, train_trivalaug,
            print_every=cli.print_every, max_batches=cli.max_batches,
        )

        results[label] = (epoch_secs, num_batches)
        print(f"  ✓ {num_batches} batches in {epoch_secs:.2f}s  "
              f"({epoch_secs/num_batches*1000:.1f} ms/batch)  avg_loss={avg_loss:.4f}\n")

        # Free GPU memory before next config
        del model, optimizer, scaler, criterion
        torch.cuda.empty_cache()

    # ---- Summary ----
    ref_label = "PyTorch Baseline: Gamma + LIF"
    ref_secs = results[ref_label][0]

    print(f"\n{'=' * 72}")
    print(f" RESULTS SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'Config':<45s} {'Time':>8s} {'ms/batch':>10s} {'Overhead':>10s}")
    print("─" * 75)
    for label, (secs, num_batches) in results.items():
        overhead_pct = ((secs - ref_secs) / ref_secs) * 100
        sign = "+" if overhead_pct >= 0 else ""
        ms_per_batch = secs / max(num_batches, 1) * 1000
        print(f"  {label:<43s} {secs:7.1f}s {ms_per_batch:9.1f}ms {sign}{overhead_pct:8.1f}%")

    print(f"\n  Reference: {ref_label}")

    # Also show PyTorch vs Triton speedups
    print(f"\n{'=' * 72}")
    print(f" TRITON SPEEDUP BREAKDOWN")
    print(f"{'=' * 72}")
    pairs = [
        ("LIF Baseline",        "PyTorch Baseline: Gamma + LIF",          "Triton Baseline:  Gamma + LIF"),
        ("Conservative C-Grad", "PyTorch C-Grad:   Gamma + conservative", "Triton C-Grad:    Gamma + conservative"),
        ("Invariant C-Grad",    "PyTorch C-Grad:   Gamma + invariant",    "Triton C-Grad:    Gamma + invariant"),
    ]
    for mode_name, py_label, tri_label in pairs:
        py_s = results[py_label][0]
        tri_s = results[tri_label][0]
        speedup = py_s / max(tri_s, 1e-6)
        print(f"  {mode_name:22s}:  PyTorch {py_s:5.1f}s  →  Triton {tri_s:5.1f}s  ({speedup:.2f}x faster)")

    print()


if __name__ == "__main__":
    main()
