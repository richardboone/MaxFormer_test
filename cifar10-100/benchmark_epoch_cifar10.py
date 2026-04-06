"""
Real-world one-epoch benchmark: conservative_cgrad vs standard LIF on CIFAR-10.

Runs one full training epoch (forward + backward + optimizer step) for each
configuration using the actual MaxFormer model, timm data pipeline, and
NativeScaler AMP, then prints a timing comparison.

Configurations
--------------
  1. Baseline (sigmoid + LIF):  dS_du=sigmoid, du_du=LIF, detach_reset=False
  2. Baseline (Gamma  + LIF):  dS_du=Gamma,   du_du=LIF, detach_reset=False
  3. C-Grad  (sigmoid + cons): dS_du=sigmoid, du_du=conservative_cgrad, detach_reset=False
  4. C-Grad  (Gamma  + cons):  dS_du=Gamma,   du_du=conservative_cgrad, detach_reset=False

Usage
-----
    conda run -n snn-maxformer python benchmark_epoch_cifar10.py
    conda run -n snn-maxformer python benchmark_epoch_cifar10.py --data-path /data/rboone/datasets/cifar10/
    conda run -n snn-maxformer python benchmark_epoch_cifar10.py --batch-size 64 --time-step 8
"""

import argparse
import os
import sys
import time
from collections import OrderedDict
from contextlib import suppress

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Ensure local imports resolve from the cifar10-100 directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import custom_neuron
import max_former  # registers the timm model

from spikingjelly.clock_driven import functional
from timm.data import create_dataset, Mixup
from timm.models import create_model
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler, random_seed
from loader import create_loader


# ---------------------------------------------------------------------------
# Timed training loop (one epoch)
# ---------------------------------------------------------------------------
def timed_train_one_epoch(model, loader, optimizer, loss_fn, loss_scaler,
                          amp_autocast, mixup_fn, prefetcher, print_every=100):
    """Train for exactly one epoch, returning (total_seconds, num_batches, avg_loss)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    # --- Warm-up: run 3 batches without timing to prime CUDA caches ---
    warmup_iter = iter(loader)
    for _ in range(3):
        try:
            inp, tgt = next(warmup_iter)
        except StopIteration:
            break
        if not prefetcher:
            inp, tgt = inp.cuda(), tgt.cuda()
            if mixup_fn is not None:
                inp, tgt = mixup_fn(inp, tgt)
        with amp_autocast():
            output = model(inp)
            loss = loss_fn(output, tgt)
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
        else:
            loss.backward()
            optimizer.step()
        functional.reset_net(model)

    torch.cuda.synchronize()
    t_start = time.perf_counter()

    for batch_idx, (inp, tgt) in enumerate(loader):
        if not prefetcher:
            inp, tgt = inp.cuda(), tgt.cuda()
            if mixup_fn is not None:
                inp, tgt = mixup_fn(inp, tgt)

        with amp_autocast():
            output = model(inp)
            loss = loss_fn(output, tgt)

        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, parameters=model.parameters())
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

    torch.cuda.synchronize()
    t_end = time.perf_counter()
    return t_end - t_start, num_batches, total_loss / max(num_batches, 1)


# ---------------------------------------------------------------------------
# Build the full args namespace that the pipeline expects
# ---------------------------------------------------------------------------
def build_full_args(cli, dS_du, du_du, detach_reset):
    """Return args namespace compatible with the cifar10-100 train.py."""
    return argparse.Namespace(
        # Model
        model="max_former",
        time_step=cli.time_step,
        layer=cli.layer,
        dim=cli.dim,
        num_heads=cli.num_heads,
        patch_size=cli.patch_size,
        mlp_ratio=cli.mlp_ratio,
        num_classes=10,
        img_size=32,
        input_size=None,
        gp=None,
        # Data
        data_path=cli.data_path,
        dataset="torch/cifar10",
        train_split="train",
        val_split="validation",
        # Loader
        batch_size=cli.batch_size,
        val_batch_size=cli.batch_size,
        workers=cli.workers,
        prefetcher=not cli.no_prefetcher,
        no_prefetcher=cli.no_prefetcher,
        pin_mem=True,
        use_multi_epochs_loader=False,
        # Optimizer
        opt="adamw",
        opt_eps=1e-8,
        opt_betas=None,
        weight_decay=0.06,
        momentum=0.9,
        lr=1.5e-3,
        clip_grad=None,
        clip_mode="norm",
        # Scheduler
        sched="cosine",
        lr_noise=None,
        lr_noise_pct=0.67,
        lr_noise_std=1.0,
        lr_cycle_mul=1.0,
        lr_cycle_limit=1,
        warmup_lr=8e-5,
        min_lr=1e-5,
        epochs=1,
        epoch_repeats=0.0,
        start_epoch=0,
        decay_epochs=30,
        warmup_epochs=0,
        cooldown_epochs=10,
        patience_epochs=10,
        decay_rate=0.1,
        # Augmentation
        no_aug=False,
        scale=[1.0, 1.0],
        ratio=[1.0, 1.0],
        hflip=0.5,
        vflip=0.0,
        color_jitter=0.0,
        aa="rand-m9-n1-mstd0.4-inc1",
        aug_splits=0,
        reprob=0.25,
        remode="const",
        recount=1,
        resplit=False,
        train_interpolation="bicubic",
        interpolation="bicubic",
        crop_pct=1.0,
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2470, 0.2435, 0.2616],
        # Mixup
        mixup=0.5,
        cutmix=0.5,
        cutmix_minmax=None,
        mixup_prob=1.0,
        mixup_switch_prob=0.5,
        mixup_mode="batch",
        mixup_off_epoch=0,
        smoothing=0.1,
        # AMP
        amp=cli.amp,
        native_amp=cli.amp,
        apex_amp=False,
        # BN / distributed
        channels_last=False,
        sync_bn=False,
        split_bn=False,
        bn_tf=False,
        bn_momentum=None,
        bn_eps=None,
        dist_bn="",
        distributed=False,
        world_size=1,
        rank=0,
        local_rank=0,
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
        # EMA / misc
        model_ema=False,
        model_ema_force_cpu=False,
        model_ema_decay=0.9998,
        seed=cli.seed,
        log_interval=100,
        log_wandb=False,
        experiment="",
        output="",
        save_images=False,
        recovery_interval=0,
        checkpoint_hist=1,
        initial_checkpoint="",
        resume="",
        no_resume_opt=False,
        eval_metric="top1",
        tta=0,
        torchscript=False,
        jsd=False,
        drop=0.0,
        drop_connect=None,
        drop_path=None,
        drop_block=None,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Real-world one-epoch timing benchmark (CIFAR-10).")
    parser.add_argument("--data-path", default="/data/rboone/datasets/cifar10/",
                        help="Path to CIFAR-10 dataset")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--time-step", type=int, default=4, help="Simulation timesteps (T)")
    parser.add_argument("--layer", type=int, default=4, help="Model depth")
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--amp", action="store_true", default=True, help="Use native AMP")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--no-prefetcher", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--print-every", type=int, default=100, help="Print progress every N batches")
    cli = parser.parse_args()

    # ---- Configurations to compare ----
    configs = [
        ("Baseline: sigmoid + LIF",            "sigmoid", "LIF",                False),
        ("Baseline: Gamma  + LIF",             "Gamma",   "LIF",                False),
        ("C-Grad:   sigmoid + conservative",    "sigmoid", "conservative_cgrad", False),
        ("C-Grad:   Gamma  + conservative",     "Gamma",   "conservative_cgrad", False),
    ]

    # ---- Build one args to create dataset (dataset doesn't depend on neuron config) ----
    sample_args = build_full_args(cli, "sigmoid", "LIF", False)
    data_config = dict(
        input_size=(3, 32, 32),
        interpolation="bicubic",
        mean=sample_args.mean,
        std=sample_args.std,
        crop_pct=sample_args.crop_pct,
    )

    print("Loading CIFAR-10 dataset...")
    t_data = time.perf_counter()
    dataset_train = create_dataset(
        "torch/cifar10", root=cli.data_path, split="train",
        is_training=True, batch_size=cli.batch_size, download=True,
    )
    print(f"Dataset loaded in {time.perf_counter() - t_data:.1f}s  ({len(dataset_train)} samples)\n")

    print(f"{'=' * 72}")
    print(f" One-Epoch Training Benchmark — CIFAR-10")
    print(f" T={cli.time_step}  batch_size={cli.batch_size}  dim={cli.dim}  "
          f"layer={cli.layer}  heads={cli.num_heads}  AMP={cli.amp}")
    print(f"{'=' * 72}\n")

    results = {}

    for label, dS_du, du_du, detach_reset in configs:
        print(f"─── {label} ───")

        args = build_full_args(cli, dS_du, du_du, detach_reset)
        custom_neuron.set_global_args(args)
        random_seed(cli.seed, 0)

        model = create_model(
            "max_former", in_channels=3, num_classes=10,
            embed_dims=cli.dim, mlp_ratios=cli.mlp_ratio,
            drop_rate=0.0, depths=cli.layer, T=cli.time_step,
        )
        model.cuda()

        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Parameters: {n_params:,}")

        optimizer = create_optimizer_v2(model, **optimizer_kwargs(cfg=args))

        # AMP
        amp_autocast = suppress
        loss_scaler = None
        if cli.amp:
            amp_autocast = torch.cuda.amp.autocast
            loss_scaler = NativeScaler()

        # Data loader (re-create per config so iterator is fresh)
        use_prefetcher = not cli.no_prefetcher
        loader_train = create_loader(
            dataset_train,
            input_size=data_config["input_size"],
            batch_size=cli.batch_size,
            is_training=True,
            use_prefetcher=use_prefetcher,
            no_aug=False,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            re_split=args.resplit,
            scale=args.scale,
            ratio=args.ratio,
            hflip=args.hflip,
            vflip=args.vflip,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            num_aug_splits=0,
            interpolation=args.train_interpolation,
            mean=data_config["mean"],
            std=data_config["std"],
            num_workers=cli.workers,
            distributed=False,
            pin_memory=True,
            use_multi_epochs_loader=False,
        )

        # Loss
        mixup_fn = None
        mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
        if mixup_active:
            mixup_args = dict(
                mixup_alpha=args.mixup, cutmix_alpha=args.cutmix,
                cutmix_minmax=args.cutmix_minmax, prob=args.mixup_prob,
                switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
                label_smoothing=args.smoothing, num_classes=args.num_classes,
            )
            if use_prefetcher:
                # With prefetcher, mixup happens in collate — don't apply separately
                from timm.data import FastCollateMixup
                # Loader was already created without collate_fn,
                # so we just apply mixup_fn here if not prefetching
                pass
            else:
                mixup_fn = Mixup(**mixup_args)

        if mixup_active:
            train_loss_fn = nn.CrossEntropyLoss(label_smoothing=args.smoothing).cuda()
        elif args.smoothing:
            from timm.loss import LabelSmoothingCrossEntropy
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing).cuda()
        else:
            train_loss_fn = nn.CrossEntropyLoss().cuda()

        epoch_secs, num_batches, avg_loss = timed_train_one_epoch(
            model, loader_train, optimizer, train_loss_fn,
            loss_scaler, amp_autocast, mixup_fn, use_prefetcher,
            print_every=cli.print_every,
        )

        results[label] = epoch_secs
        print(f"  ✓ {num_batches} batches in {epoch_secs:.2f}s  "
              f"({epoch_secs/num_batches*1000:.1f} ms/batch)  avg_loss={avg_loss:.4f}\n")

        del model, optimizer, loss_scaler, train_loss_fn
        torch.cuda.empty_cache()

    # ==================================================================
    # Summary
    # ==================================================================
    ref_label = "Baseline: sigmoid + LIF"
    ref_secs = results[ref_label]

    print(f"\n{'=' * 72}")
    print(f" RESULTS SUMMARY")
    print(f"{'=' * 72}")
    print(f"{'Config':<45s} {'Time':>8s} {'ms/batch':>10s} {'Overhead':>10s}")
    print("─" * 75)
    for label, secs in results.items():
        overhead_pct = ((secs - ref_secs) / ref_secs) * 100
        sign = "+" if overhead_pct >= 0 else ""
        ms_per_batch = secs / (len(dataset_train) // cli.batch_size) * 1000
        print(f"  {label:<43s} {secs:7.1f}s {ms_per_batch:9.1f}ms {sign}{overhead_pct:8.1f}%")

    print(f"\n  Reference: {ref_label}")

    print(f"\n{'=' * 72}")
    print(f" OVERHEAD BY SURROGATE GRADIENT")
    print(f"{'=' * 72}")
    pairs = [
        ("sigmoid", "Baseline: sigmoid + LIF", "C-Grad:   sigmoid + conservative"),
        ("Gamma",   "Baseline: Gamma  + LIF",  "C-Grad:   Gamma  + conservative"),
    ]
    for surr, base_label, test_label in pairs:
        base_s = results[base_label]
        test_s = results[test_label]
        oh = ((test_s - base_s) / base_s) * 100
        print(f"  {surr:10s}:  {base_s:.1f}s → {test_s:.1f}s  ({oh:+.1f}%)")

    print()


if __name__ == "__main__":
    main()
