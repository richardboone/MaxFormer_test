#!/usr/bin/env python3
"""Pre-flight 1-batch gate calibration probe for Stage 2 CIFAR-100 Improved C-Grad arms."""

import argparse
import copy
import os
import sys
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
CIFAR_DIR = os.path.join(REPO_ROOT, "cifar10-100")
if CIFAR_DIR not in sys.path:
    sys.path.insert(0, CIFAR_DIR)

import custom_neuron
import max_former  # noqa: F401
from loader import create_loader
from spikingjelly.clock_driven import functional
from timm.data import create_dataset
from timm.models import create_model
import train as c100_train


PROBE_ARMS = [
    {
        "name": "C100_A0_bptt_sig",
        "config": "./cifar100_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-misalign-mode=product",
            "--snnbp-harmful-only=False",
            "--snnbp-eta=0.0",
        ],
        "is_ref": True,
    },
    {
        "name": "C100_B0_bptt_gamma",
        "config": "./cifar100_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-misalign-mode=product",
            "--snnbp-harmful-only=False",
            "--snnbp-eta=0.0",
        ],
        "is_ref": True,
    },
    {
        "name": "C100_A1_T8_gentle",
        "config": "./cifar100_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-misalign-mode=product",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=25.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.0",
            "--snnbp-gamma-sharp=0.0",
            "--snnbp-corr-scale=0.5",
            "--snnbp-eta=0.35",
        ],
        "is_ref": False,
    },
    {
        "name": "C100_B1_G3_flip",
        "config": "./cifar100_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-misalign-mode=product",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=25.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.35",
            "--snnbp-gamma-sharp=8.0",
            "--snnbp-corr-scale=1.0",
            "--snnbp-eta=1.0",
        ],
        "is_ref": False,
    },
]


def parse_arm_args(arm, time_step=4):
    """Parse cifar10-100/train.py arguments for one probe arm."""
    cfg_path = os.path.join(CIFAR_DIR, arm["config"].lstrip("./"))
    orig_argv = sys.argv
    try:
        sys.argv = ["train.py", "-c", cfg_path, f"--time-step={time_step}"] + list(arm["flags"])
        args, _ = c100_train._parse_args()
        return args
    finally:
        sys.argv = orig_argv


def load_one_batch(data_path, batch_size, synthetic):
    """Load one normalized CIFAR-100 training batch (or synthetic fallback)."""
    if synthetic:
        torch.manual_seed(42)
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.arange(batch_size, dtype=torch.long) % 100
        return x, y
    dataset = create_dataset(
        "torch/cifar100",
        root=data_path,
        split="train",
        is_training=True,
        batch_size=batch_size,
        download=False,
    )
    loader = create_loader(
        dataset,
        input_size=(3, 32, 32),
        batch_size=batch_size,
        is_training=True,
        use_prefetcher=False,
        interpolation="bicubic",
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
        num_workers=0,
        persistent_workers=False,
    )
    return next(iter(loader))


def classify_arm(stats, args):
    """Classify calibrated gate rates as OK, DEAD, SATURATED, or ASYMMETRIC."""
    eff_pct = stats["cgrad/effective_rate"] * 100.0
    m_neg_pct = stats["cgrad/frac_effective_m_neg"] * 100.0
    gsharp = float(getattr(args, "snnbp_gamma_sharp", 0.0) or 0.0)
    if eff_pct < 0.5:
        return "DEAD"
    if gsharp > 0.0 and eff_pct > 35.0:
        return "SATURATED"
    if abs(m_neg_pct - 50.0) > 8.0:
        return "ASYMMETRIC"
    return "OK"


def main():
    parser = argparse.ArgumentParser(description="Pre-flight gate telemetry probe for Stage 2 CIFAR-100 arms")
    parser.add_argument("--data-path", default="/data/rboone/datasets/cifar100/")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--time-step", type=int, default=4)
    parser.add_argument("--synthetic", action="store_true")
    cli = parser.parse_args()

    os.chdir(CIFAR_DIR)
    device = torch.device(cli.device)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    x, y = load_one_batch(cli.data_path, cli.batch_size, synthetic=cli.synthetic)
    x = x.to(device)
    y = y.to(device)
    criterion = nn.CrossEntropyLoss()

    base_args = parse_arm_args(PROBE_ARMS[0], time_step=cli.time_step)
    custom_neuron.set_global_args(base_args)
    model = create_model(
        "max_former",
        in_channels=3,
        num_classes=100,
        embed_dims=384,
        mlp_ratios=4,
        drop_rate=0.0,
        depths=4,
        T=cli.time_step,
    ).to(device)
    model.train()
    init_state = copy.deepcopy(model.state_dict())

    header = f"{'arm':22s} | {'surr':7s} | {'eff_rate':>8s} | {'mean_w':>8s} | {'mean_sig':>8s} | {'near%':>6s} | {'m_neg%':>7s} | {'status':10s}"
    print(header)
    print("-" * len(header))

    bad_arms = []
    for arm in PROBE_ARMS:
        args = parse_arm_args(arm, time_step=cli.time_step)
        custom_neuron.set_global_args(args)
        model.load_state_dict(init_state)
        functional.reset_net(model)
        model.zero_grad(set_to_none=True)

        custom_neuron.enable_cgrad_stats(device, sample_prob=1.0)
        with torch.amp.autocast("cuda"):
            out = model(x)
            loss = criterion(out, y)
        loss.backward()
        stats = custom_neuron.pop_cgrad_stats()
        custom_neuron.disable_cgrad_stats()
        functional.reset_net(model)

        eff_pct = stats["cgrad/effective_rate"] * 100.0
        mean_w = stats["cgrad/mean_w"]
        mean_sig = stats["cgrad/mean_signal"]
        near_pct = stats["cgrad/near_threshold_rate"] * 100.0
        m_neg_pct = stats["cgrad/frac_effective_m_neg"] * 100.0
        status = "REF" if arm.get("is_ref") else classify_arm(stats, args)
        if not arm.get("is_ref") and status != "OK":
            bad_arms.append((arm["name"], status, eff_pct, m_neg_pct))

        print(
            f"{arm['name']:22s} | {args.dS_du:7s} | {eff_pct:7.2f}% | {mean_w:8.4f} | "
            f"{mean_sig:8.4f} | {near_pct:5.1f}% | {m_neg_pct:6.2f}% | {status:10s}"
        )

    if bad_arms:
        print(f"\n[PROBE FAIL] {len(bad_arms)} non-reference arms failed T={cli.time_step} calibration: {bad_arms}")
        sys.exit(2)
    n_checked = sum(1 for a in PROBE_ARMS if not a.get("is_ref"))
    print(f"\n[PROBE PASS] All {n_checked} Stage 2 CIFAR-100 C-Grad arms passed T={cli.time_step} gate calibration.")


if __name__ == "__main__":
    main()
