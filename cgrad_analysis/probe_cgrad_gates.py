#!/usr/bin/env python3
"""Pre-flight 1-batch gate calibration probe for Phase 1 Improved C-Grad arms."""

import argparse
import copy
import os
import sys
import torch
import torch.nn as nn

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EVENT_DIR = os.path.join(REPO_ROOT, "event")
if EVENT_DIR not in sys.path:
    sys.path.insert(0, EVENT_DIR)

import custom_neuron
import max_former  # noqa: F401
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
from timm.models import create_model
import train as event_train
import run_tune_phase1


STAGE1_REFS = [
    {
        "name": "S1_A2_default",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-eta=1.0",
        ],
        "is_ref": True,
    },
    {
        "name": "S1_B3_default",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-eta=1.0",
        ],
        "is_ref": True,
    },
]


def parse_arm_args(arm):
    cfg_path = os.path.join(EVENT_DIR, arm["config"].lstrip("./"))
    orig_argv = sys.argv
    try:
        sys.argv = ["train.py", "-c", cfg_path] + list(arm["flags"])
        return event_train.parse_args()
    finally:
        sys.argv = orig_argv


def load_one_batch(data_path, batch_size, T, synthetic):
    if synthetic:
        torch.manual_seed(42)
        x = (torch.rand(batch_size, T, 2, 128, 128) > 0.85).float()
        y = torch.arange(batch_size) % 10
        return x, y
    origin_set = cifar10_dvs.CIFAR10DVS(root=data_path, data_type="frame", frames_number=T, split_by="number")
    imgs = []
    labels = []
    for idx in range(batch_size):
        img, lbl = origin_set[idx]
        imgs.append(torch.as_tensor(img))
        labels.append(int(lbl))
    x = torch.stack(imgs, dim=0).float()
    y = torch.tensor(labels, dtype=torch.long)
    return x, y


def classify_arm(stats, args):
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
    parser = argparse.ArgumentParser(description="Pre-flight gate telemetry probe for Phase 1 / Phase 1B tuning arms")
    parser.add_argument("--data-path", default="/data/rboone/datasets/wg_dvst")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase", default="1", choices=["1", "1b"], help="Which arm set to probe")
    parser.add_argument("--synthetic", action="store_true")
    cli = parser.parse_args()

    os.chdir(EVENT_DIR)
    device = torch.device(cli.device)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    x, y = load_one_batch(cli.data_path, cli.batch_size, T=16, synthetic=cli.synthetic)
    x = x.to(device)
    y = y.to(device)
    criterion = nn.CrossEntropyLoss()

    base_args = parse_arm_args(STAGE1_REFS[0])
    custom_neuron.set_global_args(base_args)
    model = create_model(
        "max_former", in_channels=2, num_classes=10, embed_dims=256, mlp_ratios=1.0, depths=2, T=16
    ).to(device)
    model.train()
    init_state = copy.deepcopy(model.state_dict())

    if cli.phase == "1b":
        winner_names = {"T8_gentle_eta", "T7_flip_gated", "G3_gamma_flip_gated"}
        refs = [dict(a, is_ref=True) for a in run_tune_phase1.ARMS if a["name"] in winner_names]
        targets = [dict(a, is_ref=False) for a in run_tune_phase1.PHASE1B_ARMS]
        all_arms = refs + targets
    else:
        all_arms = STAGE1_REFS + [dict(a, is_ref=False) for a in run_tune_phase1.ARMS]

    header = f"{'arm':24s} | {'surr':7s} | {'eff_rate':>8s} | {'mean_w':>8s} | {'mean_sig':>8s} | {'near%':>6s} | {'m_neg%':>7s} | {'status':10s}"
    print(header)
    print("-" * len(header))

    bad_arms = []
    for arm in all_arms:
        args = parse_arm_args(arm)
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
            f"{arm['name']:24s} | {args.dS_du:7s} | {eff_pct:7.2f}% | {mean_w:8.4f} | "
            f"{mean_sig:8.4f} | {near_pct:5.1f}% | {m_neg_pct:6.2f}% | {status:10s}"
        )

    if bad_arms:
        print(f"\n[PROBE FAIL] {len(bad_arms)} non-reference arms failed calibration: {bad_arms}")
        sys.exit(2)
    print(f"\n[PROBE PASS] All {sum(1 for a in all_arms if not a.get('is_ref'))} Phase {cli.phase.upper()} tuning arms passed gate calibration.")


if __name__ == "__main__":
    main()
