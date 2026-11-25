#!/usr/bin/env python3
"""Shared-GPU-polite, disconnect-resilient Phase 1 tuning runner for CIFAR10-DVS MaxFormer."""

import argparse
import datetime
import json
import os
import signal
import subprocess
import sys
import time


ARMS = [
    {
        "name": "T1_sharp_mis",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=25.0",
            "--snnbp-alpha=3.0",
            "--snnbp-margin-m=0.0",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.0",
            "--snnbp-gamma-sharp=0.0",
            "--snnbp-corr-scale=0.5",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "T2_sharp_mis_margin",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=25.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.25",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.0",
            "--snnbp-gamma-sharp=0.0",
            "--snnbp-corr-scale=0.5",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "T3_gated_moderate",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=25.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.35",
            "--snnbp-gamma-sharp=8.0",
            "--snnbp-corr-scale=0.5",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "T4_gated_strict",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=30.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.30",
            "--snnbp-epsilon=0.25",
            "--snnbp-beta=12.0",
            "--snnbp-intervention=0.50",
            "--snnbp-gamma-sharp=10.0",
            "--snnbp-corr-scale=0.5",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "T5_mask_smooth",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=25.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.0",
            "--snnbp-gamma-sharp=0.0",
            "--snnbp-corr-scale=0.0",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "T6_mask_gated",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=25.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.35",
            "--snnbp-gamma-sharp=8.0",
            "--snnbp-corr-scale=0.0",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "T7_flip_gated",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
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
    },
    {
        "name": "T8_gentle_eta",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
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
    },
    {
        "name": "G1_gamma_sharp_mis",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=25.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.0",
            "--snnbp-gamma-sharp=0.0",
            "--snnbp-corr-scale=0.5",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "G2_gamma_mask_gated",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=25.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.35",
            "--snnbp-gamma-sharp=8.0",
            "--snnbp-corr-scale=0.0",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "G3_gamma_flip_gated",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
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
    },
    {
        "name": "G4_gamma_strict",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=30.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.30",
            "--snnbp-epsilon=0.25",
            "--snnbp-beta=12.0",
            "--snnbp-intervention=0.50",
            "--snnbp-gamma-sharp=10.0",
            "--snnbp-corr-scale=0.5",
            "--snnbp-eta=1.0",
        ],
    },
]


PHASE1B_ARMS = [
    {
        "name": "X4_T8_rms",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=rms",
            "--snnbp-misalign-mode=product",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=15.0",
            "--snnbp-alpha=4.0",
            "--snnbp-margin-m=0.25",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.0",
            "--snnbp-gamma-sharp=0.0",
            "--snnbp-corr-scale=0.5",
            "--snnbp-eta=0.35",
        ],
    },
    {
        "name": "X5_T7_rms",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=rms",
            "--snnbp-misalign-mode=product",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=15.0",
            "--snnbp-alpha=4.0",
            "--snnbp-margin-m=0.25",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.25",
            "--snnbp-gamma-sharp=8.0",
            "--snnbp-corr-scale=1.0",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "X1_T8_sign",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-misalign-mode=sign",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=6.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.0",
            "--snnbp-gamma-sharp=0.0",
            "--snnbp-corr-scale=0.5",
            "--snnbp-eta=0.35",
        ],
    },
    {
        "name": "X2_T7_sign",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=sigmoid",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-misalign-mode=sign",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=6.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.35",
            "--snnbp-gamma-sharp=8.0",
            "--snnbp-corr-scale=1.0",
            "--snnbp-eta=1.0",
        ],
    },
    {
        "name": "X3_G3_sign",
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": [
            "--du-du=improved_cgrad",
            "--dS-du=Gamma",
            "--snnbp-tau=2.0",
            "--snnbp-input-scale=coupled",
            "--detach-reset=True",
            "--snnbp-norm=local",
            "--snnbp-misalign-mode=sign",
            "--snnbp-harmful-only=False",
            "--snnbp-alpha-mis=6.0",
            "--snnbp-alpha=6.0",
            "--snnbp-margin-m=0.20",
            "--snnbp-epsilon=0.35",
            "--snnbp-beta=8.0",
            "--snnbp-intervention=0.35",
            "--snnbp-gamma-sharp=8.0",
            "--snnbp-corr-scale=1.0",
            "--snnbp-eta=1.0",
        ],
    },
]


_PHASE2_SHARED_FLAGS = [
    "--du-du=improved_cgrad",
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
]

PHASE2_ARMS = [
    {
        "name": name,
        "config": "./cifar10dvs_improved_cgrad.yaml",
        "flags": list(_PHASE2_SHARED_FLAGS) + extra_flags,
    }
    for name, extra_flags in [
        (
            "P2_T8_eta020",
            [
                "--dS-du=sigmoid",
                "--snnbp-intervention=0.0",
                "--snnbp-gamma-sharp=0.0",
                "--snnbp-corr-scale=0.5",
                "--snnbp-eta=0.20",
            ],
        ),
        (
            "P2_T8_eta055",
            [
                "--dS-du=sigmoid",
                "--snnbp-intervention=0.0",
                "--snnbp-gamma-sharp=0.0",
                "--snnbp-corr-scale=0.5",
                "--snnbp-eta=0.55",
            ],
        ),
        (
            "P2_T8_flip",
            [
                "--dS-du=sigmoid",
                "--snnbp-intervention=0.0",
                "--snnbp-gamma-sharp=0.0",
                "--snnbp-corr-scale=1.0",
                "--snnbp-eta=0.35",
            ],
        ),
        (
            "P2_G5_gamma_gentle",
            [
                "--dS-du=Gamma",
                "--snnbp-intervention=0.0",
                "--snnbp-gamma-sharp=0.0",
                "--snnbp-corr-scale=0.5",
                "--snnbp-eta=0.35",
            ],
        ),
        (
            "P2_T7_eta060",
            [
                "--dS-du=sigmoid",
                "--snnbp-intervention=0.35",
                "--snnbp-gamma-sharp=8.0",
                "--snnbp-corr-scale=1.0",
                "--snnbp-eta=0.60",
            ],
        ),
        (
            "P2_T7_strict_h45",
            [
                "--dS-du=sigmoid",
                "--snnbp-intervention=0.45",
                "--snnbp-gamma-sharp=10.0",
                "--snnbp-corr-scale=1.0",
                "--snnbp-eta=1.0",
            ],
        ),
    ]
]


def query_gpu_stats():
    for attempt in range(3):
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=index,memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=15,
            )
            stats = {}
            for line in out.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) == 3:
                    idx, mem_mb, util = int(parts[0]), int(parts[1]), int(parts[2])
                    stats[idx] = (mem_mb, util)
            if stats:
                return stats
        except (subprocess.SubprocessError, OSError, ValueError):
            pass
        if attempt < 2:
            time.sleep(2.0)
    return {}


def pick_free_gpu(
    active_jobs,
    candidate_gpus,
    jobs_per_gpu=1,
    max_gpus_override=0,
    max_mem_per_job_mb=7000,
):
    stats = query_gpu_stats()
    if not stats:
        return None, 0, 0
    ours_per_gpu = {}
    for job in active_jobs:
        g = job["gpu"]
        ours_per_gpu[g] = ours_per_gpu.get(g, 0) + 1
    owned_gpus = set(ours_per_gpu.keys())

    external_busy = 0
    for idx, (mem_mb, util) in stats.items():
        if idx in owned_gpus:
            continue
        if mem_mb >= 1000 or util >= 15:
            external_busy += 1

    if external_busy <= 3:
        max_gpus = 4
    elif external_busy == 4:
        max_gpus = 3
    else:
        max_gpus = 2

    max_gpus = min(max_gpus, len(candidate_gpus))
    if max_gpus_override > 0:
        max_gpus = min(max_gpus, max_gpus_override)

    max_slots = max_gpus * jobs_per_gpu
    if len(active_jobs) >= max_slots:
        return None, max_slots, external_busy

    # Pass 1: spread across distinct free GPUs first up to max_gpus
    if len(owned_gpus) < max_gpus:
        for g in candidate_gpus:
            if g in owned_gpus or g not in stats:
                continue
            mem_mb, util = stats[g]
            if mem_mb < 1000 and util < 10:
                time.sleep(2.0)
                stats2 = query_gpu_stats()
                m2, u2 = stats2.get(g, (99999, 100))
                if m2 < 1000 and u2 < 10:
                    return g, max_slots, external_busy

    # Pass 2: pack up to jobs_per_gpu on already-owned GPUs (guarding against external intrusion)
    if jobs_per_gpu > 1:
        packable = [
            g for g in candidate_gpus
            if g in owned_gpus and g in stats and ours_per_gpu[g] < jobs_per_gpu
        ]
        packable.sort(key=lambda g: ours_per_gpu[g])
        for g in packable:
            mem_mb, _ = stats[g]
            if mem_mb < ours_per_gpu[g] * max_mem_per_job_mb + 2000 and (mem_mb + max_mem_per_job_mb) < 75000:
                return g, max_slots, external_busy

    return None, max_slots, external_busy


def write_json_atomic(path, payload):
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)


def log_msg(master_log_path, msg):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    with open(master_log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        print(line, flush=True)
    except OSError:
        pass


def load_summary_status(summary_path):
    if not os.path.exists(summary_path):
        return None
    try:
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def both_prior_seeds_collapsed(arm_name, seed, seeds, summaries_dir, prefix):
    prior_seeds = [s for s in seeds if s < seed]
    if len(prior_seeds) < 2:
        return False
    collapsed_cnt = 0
    for ps in prior_seeds:
        sp = os.path.join(summaries_dir, f"{prefix}{arm_name}_seed{ps}.json")
        s = load_summary_status(sp)
        if not s:
            return False
        if s.get("status") == "nan_collapsed":
            collapsed_cnt += 1
        elif s.get("status") == "completed" and s.get("val_mean_last_k", 100.0) < 45.0:
            collapsed_cnt += 1
        else:
            return False
    return collapsed_cnt >= 2


def main():
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGPIPE, signal.SIG_IGN)

    parser = argparse.ArgumentParser(description="Phase 1 CIFAR10-DVS Improved C-Grad Tuning Scheduler")
    parser.add_argument("--data-path", default="/data/rboone/datasets/wg_dvst")
    parser.add_argument("--gpus", default="0,1,2,3,4")
    parser.add_argument("--jobs-per-gpu", type=int, default=1, help="Max concurrent jobs to pack per owned GPU; use 2 on A100-80GB")
    parser.add_argument("--max-gpus", type=int, default=0, help="Optional hard cap on distinct GPUs used (0 = auto polite 4/3/2)")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=None, help="DataLoader workers per job (default: 4 if jobs-per-gpu==1 else 2)")
    parser.add_argument("--cgrad-stats-prob", type=float, default=0.02)
    parser.add_argument("--out-dir", default="./logs/tune_phase1")
    parser.add_argument("--phase", default="1", choices=["1", "1b", "2"], help="Tuning phase: 1 (12 screen arms), 1b (5 matched sign/rms arms), or 2 (6 local refinement arms)")
    parser.add_argument("--prefix", default="", help="Experiment name prefix (defaults to t1_ for phase 1, t1b_ for phase 1b)")
    parser.add_argument("--arms", default="", help="Optional comma-separated subset of arm names")
    parser.add_argument("--stall-timeout", type=int, default=1200, help="Seconds of epoch progress inactivity before killing stalled run")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--wandb-online", action="store_true", help="Run wandb online instead of offline mode")
    parser.add_argument("--dry-run", action="store_true")
    cli = parser.parse_args()

    event_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(event_dir)

    candidate_gpus = [int(x) for x in cli.gpus.split(",") if x.strip()]
    seeds = [int(x) for x in cli.seeds.split(",") if x.strip()]
    arm_filter = {x.strip() for x in cli.arms.split(",") if x.strip()} if cli.arms else None

    out_dir = os.path.abspath(cli.out_dir)
    run_logs_dir = os.path.join(out_dir, "run_logs")
    summaries_dir = os.path.join(out_dir, "summaries")
    os.makedirs(run_logs_dir, exist_ok=True)
    os.makedirs(summaries_dir, exist_ok=True)
    master_log = os.path.join(out_dir, "master.log")
    status_path = os.path.join(out_dir, "status.json")

    if cli.workers is None:
        cli.workers = 4 if cli.jobs_per_gpu == 1 else 2

    if not cli.prefix:
        cli.prefix = {"1": "t1_", "1b": "t1b_", "2": "t2_"}[cli.phase]

    arm_pool = {"1": ARMS, "1b": PHASE1B_ARMS, "2": PHASE2_ARMS}[cli.phase]
    selected_arms = [a for a in arm_pool if (arm_filter is None or a["name"] in arm_filter)]
    if cli.phase == "1b":
        # Run slower PyTorch-fallback rms arms first across GPUs, then fast Triton sign arms
        rms_arms = [a for a in selected_arms if "rms" in a["name"]]
        non_rms_arms = [a for a in selected_arms if "rms" not in a["name"]]
        ordered_pairs = [(seed, arm) for seed in seeds for arm in rms_arms] + [
            (seed, arm) for seed in seeds for arm in non_rms_arms
        ]
    else:
        ordered_pairs = [(seed, arm) for seed in seeds for arm in selected_arms]

    tasks = []
    skipped = 0
    for seed, arm in ordered_pairs:
        exp_name = f"{cli.prefix}{arm['name']}_seed{seed}"
        summary_path = os.path.join(summaries_dir, f"{exp_name}.json")
        existing = load_summary_status(summary_path)
        if existing and existing.get("status") in ("completed", "nan_collapsed"):
            skipped += 1
            continue
        tasks.append({
            "arm": arm["name"],
            "seed": seed,
            "exp_name": exp_name,
            "config": arm["config"],
            "flags": arm["flags"],
            "summary_path": summary_path,
            "epochs_jsonl_path": f"{summary_path[:-5]}.epochs.jsonl",
            "log_path": os.path.join(run_logs_dir, f"{exp_name}.log"),
            "retries": 0,
        })

    if cli.dry_run:
        print(
            f"Phase {cli.phase.upper()} tuning dry-run: {len(tasks)} pending tasks ({skipped} already finished), "
            f"candidate_gpus={candidate_gpus}, jobs_per_gpu={cli.jobs_per_gpu}, max_gpus={cli.max_gpus or 'auto'}, "
            f"workers={cli.workers}, seeds={seeds}, epochs={cli.epochs}"
        )
        for idx, t in enumerate(tasks, 1):
            print(f"{idx:2d}. {t['exp_name']} | config={t['config']} | {' '.join(t['flags'])}")
        return

    log_msg(
        master_log,
        f"Phase {cli.phase.upper()} tuning scheduler starting: {len(tasks)} pending tasks ({skipped} already finished), "
        f"candidate_gpus={candidate_gpus}, jobs_per_gpu={cli.jobs_per_gpu}, max_gpus={cli.max_gpus or 'auto'}, "
        f"workers={cli.workers}, seeds={seeds}, epochs={cli.epochs}",
    )

    active_jobs = []
    completed_list = []
    failed_list = []

    while tasks or active_jobs:
        still_active = []
        now = time.time()
        for job in active_jobs:
            t = job["task"]
            ret = job["proc"].poll()
            summary = load_summary_status(t["summary_path"])
            summary_fresh = bool(
                summary
                and os.path.exists(t["summary_path"])
                and os.path.getmtime(t["summary_path"]) >= job["launch_time"] - 1.0
            )
            if ret is None:
                if summary_fresh and summary.get("status") in ("completed", "nan_collapsed"):
                    teardown_age = now - os.path.getmtime(t["summary_path"])
                    if teardown_age > 60.0:
                        log_msg(
                            master_log,
                            f"TEARDOWN {t['exp_name']} on GPU {job['gpu']} (summary completed {teardown_age:.0f}s ago) — terminating lingering process",
                        )
                        try:
                            os.killpg(job["proc"].pid, signal.SIGTERM)
                            job["proc"].wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(job["proc"].pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            try:
                                job["proc"].wait(timeout=10)
                            except Exception:
                                pass
                        except ProcessLookupError:
                            pass
                        ret = 0
                elif cli.stall_timeout > 0:
                    last_activity = job["launch_time"]
                    if os.path.exists(t["epochs_jsonl_path"]):
                        last_activity = max(last_activity, os.path.getmtime(t["epochs_jsonl_path"]))
                    idle = now - last_activity
                    if idle > cli.stall_timeout:
                        log_msg(
                            master_log,
                            f"STALL {t['exp_name']} on GPU {job['gpu']} (idle {idle:.0f}s > {cli.stall_timeout}s) — killing process group",
                        )
                        try:
                            os.killpg(job["proc"].pid, signal.SIGTERM)
                            job["proc"].wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            try:
                                os.killpg(job["proc"].pid, signal.SIGKILL)
                            except ProcessLookupError:
                                pass
                            try:
                                job["proc"].wait(timeout=10)
                            except Exception:
                                pass
                        except ProcessLookupError:
                            pass
                        ret = -99
            if ret is None:
                still_active.append(job)
                continue
            job["log_fp"].close()
            summary = load_summary_status(t["summary_path"])
            summary_fresh = bool(
                summary
                and os.path.exists(t["summary_path"])
                and os.path.getmtime(t["summary_path"]) >= job["launch_time"] - 1.0
            )
            st = summary.get("status") if summary else ("completed" if ret == 0 else "failed")
            if (ret == 0 and st == "completed") or (summary_fresh and st == "completed"):
                acc_last = summary.get("val_mean_last_k", 0.0) if summary else 0.0
                acc_max = summary.get("max_val_acc1", 0.0) if summary else 0.0
                log_msg(
                    master_log,
                    f"DONE  {t['exp_name']} on GPU {job['gpu']} | mean_last10={acc_last:.2f} max={acc_max:.2f}",
                )
                completed_list.append(t["exp_name"])
            elif st == "nan_collapsed":
                log_msg(master_log, f"NAN   {t['exp_name']} on GPU {job['gpu']} (recorded as nan_collapsed)")
                completed_list.append(t["exp_name"])
            else:
                if t["retries"] < 2:
                    t["retries"] += 1
                    log_msg(master_log, f"RETRY {t['exp_name']} after exit={ret} on GPU {job['gpu']} (attempt {t['retries'] + 1})")
                    tasks.insert(0, t)
                else:
                    log_msg(master_log, f"FAIL  {t['exp_name']} permanently after retries (exit={ret})")
                    failed_list.append(t["exp_name"])
        active_jobs = still_active

        while tasks:
            next_task = tasks[0]
            if both_prior_seeds_collapsed(next_task["arm"], next_task["seed"], seeds, summaries_dir, cli.prefix):
                tasks.pop(0)
                skipped += 1
                log_msg(master_log, f"SKIP  {next_task['exp_name']} (prior 2 seeds collapsed)")
                continue
            gpu_id, max_slots, ext_busy = pick_free_gpu(
                active_jobs,
                candidate_gpus,
                jobs_per_gpu=cli.jobs_per_gpu,
                max_gpus_override=cli.max_gpus,
            )
            if gpu_id is None:
                break
            t = tasks.pop(0)
            cmd = [
                sys.executable,
                "train.py",
                "-c",
                t["config"],
                f"--data-path={cli.data_path}",
                "--dataset=cifar10dvs",
                "--model=max_former",
                "--T=16",
                f"--batch-size={cli.batch_size}",
                f"--epochs={cli.epochs}",
                f"--workers={cli.workers}",
                "--lr=0.0045",
                "--weight-decay=0.0079",
                "--amp=True",
                f"--seed={t['seed']}",
                f"--experiment={t['exp_name']}",
                f"--summary-json={t['summary_path']}",
                "--use-custom-neuron=True",
                "--reset-mode=hard",
                "--log-cgrad-stats=True",
                f"--cgrad-stats-prob={cli.cgrad_stats_prob}",
                "--metric-window=10",
                "--early-stop-patience=-1",
                "--device=cuda:0",
            ]
            if not cli.no_wandb:
                cmd.append("--log-wandb")
            cmd.extend(t["flags"])

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            env["PYTHONUNBUFFERED"] = "1"
            if not cli.wandb_online and not cli.no_wandb:
                env["WANDB_MODE"] = "offline"

            log_fp = open(t["log_path"], "a", encoding="utf-8")
            log_fp.write(f"\n=== Launching on GPU {gpu_id} at {datetime.datetime.now().isoformat()} ===\n")
            log_fp.write("CMD: " + " ".join(cmd) + "\n")
            log_fp.flush()

            proc = subprocess.Popen(
                cmd,
                cwd=event_dir,
                env=env,
                stdout=log_fp,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            active_jobs.append({"task": t, "proc": proc, "gpu": gpu_id, "log_fp": log_fp, "launch_time": time.time()})
            log_msg(
                master_log,
                f"START {t['exp_name']} on GPU {gpu_id} (active={len(active_jobs)}/{max_slots}, ext_busy={ext_busy})",
            )

        write_json_atomic(
            status_path,
            {
                "updated_at": datetime.datetime.now().isoformat(),
                "active": [{"exp": j["task"]["exp_name"], "gpu": j["gpu"], "pid": j["proc"].pid} for j in active_jobs],
                "pending_count": len(tasks),
                "completed_count": len(completed_list) + skipped,
                "failed": failed_list,
            },
        )
        if tasks or active_jobs:
            time.sleep(15.0)

    log_msg(
        master_log,
        f"Phase 1 tuning scheduler finished! completed={len(completed_list) + skipped}, failed={len(failed_list)} {failed_list}",
    )
    if failed_list:
        sys.exit(1)


if __name__ == "__main__":
    main()
