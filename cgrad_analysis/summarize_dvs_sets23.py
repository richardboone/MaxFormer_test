#!/usr/bin/env python3
"""Summarize CIFAR10-DVS Set 2 (HP Tuning) and Set 3 (Timestep Sweep) results."""

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
EVENT_DIR = os.path.join(REPO_ROOT, "event")
if EVENT_DIR not in sys.path:
    sys.path.insert(0, EVENT_DIR)

from run_dvs_set2_tune import SET2_ARMS  # noqa: E402

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def mean_sd(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


def load_summaries(summaries_dir, prefix):
    by_arm = defaultdict(dict)
    if not os.path.isdir(summaries_dir):
        return by_arm
    for path in sorted(glob.glob(os.path.join(summaries_dir, f"{prefix}*.json"))):
        if path.endswith(".epochs.jsonl"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        exp = rec.get("experiment", os.path.basename(path)[:-5])
        body = exp[len(prefix):] if exp.startswith(prefix) else exp
        arm = body.rsplit("_seed", 1)[0] if "_seed" in body else body
        by_arm[arm][rec.get("seed")] = rec
    return by_arm


def paired_delta(by_arm, lhs, rhs):
    if lhs not in by_arm or rhs not in by_arm:
        return None
    common = sorted(set(by_arm[lhs]).intersection(by_arm[rhs]))
    d_last = []
    d_max = []
    for s in common:
        rl, rr = by_arm[lhs][s], by_arm[rhs][s]
        if rl.get("status") == "completed" and rr.get("status") == "completed":
            d_last.append(rl["val_mean_last_k"] - rr["val_mean_last_k"])
            d_max.append(rl["max_val_acc1"] - rr["max_val_acc1"])
    if not d_last:
        return None
    m_d, sd_d = mean_sd(d_last)
    se_d = sd_d / math.sqrt(len(d_last))
    m_dm, _ = mean_sd(d_max)
    wins = sum(1 for d in d_last if d > 0.0)
    if len(d_last) >= 2:
        tcrit = T_CRIT_95.get(len(d_last) - 1, 1.96)
        ci = f"[{m_d - tcrit * se_d:+.2f}, {m_d + tcrit * se_d:+.2f}]"
    else:
        ci = "N/A"
    return len(d_last), m_d, se_d, ci, m_dm, wins, d_last


def summarize_set2(set2_dir):
    by_arm = load_summaries(set2_dir, "d2_")
    if not by_arm:
        return ["## Set 2 (HP Tuning @ T=16)", "", "_No Set 2 summaries found yet._"]

    rows = []
    arm_order = [a["name"] for a in SET2_ARMS]
    extra_arms = sorted(k for k in by_arm if k not in arm_order)
    for arm in arm_order + extra_arms:
        recs = [r for r in by_arm.get(arm, {}).values() if r.get("status") == "completed"]
        n_nan = sum(1 for r in by_arm.get(arm, {}).values() if r.get("status") == "nan_collapsed")
        if not recs and not n_nan:
            continue
        if not recs:
            rows.append((-999.0, arm, None))
            continue
        m_last, sd_last = mean_sd([r["val_mean_last_k"] for r in recs])
        m_max, sd_max = mean_sd([r["max_val_acc1"] for r in recs])
        m_ep, _ = mean_sd([r.get("mean_epoch_time_sec", 0.0) for r in recs])
        m_eff, _ = mean_sd([r.get("cgrad_stats_mean", {}).get("cgrad/effective_rate", 0.0) * 100.0 for r in recs])
        m_near, _ = mean_sd([r.get("cgrad_stats_mean", {}).get("cgrad/near_threshold_rate", 0.0) * 100.0 for r in recs])
        m_mneg, _ = mean_sd([r.get("cgrad_stats_mean", {}).get("cgrad/frac_effective_m_neg", 0.0) * 100.0 for r in recs])
        d_bptt = paired_delta(by_arm, arm, "S2_A0_bptt")
        d_paper = paired_delta(by_arm, arm, "S2_T8_paper")
        rows.append((
            m_last,
            arm,
            {
                "n": len(recs),
                "nan": n_nan,
                "m_last": m_last,
                "sd_last": sd_last,
                "m_max": m_max,
                "sd_max": sd_max,
                "d_bptt": d_bptt[1] if d_bptt else float("nan"),
                "d_paper": d_paper[1] if d_paper else float("nan"),
                "eff": m_eff,
                "near": m_near,
                "mneg": m_mneg,
                "ep_s": m_ep,
            },
        ))

    rows.sort(key=lambda x: x[0], reverse=True)
    lines = [
        "## Set 2: CIFAR10-DVS Targeted HP Tuning (T=16, 50+10 ep, paper lr=0.005/wd=0.06 anchor)",
        "",
        "| Rank | Arm | n (nan) | mean_last10 ± SD | max_val ± SD | Δ vs BPTT | Δ vs T8_paper | eff_rate% | near% | m_neg% | s/ep |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, (_, arm, info) in enumerate(rows, 1):
        if info is None:
            lines.append(f"| {idx} | `{arm}` | 0 (all NaN) | — | — | — | — | — | — | — | — |")
            continue
        db = f"{info['d_bptt']:+.2f}" if not math.isnan(info["d_bptt"]) else "—"
        dp = f"{info['d_paper']:+.2f}" if not math.isnan(info["d_paper"]) else "—"
        lines.append(
            f"| {idx} | `{arm}` | {info['n']} ({info['nan']}) | {info['m_last']:.2f} ± {info['sd_last']:.2f} | "
            f"{info['m_max']:.2f} ± {info['sd_max']:.2f} | {db} | {dp} | {info['eff']:.2f}% | "
            f"{info['near']:.1f}% | {info['mneg']:.1f}% | {info['ep_s']:.1f} |"
        )

    contrasts = [
        ("S2_lr003_coupled", "S2_lr003_fixed", "Coupled (eps∝lr, beta∝1/lr) vs Fixed gates at lr=0.003"),
        ("S2_lr007_coupled", "S2_lr007_fixed", "Coupled (eps∝lr, beta∝1/lr) vs Fixed gates at lr=0.007"),
        ("S2_T8_paper", "S2_A0_bptt", "T8_gentle vs BPTT under paper lr=0.005, wd=0.06"),
    ]
    c_rows = []
    for lhs, rhs, desc in contrasts:
        res = paired_delta(by_arm, lhs, rhs)
        if res:
            n_p, m_d, se_d, ci, m_dm, wins, _ = res
            c_rows.append(
                f"| `{lhs} - {rhs}` | {desc} | {n_p} | {m_d:+.2f} ({se_d:.2f}) | {ci} | {m_dm:+.2f} | {wins}/{n_p} |"
            )
    if c_rows:
        lines.extend([
            "",
            "### Key Set 2 Paired Contrasts",
            "",
            "| Contrast | Hypothesis | n | Δ mean_last10 (SE) | 95% CI | Δ val/max | wins |",
            "|---|---|---:|---:|---:|---:|---:|",
        ])
        lines.extend(c_rows)
    return lines


def summarize_set3(set3_dir):
    by_arm = load_summaries(set3_dir, "d3_")
    if not by_arm:
        return ["## Set 3 (Timestep Sweep)", "", "_No Set 3 summaries found yet._"]

    timesteps = sorted({
        int(arm.split("_")[1][1:])
        for arm in by_arm
        if arm.startswith("S3_T") and "_" in arm
    })
    lines = [
        "## Set 3: CIFAR10-DVS Timestep Sweep (BPTT vs T8_gentle C-Grad, 50+10 ep, lr=0.005/wd=0.06)",
        "",
        "| T | n_pairs | BPTT mean_last10 ± SD | C-Grad mean_last10 ± SD | Paired Δ(T) ± SD | 95% CI | wins | BPTT max | C-Grad max | eff_rate(T)% | s/ep (BPTT / CG) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for T in timesteps:
        b_arm = f"S3_T{T}_bptt"
        c_arm = f"S3_T{T}_cgrad"
        b_recs = [r for r in by_arm.get(b_arm, {}).values() if r.get("status") == "completed"]
        c_recs = [r for r in by_arm.get(c_arm, {}).values() if r.get("status") == "completed"]
        bm, bsd = mean_sd([r["val_mean_last_k"] for r in b_recs])
        cm, csd = mean_sd([r["val_mean_last_k"] for r in c_recs])
        bmax, _ = mean_sd([r["max_val_acc1"] for r in b_recs])
        cmax, _ = mean_sd([r["max_val_acc1"] for r in c_recs])
        bs, _ = mean_sd([r.get("mean_epoch_time_sec", 0.0) for r in b_recs])
        cs, _ = mean_sd([r.get("mean_epoch_time_sec", 0.0) for r in c_recs])
        ceff, _ = mean_sd([r.get("cgrad_stats_mean", {}).get("cgrad/effective_rate", 0.0) * 100.0 for r in c_recs])
        res = paired_delta(by_arm, c_arm, b_arm)
        if res:
            n_p, m_d, se_d, ci, _, wins, d_list = res
            _, sd_d = mean_sd(d_list)
            d_str = f"{m_d:+.2f} ± {sd_d:.2f}"
            w_str = f"{wins}/{n_p}"
        else:
            n_p = min(len(b_recs), len(c_recs))
            d_str = "—"
            ci = "—"
            w_str = "—"
        b_str = f"{bm:.2f} ± {bsd:.2f}" if b_recs else "—"
        c_str = f"{cm:.2f} ± {csd:.2f}" if c_recs else "—"
        lines.append(
            f"| {T} | {n_p} | {b_str} | {c_str} | {d_str} | {ci} | {w_str} | "
            f"{bmax:.2f} | {cmax:.2f} | {ceff:.2f}% | {bs:.1f} / {cs:.1f} |"
        )
    return lines


def main():
    parser = argparse.ArgumentParser(description="Summarize CIFAR10-DVS Set 2 and Set 3 results")
    parser.add_argument("--set2-dir", default="/home/rboone/work/MaxFormer_test/event/logs/dvs_set2_tune/summaries")
    parser.add_argument("--set3-dir", default="/home/rboone/work/MaxFormer_test/event/logs/dvs_set3_tsweep/summaries")
    parser.add_argument("--out-md", default="")
    args = parser.parse_args()

    out_lines = []
    out_lines.extend(summarize_set2(args.set2_dir))
    out_lines.append("")
    out_lines.extend(summarize_set3(args.set3_dir))
    report = "\n".join(out_lines) + "\n"
    print(report)

    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w", encoding="utf-8") as f:
            f.write(report)


if __name__ == "__main__":
    main()
