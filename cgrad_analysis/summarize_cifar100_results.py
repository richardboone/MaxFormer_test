#!/usr/bin/env python3
"""Aggregate and analyze Stage 2 CIFAR-100 Improved C-Grad evaluation summaries."""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


CONTRASTS = [
    ("C100_A1_T8_gentle", "C100_A0_bptt_sig", "Improved C-Grad smooth (T8) vs BPTT [sigmoid]"),
    ("C100_B1_G3_flip", "C100_B0_bptt_gamma", "Improved C-Grad gated flip (G3) vs BPTT [Gamma]"),
    ("C100_A1_T8_gentle", "C100_B0_bptt_gamma", "Improved C-Grad smooth (T8) vs Gamma BPTT"),
    ("C100_A1_T8_gentle", "C100_B1_G3_flip", "Smooth sigmoid (T8) vs Gated flip Gamma (G3)"),
]

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def mean_sd(vals):
    """Compute sample mean and SD."""
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


def parse_arm_from_exp(exp_name, prefix="c100_"):
    """Strip run prefix and _seedN suffix."""
    body = exp_name[len(prefix):] if prefix and exp_name.startswith(prefix) else exp_name
    if "_seed" in body:
        return body.rsplit("_seed", 1)[0]
    return body


def load_summaries(summary_dir, prefix="c100_"):
    """Load completed/nan summaries indexed by {arm: {seed: rec}}."""
    by_arm = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(summary_dir, f"{prefix}*.json"))):
        if path.endswith(".epochs.jsonl"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        arm = parse_arm_from_exp(rec.get("experiment", os.path.basename(path)), prefix=prefix)
        by_arm[arm][rec.get("seed")] = rec
    return by_arm


def paired_delta(by_arm, lhs, rhs):
    """Compute paired-seed difference statistics for lhs minus rhs."""
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
    return len(d_last), m_d, se_d, ci, m_dm, wins


def render_report(by_arm, time_step=4):
    """Render Markdown summary report."""
    def rank_key(arm):
        ok = [r for r in by_arm[arm].values() if r.get("status") == "completed"]
        return -(mean_sd([r["val_mean_last_k"] for r in ok])[0] if ok else -1.0)

    ordered = sorted(by_arm.keys(), key=rank_key)
    lines = [
        f"# Stage 2 CIFAR-100 MaxFormer Results (`T={time_step}`, `400+10` Epochs)",
        "",
        "## Per-Arm Summary (Ranked by `val/mean_last10`)",
        "",
        "| Arm | Surr | n (ok/nan) | val/mean_last10 | val/max | final_acc1 | eff_rate | mean_w | m_neg | s/ep |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in ordered:
        recs = list(by_arm[arm].values())
        ok_recs = [r for r in recs if r.get("status") == "completed"]
        nan_cnt = sum(1 for r in recs if r.get("status") == "nan_collapsed")
        surr = ok_recs[0].get("args", {}).get("dS_du", "?") if ok_recs else recs[0].get("args", {}).get("dS_du", "?")
        if ok_recs:
            m_last, sd_last = mean_sd([r["val_mean_last_k"] for r in ok_recs])
            m_max, sd_max = mean_sd([r["max_val_acc1"] for r in ok_recs])
            m_fin, _ = mean_sd([r["final_val_acc1"] for r in ok_recs])
            m_ep, _ = mean_sd([r["mean_epoch_time_sec"] for r in ok_recs])
            er, _ = mean_sd([r.get("cgrad_stats_mean", {}).get("cgrad/effective_rate", 0.0) * 100.0 for r in ok_recs])
            mw, _ = mean_sd([r.get("cgrad_stats_mean", {}).get("cgrad/mean_w", 0.0) for r in ok_recs])
            mn, _ = mean_sd([r.get("cgrad_stats_mean", {}).get("cgrad/frac_effective_m_neg", 0.0) * 100.0 for r in ok_recs])
            lines.append(
                f"| `{arm}` | {surr} | {len(ok_recs)}/{nan_cnt} | {m_last:.2f} ± {sd_last:.2f} | "
                f"{m_max:.2f} ± {sd_max:.2f} | {m_fin:.2f} | {er:.2f}% | {mw:.4f} | {mn:.1f}% | {m_ep:.1f}s |"
            )
        else:
            lines.append(f"| `{arm}` | {surr} | 0/{nan_cnt} | COLLAPSED | COLLAPSED | - | - | - | - | - |")

    lines.extend([
        "",
        "## Paired Seed Contrasts",
        "",
        "| Contrast | Question | n_pairs | Δ mean_last10 (SE) | 95% CI | Δ val/max | wins |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for lhs, rhs, desc in CONTRASTS:
        res = paired_delta(by_arm, lhs, rhs)
        if not res:
            continue
        n_p, m_d, se_d, ci, m_dm, wins = res
        lines.append(f"| `{lhs} - {rhs}` | {desc} | {n_p} | {m_d:+.2f} ({se_d:.2f}) | {ci} | {m_dm:+.2f} | {wins}/{n_p} |")

    lines.extend([
        "",
        "## Seed-Level Details",
        "",
        "| Arm | Seed | Status | mean_last10 | val/max | final_acc1 | eff_rate | mean_w | s/ep |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ])
    for arm in ordered:
        for seed in sorted(by_arm[arm]):
            r = by_arm[arm][seed]
            er = r.get("cgrad_stats_mean", {}).get("cgrad/effective_rate", 0.0) * 100.0
            mw = r.get("cgrad_stats_mean", {}).get("cgrad/mean_w", 0.0)
            lines.append(
                f"| `{arm}` | {seed} | {r.get('status')} | {r.get('val_mean_last_k', 0.0):.2f} | "
                f"{r.get('max_val_acc1', 0.0):.2f} | {r.get('final_val_acc1', 0.0):.2f} | "
                f"{er:.2f}% | {mw:.4f} | {r.get('mean_epoch_time_sec', 0.0):.1f}s |"
            )
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize Stage 2 CIFAR-100 evaluation results")
    parser.add_argument("--summary-dir", default="cifar10-100/logs/stage2_cifar100/summaries")
    parser.add_argument("--prefix", default="c100_")
    parser.add_argument("--time-step", type=int, default=4)
    parser.add_argument("--write-md", default="")
    args = parser.parse_args()

    by_arm = load_summaries(args.summary_dir, prefix=args.prefix)
    if not by_arm:
        print(f"No CIFAR-100 summaries found in {args.summary_dir}")
        return
    report = render_report(by_arm, time_step=args.time_step)
    print(report)
    if args.write_md:
        parent = os.path.dirname(os.path.abspath(args.write_md))
        os.makedirs(parent, exist_ok=True)
        with open(args.write_md, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote CIFAR-100 summary to {args.write_md}")


if __name__ == "__main__":
    main()
