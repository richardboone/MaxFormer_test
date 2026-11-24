#!/usr/bin/env python3
"""Aggregate and analyze Phase 1 / Phase 1B CIFAR10-DVS Improved C-Grad tuning summaries."""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


STAGE1_BASELINES = [
    ("A0_bptt_sig", "BPTT baseline (sigmoid, tau=2, coupled, detach=T)"),
    ("A2_impr_twosided_sig", "Stage 1 default Improved C-Grad (two-sided, sigmoid)"),
    ("B0_bptt_gamma", "BPTT baseline (Gamma, tau=2, coupled, detach=T)"),
    ("B2_prior_inv_cgrad", "Prior invariant_cgrad (sign+flip, h=0.24, Gamma)"),
    ("B3_impr_twosided_gamma", "Stage 1 default Improved C-Grad (two-sided, Gamma)"),
]

T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306}


def mean_sd(vals):
    n = len(vals)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(vals) / n
    if n == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return m, math.sqrt(var)


def parse_arm_from_exp(exp_name):
    for prefix in ("s1_", "t1b_", "t1_"):
        if exp_name.startswith(prefix):
            body = exp_name[len(prefix):]
            if "_seed" in body:
                return body.rsplit("_seed", 1)[0]
    if "_seed" in exp_name:
        return exp_name.rsplit("_seed", 1)[0]
    return exp_name


def load_stage1_baselines(stage1_dir):
    by_arm = defaultdict(dict)
    keep = {name for name, _ in STAGE1_BASELINES}
    for path in sorted(glob.glob(os.path.join(stage1_dir, "*.json"))):
        if path.endswith(".epochs.jsonl"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        arm = parse_arm_from_exp(rec.get("experiment", os.path.basename(path)))
        if arm in keep:
            by_arm[arm][rec.get("seed")] = rec
    return by_arm


def load_tune_dir(tune_dir):
    by_arm = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(tune_dir, "*.json"))):
        if path.endswith(".epochs.jsonl"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        arm = parse_arm_from_exp(rec.get("experiment", os.path.basename(path)))
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
    if len(d_last) >= 2:
        tcrit = T_CRIT_95.get(len(d_last) - 1, 1.96)
        ci = f"[{m_d - tcrit * se_d:+.2f}, {m_d + tcrit * se_d:+.2f}]"
    else:
        ci = "N/A"
    return len(d_last), m_d, se_d, ci, m_dm


def render_report(tune_arms, all_arms):
    def rank_key(arm):
        ok = [r for r in all_arms[arm].values() if r.get("status") == "completed"]
        return -(mean_sd([r["val_mean_last_k"] for r in ok])[0] if ok else -1.0)

    ordered = sorted(all_arms.keys(), key=rank_key)

    lines = []
    lines.append("# Improved C-Grad Phase 1 Tuning Results (CIFAR10-DVS MaxFormer)")
    lines.append("")
    lines.append("## Per-Arm Summary (Ranked by `val/mean_last10`)")
    lines.append("")
    lines.append("| Arm | Surr | n (ok/nan) | val/mean_last10 | val/max | final_acc1 | eff_rate | mean_w | m_neg | s/ep |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for arm in ordered:
        recs = list(all_arms[arm].values())
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

    lines.append("")
    lines.append("## Paired Seed Contrasts (Tuning Arms vs Matched Baselines & Best BPTT)")
    lines.append("")
    lines.append("| Tuning Arm | n_pairs | Δ vs Matched BPTT | Δ vs Stage 1 Default | Δ vs `B0_bptt_gamma` | Δ val/max (vs BPTT) |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    for arm in sorted(tune_arms.keys(), key=rank_key):
        ok_recs = [r for r in tune_arms[arm].values() if r.get("status") == "completed"]
        surr = ok_recs[0].get("args", {}).get("dS_du", "sigmoid") if ok_recs else "sigmoid"
        bptt_ref = "B0_bptt_gamma" if surr == "Gamma" else "A0_bptt_sig"
        s1_ref = "B3_impr_twosided_gamma" if surr == "Gamma" else "A2_impr_twosided_sig"

        r_bptt = paired_delta(all_arms, arm, bptt_ref)
        r_s1 = paired_delta(all_arms, arm, s1_ref)
        r_b0 = paired_delta(all_arms, arm, "B0_bptt_gamma")

        n_p = r_bptt[0] if r_bptt else 0
        s_bptt = f"{r_bptt[1]:+.2f} ({r_bptt[2]:.2f}) {r_bptt[3]}" if r_bptt else "N/A"
        s_s1 = f"{r_s1[1]:+.2f} ({r_s1[2]:.2f})" if r_s1 else "N/A"
        s_b0 = f"{r_b0[1]:+.2f} ({r_b0[2]:.2f})" if r_b0 else "N/A"
        s_max = f"{r_bptt[4]:+.2f}" if r_bptt else "N/A"
        lines.append(f"| `{arm}` | {n_p} | {s_bptt} | {s_s1} | {s_b0} | {s_max} |")

    lines.append("")
    lines.append("## Seed-Level Details (Tuning Runs)")
    lines.append("")
    lines.append("| Arm | Seed | Status | mean_last10 | val/max | final_acc1 | eff_rate | mean_w | s/ep |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|")
    for arm in sorted(tune_arms.keys(), key=rank_key):
        for seed in sorted(tune_arms[arm]):
            r = tune_arms[arm][seed]
            er = r.get("cgrad_stats_mean", {}).get("cgrad/effective_rate", 0.0) * 100.0
            mw = r.get("cgrad_stats_mean", {}).get("cgrad/mean_w", 0.0)
            lines.append(
                f"| `{arm}` | {seed} | {r.get('status')} | {r.get('val_mean_last_k', 0.0):.2f} | "
                f"{r.get('max_val_acc1', 0.0):.2f} | {r.get('final_val_acc1', 0.0):.2f} | "
                f"{er:.2f}% | {mw:.4f} | {r.get('mean_epoch_time_sec', 0.0):.1f}s |"
            )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize Phase 1 Improved C-Grad tuning results")
    parser.add_argument("--tune-dir", default="event/logs/tune_phase1/summaries")
    parser.add_argument("--stage1-dir", default="event/logs/stage1_eval/summaries")
    parser.add_argument("--write-md", default="")
    args = parser.parse_args()

    stage1_arms = load_stage1_baselines(args.stage1_dir)
    tune_arms = load_tune_dir(args.tune_dir)
    if not tune_arms:
        print(f"No tuning JSON summaries found in {args.tune_dir}")
        return

    all_arms = {}
    all_arms.update(stage1_arms)
    all_arms.update(tune_arms)

    report = render_report(tune_arms, all_arms)
    print(report)
    if args.write_md:
        parent = os.path.dirname(os.path.abspath(args.write_md))
        os.makedirs(parent, exist_ok=True)
        with open(args.write_md, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote tuning summary to {args.write_md}")


if __name__ == "__main__":
    main()
