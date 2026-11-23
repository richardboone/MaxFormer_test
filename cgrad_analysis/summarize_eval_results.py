"""Aggregate and analyze Stage 1 CIFAR10-DVS evaluation JSON summaries."""

import argparse
import glob
import json
import math
import os
from collections import defaultdict


ARM_ORDER = [
    "A0_bptt_sig",
    "A1_impr_harmful_sig",
    "A2_impr_twosided_sig",
    "B0_bptt_gamma",
    "B1_impr_harmful_gamma",
    "B2_prior_inv_cgrad",
    "C0_bptt_undetached",
    "C1_impr_undetached",
    "D0_bptt_tau20_scale1",
    "D1_impr_tau20_scale1",
]

ARM_DESCRIPTIONS = {
    "A0_bptt_sig": "Matched BPTT (eta=0, sigmoid, tau=2, coupled, detach=T)",
    "A1_impr_harmful_sig": "Improved C-Grad (harmful_only=T, sigmoid, tau=2, coupled, detach=T)",
    "A2_impr_twosided_sig": "Improved C-Grad (harmful_only=F, sigmoid, tau=2, coupled, detach=T)",
    "B0_bptt_gamma": "Matched BPTT (eta=0, Gamma, tau=2, coupled, detach=T)",
    "B1_impr_harmful_gamma": "Improved C-Grad (harmful_only=T, Gamma, tau=2, coupled, detach=T)",
    "B2_prior_inv_cgrad": "Prior invariant_cgrad (sign+flip, h=0.24, Gamma, tau=2, detach=T)",
    "C0_bptt_undetached": "Matched BPTT (eta=0, sigmoid, tau=2, coupled, detach=F)",
    "C1_impr_undetached": "Improved C-Grad (harmful_only=T, sigmoid, tau=2, coupled, detach=F)",
    "D0_bptt_tau20_scale1": "Matched BPTT (eta=0, sigmoid, tau=20, scale=1.0, detach=F)",
    "D1_impr_tau20_scale1": "Improved C-Grad (harmful_only=T, sigmoid, tau=20, scale=1.0, detach=F)",
}

PAIRED_CONTRASTS = [
    ("A1_impr_harmful_sig", "A0_bptt_sig", "Improved C-Grad (harmful) vs Matched BPTT [sigmoid]"),
    ("A2_impr_twosided_sig", "A0_bptt_sig", "Improved C-Grad (two-sided) vs Matched BPTT [sigmoid]"),
    ("A2_impr_twosided_sig", "A1_impr_harmful_sig", "Two-sided vs Harmful-only gating [sigmoid]"),
    ("B1_impr_harmful_gamma", "B0_bptt_gamma", "Improved C-Grad (harmful) vs Matched BPTT [Gamma]"),
    ("B1_impr_harmful_gamma", "B2_prior_inv_cgrad", "Improved C-Grad vs Prior invariant_cgrad [Gamma]"),
    ("C0_bptt_undetached", "A0_bptt_sig", "detach_reset=False vs True [BPTT, sigmoid]"),
    ("C1_impr_undetached", "C0_bptt_undetached", "Improved C-Grad vs Matched BPTT [detach_reset=False]"),
    ("D0_bptt_tau20_scale1", "C0_bptt_undetached", "tau=20/scale=1.0 vs tau=2/coupled [BPTT, detach=F]"),
    ("D1_impr_tau20_scale1", "D0_bptt_tau20_scale1", "Improved C-Grad vs Matched BPTT [tau=20, scale=1.0]"),
    ("A0_bptt_sig", "B0_bptt_gamma", "Surrogate sigmoid vs Gamma [BPTT]"),
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
    if exp_name.startswith("s1_"):
        body = exp_name[3:]
        if "_seed" in body:
            return body.rsplit("_seed", 1)[0]
    return exp_name


def load_summaries(summary_dir):
    by_arm = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(summary_dir, "*.json"))):
        if path.endswith(".epochs.jsonl"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rec = json.load(f)
        arm = parse_arm_from_exp(rec.get("experiment", os.path.basename(path)))
        seed = rec.get("seed")
        by_arm[arm][seed] = rec
    return by_arm


def render_report(by_arm):
    lines = []
    lines.append("# Stage 1 CIFAR10-DVS Evaluation Summary")
    lines.append("")
    lines.append("## Per-Arm Results (Mean ± SD across matched seeds)")
    lines.append("")
    lines.append(
        "| Arm | n (ok/nan) | val/mean_last10 | val/max | final_acc1 | eff_rate | mean_w | frac_eff_m_neg | s/ep |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    ordered_arms = [a for a in ARM_ORDER if a in by_arm] + [a for a in sorted(by_arm) if a not in ARM_ORDER]
    for arm in ordered_arms:
        recs = list(by_arm[arm].values())
        ok_recs = [r for r in recs if r.get("status") == "completed"]
        nan_cnt = sum(1 for r in recs if r.get("status") == "nan_collapsed")
        m_last, sd_last = mean_sd([r["val_mean_last_k"] for r in ok_recs])
        m_max, sd_max = mean_sd([r["max_val_acc1"] for r in ok_recs])
        m_fin, _ = mean_sd([r["final_val_acc1"] for r in ok_recs])
        m_ep, _ = mean_sd([r["mean_epoch_time_sec"] for r in ok_recs])

        eff_rates = [r.get("cgrad_stats_mean", {}).get("cgrad/effective_rate", 0.0) * 100.0 for r in ok_recs]
        mean_ws = [r.get("cgrad_stats_mean", {}).get("cgrad/mean_w", 0.0) for r in ok_recs]
        m_negs = [r.get("cgrad_stats_mean", {}).get("cgrad/frac_effective_m_neg", 0.0) * 100.0 for r in ok_recs]
        er, _ = mean_sd(eff_rates)
        mw, _ = mean_sd(mean_ws)
        mn, _ = mean_sd(m_negs)

        last_str = f"{m_last:.2f} ± {sd_last:.2f}" if ok_recs else "COLLAPSED"
        max_str = f"{m_max:.2f} ± {sd_max:.2f}" if ok_recs else "COLLAPSED"
        fin_str = f"{m_fin:.2f}" if ok_recs else "-"
        lines.append(
            f"| `{arm}` | {len(ok_recs)}/{nan_cnt} | {last_str} | {max_str} | {fin_str} | "
            f"{er:.2f}% | {mw:.4f} | {mn:.1f}% | {m_ep:.1f}s |"
        )

    lines.append("")
    lines.append("## Paired Seed Contrasts (`val/mean_last10` and `val/max`)")
    lines.append("")
    lines.append("| Contrast | Question | n_pairs | Δ mean_last10 (SE) | 95% CI | Δ val/max |")
    lines.append("|---|---|---:|---:|---:|---:|")

    for a_lhs, a_rhs, desc in PAIRED_CONTRASTS:
        if a_lhs not in by_arm or a_rhs not in by_arm:
            continue
        common_seeds = sorted(set(by_arm[a_lhs]).intersection(by_arm[a_rhs]))
        d_last = []
        d_max = []
        for s in common_seeds:
            rl, rr = by_arm[a_lhs][s], by_arm[a_rhs][s]
            if rl.get("status") == "completed" and rr.get("status") == "completed":
                d_last.append(rl["val_mean_last_k"] - rr["val_mean_last_k"])
                d_max.append(rl["max_val_acc1"] - rr["max_val_acc1"])
        if not d_last:
            lines.append(f"| `{a_lhs} - {a_rhs}` | {desc} | 0 | N/A | N/A | N/A |")
            continue
        m_d, sd_d = mean_sd(d_last)
        se_d = sd_d / math.sqrt(len(d_last))
        m_dm, _ = mean_sd(d_max)
        if len(d_last) >= 2:
            tcrit = T_CRIT_95.get(len(d_last) - 1, 1.96)
            ci_str = f"[{m_d - tcrit * se_d:+.2f}, {m_d + tcrit * se_d:+.2f}]"
        else:
            ci_str = "N/A (n=1)"
        lines.append(
            f"| `{a_lhs} - {a_rhs}` | {desc} | {len(d_last)} | {m_d:+.2f} ({se_d:.2f}) | {ci_str} | {m_dm:+.2f} |"
        )

    lines.append("")
    lines.append("## Seed-Level Details")
    lines.append("")
    lines.append("| Arm | Seed | Status | mean_last10 | val/max | final_acc1 | eff_rate | s/ep |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|")
    for arm in ordered_arms:
        for seed in sorted(by_arm[arm]):
            r = by_arm[arm][seed]
            er = r.get("cgrad_stats_mean", {}).get("cgrad/effective_rate", 0.0) * 100.0
            lines.append(
                f"| `{arm}` | {seed} | {r.get('status')} | {r.get('val_mean_last_k', 0.0):.2f} | "
                f"{r.get('max_val_acc1', 0.0):.2f} | {r.get('final_val_acc1', 0.0):.2f} | "
                f"{er:.2f}% | {r.get('mean_epoch_time_sec', 0.0):.1f}s |"
            )

    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Summarize Stage 1 evaluation JSONs")
    parser.add_argument("--summary-dir", default="event/logs/stage1_eval/summaries")
    parser.add_argument("--write-md", default="")
    args = parser.parse_args()

    by_arm = load_summaries(args.summary_dir)
    if not by_arm:
        print(f"No summary JSON files found in {args.summary_dir}")
        return
    report = render_report(by_arm)
    print(report)
    if args.write_md:
        parent = os.path.dirname(os.path.abspath(args.write_md))
        os.makedirs(parent, exist_ok=True)
        with open(args.write_md, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Wrote summary markdown to {args.write_md}")


if __name__ == "__main__":
    main()
