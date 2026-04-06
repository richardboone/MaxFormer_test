"""
Grid Benchmark: Backward-pass overhead of conservative_cgrad vs standard LIF.

Sweeps over T, batch-size, and channel dimensions, then reports:
  • Full results table
  • Best-case  scenario (smallest overhead %)
  • Worst-case scenario (largest  overhead %)

Configurations compared per grid point
---------------------------------------
  Baseline 1 — dS_du=sigmoid, du_du=LIF,               detach_reset=False
  Baseline 2 — dS_du=Gamma,   du_du=LIF,               detach_reset=False
  Test 1     — dS_du=sigmoid, du_du=conservative_cgrad, detach_reset=False
  Test 2     — dS_du=Gamma,   du_du=conservative_cgrad, detach_reset=False

Usage
-----
    conda run -n snn-maxformer python benchmark_grid.py
    conda run -n snn-maxformer python benchmark_grid.py --device cuda
    conda run -n snn-maxformer python benchmark_grid.py --warmup 20 --trials 100
"""

import argparse
import itertools
import statistics
import time
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_neuron import LIFSpikeLayer_Cons


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_args(dS_du: str, du_du: str, detach_reset: bool = False):
    return argparse.Namespace(
        dS_du=dS_du,
        du_du=du_du,
        surrogate_alpha=4.0,
        use_custom_neuron=True,
        detach_reset=detach_reset,
        snnbp_epsilon=0.3,
        snnbp_alpha=2.0,
        snnbp_beta=2.0,
        snnbp_intervention=0.8,
    )


CONFIG_SPECS = {
    "sigmoid+LIF":              make_args("sigmoid", "LIF"),
    "Gamma+LIF":                make_args("Gamma",   "LIF"),
    "sigmoid+conservative":     make_args("sigmoid", "conservative_cgrad"),
    "Gamma+conservative":       make_args("Gamma",   "conservative_cgrad"),
}

BASELINE_KEYS = ["sigmoid+LIF", "Gamma+LIF"]
TEST_KEYS     = ["sigmoid+conservative", "Gamma+conservative"]


def time_backward(layer, x_template, warmup, trials, device):
    durations = []
    use_cuda = (device == "cuda" or
                (isinstance(device, torch.device) and device.type == "cuda"))
    for i in range(warmup + trials):
        inp = x_template.clone().detach().requires_grad_(True)
        out = layer(inp)
        loss = out.sum()
        if use_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss.backward()
        if use_cuda:
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        if i >= warmup:
            durations.append(t1 - t0)
    return durations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Grid benchmark for LIF backward pass.")
    parser.add_argument("--device",  default="cpu",  help="cpu or cuda")
    parser.add_argument("--warmup",  type=int, default=10)
    parser.add_argument("--trials",  type=int, default=50)
    parser.add_argument("--tau",     type=float, default=2.0)
    cli = parser.parse_args()

    device = torch.device(cli.device)

    # ---- Grid axes ----
    T_values = [4, 16, 64]
    B_values = [8, 16, 32]
    C_values = [64, 128, 256, 512]

    total_combos = len(T_values) * len(B_values) * len(C_values)

    print(f"\n{'=' * 80}")
    print(f" Grid Backward-Pass Benchmark   device={device}")
    print(f" T  ∈ {T_values}")
    print(f" B  ∈ {B_values}")
    print(f" C  ∈ {C_values}")
    print(f" Warm-up: {cli.warmup}   Trials: {cli.trials}")
    print(f" Total grid points: {total_combos}")
    print(f"{'=' * 80}\n")

    # ---- Storage ----
    # Each entry: {config_name: mean_ms}
    all_results = []  # list of dicts with keys: T, B, C, config, mean_ms, std_ms

    overhead_records = []  # (overhead_pct, T, B, C, baseline_name, test_name, base_ms, test_ms)

    combo_idx = 0
    for T, B, C in itertools.product(T_values, B_values, C_values):
        combo_idx += 1
        x_template = torch.randn(T, B, C, device=device)

        print(f"[{combo_idx}/{total_combos}]  T={T:3d}  B={B:3d}  C={C:4d}")

        timings = {}  # config_name -> mean_ms

        for cfg_name, cfg_args in CONFIG_SPECS.items():
            layer = LIFSpikeLayer_Cons(
                thresh=1.0, tau=cli.tau, gama=1.0,
                detach_reset=cfg_args.detach_reset, args=cfg_args,
            ).to(device)

            durs = time_backward(layer, x_template, cli.warmup, cli.trials, device)
            mean_ms = statistics.mean(durs) * 1000
            std_ms  = statistics.stdev(durs) * 1000 if len(durs) > 1 else 0.0

            timings[cfg_name] = mean_ms

            all_results.append(dict(
                T=T, B=B, C=C,
                config=cfg_name,
                mean_ms=mean_ms,
                std_ms=std_ms,
            ))

        # Compute overhead for each (baseline, test) pair
        for bk in BASELINE_KEYS:
            for tk in TEST_KEYS:
                base_ms = timings[bk]
                test_ms = timings[tk]
                overhead_pct = ((test_ms - base_ms) / base_ms) * 100
                overhead_records.append((
                    overhead_pct, T, B, C, bk, tk, base_ms, test_ms
                ))

        # Short inline summary
        for tk in TEST_KEYS:
            for bk in BASELINE_KEYS:
                oh = ((timings[tk] - timings[bk]) / timings[bk]) * 100
                print(f"    {tk:30s} vs {bk:15s}  →  {oh:+7.1f}% overhead")
        print()

    # ==================================================================
    # Summary Table
    # ==================================================================
    print(f"\n{'=' * 80}")
    print(" FULL RESULTS TABLE")
    print(f"{'=' * 80}")
    hdr = f"{'T':>4s} {'B':>4s} {'C':>5s} | {'sigmoid+LIF':>12s} {'Gamma+LIF':>12s} | {'sig+cons':>12s} {'Gam+cons':>12s} | {'OH(sig)':>8s} {'OH(Gam)':>8s}"
    print(hdr)
    print("─" * len(hdr))

    for T, B, C in itertools.product(T_values, B_values, C_values):
        row = {r["config"]: r["mean_ms"]
               for r in all_results
               if r["T"] == T and r["B"] == B and r["C"] == C}

        oh_sig = ((row["sigmoid+conservative"] - row["sigmoid+LIF"]) / row["sigmoid+LIF"]) * 100
        oh_gam = ((row["Gamma+conservative"]   - row["Gamma+LIF"])   / row["Gamma+LIF"])   * 100

        print(f"{T:4d} {B:4d} {C:5d} | "
              f"{row['sigmoid+LIF']:11.3f}ms {row['Gamma+LIF']:11.3f}ms | "
              f"{row['sigmoid+conservative']:11.3f}ms {row['Gamma+conservative']:11.3f}ms | "
              f"{oh_sig:+7.1f}% {oh_gam:+7.1f}%")

    # ==================================================================
    # Best / Worst
    # ==================================================================
    overhead_records.sort(key=lambda x: x[0])

    print(f"\n{'=' * 80}")
    print(" BEST-CASE SCENARIO  (smallest overhead)")
    print(f"{'=' * 80}")
    for rec in overhead_records[:3]:
        oh, T, B, C, bk, tk, bms, tms = rec
        print(f"  {oh:+7.1f}%  T={T:3d} B={B:3d} C={C:4d}  {tk} vs {bk}  "
              f"({bms:.3f}ms → {tms:.3f}ms)")

    print(f"\n{'=' * 80}")
    print(" WORST-CASE SCENARIO  (largest overhead)")
    print(f"{'=' * 80}")
    for rec in overhead_records[-3:]:
        oh, T, B, C, bk, tk, bms, tms = rec
        print(f"  {oh:+7.1f}%  T={T:3d} B={B:3d} C={C:4d}  {tk} vs {bk}  "
              f"({bms:.3f}ms → {tms:.3f}ms)")

    # ==================================================================
    # Quick statistics
    # ==================================================================
    all_oh = [r[0] for r in overhead_records]
    print(f"\n{'=' * 80}")
    print(" OVERHEAD STATISTICS  (across all grid points & pairings)")
    print(f"{'=' * 80}")
    print(f"  Mean overhead:   {statistics.mean(all_oh):+.1f}%")
    print(f"  Median overhead: {statistics.median(all_oh):+.1f}%")
    print(f"  Min overhead:    {min(all_oh):+.1f}%")
    print(f"  Max overhead:    {max(all_oh):+.1f}%")
    print(f"  Std deviation:   {statistics.stdev(all_oh):.1f}%")
    print()


if __name__ == "__main__":
    main()
