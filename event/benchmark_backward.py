"""
Benchmark: Backward-pass wall-clock time for a single LIF neuron layer.

Configurations compared
-----------------------
Baseline 1 — dS_du=sigmoid,  du_du=LIF (default else branch), detach_reset=False
Baseline 2 — dS_du=Gamma,    du_du=LIF (default else branch), detach_reset=False
Test       — dS_du=sigmoid,  du_du=conservative_cgrad,        detach_reset=False
Test       — dS_du=Gamma,    du_du=conservative_cgrad,        detach_reset=False

Usage
-----
    python benchmark_backward.py                          # uses defaults
    python benchmark_backward.py --T 16 --batch 64 --C 128 --warmup 20 --trials 100 --device cuda
"""

import argparse
import time
import statistics
import torch
import sys, os

# Make sure we can import the custom neuron from the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from custom_neuron import LIFSpikeLayer_Cons


# ---------------------------------------------------------------------------
# Lightweight args namespace used to configure the neuron backward pass
# ---------------------------------------------------------------------------
def make_args(dS_du: str, du_du: str, detach_reset: bool = False, **extra):
    """Build a minimal namespace that the backward function reads via getattr."""
    d = dict(
        dS_du=dS_du,
        du_du=du_du,
        surrogate_alpha=4.0,
        use_custom_neuron=True,
        detach_reset=detach_reset,
        # conservative_cgrad defaults
        snnbp_epsilon=0.3,
        snnbp_alpha=2.0,
        snnbp_beta=2.0,
        snnbp_intervention=0.8,
    )
    d.update(extra)
    return argparse.Namespace(**d)


# ---------------------------------------------------------------------------
# Core timing helper
# ---------------------------------------------------------------------------
def time_backward(neuron_layer, x, num_warmup=10, num_trials=50, device="cpu"):
    """Return a list of backward-pass durations (seconds) for *num_trials* runs.

    Each trial:
      1. Re-creates a fresh input (clone + requires_grad)
      2. Forward pass  (not timed)
      3. Creates a dummy loss  (not timed)
      4. Backward pass (** timed **)
    """
    durations = []

    for i in range(num_warmup + num_trials):
        inp = x.clone().detach().requires_grad_(True)
        out = neuron_layer(inp)
        loss = out.sum()

        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        loss.backward()
        if device == "cuda" or (isinstance(device, torch.device) and device.type == "cuda"):
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        if i >= num_warmup:
            durations.append(t1 - t0)

    return durations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Benchmark LIF neuron backward pass.")
    parser.add_argument("--T", type=int, default=8, help="Number of timesteps")
    parser.add_argument("--batch", type=int, default=32, help="Batch size")
    parser.add_argument("--C", type=int, default=256, help="Channel / spatial dimension")
    parser.add_argument("--warmup", type=int, default=10, help="Warm-up iterations (not timed)")
    parser.add_argument("--trials", type=int, default=50, help="Timed iterations")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--tau", type=float, default=2.0, help="LIF time constant tau")
    cli = parser.parse_args()

    device = torch.device(cli.device)
    print(f"\n{'=' * 70}")
    print(f" Backward-pass benchmark  —  device={device}")
    print(f" Input shape: [{cli.T}, {cli.batch}, {cli.C}]")
    print(f" Warm-up: {cli.warmup}   Trials: {cli.trials}")
    print(f"{'=' * 70}\n")

    # Shared random input tensor (same data for every config)
    x_template = torch.randn(cli.T, cli.batch, cli.C, device=device)

    # ---- Configurations to benchmark ----
    configs = {
        "Baseline: sigmoid + LIF (no detach)": make_args(
            dS_du="sigmoid", du_du="LIF", detach_reset=False
        ),
        "Baseline: Gamma  + LIF (no detach)": make_args(
            dS_du="Gamma", du_du="LIF", detach_reset=False
        ),
        "Test:     sigmoid + conservative_cgrad": make_args(
            dS_du="sigmoid", du_du="conservative_cgrad", detach_reset=False
        ),
        "Test:     Gamma  + conservative_cgrad": make_args(
            dS_du="Gamma", du_du="conservative_cgrad", detach_reset=False
        ),
    }

    results = {}

    for label, args in configs.items():
        layer = LIFSpikeLayer_Cons(
            thresh=1.0, tau=cli.tau, gama=1.0,
            detach_reset=args.detach_reset, args=args,
        )
        layer.to(device)

        durations = time_backward(
            layer, x_template,
            num_warmup=cli.warmup,
            num_trials=cli.trials,
            device=device,
        )

        mean_ms = statistics.mean(durations) * 1000
        std_ms = statistics.stdev(durations) * 1000 if len(durations) > 1 else 0.0
        med_ms = statistics.median(durations) * 1000
        results[label] = (mean_ms, std_ms, med_ms)

        print(f"  {label}")
        print(f"      mean = {mean_ms:8.3f} ms   std = {std_ms:6.3f} ms   median = {med_ms:8.3f} ms\n")

    # ---- Comparison summary ----
    # Use first baseline (sigmoid + LIF) as the reference
    ref_label = "Baseline: sigmoid + LIF (no detach)"
    ref_mean = results[ref_label][0]

    print(f"{'─' * 70}")
    print(f" Overhead relative to  « {ref_label} »")
    print(f"{'─' * 70}")
    for label, (mean_ms, std_ms, med_ms) in results.items():
        overhead_pct = ((mean_ms - ref_mean) / ref_mean) * 100
        sign = "+" if overhead_pct >= 0 else ""
        print(f"  {label:50s}  {sign}{overhead_pct:6.1f}%")

    print()


if __name__ == "__main__":
    main()
