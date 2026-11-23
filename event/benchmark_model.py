"""
End-to-End Model Training Step Benchmark for MaxFormer
======================================================
Measures full training loop throughput (Forward + Loss + Backward + Step)
on realistic synthetic CIFAR10-DVS batches (T=16, Batch=16, Channels=2, H=128, W=128).

Compares:
  1. PyTorch Baseline:  Gamma + LIF
  2. PyTorch C-Grad:    Gamma + conservative
  3. PyTorch C-Grad:    Gamma + invariant
  4. Triton Baseline:   Gamma + LIF
  5. Triton C-Grad:     Gamma + conservative
  6. Triton C-Grad:     Gamma + invariant
"""

import argparse
import time
import os
import sys
import statistics
import torch
import torch.nn as nn
from timm.models import create_model
from timm.optim import create_optimizer
from spikingjelly.clock_driven import functional
from torch.cuda import amp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import max_former
import custom_neuron


def build_args(dS_du, du_du, detach_reset=False, lr=1e-3, weight_decay=1e-4):
    return argparse.Namespace(
        model="max_former",
        dataset="cifar10dvs",
        num_classes=10,
        dS_du=dS_du,
        du_du=du_du,
        surrogate_alpha=4.0,
        use_custom_neuron=True,
        detach_reset=detach_reset,
        snnbp_epsilon=0.3,
        snnbp_alpha=2.0,
        snnbp_beta=2.0,
        snnbp_intervention=0.8,
        snnbp_tau=0.5,
        snnbp_p=47.88,
        # Optimizer args
        opt="adamw",
        lr=lr,
        weight_decay=weight_decay,
        momentum=0.9,
        sched=None,
    )


def benchmark_config(label, dS_du, du_du, detach_reset, disable_triton, device, T=16, batch_size=16,
                     dim=128, depth=2, num_warmup=10, num_steps=50, use_amp=True, log_wandb=False):
    os.environ['DISABLE_TRITON'] = disable_triton
    args = build_args(dS_du, du_du, detach_reset)
    custom_neuron.set_global_args(args)

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = create_model(
        "max_former", in_channels=2, num_classes=10,
        embed_dims=dim, mlp_ratios=1.0, depths=depth, T=T,
    ).to(device)

    model.train()
    optimizer = create_optimizer(args, model)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = amp.GradScaler() if use_amp else None

    # Pre-generate synthetic batch on GPU (avoids disk IO / worker queuing jitter)
    x = torch.randn(batch_size, 2, 128, 128, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)

    # Warm-up iterations
    for _ in range(num_warmup):
        optimizer.zero_grad()
        with amp.autocast(enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        functional.reset_net(model)

    torch.cuda.synchronize(device)

    # Timed iterations
    step_times = []
    forward_times = []
    backward_times = []

    for step_idx in range(num_steps):
        t0 = time.perf_counter()

        optimizer.zero_grad()
        t_fwd_start = time.perf_counter()
        with amp.autocast(enabled=use_amp):
            out = model(x)
            loss = criterion(out, y)
        torch.cuda.synchronize(device)
        t_fwd_end = time.perf_counter()

        t_bwd_start = time.perf_counter()
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        functional.reset_net(model)
        torch.cuda.synchronize(device)
        t_bwd_end = time.perf_counter()

        t1 = time.perf_counter()
        s_ms = (t1 - t0) * 1000
        f_ms = (t_fwd_end - t_fwd_start) * 1000
        b_ms = (t_bwd_end - t_bwd_start) * 1000

        step_times.append(t1 - t0)
        forward_times.append(t_fwd_end - t_fwd_start)
        backward_times.append(t_bwd_end - t_bwd_start)

        if log_wandb:
            import wandb
            wandb.log({
                f"{label}/step_ms": s_ms,
                f"{label}/fwd_ms": f_ms,
                f"{label}/bwd_ms": b_ms,
                f"{label}/loss": loss.item(),
            })

    # Cleanup
    del model, optimizer, scaler, criterion, x, y
    torch.cuda.empty_cache()

    return {
        "step_mean_ms": statistics.mean(step_times) * 1000,
        "step_median_ms": statistics.median(step_times) * 1000,
        "fwd_median_ms": statistics.median(forward_times) * 1000,
        "bwd_median_ms": statistics.median(backward_times) * 1000,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--T", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=True)
    parser.add_argument("--log-wandb", action="store_true", default=False, help="Log to wandb")
    parser.add_argument("--wandb-project", type=str, default="maxformer_event", help="WandB project")
    parser.add_argument("--wandb-name", type=str, default="maxformer_triton_benchmark", help="WandB run name")
    cli = parser.parse_args()

    if cli.log_wandb:
        try:
            import wandb
            wandb.init(
                project=cli.wandb_project,
                name=cli.wandb_name,
                config=vars(cli),
            )
            print(f"[WandB] Initialized run: {cli.wandb_name} in project: {cli.wandb_project}")
        except Exception as e:
            print(f"[WandB] Warning: Failed to initialize wandb: {e}")
            cli.log_wandb = False

    device = torch.device(cli.device)
    print(f"{'=' * 80}")
    print(f" MaxFormer End-to-End Training Benchmark (Forward + Backward + Step)")
    print(f" Device: {torch.cuda.get_device_name(device)} | T={cli.T} | Batch={cli.batch_size} | Dim={cli.dim} | Depth={cli.depth} | AMP={cli.amp} | WandB={cli.log_wandb}")
    print(f"{'=' * 80}\n")

    configs = [
        ("PyTorch Baseline: Gamma + LIF",           "Gamma", "LIF",               False, "1"),
        ("PyTorch C-Grad:   Gamma + conservative",  "Gamma", "conservative_cgrad", False, "1"),
        ("PyTorch C-Grad:   Gamma + invariant",     "Gamma", "invariant_cgrad",    False, "1"),
        ("Triton Baseline:  Gamma + LIF",           "Gamma", "LIF",               False, "0"),
        ("Triton C-Grad:    Gamma + conservative",  "Gamma", "conservative_cgrad", False, "0"),
        ("Triton C-Grad:    Gamma + invariant",     "Gamma", "invariant_cgrad",    False, "0"),
    ]

    results = {}
    for label, dS_du, du_du, detach_reset, disable_triton in configs:
        print(f"Running: {label} ...", flush=True)
        res = benchmark_config(
            label, dS_du, du_du, detach_reset, disable_triton, device,
            T=cli.T, batch_size=cli.batch_size, dim=cli.dim, depth=cli.depth,
            num_warmup=cli.warmup, num_steps=cli.steps, use_amp=cli.amp,
            log_wandb=cli.log_wandb
        )
        results[label] = res
        print(f"  -> Total: {res['step_median_ms']:.2f} ms/step  (Fwd: {res['fwd_median_ms']:.2f} ms | Bwd+Opt: {res['bwd_median_ms']:.2f} ms)\n", flush=True)

    ref_label = "PyTorch Baseline: Gamma + LIF"
    ref_step = results[ref_label]['step_median_ms']

    print(f"\n{'=' * 80}")
    print(f" COMPLETE RESULTS TABLE (Median ms per training step)")
    print(f"{'=' * 80}")
    print(f"{'Configuration':<42s} {'Total (ms)':>11s} {'Fwd (ms)':>10s} {'Bwd+Opt (ms)':>13s} {'vs PyTorch Base':>16s}")
    print("-" * 96)

    for label, res in results.items():
        overhead_pct = ((res['step_median_ms'] - ref_step) / ref_step) * 100
        sign = "+" if overhead_pct >= 0 else ""
        print(f"  {label:<40s} {res['step_median_ms']:10.2f} {res['fwd_median_ms']:9.2f} {res['bwd_median_ms']:12.2f} {sign}{overhead_pct:14.1f}%")

    print(f"\n{'=' * 80}")
    print(f" TRITON SPEEDUP SUMMARY (PyTorch vs Triton)")
    print(f"{'=' * 80}")
    pairs = [
        ("Baseline LIF",         "PyTorch Baseline: Gamma + LIF",          "Triton Baseline:  Gamma + LIF"),
        ("Conservative C-Grad",  "PyTorch C-Grad:   Gamma + conservative", "Triton C-Grad:    Gamma + conservative"),
        ("Invariant C-Grad",     "PyTorch C-Grad:   Gamma + invariant",    "Triton C-Grad:    Gamma + invariant"),
    ]
    for mode_name, py_label, tri_label in pairs:
        py_t = results[py_label]['step_median_ms']
        tri_t = results[tri_label]['step_median_ms']
        py_bwd = results[py_label]['bwd_median_ms']
        tri_bwd = results[tri_label]['bwd_median_ms']
        speedup_total = py_t / tri_t
        speedup_bwd = py_bwd / tri_bwd
        print(f"  * {mode_name:<22s}: Total step {py_t:6.2f} ms -> {tri_t:6.2f} ms ({speedup_total:.2f}x speedup) | Bwd+Opt {py_bwd:6.2f} ms -> {tri_bwd:6.2f} ms ({speedup_bwd:.2f}x speedup)")

        if cli.log_wandb:
            import wandb
            wandb.summary[f"speedup_total_{mode_name}"] = speedup_total
            wandb.summary[f"speedup_bwd_{mode_name}"] = speedup_bwd

    if cli.log_wandb:
        import wandb
        wandb.finish()

    print()


if __name__ == "__main__":
    main()
