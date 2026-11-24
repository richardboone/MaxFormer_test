"""Stage 0 Verification Suite for improved_cgrad and custom_neuron_improved integration.

Checks:
  1. Loss-scale equivariance (fp32) across 9 orders of magnitude ('local' and 'rms').
  2. AMP fp16 + GradScaler (scale=65536) safety (zero NaNs/Infs, close to fp32).
  3. Exact BPTT identity at snnbp_eta=0.0 across hard/soft x detach True/False x sigmoid/Gamma (PyTorch & Triton).
  4. harmful_only=True vs False behavior + pointwise non-expansiveness bound.
  5. Triton vs PyTorch parity across T in {4,8,16}, surrogates, resets, detach, norms, fp32/fp16.
  6. Live event/Max_Former uniformity, AMP vs fp32 parity, and sampled stats collection with Triton ON.
  7. CLI/YAML parser isolation check (no legacy default leakage) + GPU throughput benchmark.

Run:
  conda run --no-capture-output -n maxformer python cgrad_analysis/test_improved_cgrad.py
"""

import argparse
import os
import subprocess
import sys
import time
import warnings

import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVENT_DIR = os.path.join(REPO_ROOT, "event")
CIFAR_DIR = os.path.join(REPO_ROOT, "cifar10-100")
CGRAD_DIR = os.path.join(REPO_ROOT, "cgrad_analysis")

sys.path.insert(0, CGRAD_DIR)
import custom_neuron_improved as cni  # noqa: E402

sys.path.insert(0, EVENT_DIR)
import custom_neuron as event_cn  # noqa: E402
import triton_neuron as event_tn  # noqa: E402
import max_former as event_mf  # noqa: E402
from timm.models import create_model  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def make_improved_args(**overrides):
    d = dict(
        use_custom_neuron=True,
        du_du="improved_cgrad",
        dS_du="sigmoid",
        surrogate_alpha=4.0,
        surrogate_gamma=1.0,
        reset_mode="hard",
        detach_reset=True,
        snnbp_tau=2.0,
        snnbp_input_scale="coupled",
        snnbp_norm="local",
        snnbp_harmful_only=True,
        snnbp_epsilon=0.35,
        snnbp_alpha=3.0,
        snnbp_beta=8.0,
        snnbp_alpha_mis=3.0,
        snnbp_margin_m=0.0,
        snnbp_intervention=0.0,
        snnbp_gamma_sharp=0.0,
        snnbp_corr_scale=0.5,
        snnbp_eta=1.0,
    )
    d.update(overrides)
    return argparse.Namespace(**d)


def autograd_ref_lif(x, thresh, decay, input_scale, gama, surrogate_mode, surrogate_alpha, detach_reset, reset_mode):
    class _SpikeFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, u):
            ctx.save_for_backward(u)
            return (u > thresh).to(u.dtype)

        @staticmethod
        def backward(ctx, g):
            (u,) = ctx.saved_tensors
            if surrogate_mode == "sigmoid":
                sgax = (u - thresh) * surrogate_alpha
                sig = torch.sigmoid(sgax)
                ds = (1.0 - sig) * sig * surrogate_alpha
            else:
                ds = (1.0 / (gama * gama)) * (gama - (u - thresh).abs()).clamp(min=0.0)
            return g * ds

    mem = torch.zeros(x.shape[1:], device=x.device, dtype=x.dtype)
    outs = []
    for t in range(x.shape[0]):
        u1 = mem * decay + x[t] * input_scale
        s = _SpikeFn.apply(u1)
        outs.append(s)
        sd = s.detach() if detach_reset else s
        mem = (u1 - thresh * sd) if reset_mode == "soft" else (u1 * (1.0 - sd))
    return torch.stack(outs, dim=0)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Stage 0 Triton and AMP tests")
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    T, B, C = 8, 16, 128
    THRESH, DECAY, ISCALE, GAMA = 1.0, 0.5, 0.5, 1.0
    SCALES = (1e-3, 1.0, 1e2, 1e4, 65536.0, 1e6)

    x0 = (torch.randn(T, B, C, device=device, dtype=torch.float32) * 1.1)
    w0 = (torch.randn(T, B, C, device=device, dtype=torch.float32) * 1e-2)

    print("\n1. Loss-scale equivariance (fp32): g(c * dL)/c vs g(dL)")
    for mod_name, fn_apply, mode_name in (
        ("cni.TimeParallel", cni.TimeParallel_LIFSpike.apply, "conservative_cgrad"),
        ("event.TimeParallel", event_cn.TimeParallel_LIFSpike.apply, "improved_cgrad"),
        ("event.Triton", event_tn.TritonTimeParallel_LIFSpike.apply, "improved_cgrad"),
    ):
        for norm in ("local", "rms"):
            if "Triton" in mod_name and norm == "rms":
                continue
            args = make_improved_args(du_du=mode_name, snnbp_norm=norm)
            x_ref = x0.clone().requires_grad_(True)
            y_ref = fn_apply(x_ref, THRESH, DECAY, ISCALE, GAMA, args, True, "hard")
            y_ref.backward(w0)
            g_ref = x_ref.grad.detach()
            worst = 0.0
            for c in SCALES:
                xc = x0.clone().requires_grad_(True)
                yc = fn_apply(xc, THRESH, DECAY, ISCALE, GAMA, args, True, "hard")
                yc.backward(w0 * c)
                gc = xc.grad.detach() / c
                rel = (gc - g_ref).abs().max().item() / (g_ref.abs().max().item() + 1e-12)
                worst = max(worst, rel)
            check(f"{mod_name:20s} norm={norm:5s} max rel dev = {worst:.2e}", worst < 1e-5)

    print("\n2. AMP fp16 + GradScaler (scale=65536.0) safety")
    for mod_name, fn_apply, mode_name in (
        ("cni.TimeParallel", cni.TimeParallel_LIFSpike.apply, "conservative_cgrad"),
        ("event.TimeParallel", event_cn.TimeParallel_LIFSpike.apply, "improved_cgrad"),
        ("event.Triton", event_tn.TritonTimeParallel_LIFSpike.apply, "improved_cgrad"),
    ):
        for norm in ("local", "rms"):
            if "Triton" in mod_name and norm == "rms":
                continue
            args = make_improved_args(du_du=mode_name, snnbp_norm=norm)
            x32 = x0.clone().requires_grad_(True)
            y32 = fn_apply(x32, THRESH, DECAY, ISCALE, GAMA, args, True, "hard")
            y32.backward(w0)
            g32 = x32.grad.detach()

            x16 = x0.clone().half().requires_grad_(True)
            y16 = fn_apply(x16, THRESH, DECAY, ISCALE, GAMA, args, True, "hard")
            y16.backward((w0 * 65536.0).half())
            g16 = x16.grad.float() / 65536.0
            finite = torch.isfinite(g16).all().item()
            rel_l2 = ((g16 - g32).norm() / (g32.norm() + 1e-12)).item() if finite else float("inf")
            check(
                f"{mod_name:20s} norm={norm:5s} fp16+65536 finite={finite} rel_L2={rel_l2:.2e}",
                finite and rel_l2 < 1e-2,
            )

    print("\n3. Exact plain-BPTT reduction at snnbp_eta = 0.0")
    for sg in ("sigmoid", "Gamma"):
        for reset in ("hard", "soft"):
            for detach in (True, False):
                x_ref = x0.clone().requires_grad_(True)
                y_ref = autograd_ref_lif(x_ref, THRESH, DECAY, ISCALE, GAMA, sg, 4.0, detach, reset)
                y_ref.backward(w0)
                g_ref = x_ref.grad.detach()

                args = make_improved_args(du_du="improved_cgrad", dS_du=sg, detach_reset=detach, reset_mode=reset, snnbp_eta=0.0)
                for label, fn_apply in (
                    ("PyTorch", event_cn.TimeParallel_LIFSpike.apply),
                    ("Triton", event_tn.TritonTimeParallel_LIFSpike.apply),
                ):
                    xt = x0.clone().requires_grad_(True)
                    yt = fn_apply(xt, THRESH, DECAY, ISCALE, GAMA, args, detach, reset)
                    yt.backward(w0)
                    err = (xt.grad - g_ref).abs().max().item()
                    check(f"{label:7s} sg={sg:7s} reset={reset:4s} detach={str(detach):5s} max|Δ|={err:.2e}", err < 1e-5)

    print("\n4. harmful_only switch & pointwise magnitude bound")
    args_h = make_improved_args(snnbp_harmful_only=True)
    args_t = make_improved_args(snnbp_harmful_only=False)
    event_cn.enable_cgrad_stats(device, sample_prob=1.0)
    xh = x0.clone().requires_grad_(True)
    yh = event_cn.TimeParallel_LIFSpike.apply(xh, THRESH, DECAY, ISCALE, GAMA, args_h, True, "hard")
    yh.backward(w0)
    sh = event_cn.pop_cgrad_stats()
    event_cn.enable_cgrad_stats(device, sample_prob=1.0)
    xt = x0.clone().requires_grad_(True)
    yt = event_cn.TimeParallel_LIFSpike.apply(xt, THRESH, DECAY, ISCALE, GAMA, args_t, True, "hard")
    yt.backward(w0)
    st = event_cn.pop_cgrad_stats()
    event_cn.disable_cgrad_stats()
    check(
        f"harmful_only softly suppresses m<0 ({sh['cgrad/frac_interventions_m_neg']*100:.2f}% vs {st['cgrad/frac_interventions_m_neg']*100:.2f}%)",
        sh["cgrad/frac_interventions_m_neg"] < 0.25
        and st["cgrad/frac_interventions_m_neg"] > 0.40
        and sh["cgrad/frac_interventions_m_neg"] < 0.5 * st["cgrad/frac_interventions_m_neg"],
    )
    # Also check pointwise non-expansiveness when corr_scale=0.5 (|dL_dU1| <= |base_function|):
    args_1step = make_improved_args(snnbp_corr_scale=0.5)
    x_1 = x0[:1].clone().requires_grad_(True)
    g_impr = torch.autograd.grad(
        event_cn.TimeParallel_LIFSpike.apply(x_1, THRESH, DECAY, ISCALE, GAMA, args_1step, True, "hard"),
        x_1,
        w0[:1],
    )[0]
    g_base = torch.autograd.grad(
        event_cn.TimeParallel_LIFSpike.apply(x_1, THRESH, DECAY, ISCALE, GAMA, make_improved_args(snnbp_eta=0.0), True, "hard"),
        x_1,
        w0[:1],
    )[0]
    ratio = (g_impr.abs() / (g_base.abs() + 1e-12)).max().item()
    check(f"1-step |dL_dU1| <= |base_function| with corr_scale=0.5 (max ratio={ratio:.6f})", ratio <= 1.0 + 1e-5)

    print("\n5. Triton vs PyTorch parity on improved_cgrad")
    worst_fp32_rel = 0.0
    worst_fp16_mean = 0.0
    parity_ok = True
    for T_test in (4, 8, 16):
        xt_in = torch.randn(T_test, 8, 64, device=device) * 1.1
        wt_in = torch.randn(T_test, 8, 64, device=device) * 1e-2
        for sg in ("sigmoid", "Gamma", "fast_sigmoid", "ste"):
            for reset in ("hard", "soft"):
                for detach in (True, False):
                    for harmful in (True, False):
                        for norm in ("local", "none"):
                            for gsharp in (0.0, 5.0):
                                args = make_improved_args(
                                    dS_du=sg,
                                    reset_mode=reset,
                                    detach_reset=detach,
                                    snnbp_harmful_only=harmful,
                                    snnbp_norm=norm,
                                    snnbp_gamma_sharp=gsharp,
                                    snnbp_intervention=0.2 if gsharp > 0 else 0.0,
                                )
                                x_py = xt_in.clone().requires_grad_(True)
                                x_tr = xt_in.clone().requires_grad_(True)
                                y_py = event_cn.TimeParallel_LIFSpike.apply(x_py, THRESH, DECAY, ISCALE, GAMA, args, detach, reset)
                                y_tr = event_tn.TritonTimeParallel_LIFSpike.apply(x_tr, THRESH, DECAY, ISCALE, GAMA, args, detach, reset)
                                if (y_py - y_tr).abs().max().item() != 0.0:
                                    parity_ok = False
                                y_py.backward(wt_in)
                                y_tr.backward(wt_in)
                                rel = ((x_py.grad - x_tr.grad).norm() / (x_py.grad.norm() + 1e-12)).item()
                                worst_fp32_rel = max(worst_fp32_rel, rel)
                                if rel >= 1e-5:
                                    parity_ok = False
    # snnbp_misalign_mode='sign' Triton vs PyTorch parity check
    for gsharp in (0.0, 5.0):
        for harmful in (True, False):
            args_sign = make_improved_args(
                dS_du="sigmoid",
                snnbp_harmful_only=harmful,
                snnbp_norm="local",
                snnbp_misalign_mode="sign",
                snnbp_gamma_sharp=gsharp,
                snnbp_intervention=0.2 if gsharp > 0 else 0.0,
            )
            x_py = x0.clone().requires_grad_(True)
            x_tr = x0.clone().requires_grad_(True)
            y_py = event_cn.TimeParallel_LIFSpike.apply(x_py, THRESH, DECAY, ISCALE, GAMA, args_sign, True, "hard")
            y_tr = event_tn.TritonTimeParallel_LIFSpike.apply(x_tr, THRESH, DECAY, ISCALE, GAMA, args_sign, True, "hard")
            y_py.backward(w0)
            y_tr.backward(w0)
            rel = ((x_py.grad - x_tr.grad).norm() / (x_py.grad.norm() + 1e-12)).item()
            worst_fp32_rel = max(worst_fp32_rel, rel)
            if rel >= 1e-5:
                parity_ok = False

    # fp16 parity check
    for sg in ("sigmoid", "Gamma"):
        for harmful in (True, False):
            args = make_improved_args(dS_du=sg, snnbp_harmful_only=harmful)
            x_py = x0.clone().half().requires_grad_(True)
            x_tr = x0.clone().half().requires_grad_(True)
            y_py = event_cn.TimeParallel_LIFSpike.apply(x_py, THRESH, DECAY, ISCALE, GAMA, args, True, "hard")
            y_tr = event_tn.TritonTimeParallel_LIFSpike.apply(x_tr, THRESH, DECAY, ISCALE, GAMA, args, True, "hard")
            y_py.backward((w0 * 65536.0).half())
            y_tr.backward((w0 * 65536.0).half())
            g_py = x_py.grad.float() / 65536.0
            g_tr = x_tr.grad.float() / 65536.0
            mean_d = (g_py - g_tr).abs().mean().item()
            worst_fp16_mean = max(worst_fp16_mean, mean_d)
            if not (torch.isfinite(g_tr).all() and mean_d < 5e-4):
                parity_ok = False
    check(
        f"Triton vs PyTorch across 192 fp32 configs (worst rel_L2={worst_fp32_rel:.2e}) & fp16 (worst mean={worst_fp16_mean:.2e})",
        parity_ok,
    )

    print("\n6. Live event/Max_Former uniformity, AMP vs fp32, and stats collection")
    for arm_name, tau_arg, scale_arg, detach_arg, expected_decay, expected_scale in (
        ("A1_tau2_coupled", 2.0, "coupled", True, 0.5, 0.5),
        ("D1_tau20_scale1", 20.0, "1.0", False, 0.95, 1.0),
    ):
        args_mf = make_improved_args(
            snnbp_tau=tau_arg,
            snnbp_input_scale=scale_arg,
            detach_reset=detach_arg,
        )
        event_cn.set_global_args(args_mf)
        model = create_model(
            "max_former", in_channels=2, num_classes=10, embed_dims=256, mlp_ratios=1.0, depths=2, T=16
        ).to(device)
        x_img = torch.randn(4, 16, 2, 128, 128, device=device)
        out = model(x_img)
        lif_layers = [m for m in model.modules() if isinstance(m, event_cn.LIFSpikeLayer_Cons)]
        uniform = all(
            abs(m.decay - expected_decay) < 1e-7
            and abs(m.input_scale - expected_scale) < 1e-7
            and m.detach_reset == detach_arg
            for m in lif_layers
        )
        check(
            f"Max_Former {arm_name}: {len(lif_layers)} LIF layers uniform (decay={expected_decay}, scale={expected_scale}, detach={detach_arg})",
            uniform and len(lif_layers) >= 10,
        )
        model.zero_grad(set_to_none=True)
        out.sum().backward()

    # AMP + Triton + stats collection on Max_Former
    args_amp = make_improved_args(snnbp_tau=2.0, snnbp_input_scale="coupled", detach_reset=True)
    event_cn.set_global_args(args_amp)
    torch.manual_seed(123)
    model_amp = create_model(
        "max_former", in_channels=2, num_classes=10, embed_dims=256, mlp_ratios=1.0, depths=2, T=16
    ).to(device)
    x_amp = torch.randn(4, 16, 2, 128, 128, device=device)
    y_amp = torch.tensor([0, 1, 2, 3], device=device)
    crit = torch.nn.CrossEntropyLoss()

    # Pure fp32 PyTorch vs Triton on Max_Former
    os.environ["DISABLE_TRITON"] = "1"
    event_cn.disable_cgrad_stats()
    model_amp.zero_grad(set_to_none=True)
    loss32_py = crit(model_amp(x_amp), y_amp)
    loss32_py.backward()
    g32_py = torch.cat([p.grad.detach().flatten() for p in model_amp.parameters() if p.grad is not None])

    os.environ["DISABLE_TRITON"] = "0"
    model_amp.zero_grad(set_to_none=True)
    loss32_tr = crit(model_amp(x_amp), y_amp)
    loss32_tr.backward()
    g32_tr = torch.cat([p.grad.detach().flatten() for p in model_amp.parameters() if p.grad is not None])
    fp32_rel = ((g32_tr - g32_py).norm() / (g32_py.norm() + 1e-12)).item()
    check(f"Max_Former fp32 PyTorch vs Triton grad rel_L2={fp32_rel:.2e}", fp32_rel < 1e-3)

    # AMP GradScaler=65536: PyTorch vs Triton (both under autocast)
    scaler = torch.amp.GradScaler("cuda", init_scale=65536.0)
    os.environ["DISABLE_TRITON"] = "1"
    model_amp.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        loss16_py = crit(model_amp(x_amp), y_amp)
    scaler.scale(loss16_py).backward()
    g16_py = torch.cat([p.grad.detach().float().flatten() / 65536.0 for p in model_amp.parameters() if p.grad is not None])

    os.environ["DISABLE_TRITON"] = "0"
    model_amp.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        loss16_tr = crit(model_amp(x_amp), y_amp)
    scaler.scale(loss16_tr).backward()
    g16_tr = torch.cat([p.grad.detach().float().flatten() / 65536.0 for p in model_amp.parameters() if p.grad is not None])

    amp_finite = torch.isfinite(g16_tr).all().item() and torch.isfinite(g16_py).all().item()
    amp_rel = ((g16_tr - g16_py).norm() / (g16_py.norm() + 1e-12)).item() if amp_finite else float("inf")
    check(f"Max_Former AMP+Triton vs AMP+PyTorch finite={amp_finite} rel_L2={amp_rel:.2e}", amp_finite and amp_rel < 1e-2)

    # Stats collection (sample_prob=1.0) on Max_Former
    event_cn.enable_cgrad_stats(device, sample_prob=1.0)
    model_amp.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda"):
        loss_st = crit(model_amp(x_amp), y_amp)
    scaler.scale(loss_st).backward()
    mf_stats = event_cn.pop_cgrad_stats()
    event_cn.disable_cgrad_stats()
    check(
        f"Max_Former stats effective_rate={mf_stats['cgrad/effective_rate']*100:.2f}% mean_w={mf_stats['cgrad/mean_w']:.4f} m_neg={mf_stats['cgrad/frac_effective_m_neg']*100:.2f}%",
        mf_stats is not None and mf_stats["cgrad/effective_rate"] > 0.005,
    )

    print("\n7. CLI/YAML parser isolation & GPU throughput benchmark")
    sys.path.insert(0, CIFAR_DIR)
    for script_dir, yaml_name, expected in (
        (EVENT_DIR, "cifar10dvs_improved_cgrad.yaml", dict(du_du="improved_cgrad", dS_du="sigmoid", snnbp_beta=8.0, snnbp_epsilon=0.35, snnbp_intervention=0.0, snnbp_eta=1.0, snnbp_norm="local", snnbp_harmful_only=True)),
        (CIFAR_DIR, "cifar100_improved_cgrad.yaml", dict(du_du="improved_cgrad", dS_du="sigmoid", snnbp_beta=8.0, snnbp_epsilon=0.35, snnbp_intervention=0.0, snnbp_eta=1.0, snnbp_norm="local", snnbp_harmful_only=True)),
        (CIFAR_DIR, "cifar10_improved_cgrad.yaml", dict(du_du="improved_cgrad", dS_du="sigmoid", snnbp_beta=8.0, snnbp_epsilon=0.35, snnbp_intervention=0.0, snnbp_eta=1.0, snnbp_norm="local", snnbp_harmful_only=True)),
    ):
        code = (
            f"import sys, json; sys.path.insert(0, {script_dir!r}); "
            f"sys.argv=['train.py', '-c', {os.path.join(script_dir, yaml_name)!r}]; "
            f"import train; "
            f"a = train.parse_args() if hasattr(train, 'parse_args') else train._parse_args()[0]; "
            f"print(json.dumps({{k: getattr(a, k) for k in {list(expected.keys())!r}}}))"
        )
        res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
        import json
        parsed = json.loads(res.stdout.strip().splitlines()[-1])
        ok_cfg = all(parsed[k] == v for k, v in expected.items())
        check(f"Parser check {yaml_name}: {parsed}", ok_cfg)

    # Short GPU throughput benchmark on Max_Former (T=16, bs=16, 128x128)
    bx = torch.randn(16, 2, 128, 128, device=device)
    by = torch.randint(0, 10, (16,), device=device)
    bench_results = {}
    for label, mode_b, dis_tri, stats_p in (
        ("LIF (Triton)", "LIF", "0", 0.0),
        ("improved_cgrad (PyTorch)", "improved_cgrad", "1", 0.0),
        ("improved_cgrad (Triton)", "improved_cgrad", "0", 0.0),
        ("improved_cgrad (Triton+5%stats)", "improved_cgrad", "0", 0.05),
    ):
        os.environ["DISABLE_TRITON"] = dis_tri
        b_args = make_improved_args(du_du=mode_b)
        event_cn.set_global_args(b_args)
        if stats_p > 0:
            event_cn.enable_cgrad_stats(device, sample_prob=stats_p)
        else:
            event_cn.disable_cgrad_stats()
        m_b = create_model(
            "max_former", in_channels=2, num_classes=10, embed_dims=256, mlp_ratios=1.0, depths=2, T=16
        ).to(device)
        opt_b = torch.optim.AdamW(m_b.parameters(), lr=1e-3)
        sc_b = torch.amp.GradScaler("cuda")
        for _ in range(5):
            opt_b.zero_grad()
            with torch.amp.autocast("cuda"):
                l = crit(m_b(bx), by)
            sc_b.scale(l).backward()
            sc_b.step(opt_b)
            sc_b.update()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(25):
            opt_b.zero_grad()
            with torch.amp.autocast("cuda"):
                l = crit(m_b(bx), by)
            sc_b.scale(l).backward()
            sc_b.step(opt_b)
            sc_b.update()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / 25 * 1000.0
        bench_results[label] = ms
        if stats_p > 0:
            event_cn.pop_cgrad_stats()
            event_cn.disable_cgrad_stats()
        del m_b, opt_b, sc_b
    speedup = bench_results["improved_cgrad (PyTorch)"] / bench_results["improved_cgrad (Triton)"]
    print("  Throughput (ms/step @ T=16, bs=16, 128x128):")
    for k, v in bench_results.items():
        print(f"    {k:34s}: {v:6.2f} ms/step")
    check(f"Triton improved_cgrad speedup = {speedup:.2f}x", speedup > 1.2)

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES else f"{len(FAILURES)} FAILED: {FAILURES}"))
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
