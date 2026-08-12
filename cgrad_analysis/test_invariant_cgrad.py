"""Property tests for the invariant_cgrad mode in event/custom_neuron.py.

Checks, in order:
  1. equivariance under loss rescaling   (the point of the mode)
  2. firing rate is constant across loss scales
  3. equivariance survives the real AMP fp16 path without overflow
  4. the correction never exceeds the base gradient in magnitude
  5. gating reduces to standard BPTT when the gates are shut
  6. the two_sided switch does what it says

Run:  python cgrad_analysis/test_invariant_cgrad.py
"""
import argparse
import os as _os
import sys

import torch

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_REPO, "event"))
from custom_neuron import TimeParallel_LIFSpike  # noqa: E402

torch.manual_seed(0)
T, B, C = 8, 32, 256
THRESH, DECAY, ISCALE, GAMA = 1.0, 0.5, 0.5, 1.0
SCALES = (1e-3, 1.0, 1e2, 1e4, 65536.0, 1e6)

HP = dict(snnbp_epsilon=0.3, snnbp_alpha=2.0, snnbp_beta=2.0,
          snnbp_alpha_cons=2.0, snnbp_kappa=1.0, snnbp_h=0.5,
          snnbp_eta=0.5, snnbp_gamma=5.0, snnbp_two_sided=True,
          snnbp_intervention=0.8)   # inherited default; must be IGNORED by the mode

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def args(mode, **over):
    d = dict(dS_du="Gamma", du_du=mode, detach_reset=True, reset_mode="hard", **HP)
    d.update(over)
    return argparse.Namespace(**d)


def backward_of(mode, scale, x0, w0, dtype=torch.float32, **over):
    x = x0.clone().to(dtype).requires_grad_(True)
    y = TimeParallel_LIFSpike.apply(x, THRESH, DECAY, ISCALE, GAMA,
                                    args(mode, **over), True, "hard")
    y.backward((w0 * scale).to(dtype))
    return x.grad.float() / scale


x0 = torch.randn(T, B, C) * 1.2
w0 = torch.randn(T, B, C) * 1e-3


def surrogate_tri(u):
    return (1 / GAMA ** 2) * (GAMA - (u - THRESH).abs()).clamp(min=0)


class _SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > THRESH).float()

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g * surrogate_tri(u)


def detached_bptt_reference(scale):
    """Autograd LIF with a detached reset -- the base gradient invariant_cgrad
    reduces to when its gates are shut. Deliberately not du_du='LIF', which
    ignores detach_reset and would give the non-detached gradient instead."""
    x = x0.clone().requires_grad_(True)
    mem = torch.zeros(B, C)
    out = []
    for t in range(T):
        u1 = mem * DECAY + x[t] * ISCALE
        s = _SpikeFn.apply(u1)
        out.append(s)
        mem = u1 * (1 - s.detach())
    torch.stack(out).backward(w0 * scale)
    return x.grad.clone() / scale

print("1. Equivariance:  g(c*grad)/c  vs  g(grad)")
# conservative_cgrad needs its own hyperparameters here: with the shared default
# snnbp_intervention=0.8 its gate can never open (ceiling 0.646 < 0.8), so it
# degenerates to plain BPTT and would look trivially equivariant. These are the
# values from event/cifar10dvs_cgrad.yaml, i.e. a configuration that does fire.
CONS_HP = dict(snnbp_alpha=3.11, snnbp_beta=1.21, snnbp_epsilon=0.17,
               snnbp_intervention=0.24)
for mode in ("LIF", "conservative_cgrad", "invariant_cgrad"):
    over = CONS_HP if mode == "conservative_cgrad" else {}
    ref = backward_of(mode, 1.0, x0, w0, **over)
    worst = max((backward_of(mode, c, x0, w0, **over) - ref).abs().max().item()
                / ref.abs().max().item() for c in SCALES)
    expect_invariant = mode != "conservative_cgrad"
    check(f"{mode:<20} max rel. deviation = {worst:.2e}",
          (worst < 1e-5) == expect_invariant,
          "" if expect_invariant else "known non-equivariant, shown for contrast")

print("\n2. Firing rate vs loss scale (invariant_cgrad)")
surro = lambda u: (1 / GAMA ** 2) * (GAMA - (u - THRESH).abs()).clamp(min=0)
mem = torch.zeros(B, C)
u1s, sps = [], []
for t in range(T):
    u1 = mem * DECAY + x0[t] * ISCALE
    s = (u1 > THRESH).float()
    u1s.append(u1)
    sps.append(s)
    mem = u1 * (1 - s)
u1s, sps = torch.stack(u1s), torch.stack(sps)


def firing_rate(scale, two_sided=True):
    gml = torch.zeros(B, C)
    fire = tot = neg = 0
    for t in reversed(range(T)):
        u1, s = u1s[t], sps[t]
        dL_dS, dL_dU2 = w0[t] * scale, gml * DECAY
        dS, dU2 = surro(u1), 1 - s
        m = torch.where(u1 < THRESH, dL_dS - u1 * dL_dU2, THRESH * dL_dU2 - dL_dS)
        mg = torch.where(u1 < THRESH, m, -m)
        bf = dL_dS * dS + dL_dU2 * dU2
        rho, d = THRESH - u1, u1 - THRESH
        pred = rho * bf
        norm = torch.sqrt(m.pow(2).mean() + pred.pow(2).mean()).clamp(min=1e-30)
        mh, ph = m / norm, pred / norm
        g_m = torch.sigmoid(HP['snnbp_alpha'] * ((mh.abs() if two_sided else mh)
                                                 - HP['snnbp_kappa']))
        g_d = torch.sigmoid(HP['snnbp_beta'] * (HP['snnbp_epsilon'] - d.abs()))
        g_c = torch.sigmoid(HP['snnbp_alpha_cons'] * (-mh * ph))
        sig = (g_m * g_d * g_c).pow(1.0 / 3.0)
        do = sig > HP['snnbp_h']
        fire += do.sum().item()
        tot += do.numel()
        neg += (do & (m < 0)).sum().item()
        w = HP['snnbp_eta'] * do.float() * torch.sigmoid(
            HP['snnbp_gamma'] * (sig - HP['snnbp_h']))
        gml = (1 - w) * bf + w * torch.sign(mg) * bf.abs()
    return 100 * fire / tot, 100 * neg / max(fire, 1)


rates = [firing_rate(c)[0] for c in SCALES]
for c, r in zip(SCALES, rates):
    print(f"      scale {c:>10.0e}   fire {r:7.3f}%")
check("firing rate constant across 9 orders of magnitude",
      max(rates) - min(rates) < 1e-9, f"spread = {max(rates) - min(rates):.2e}")

print("\n3. AMP fp16 path (loss scale 65536, half precision)")
g16 = backward_of("invariant_cgrad", 65536.0, x0, w0, dtype=torch.float16)
g32 = backward_of("invariant_cgrad", 1.0, x0, w0, dtype=torch.float32)
finite = torch.isfinite(g16).all().item()
check("no inf/nan in fp16 backward", finite)
if finite:
    rel = ((g16 - g32).norm() / g32.norm()).item()
    check("fp16 result matches fp32 within half-precision tolerance",
          rel < 1e-2, f"relative L2 error = {rel:.2e}")

print("\n4. Correction is non-expansive per timestep")
# g_corr = sign(m*rho)*|g_base| has the same magnitude as g_base, so the blend
# (1-w)*g_base + w*g_corr is either g_base (signs agree) or (1-2w)*g_base (signs
# disagree). Either way |dL_dU1| <= |base_function| pointwise, for any w in [0,1]:
# the correction can flip or shrink the local gradient but never amplify it.
# This is what backs the "cannot introduce large gradient spikes" claim of Sec 6.2.
worst_ratio = 0.0
gml = torch.zeros(B, C)
for t in reversed(range(T)):
    u1, s = u1s[t], sps[t]
    dL_dS, dL_dU2 = w0[t], gml * DECAY
    dS, dU2 = surrogate_tri(u1), 1 - s
    m = torch.where(u1 < THRESH, dL_dS - u1 * dL_dU2, THRESH * dL_dU2 - dL_dS)
    mg = torch.where(u1 < THRESH, m, -m)
    bf = dL_dS * dS + dL_dU2 * dU2
    rho, d = THRESH - u1, u1 - THRESH
    pred = rho * bf
    norm = torch.sqrt(m.pow(2).mean() + pred.pow(2).mean()).clamp(min=1e-30)
    mh, ph = m / norm, pred / norm
    sig = (torch.sigmoid(HP['snnbp_alpha'] * (mh.abs() - HP['snnbp_kappa']))
           * torch.sigmoid(HP['snnbp_beta'] * (HP['snnbp_epsilon'] - d.abs()))
           * torch.sigmoid(HP['snnbp_alpha_cons'] * (-mh * ph))).pow(1.0 / 3.0)
    w = HP['snnbp_eta'] * (sig > HP['snnbp_h']).float() * torch.sigmoid(
        HP['snnbp_gamma'] * (sig - HP['snnbp_h']))
    out = (1 - w) * bf + w * torch.sign(mg) * bf.abs()
    worst_ratio = max(worst_ratio, (out.abs() / (bf.abs() + 1e-30)).max().item())
    gml = out
check("|dL_dU1| <= |base_function| pointwise at every timestep",
      worst_ratio <= 1.0 + 1e-5, f"max ratio = {worst_ratio:.6f}")

# End-to-end amplification is NOT bounded by the above: a correction at time t
# feeds the recursion at t-1, so the input gradient can move much further. Report
# it rather than asserting on it.
base = backward_of("invariant_cgrad", 1.0, x0, w0, snnbp_eta=0.0)
gc = backward_of("invariant_cgrad", 1.0, x0, w0)
print(f"      (informational) end-to-end ||g_cgrad|| / ||g_base|| = "
      f"{(gc.norm() / base.norm()).item():.3f}, "
      f"max elementwise ratio = {(gc.abs() / (base.abs() + 1e-12)).max().item():.1f}")

print("\n5. Shutting the gates recovers standard BPTT")
ref = detached_bptt_reference(1.0)
check("eta = 0 reproduces detached BPTT exactly",
      (base - ref).abs().max().item() < 1e-6,
      f"max |delta| = {(base - ref).abs().max().item():.2e}")
closed = backward_of("invariant_cgrad", 1.0, x0, w0, snnbp_h=0.999)
check("h -> 1 reproduces detached BPTT exactly",
      (closed - ref).abs().max().item() < 1e-6,
      f"max |delta| = {(closed - ref).abs().max().item():.2e}")
# snnbp_intervention is conservative_cgrad's knob; invariant_cgrad must ignore it,
# otherwise its 0.8 default would silently make this mode a no-op.
ignored = backward_of("invariant_cgrad", 1.0, x0, w0, snnbp_intervention=0.999)
check("snnbp_intervention does not affect invariant_cgrad",
      (ignored - gc).abs().max().item() < 1e-9,
      f"max |delta| = {(ignored - gc).abs().max().item():.2e}")
# Guard on the finding that du_du='LIF'/'TET' ignore detach_reset.
lif = backward_of("LIF", 1.0, x0, w0)
check("du_du='LIF' still ignores detach_reset (known bug, not yet fixed)",
      (lif - ref).abs().max().item() > 1e-6,
      f"max |delta| vs detached reference = {(lif - ref).abs().max().item():.2e}")

print("\n6. two_sided switch")
_, neg_two = firing_rate(65536.0, two_sided=True)
_, neg_one = firing_rate(65536.0, two_sided=False)
print(f"      two_sided=True    interventions with m<0: {neg_two:6.2f}%")
print(f"      two_sided=False   interventions with m<0: {neg_one:6.2f}%")
check("one-sided mode suppresses m<0 interventions", neg_one < neg_two)

print("\n" + ("ALL CHECKS PASSED" if not FAILURES
             else f"{len(FAILURES)} FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
