"""Demonstrate that conservative_cgrad is not equivariant to loss rescaling.

Equivariance test: for standard BPTT, scaling the loss by c scales every gradient
by exactly c, so g(c*grad_out)/c == g(grad_out).  Any rule with that property is
"scale-invariant" -- rescaling the loss is undone by rescaling the learning rate.
We check whether C-Grad has it.
"""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, sys
import torch

sys.path.insert(0, _os.path.join(_REPO, "event"))
from custom_neuron import TimeParallel_LIFSpike

torch.manual_seed(0)
T, B, C = 8, 32, 256
THRESH, DECAY, ISCALE, GAMA = 1.0, 0.5, 0.5, 1.0

# cifar10_best_cons.yaml -- the config behind the CIFAR-10/100 + ImageNet results
HP = dict(snnbp_epsilon=0.357430, snnbp_alpha=1.209609,
          snnbp_beta=1.366561, snnbp_intervention=0.531940)


def args(mode):
    return argparse.Namespace(dS_du="Gamma", du_du=mode, detach_reset=True,
                              reset_mode="hard", **HP)


x0 = (torch.randn(T, B, C) * 1.2).requires_grad_(False)
w0 = torch.randn(T, B, C) * 1e-3          # realistic per-neuron dL/dS magnitude


def cgrad_out(mode, scale):
    x = x0.clone().requires_grad_(True)
    y = TimeParallel_LIFSpike.apply(x, THRESH, DECAY, ISCALE, GAMA,
                                    args(mode), True, "hard")
    y.backward(w0 * scale)
    return x.grad.clone() / scale       # undo the scaling; should be scale-free


print("=" * 76)
print("EQUIVARIANCE TEST:  g(c*grad_out)/c  vs  g(grad_out)")
print("=" * 76)
for mode in ("LIF", "conservative_cgrad"):
    ref = cgrad_out(mode, 1.0)
    print(f"\n  du_du = {mode}")
    print(f"    {'loss scale c':>14} {'max rel. deviation from c=1':>30}")
    for c in (1e-2, 1.0, 1e2, 1e4, 65536.0, 1e6):
        g = cgrad_out(mode, c)
        rel = ((g - ref).abs().max() / ref.abs().max()).item()
        tag = "  <-- AMP GradScaler default" if c == 65536.0 else ""
        print(f"    {c:>14.0e} {rel:>30.4f}{tag}")

# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("WHAT THE GATES DO AS THE LOSS SCALE CHANGES")
print("=" * 76)
eps, am, ad, h = HP['snnbp_epsilon'], HP['snnbp_alpha'], HP['snnbp_beta'], HP['snnbp_intervention']

# forward once
mem = torch.zeros(B, C); u1s, sps = [], []
for t in range(T):
    u1 = mem * DECAY + x0[t] * ISCALE
    s = (u1 > THRESH).float()
    u1s.append(u1); sps.append(s); mem = u1 * (1 - s)
u1s, sps = torch.stack(u1s), torch.stack(sps)
surro = lambda u: (1 / GAMA**2) * (GAMA - (u - THRESH).abs()).clamp(min=0)

print(f"\n  {'scale c':>10} {'fire %':>9} {'mean g_m':>10} {'mean g_d':>10} "
      f"{'g_mis binary %':>16} {'inconsistent %':>16}")
for c in (1e-2, 1.0, 1e2, 1e4, 65536.0, 1e6):
    gml = torch.zeros(B, C); fire = tot = binm = incon = 0
    gm_s = gd_s = 0.0
    for t in reversed(range(T)):
        u1, s = u1s[t], sps[t]
        dL_dS, dL_dU2 = w0[t] * c, gml * DECAY
        dS = surro(u1); dU2 = 1 - s
        m = torch.where(u1 < THRESH, dL_dS - u1 * dL_dU2, THRESH * dL_dU2 - dL_dS)
        mg = torch.where(u1 < THRESH, m, -m)
        bf = dL_dS * dS + dL_dU2 * dU2
        d = u1 - THRESH
        g_m = torch.sigmoid(am * (m.abs() - 0.1))
        g_d = torch.sigmoid(ad * (eps - d.abs()))
        g_mis = torch.sigmoid(am * (-mg * bf))
        sig = g_m * g_d * g_mis
        do = sig > h
        fire += do.sum().item(); tot += do.numel()
        binm += ((g_mis > 0.99) | (g_mis < 0.01)).sum().item()
        incon += ((-mg * bf > 0) & (d.abs() <= eps)).sum().item()
        gm_s += g_m.mean().item(); gd_s += g_d.mean().item()
        blend = do.float() * torch.sigmoid(5 * (sig - h))
        gml = (1 - 0.5 * blend) * bf + 0.5 * blend * torch.sign(mg) * bf.abs()
    print(f"  {c:>10.0e} {100*fire/tot:>8.3f}% {gm_s/T:>10.4f} {gd_s/T:>10.4f} "
          f"{100*binm/tot:>15.1f}% {100*incon/tot:>15.1f}%")

print("""
  'inconsistent %' = fraction of neuron-timesteps that are near threshold AND
  genuinely violate Eq.(10).  It is scale-INVARIANT (a sign test), and it is the
  set C-Grad is supposed to act on.  'fire %' is the set it actually acts on.""")

# ---------------------------------------------------------------------------
print()
print("=" * 76)
print("PROPOSED SCALE-FREE GATE:  r = -m / (rho * g_base)   [inconsistent <=> r>0]")
print("=" * 76)
print(f"\n  {'scale c':>10} {'fire % (r-gate)':>18}")
for c in (1e-2, 1.0, 1e2, 1e4, 65536.0, 1e6):
    gml = torch.zeros(B, C); fire = tot = 0
    for t in reversed(range(T)):
        u1, s = u1s[t], sps[t]
        dL_dS, dL_dU2 = w0[t] * c, gml * DECAY
        dS = surro(u1); dU2 = 1 - s
        m = torch.where(u1 < THRESH, dL_dS - u1 * dL_dU2, THRESH * dL_dU2 - dL_dS)
        bf = dL_dS * dS + dL_dU2 * dU2
        rho, d = THRESH - u1, u1 - THRESH
        r = -m / (rho * bf + 1e-30 * torch.sign(rho * bf + 1e-30))
        g_r = torch.sigmoid(am * torch.log1p(r.clamp(min=0)))   # 0.5 at r=0, ->1 for r>>0
        g_d = torch.sigmoid(ad * (eps - d.abs()))
        sig = g_r * g_d
        do = sig > h * 0.6197 / 1.0     # same relative bar as before
        fire += do.sum().item(); tot += do.numel()
        blend = do.float() * torch.sigmoid(5 * (sig - h))
        mg = torch.where(u1 < THRESH, m, -m)
        gml = (1 - 0.5 * blend) * bf + 0.5 * blend * torch.sign(mg) * bf.abs()
    print(f"  {c:>10.0e} {100*fire/tot:>17.3f}%")
print("\n  Identical at every scale -- which is the point.")
