"""Verification of the C-Grad implementation in event/custom_neuron.py against
(a) the paper's equations and (b) an autograd reference LIF."""
import os as _os
_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
import argparse, sys, os
import torch

sys.path.insert(0, _os.path.join(_REPO, "event"))
from custom_neuron import TimeParallel_LIFSpike, LIFSpikeLayer_Cons

torch.manual_seed(0)
DEV = "cpu"
T, B, C = 6, 4, 32
THRESH, DECAY, ISCALE, GAMA = 1.0, 0.5, 0.5, 1.0


def A(du_du, detach=False, reset="hard", **kw):
    d = dict(dS_du="Gamma", du_du=du_du, detach_reset=detach, reset_mode=reset,
             snnbp_epsilon=0.3, snnbp_alpha=2.0, snnbp_beta=2.0,
             snnbp_intervention=0.8, snnbp_eta=0.5, snnbp_p=0.5, snnbp_blend=0.5)
    d.update(kw)
    return argparse.Namespace(**d)


def surrogate_tri(u):
    return (1 / GAMA**2) * (GAMA - (u - THRESH).abs()).clamp(min=0)


class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u):
        ctx.save_for_backward(u)
        return (u > THRESH).float()

    @staticmethod
    def backward(ctx, g):
        (u,) = ctx.saved_tensors
        return g * surrogate_tri(u)


def ref_lif(x, detach=False, reset="hard"):
    """Pure-autograd reference implementing eq. (1) of the paper."""
    mem = torch.zeros(x.shape[1:], dtype=x.dtype)
    out = []
    for t in range(x.shape[0]):
        u1 = mem * DECAY + x[t] * ISCALE
        s = SpikeFn.apply(u1)
        out.append(s)
        sd = s.detach() if detach else s
        mem = u1 - THRESH * sd if reset == "soft" else u1 * (1 - sd)
    return torch.stack(out)


def grads(fn, x0, w):
    x = x0.clone().requires_grad_(True)
    (fn(x) * w).sum().backward()
    return x.grad.clone()


x0 = torch.randn(T, B, C) * 1.5
w = torch.randn(T, B, C)

print("=" * 78)
print("1. Hand-written backward (du_du='LIF') vs autograd reference")
print("=" * 78)
for reset in ("hard", "soft"):
    for detach in (False, True):
        gr = grads(lambda x: ref_lif(x, detach, reset), x0, w)
        gc = grads(lambda x: TimeParallel_LIFSpike.apply(
            x, THRESH, DECAY, ISCALE, GAMA, A("LIF", detach, reset), detach, reset), x0, w)
        err = (gr - gc).abs().max().item()
        print(f"  reset={reset:4s} detach={str(detach):5s}  max|Δ| = {err:.3e}   "
              f"{'MATCH' if err < 1e-6 else '**MISMATCH**'}")

print()
print("=" * 78)
print("2. Do snnbp_eta / snnbp_blend / snnbp_p change conservative_cgrad?")
print("=" * 78)
base = grads(lambda x: TimeParallel_LIFSpike.apply(
    x, THRESH, DECAY, ISCALE, GAMA, A("conservative_cgrad", True), True, "hard"), x0, w)
for name, vals in [("snnbp_eta", [0.05, 0.95]), ("snnbp_blend", [0.05, 0.95]),
                   ("snnbp_p", [0.05, 20.0]), ("snnbp_alpha", [0.5, 8.0]),
                   ("snnbp_beta", [0.5, 8.0]), ("snnbp_epsilon", [0.05, 0.9]),
                   ("snnbp_intervention", [0.2, 0.95])]:
    deltas = []
    for v in vals:
        g = grads(lambda x: TimeParallel_LIFSpike.apply(
            x, THRESH, DECAY, ISCALE, GAMA, A("conservative_cgrad", True, **{name: v}),
            True, "hard"), x0, w)
        deltas.append((g - base).abs().max().item())
    flag = "NO EFFECT (dead hyperparameter)" if max(deltas) == 0.0 else "active"
    print(f"  {name:20s} {str(vals):14s} max|Δgrad| = {max(deltas):.3e}   {flag}")

print()
print("=" * 78)
print("3. conservative_cgrad_2 (eta/p wired in) — same test")
print("=" * 78)
base2 = grads(lambda x: TimeParallel_LIFSpike.apply(
    x, THRESH, DECAY, ISCALE, GAMA, A("conservative_cgrad_2", True), True, "hard"), x0, w)
for name, vals in [("snnbp_eta", [0.05, 0.95]), ("snnbp_p", [0.05, 20.0]),
                   ("snnbp_blend", [0.05, 0.95])]:
    deltas = [ (grads(lambda x: TimeParallel_LIFSpike.apply(
        x, THRESH, DECAY, ISCALE, GAMA, A("conservative_cgrad_2", True, **{name: v}),
        True, "hard"), x0, w) - base2).abs().max().item() for v in vals]
    print(f"  {name:20s} max|Δgrad| = {max(deltas):.3e}   "
          f"{'NO EFFECT' if max(deltas)==0 else 'active'}")

print()
print("=" * 78)
print("4. Intervention statistics: sign of m when C-Grad fires")
print("=" * 78)
# replicate the backward internals to count events
stats = {"fired": 0, "fired_m_neg": 0, "total": 0}
mem = torch.zeros(B, C)
u1s, sps = [], []
for t in range(T):
    u1 = mem * DECAY + x0[t] * ISCALE
    s = (u1 > THRESH).float()
    u1s.append(u1); sps.append(s)
    mem = u1 * (1 - s)
u1s = torch.stack(u1s); sps = torch.stack(sps)

gml = torch.zeros(B, C)
eps, al, be, h = 0.3, 2.0, 2.0, 0.8
for t in reversed(range(T)):
    u1, s = u1s[t], sps[t]
    dL_dS, dL_dU2 = w[t], gml * DECAY
    dS = surrogate_tri(u1)
    dU2 = 1 - s                                    # detach_reset=True
    m = torch.where(u1 < THRESH, dL_dS - u1 * dL_dU2, THRESH * dL_dU2 - dL_dS)
    mg = torch.where(u1 < THRESH, m, -m)
    bf = dL_dS * dS + dL_dU2 * dU2
    mis = -mg * bf
    d = u1 - THRESH
    sig = torch.sigmoid(al * (m.abs() - 0.1)) * torch.sigmoid(be * (eps - d.abs())) \
        * torch.sigmoid(al * mis)
    do = (sig > h)
    stats["fired"] += do.sum().item()
    stats["fired_m_neg"] += (do & (m < 0)).sum().item()
    stats["total"] += do.numel()
    blend = do.float() * torch.sigmoid(5 * (sig - h))
    gml = (1 - 0.5 * blend) * bf + 0.5 * blend * torch.sign(mg) * bf.abs()

print(f"  neuron-timesteps: {stats['total']}")
print(f"  interventions fired: {stats['fired']} "
      f"({100*stats['fired']/stats['total']:.2f}%)")
print(f"  of those, m < 0 (flip would REDUCE loss; paper says do not intervene): "
      f"{stats['fired_m_neg']}")

print()
print("=" * 78)
print("5. Correction direction: code vs paper Eq.(18) vs Algorithm 1")
print("=" * 78)
rho = THRESH - u1s
mm = torch.where(u1s < THRESH, torch.ones_like(rho), -torch.ones_like(rho))
print("  Eq.(18):     g_corr = -sign(rho)*|g_base|   -> descent step moves u1 TOWARD threshold")
print("  Algorithm 1: g_corr = +sign(m*rho)*|g_base| -> moves u1 AWAY (when m>0)")
print("  Code:        g_corr = sign(m_grad)*|g_base| = sign(m*rho)*|g_base|  == Algorithm 1")
print("  => Eq.(18) and Algorithm 1 disagree in sign; the code follows Algorithm 1.")
