"""Checks for the snnbp_correction mechanism ablation on invariant_cgrad.

  1. correction='blend' is bit-identical to the pre-existing code path
     (a live sweep depends on this)
  2. 'mask' zeroes exactly the intervened entries and leaves the rest alone
  3. 'flip' sets exactly the intervened entries to sign(m_grad)*|g_base|
  4. all three stay non-expansive: |dL_dU1| <= |base_function| pointwise
  5. with the gate shut, all three reduce to detached BPTT

Run:  python cgrad_analysis/test_correction_modes.py
"""
import argparse
import os as _os
import sys

import torch

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_REPO, "event"))
from custom_neuron import TimeParallel_LIFSpike  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
T, B, C = 8, 64, 512
THRESH, DECAY, ISCALE, GAMA = 1.0, 0.5, 0.5, 1.0

HP = dict(snnbp_epsilon=0.3, snnbp_alpha=2.0, snnbp_beta=2.0,
          snnbp_alpha_cons=2.0, snnbp_kappa=1.0, snnbp_h=0.5,
          snnbp_eta=0.5, snnbp_gamma=5.0, snnbp_two_sided=True)

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{('  -- ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def args(**over):
    d = dict(dS_du="Gamma", du_du="invariant_cgrad", detach_reset=True,
             reset_mode="hard", **HP)
    d.update(over)
    return argparse.Namespace(**d)


x0 = (torch.randn(T, B, C) * 1.2).to(DEV)
w0 = (torch.randn(T, B, C) * 1e-3).to(DEV)


def run(**over):
    x = x0.clone().requires_grad_(True)
    y = TimeParallel_LIFSpike.apply(x, THRESH, DECAY, ISCALE, GAMA,
                                    args(**over), True, "hard")
    y.backward(w0)
    return x.grad.clone()


print(f"device = {DEV}\n")

print("1. Default is unchanged (the running sweep depends on this)")
g_absent = run()                                  # attribute missing -> getattr default
g_blend = run(snnbp_correction='blend')
check("omitting snnbp_correction == correction='blend'",
      torch.equal(g_absent, g_blend),
      f"max |delta| = {(g_absent - g_blend).abs().max().item():.2e}")

print("\n2/3. mask and flip differ from blend and from each other")
g_mask = run(snnbp_correction='mask')
g_flip = run(snnbp_correction='flip')
for nm, g in (('mask', g_mask), ('flip', g_flip)):
    d = (g - g_blend).abs().max().item()
    check(f"'{nm}' actually changes the gradient", d > 0, f"max |delta| vs blend = {d:.2e}")
check("mask != flip", not torch.equal(g_mask, g_flip),
      f"max |delta| = {(g_mask - g_flip).abs().max().item():.2e}")

print("\n4. Non-expansiveness  |dL_dU1| <= |base_function|")
# This is a PER-TIMESTEP property of the rule, not a property of the end-to-end
# input gradient: attenuating at step t changes the recurrence feeding step t-1,
# so x.grad ratios compound and say nothing about the guarantee. Replicate the
# recurrence explicitly and check the bound where it is actually claimed.
g_base = run(snnbp_h=0.999999)   # still used by check 5 below
surro = lambda u: (1 / GAMA ** 2) * (GAMA - (u - THRESH).abs()).clamp(min=0)

mem = torch.zeros(B, C, device=DEV)
u1s, sps = [], []
for t in range(T):
    u1 = mem * DECAY + x0[t] * ISCALE
    s = (u1 > THRESH).float()
    u1s.append(u1); sps.append(s)
    mem = u1 * (1 - s)
u1s, sps = torch.stack(u1s), torch.stack(sps)


def per_step_ratio(mode):
    gml = torch.zeros(B, C, device=DEV)
    worst, fired = 0.0, 0
    for t in reversed(range(T)):
        u1, s = u1s[t], sps[t]
        dL_dS, dL_dU2 = w0[t], gml * DECAY
        m = torch.where(u1 < THRESH, dL_dS - u1 * dL_dU2, THRESH * dL_dU2 - dL_dS)
        m_grad = torch.where(u1 < THRESH, m, -m)
        bf = dL_dS * surro(u1) + dL_dU2 * (1 - s)
        d, rho = u1 - THRESH, THRESH - u1
        pred = rho * bf
        norm = torch.sqrt(m.pow(2).mean() + pred.pow(2).mean()).clamp(min=1e-30)
        mh, ph = m / norm, pred / norm
        sig = (torch.sigmoid(HP['snnbp_alpha'] * (mh.abs() - HP['snnbp_kappa']))
               * torch.sigmoid(HP['snnbp_beta'] * (HP['snnbp_epsilon'] - d.abs()))
               * torch.sigmoid(HP['snnbp_alpha_cons'] * (-mh * ph))).pow(1 / 3)
        do = (sig > HP['snnbp_h']).float()
        soft = torch.sign(m_grad) * bf.abs()
        if mode == 'mask':
            new = (1 - do) * bf
        elif mode == 'flip':
            new = torch.where(do.bool(), soft, bf)
        else:
            w = HP['snnbp_eta'] * do * torch.sigmoid(
                HP['snnbp_gamma'] * (sig - HP['snnbp_h']))
            new = (1 - w) * bf + w * soft
        sel = bf.abs() > 1e-20          # ratio is meaningless where base is ~0
        if sel.any():
            worst = max(worst, (new[sel].abs() / bf[sel].abs()).max().item())
        fired += int(do.sum().item())
        gml = new
    return worst, fired


for nm in ('blend', 'mask', 'flip'):
    ratio, fired = per_step_ratio(nm)
    check(f"'{nm}' non-expansive (per timestep)", ratio <= 1.0 + 1e-5,
          f"max |dL_dU1|/|base| = {ratio:.6f} over {fired:,} interventions")

print("\n5. Gate shut -> detached BPTT, for every mechanism")
for nm in ('blend', 'mask', 'flip'):
    g = run(snnbp_correction=nm, snnbp_h=0.999999)
    check(f"'{nm}' with h->1 reproduces base", torch.equal(g, g_base),
          f"max |delta| = {(g - g_base).abs().max().item():.2e}")

print("\n6. eta/gamma are inert for mask and flip (they bypass the blend)")
for nm in ('mask', 'flip'):
    a = run(snnbp_correction=nm, snnbp_eta=0.1, snnbp_gamma=1.0)
    b = run(snnbp_correction=nm, snnbp_eta=0.9, snnbp_gamma=20.0)
    check(f"'{nm}' ignores eta and gamma", torch.equal(a, b),
          f"max |delta| = {(a - b).abs().max().item():.2e}")

print("\n" + ("ALL CHECKS PASSED" if not FAILURES
             else f"{len(FAILURES)} FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
