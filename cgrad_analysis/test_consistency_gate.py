"""Does the revised consistency gate select FOR inconsistency, or against it?

Ground truth (paper Eq. 10, independent of any smoothing): an update is
inconsistent when sign(rho*g_base) != sign(m), i.e. rho*g_base*m < 0.

The bar: a gate that fires on X% of entries should find genuinely inconsistent
entries at a rate ABOVE the population base rate. The shipped 'product' gate
sits below it, which is worse than firing at random.

Every reported number comes from a hand-rolled replica of the backward pass, so
step 0 cross-checks that replica against the real autograd Function -- a replica
that has drifted from the module would invalidate everything downstream.

Run:  CUDA_VISIBLE_DEVICES=3 python cgrad_analysis/test_consistency_gate.py
"""
import argparse
import os as _os
import sys

import torch

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_REPO, "event"))
import custom_neuron  # noqa: E402
from custom_neuron import TimeParallel_LIFSpike  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
T, B, C = 16, 128, 512
THRESH, DECAY, ISCALE, GAMA = 1.0, 0.5, 0.5, 1.0

# hyperparameters of the highest-firing real sweep run
HP = dict(snnbp_alpha=0.5100895272622367, snnbp_beta=2.911823363394474,
          snnbp_alpha_cons=2.2730509544985, snnbp_kappa=0.7279949600359477,
          snnbp_epsilon=0.4698587395133217, snnbp_eta=0.6034655464254214,
          snnbp_gamma=5.0, snnbp_two_sided=True)

g = torch.Generator().manual_seed(0)
X = (torch.randn(T, B, C, generator=g) * 1.2).to(DEV)
W = (torch.randn(T, B, C, generator=g) * 1e-3).to(DEV)
surro = lambda u: (1 / GAMA ** 2) * (GAMA - (u - THRESH).abs()).clamp(min=0)


def ns(**over):
    d = dict(dS_du="Gamma", du_du="invariant_cgrad", detach_reset=True,
             reset_mode="hard", snnbp_h=0.5, snnbp_consistency='product',
             snnbp_correction='blend', **HP)
    d.update(over)
    return argparse.Namespace(**d)


def real_rate(**over):
    """Intervention rate straight from the shipped autograd Function."""
    custom_neuron.enable_cgrad_stats(DEV, sample_prob=1.0)
    custom_neuron._INVARIANT_GATE_CHECKED = True
    x = X.clone().requires_grad_(True)
    TimeParallel_LIFSpike.apply(x, THRESH, DECAY, ISCALE, GAMA,
                                ns(**over), True, "hard").backward(W)
    s = custom_neuron.pop_cgrad_stats()
    custom_neuron.disable_cgrad_stats()
    return s['cgrad/intervention_rate']


def replica(consistency, h):
    """Hand-rolled replica; returns (rate, frac_fired_inconsistent, base_rate)."""
    mem = torch.zeros(B, C, device=DEV)
    u1s, sps = [], []
    for t in range(T):
        u1 = mem * DECAY + X[t] * ISCALE
        s = (u1 > THRESH).float()
        u1s.append(u1); sps.append(s); mem = u1 * (1 - s)
    u1s, sps = torch.stack(u1s), torch.stack(sps)

    gml = torch.zeros(B, C, device=DEV)
    fired = inc_fired = inc_all = total = 0
    for t in reversed(range(T)):
        u1, s = u1s[t], sps[t]
        dL_dS, dL_dU2 = W[t], gml * DECAY
        m = torch.where(u1 < THRESH, dL_dS - u1 * dL_dU2, THRESH * dL_dU2 - dL_dS)
        m_grad = torch.where(u1 < THRESH, m, -m)
        bf = dL_dS * surro(u1) + dL_dU2 * (1 - s)
        d, rho = u1 - THRESH, THRESH - u1

        # ground truth, always from the unmodified Eq. 10
        truth = (rho * bf * m) < 0

        if consistency == 'product':
            pred = rho * bf
        else:
            pred = torch.where(u1 < THRESH, bf, -bf) * THRESH

        norm = torch.sqrt(m.pow(2).mean() + pred.pow(2).mean()).clamp(min=1e-30)
        mh, ph = m / norm, pred / norm
        gm = torch.sigmoid(HP['snnbp_alpha'] * (mh.abs() - HP['snnbp_kappa']))
        gd = torch.sigmoid(HP['snnbp_beta'] * (HP['snnbp_epsilon'] - d.abs()))
        if consistency == 'sign':
            gc = torch.sigmoid(HP['snnbp_alpha_cons'] * (-torch.sign(mh) * torch.sign(ph)))
        else:
            gc = torch.sigmoid(HP['snnbp_alpha_cons'] * (-mh * ph))
        sig = (gm * gd * gc).pow(1 / 3)
        do = sig > h

        fired += do.sum().item()
        inc_fired += (do & truth).sum().item()
        inc_all += truth.sum().item()
        total += do.numel()

        w = HP['snnbp_eta'] * do.float() * torch.sigmoid(
            HP['snnbp_gamma'] * (sig - h))
        gml = (1 - w) * bf + w * torch.sign(m_grad) * bf.abs()

    return (fired / total,
            (inc_fired / fired) if fired else float('nan'),
            inc_all / total)


print(f"device = {DEV}\n")
print("0. Replica agrees with the shipped module (validates everything below)")
ok = True
for cons in ('product', 'ref', 'sign'):
    for h in (0.35, 0.50):
        a = real_rate(snnbp_consistency=cons, snnbp_h=h)
        b, _, _ = replica(cons, h)
        match = abs(a - b) < 1e-9
        ok &= match
        print(f"  [{'PASS' if match else 'FAIL'}] {cons:8} h={h:.2f}  "
              f"module={a*100:7.3f}%  replica={b*100:7.3f}%")
if not ok:
    sys.exit("replica does not match the module -- results below are invalid")

_, _, base = replica('product', 0.5)
print(f"\nBASE RATE: {base*100:.1f}% of all neuron-timesteps are genuinely inconsistent")
print("A useful gate fires on MORE than this. The shipped one fires on fewer.\n")

print(f"{'gate':>9} {'h':>5} {'fires':>9} {'% of fired inconsistent':>24} {'vs base':>9}")
print("-" * 62)
for cons in ('product', 'ref', 'sign'):
    for h in (0.30, 0.35, 0.40, 0.50, 0.60):
        rate, sel, _ = replica(cons, h)
        if rate < 1e-6:
            print(f"{cons:>9} {h:5.2f} {rate*100:8.3f}%  {'(never fires)':>23}")
            continue
        print(f"{cons:>9} {h:5.2f} {rate*100:8.3f}% {sel*100:23.1f}% "
              f"{(sel-base)*100:+8.1f}")
    print()
