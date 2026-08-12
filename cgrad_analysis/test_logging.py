"""Quick check of the P0/P1 additions before committing a sweep to them.

  1. stats collection does not change any gradient
  2. the reported intervention rate matches an independent recount
  3. a dead-gate configuration is correctly reported as 0% (the case a sweep
     must be able to detect)
  4. backward-pass overhead of collection is acceptable
  5. val/mean_last{k} behaves as intended on a synthetic noisy accuracy curve

Run:  python cgrad_analysis/test_logging.py
"""
import argparse
import os as _os
import sys
import time

import torch

_REPO = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
sys.path.insert(0, _os.path.join(_REPO, "event"))
import custom_neuron  # noqa: E402
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

print("1. Collection does not perturb the gradient")
custom_neuron.disable_cgrad_stats()
g_off = run()
custom_neuron.enable_cgrad_stats(DEV, sample_prob=1.0)
g_on = run()
stats = custom_neuron.pop_cgrad_stats()
check("gradient identical with stats on vs off",
      torch.equal(g_off, g_on),
      f"max |delta| = {(g_off - g_on).abs().max().item():.2e}")

print("\n2. Reported rate matches an independent recount")
print(f"      reported intervention_rate = {stats['cgrad/intervention_rate']*100:.4f}%")
print(f"      reported mean_signal       = {stats['cgrad/mean_signal']:.4f}")
print(f"      reported near_threshold    = {stats['cgrad/near_threshold_rate']*100:.2f}%")
print(f"      frac interventions m<0     = {stats['cgrad/frac_interventions_m_neg']*100:.2f}%")

surro = lambda u: (1 / GAMA ** 2) * (GAMA - (u - THRESH).abs()).clamp(min=0)
mem = torch.zeros(B, C, device=DEV)
u1s, sps = [], []
for t in range(T):
    u1 = mem * DECAY + x0[t] * ISCALE
    s = (u1 > THRESH).float()
    u1s.append(u1)
    sps.append(s)
    mem = u1 * (1 - s)
u1s, sps = torch.stack(u1s), torch.stack(sps)

gml = torch.zeros(B, C, device=DEV)
fire = tot = 0
for t in reversed(range(T)):
    u1, s = u1s[t], sps[t]
    dL_dS, dL_dU2 = w0[t], gml * DECAY
    dS, dU2 = surro(u1), 1 - s
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
    do = sig > HP['snnbp_h']
    fire += do.sum().item()
    tot += do.numel()
    w = HP['snnbp_eta'] * do.float() * torch.sigmoid(
        HP['snnbp_gamma'] * (sig - HP['snnbp_h']))
    gml = (1 - w) * bf + w * torch.sign(mg) * bf.abs()
recount = fire / tot
check("reported rate matches recount",
      abs(recount - stats['cgrad/intervention_rate']) < 1e-9,
      f"recount = {recount*100:.4f}%")

print("\n3. A dead-gate config is reported as exactly 0%")
custom_neuron.enable_cgrad_stats(DEV, sample_prob=1.0)
custom_neuron._INVARIANT_GATE_CHECKED = True   # suppress the one-time warning
run(snnbp_h=0.99)
dead = custom_neuron.pop_cgrad_stats()
check("dead gate reports 0% intervention rate",
      dead['cgrad/intervention_rate'] == 0.0,
      f"rate = {dead['cgrad/intervention_rate']*100:.4f}%  "
      f"(this is the case a sweep must be able to detect)")

print("\n4. Overhead of collection")


def timeit(enabled, n=30):
    custom_neuron.enable_cgrad_stats(DEV) if enabled else custom_neuron.disable_cgrad_stats()  # default sampling
    for _ in range(5):
        run()
    if DEV == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        run()
    if DEV == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


off_ms, on_ms = timeit(False), timeit(True)
custom_neuron.disable_cgrad_stats()
over = 100 * (on_ms - off_ms) / off_ms
print(f"      stats off: {off_ms:.2f} ms/iter      stats on: {on_ms:.2f} ms/iter")
check("collection overhead under 8%", over < 8.0, f"overhead = {over:+.1f}%")

print("\n5. val/mean_last{k} vs val/max on a noisy curve")
torch.manual_seed(1)
# flat-mean curve with occasional lucky spikes, i.e. the situation where max
# rewards volatility rather than quality
curve = (80.0 + torch.randn(96) * 0.8).tolist()
k = 10
print(f"      val/max        = {max(curve):.2f}")
print(f"      val/mean_last{k} = {sum(curve[-k:]) / k:.2f}")
print(f"      true mean      = {sum(curve) / len(curve):.2f}")
spread_max, spread_mean = [], []
for trial in range(200):
    c = (80.0 + torch.randn(96) * 0.8).tolist()
    spread_max.append(max(c))
    spread_mean.append(sum(c[-k:]) / k)
sd_max = torch.tensor(spread_max).std().item()
sd_mean = torch.tensor(spread_mean).std().item()
bias_max = torch.tensor(spread_max).mean().item() - 80.0
print(f"      over 200 synthetic runs of identical quality:")
print(f"        val/max        bias {bias_max:+.2f}, sd {sd_max:.3f}")
print(f"        val/mean_last{k} bias {torch.tensor(spread_mean).mean().item()-80.0:+.2f}, "
      f"sd {sd_mean:.3f}")
check("trailing mean has lower run-to-run spread than max",
      sd_mean < sd_max, f"{sd_mean:.3f} vs {sd_max:.3f}")

print("\n" + ("ALL CHECKS PASSED" if not FAILURES
             else f"{len(FAILURES)} FAILED: {FAILURES}"))
sys.exit(1 if FAILURES else 0)
