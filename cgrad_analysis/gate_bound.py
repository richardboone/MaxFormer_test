"""Analytic reachability of the C-Grad intervention gate for the configs
actually used in the paper's experiments."""
import math

sig = lambda z: 1 / (1 + math.exp(-z))

CFG = [
    ("cifar10_best_cons.yaml / cifar100_best_cons.yaml / imagenet 10_512_t4_cons.yml",
     1.209609, 1.366561, 0.357430, 0.531940),
    ("event/cifar10dvs_cgrad.yaml  (the 84.2% CIFAR10-DVS result)",
     3.11, 1.21, 0.17, 0.24),
    ("event/cifar10dvs_hp_ablation.yaml (paper Table 3 defaults)",
     1.2, 1.3, 0.3, 0.5),
    ("event/cifar10dvs_rebuttal_ablate.yaml (gate-ablation runs, Table 2)",
     2.01, 4.06, 0.437, 0.562),
    ("benchmark_*.py defaults", 2.0, 2.0, 0.3, 0.8),
]

print(f"{'config':<62} {'a_m':>6} {'a_d':>6} {'eps':>6} {'h':>6} "
      f"{'max g_d':>8} {'max sig':>8} {'req g_m*g_mis':>14}")
print("-" * 128)
for name, am, ad, eps, h in CFG:
    gd_max = sig(ad * eps)          # attained only exactly at delta = 0
    sig_max = gd_max * 1.0 * 1.0    # g_m, g_mis -> 1 in the limit
    if sig_max <= h:
        req = "UNREACHABLE"
    else:
        req = f"{h / gd_max:.4f}"
    print(f"{name:<62} {am:6.3f} {ad:6.3f} {eps:6.3f} {h:6.3f} "
          f"{gd_max:8.4f} {sig_max:8.4f} {req:>14}")

print()
print("Interpretation: intervention_signal = g_m * g_d * g_mis, and g_d = sigmoid(a_d*(eps-|delta|))")
print("is bounded above by sigmoid(a_d*eps) < 1. C-Grad fires only where signal > h, so the")
print("supremum of the signal over ALL states is sigmoid(a_d*eps). The last column is the")
print("product g_m*g_mis needed at delta=0 exactly; away from threshold the bar is higher still.")
print()
print("Required |m| at delta=0, assuming the misalignment gate is fully open (g_mis -> 1):")
for name, am, ad, eps, h in CFG:
    gd_max = sig(ad * eps)
    if h / gd_max >= 1:
        print(f"  {name[:58]:<58}  impossible (gate can never open)")
        continue
    y = h / gd_max
    m_req = math.log(y / (1 - y)) / am + 0.1
    print(f"  {name[:58]:<58}  |m| > {m_req:8.3f}")
