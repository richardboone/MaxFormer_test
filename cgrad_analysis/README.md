# C-Grad implementation analysis

Verification scripts for the C-Grad implementation in `*/custom_neuron.py`.
Each is standalone; run from anywhere.

```bash
conda activate maxformer          # or: /home/rboone/.conda/envs/maxformer/bin/python
python cgrad_analysis/gate_bound.py                       # no GPU needed
CUDA_VISIBLE_DEVICES=0 python cgrad_analysis/scale_demo.py
```

## Scripts

| script | GPU | what it checks |
|---|---|---|
| `verify_cgrad.py` | no | Hand-written backward vs an independent autograd LIF; dead-hyperparameter test |
| `gate_bound.py` | no | Analytic ceiling on the intervention signal, per published config |
| `scale_demo.py` | no | Equivariance of the backward rule under loss rescaling |
| `measure_firing.py` | yes | Intervention rate in a real Max-Former backward, **without** AMP |
| `measure_firing_amp.py` | yes | Same, through the real AMP + GradScaler path the configs use |

The two `measure_firing*` scripts monkey-patch `TimeParallel_LIFSpike.backward` to
record gate statistics, then delegate to the original — training math is unchanged.
They run an untrained model on random inputs for a few steps, so absolute rates are
initialization-time figures, not mid-training ones.

## Findings

**1. The backward pass is correct.** `du_du: LIF` reproduces an independent autograd
reference to `1.2e-7` for both hard and soft reset. `m[t]` matches paper Eq. 13 / 22.

**2. `du_du: LIF` and `TET` ignore `detach_reset`.** Both recompute `dU2_dU1` in their
own branch (`custom_neuron.py`, the `elif mode == "TET"` / `else` tail) instead of using
the correctly-computed `dU2_dU1_standard` above, so the flag has no effect. This blocks
the matched in-framework baseline.

**3. `snnbp_eta`, `snnbp_blend`, `snnbp_p` are dead in `conservative_cgrad`** — read via
`getattr`, never used (η and the blend are hardcoded to `0.5`, γ to `5`, and
`g_misalign` uses `alpha` rather than `p`). Sweeping each over a 20x range gives
bitwise-identical gradients. `conservative_cgrad_2` wires η and p in, but nothing uses
that mode. Consequence: `event/run_hp_ablation.sh` sweeps `--snnbp-blend` and
`--snnbp-eta`, so those runs differ only by nondeterminism.

**4. The gates are not scale-invariant.** `g_m = sigmoid(a_m*(|m| - 0.1))` compares a
quantity with units of loss against a constant; `g_mis` is quadratic in the loss scale.
The correction direction `sign(m*rho)*|g_base|` *is* correctly equivariant, so the defect
is in *when* the rule fires, not *what* it does. Measured intervention rate on the same
gradients:

| loss scale | fire % | genuinely inconsistent % |
|---:|---:|---:|
| 1e-2 | 0.000% | 1.3% |
| 1e+0 | 0.000% | 1.3% |
| 1e+4 | 0.385% | 1.3% |
| 65536 (AMP default) | 0.752% | 1.3% |

Consistency is a sign test and is scale-free (right column). The gate is not (left).
Every config in the repo sets `amp: True`, and `GradScaler` halves on overflow and
doubles every 2000 steps, so the effective aggressiveness drifts *within* a run.

Structural ceiling: `intervention_signal <= sigmoid(a_d * eps)` regardless of state.
For the `benchmark_*.py` defaults (`a_d=2.0, eps=0.3, h=0.8`) that ceiling is `0.646 < h`,
so those timing benchmarks measure a path that never fires.

**5. Half of all interventions have `m < 0`.** Paper Sec 6.1 states "we only intervene
when m[t] > 0", but `g_m` uses `|m|`, so C-Grad fires as often to accelerate a
loss-reducing crossing as to suppress a harmful one. Arguably the better algorithm —
two-sided consistency enforcement — but not what the paper describes.

**6. Baselines are not matched.** `cifar10.yaml` / `cifar100.yaml` / `cifar10dvs.yaml`
set `use_custom_neuron: False` (SpikingJelly cupy, sigmoid surrogate a=4); C-Grad configs
set `True` (custom PyTorch neuron, triangular surrogate). Reported deltas bundle an
implementation swap and a surrogate swap with the algorithm.

**7. `harvest_conservative_metrics.py` is broken.** `set_metrics_collector` populates
`METRICS_COLLECTOR`, which is never read in any `custom_neuron.py` — the collection code
was removed, so the Appendix E figures cannot currently be regenerated. It also runs
without AMP, i.e. in the regime where C-Grad never fires.

## The `invariant_cgrad` mode

Implemented in `event/custom_neuron.py` as a new `du_du` mode; no existing mode is
touched. Config: `event/cifar10dvs_invariant.yaml`. Tests: `test_invariant_cgrad.py`.

```bash
conda run -n maxformer python cgrad_analysis/test_invariant_cgrad.py
cd event && python train.py -c ./cifar10dvs_invariant.yaml --data-path <DVS> --model max_former
```

Both `m[t]` and `dL_pred = rho[t]*base_function` carry units of loss, so normalising
them by a common RMS statistic of the current tensor makes every gate dimensionless:

```
norm     = sqrt(mean(m^2) + mean(dL_pred^2))       # per layer, per timestep, in fp32
g_m      = sigmoid(alpha_m * (|m/norm| - kappa))   # kappa is now relative
g_d      = sigmoid(alpha_d * (eps - |delta|))      # unchanged; delta is in units of V_th
g_cons   = sigmoid(alpha_c * (-(m/norm)*(dL_pred/norm)))   # < 0 exactly when Eq.(10) is violated
signal   = (g_m * g_d * g_cons)^(1/3)
```

Three changes beyond normalisation:

- **Geometric mean, not raw product.** A product of three sigmoids each bounded near 0.6
  has a ceiling near 0.3, so any `h` above that makes the rule silently dead — which is
  what `conservative_cgrad` does at the `benchmark_*.py` defaults. The geometric mean
  keeps the "all three gates agree" semantics while staying reachable, and makes `h`
  readable as "the typical gate value required to intervene".
- **`snnbp_h`, not `snnbp_intervention`.** The latter defaults to 0.8 for
  `conservative_cgrad`; reusing it would give a 0.00% intervention rate. A startup
  `RuntimeWarning` fires if `h` exceeds the attainable ceiling.
- **`eta` is applied**, per Algorithm 1 line 12. `snnbp_blend` is deliberately not read:
  Table 3 lists `b` as a hyperparameter but Algorithm 1 computes it, and a separate
  constant would be redundant with `eta`.

Verified properties (all in `test_invariant_cgrad.py`):

| property | result |
|---|---|
| equivariance, `g(c*grad)/c` vs `g(grad)` | `1.4e-07` (vs `3.9e-01` for `conservative_cgrad`) |
| firing rate across 9 orders of magnitude of loss scale | `2.972%`, spread `0.0` |
| fp16 / AMP path | finite, `2.4e-03` relative L2 vs fp32 |
| `\|dL_dU1\| <= \|base_function\|` pointwise | holds exactly (max ratio `1.000000`) |
| `eta=0` or `h -> 1` | reproduces detached BPTT bit-exactly |
| `snnbp_intervention` | correctly ignored |

The last two rows matter for review: the mode is a strict superset of the baseline, so
any accuracy difference is attributable to the intervention and nothing else.

`two_sided=True` (default) gates on `|m|`, matching what `conservative_cgrad` actually
does — roughly half of interventions have `m < 0`. Setting `two_sided=false` implements
Sec. 6.1 as literally written (harmful crossings only) and gives the ablation for free.
