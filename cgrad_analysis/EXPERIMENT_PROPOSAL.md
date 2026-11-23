# Proposed confirmatory experiment

Addresses two defects in the v2 sweep design, plus a measurement bug found while
checking them. All rate/selectivity numbers below are measured with the replica
in `test_consistency_gate.py`, which is cross-validated against the shipped
autograd Function to exact equality.

---

## Concern 1 — the `h` range

You are right that v2 reused v1's floor. The diagnosis in the v2 config header
was also wrong about *why* it mattered.

### What the range should have delivered vs. what it did

Measured rate/selectivity for the `sign` gate with `g_m`/`g_d` held permissive:

| h | reported rate | of which `bf=0` | selectivity (excl. `bf=0`) | **effective rate** | grad L1 moved |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 30.3% | 20.5% | 100% | 24.1% | 29.2% |
| 0.24 | 26.1% | 16.9% | 100% | 21.7% | 27.7% |
| 0.30 | 19.6% | 9.9% | 100% | 17.6% | 24.5% |
| 0.35 | 14.3% | 1.8% | 100% | 14.0% | 21.3% |
| 0.40 | 10.4% | 0.0% | 100% | 10.4% | 17.7% |
| 0.50 | 4.6% | 0.0% | 100% | 4.6% | 10.4% |

So `h ∈ [0.30, 0.60]` *should* have spanned 4.6–17.6% effective. The realized
sweep gave **median 2.3%, with 36% of `sign` runs below 1%** and only one run
above 20%.

**The range was not the binding problem — the uncontrolled nuisance gates were.**
`alpha`, `beta`, `kappa`, `epsilon` were swept alongside `h`, and `g_m·g_d`
shut the gate down independently of `h`. The intervention rate was therefore a
*random variable I could not control*, and most of the sweep landed at
near-baseline. That is why v2 has 39 usable runs mostly clustered at low rate.

### Fix

1. **Pin `alpha=0.51, beta=2.91, kappa=0.728, epsilon=0.470`** so `h` alone sets
   the rate. This is a mechanism test, not a hyperparameter search; the nuisance
   parameters should be controlled, not randomised.
2. **Set `alpha_c = 6.0`.** The selectivity boundary is
   `sigmoid(−alpha_c)^(1/3)` = 0.135, far below any operating `h`, so
   selectivity is exactly 100% across the whole usable range with headroom.
3. **Operating point `h = 0.24`** — the largest perturbation available at 100%
   selectivity: 21.7% effective rate, 27.7% of gradient L1 moved. This is the
   theory's own best case. `h = 0.40` (10.4%, 17.7% moved) as the secondary
   point if stage 1 is non-null.

Caveat: the table is measured on Gaussian synthetic input. Real gradient
distributions differ, so stage 0 reports the measured effective rate and `h` is
recalibrated before stage 1 if it lands far from 21.7%.

---

## Concern 2 — plain BPTT is not a sufficient control

Correct, and it rules out the design I proposed last message. Comparing C-Grad
to BPTT confounds *"the consistency information helps"* with *"the gradient
magnitude changed."* Two separate problems:

**(a) The obvious baseline is broken.** `du_du=<unrecognised>` falls through to a
branch that ignores `detach_reset` and uses `(1−S) − u₁·dS/du₁`, while
`invariant_cgrad`'s `base_function` uses `(1−S)`. Measured difference: **43.6%
relative** on the input gradient. That arm would have been useless.

The correct baseline is `invariant_cgrad` with **`snnbp_h = 1.01`**. The signal
is a geometric mean of sigmoids so `sig ≤ 1` always, the gate never fires, and
`dL_dU1 = base_function` exactly. Verified: max|diff| = **0.000e+00** against all
three correction modes. Same code path, same everything, zero intervention.
(It emits one reachability warning, which is the correct signal.)

**(b) Magnitude still confounds it.** Even with a correct baseline, `mask`
shrinks the gradient and `blend` rescales it, so any difference could be an
effective-learning-rate change rather than the consistency signal.

### Fix — use `flip`, and add a rate-matched random control

`flip` sets `dL_dU1 = sign(m_grad)·|base_function|` on fired entries. Since
`|sign(m)·|bf|| = |bf|`, **`flip` preserves the per-entry gradient magnitude
exactly.** It is the only correction mode with no magnitude confound at the
point of application.

That leaves exactly one thing that distinguishes C-Grad from a null: *which*
entries get flipped. So the control is:

**`randgate`** — flip a random subset of entries, rate-matched per batch to what
the real gate fired, with the consistency signal ignored.

| arm | what it isolates |
|---|---|
| `base` (h=1.01) | no intervention; anchor |
| `cgrad` (h=0.24) | the full method |
| `randgate` | same count, same magnitudes, same flip operation — random selection |

Readings:
- `cgrad ≈ randgate ≈ base` → the mechanism does nothing.
- `cgrad ≈ randgate`, both `≠ base` → it is the perturbation, not the consistency information.
- `cgrad > randgate` → **selecting on inconsistency is what helps.** This is the paper's claim, and this is the only comparison that can establish it.

Note: `flip` preserves magnitude *per timestep at the point of application*;
downstream gradients still differ because `dL_dU1` feeds the next step. But
`randgate` is subject to the identical downstream effect, which is precisely why
the control is needed rather than a bare BPTT comparison.

`randgate` is only valid with `flip` or `mask` (both use `do_intervene` alone).
It is invalid with `blend`, whose `w` couples to the real signal — assert on this.

---

## Bug found — `cgrad/intervention_rate` overcounts

**10.6% of all neuron-timesteps have `base_function == 0` exactly**: the Gamma
surrogate is zero for `|u₁−V_th| > γ`, and `(1−S) = 0` for spiking neurons, so
both terms vanish. For those entries `sign(pred) = 0`, hence
`g_cons = sigmoid(0) = 0.5`, hence `sig` can clear `h` — and the correction
`sign(m_grad)·|bf| = 0` is a **no-op**.

Consequences:
- Reported rate overstates the real one by up to ~20% relative (31.1% vs 24.9% at h=0.20).
- Apparent selectivity below 100% above the theoretical boundary was entirely
  this artifact. Excluding these entries, selectivity is exactly 100%, as derived.
- Every intervention rate in v1, v2, and both my reports is inflated.

Fix: add `cgrad/intervention_rate_effective` counting only
`do_intervene & (base_function != 0)`.

---

## Power analysis

Best available estimate of run-to-run sd is the **pooled within-cell sd of the
v2 two-sided runs, excluding `product/mask`**:

```
sign/blend     n= 8  sd=0.404
sign/mask      n= 6  sd=0.599
sign/flip      n=12  sd=0.723
product/blend  n= 5  sd=0.823
product/flip   n= 2  sd=0.700
pooled sd = 0.652   (df=28)
```

This is an **upper bound** on seed sd — each cell still spans the full range of
`h, kappa, eta, alpha, beta, epsilon`, so it contains hyperparameter variance
that the confirmatory design removes by pinning. True seed sd is plausibly
0.30–0.45, but **nothing in the existing data measures it**, because all 80 v2
configs are distinct and no configuration was ever repeated.

n per arm, 80% power, Bonferroni-corrected for 2 comparisons (α=0.025 two-sided):

| seed sd | δ=0.25 | δ=0.30 | δ=0.40 | δ=0.50 | δ=0.75 | δ=1.00 |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 19 | 14 | 8 | 5 | 3 | 2 |
| 0.30 | 28 | 19 | 11 | 7 | 4 | 2 |
| 0.40 | 49 | 34 | 19 | 13 | 6 | 4 |
| 0.50 | 76 | 53 | 30 | 19 | 9 | 5 |
| 0.652 | 129 | 90 | 51 | 33 | 15 | 9 |

**My earlier "15 runs" was indeed unbounded**: it detects δ = 0.34 if σ=0.30, but
only δ = 0.70 if σ=0.62. Since σ is unknown, the design has to measure it first.

**Minimum interesting effect: δ = 0.5 points.** Below that the method is not
worth its complexity, and it is smaller than the ±0.3 bound the existing data
already places on the gate repair.

---

## The runs

### Stage 0 — measure the seed sd (16 runs, ~62 GPU-h, ~8 h wall)

| arm | `snnbp_h` | seeds | n |
|---|---:|---|---:|
| `base` | 1.01 | 1–8 | 8 |
| `cgrad` | 0.24 | 1–8 | 8 |

Everything else identical and pinned:

```
--du-du invariant_cgrad --dS-du Gamma --detach-reset true --use-custom-neuron true
--snnbp-consistency sign --snnbp-two-sided true --snnbp-correction flip
--snnbp-alpha 0.51 --snnbp-beta 2.91 --snnbp-kappa 0.728 --snnbp-epsilon 0.470
--snnbp-alpha-cons 6.0 --snnbp-tau 0.5
--epochs 50 --lr 0.0045 --weight-decay 0.0079 --metric-window 10
--log-cgrad-stats true --cgrad-stats-prob 0.125
```

`eta` and `gamma` are unset — inert under `flip`.

Delivers: (i) seed sd to ±27% (90% CI roughly [0.56×, 1.44×]); (ii) the real
effective intervention rate, to recalibrate `h`; (iii) a first estimate of
`cgrad − base` with SE = 0.5σ, i.e. already conclusive if the true effect
exceeds ~0.87 points.

**Stop rule:** if stage 0 shows |cgrad − base| > 1.0 points, skip straight to
stage 1 at the implied n.

### Stage 1 — the controlled comparison (sized from stage 0)

Three arms at the stage-0 `h`, reusing the stage-0 runs as seeds 1–8:

| arm | `snnbp_h` | `snnbp_control` |
|---|---:|---|
| `base` | 1.01 | none |
| `cgrad` | 0.24 (or recalibrated) | none |
| `randgate` | same as `cgrad` | randgate |

Additional runs needed beyond stage 0, for δ=0.5 at 80% power:

| seed sd from stage 0 | n/arm | new runs | GPU-h | wall (8 concurrent) |
|---:|---:|---:|---:|---:|
| ≤ 0.30 | 7 | 7 | 27 | 3.9 h |
| 0.40 | 13 | 18 | 70 | 11.6 h |
| 0.50 | 19 | 30 | 116 | 19.4 h |
| 0.652 | 33 | 83 | 322 | 42.7 h |

Two pre-specified comparisons: `cgrad − base` and `cgrad − randgate`. No others.

This is an internal-pilot design (σ estimated from stage 0, which is then pooled
into the stage-1 test). Type-I inflation from reusing the pilot is small and
well characterised, but it is real — worth stating in any writeup.

### Stage 2 — conditional, only if stage 1 is non-null

Rate ladder at `h ∈ {0.40, 0.30, 0.20}`, `cgrad` and `randgate` arms, same n.
Establishes whether the effect scales with intervention rate as the theory
predicts. Not worth running against a null.

### Dropped

- `two_sided=false` — grade A established, 41/41 collapse. Further runs are waste.
- `blend` — `w` couples to the signal, so no clean rate-matched control exists.
- `product` gate — the v2 data bounds any `sign`-vs-`product` difference at
  ±0.83 outside `mask`; not the question anymore.

---

## Code required before stage 1

Three changes to `event/custom_neuron.py` (stage 0 needs none):

1. `--snnbp-control {none,randgate}`, applied after `do_intervene` is computed:
   ```python
   if control == 'randgate':
       p = do_intervene.to(signal.dtype).mean()
       do_intervene = (torch.rand_like(signal) < p).to(do_intervene.dtype)
   ```
   Per-batch rate matching, so no calibration run and no cross-run dependency.
2. Assert `control == 'none'` unless `correction in ('flip','mask')`.
3. `cgrad/intervention_rate_effective` excluding `base_function == 0`.

Stage 0 can start immediately — it uses only existing flags.
