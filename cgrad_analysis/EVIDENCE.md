# C-Grad sweep evidence audit

Scope: sweeps `ojc4wc56` (v1, n=141 finished) and `eao1m37k` (v2, n=80 finished
at time of writing, 8 still running). Task: CIFAR10-DVS, `max_former`,
50 epochs, `val/mean_last10`. Chance = 10%.

Every claim below is graded. Grades mean:

- **A** — complete separation or a tight direct measurement; would need a bug,
  not noise, to be wrong.
- **B** — effect estimate whose 95% CI excludes zero, but resting on a post-hoc
  subgroup, small *n*, or an unrandomised split.
- **C** — suggestive; CI touches zero, or *n* < 10, or found by looking.
- **Not tested** — I asserted it earlier and the data does not address it.

---

## The single most important limitation

**There is no plain-BPTT baseline arm anywhere in this data.**

All 842 runs in the project use some C-Grad variant. The 229 `invariant_cgrad`
runs are the only ones reporting `val/mean_last10`; the other arms
(`conservative_cgrad`, `conservative_ablate`, `complex54`, …) ran 96 epochs
under a different metric and are not comparable.

Consequence: **no comparison in this report is C-Grad vs. no-C-Grad.** Every
number is C-Grad vs. C-Grad. The nearest available proxy is the subgroup of runs
whose intervention rate is near zero — those train as plain BPTT — but that is a
post-hoc split on an outcome variable, not a randomised control.

Second limitation: **n = 1 per configuration.** 80 distinct configs, 80 runs,
zero repeats. There is no estimate of pure seed variance anywhere, so every
standard deviation quoted below conflates hyperparameter effect with seed noise.

Third: single dataset, single architecture, single lr/wd. Nothing here
generalises without further runs.

---

## A1 — `two_sided=false` destroys training

| gate | two_sided | n | collapsed (<40%) | mean | range |
|---|---|---:|---:|---:|---|
| sign | True | 25 | 0 | 79.00 | 77.88 – 80.25 |
| sign | False | 19 | **19** | 17.31 | 9.76 – 31.95 |
| product | True | 14 | 0 | 71.90 | 50.16 – 80.10 |
| product | False | 22 | **22** | 14.27 | 9.99 – 24.76 |

41/41 vs 0/39. Complete separation, both gates, all correction modes.

**Grade A.** Caveat: one-sided gating changes *both* the sign symmetry and which
entries fire, so this does not isolate "asymmetry" as the cause — only that this
code path is fatal.

## A2 — corrections split evenly on the sign of `m` when two-sided

`cgrad/frac_interventions_m_neg`:

| gate | two_sided | n | mean | sd |
|---|---|---:|---:|---:|
| sign | True | 25 | 0.4980 | 0.0041 |
| product | True | 14 | 0.4970 | 0.0050 |
| sign | False | 19 | 0.1039 | 0.1397 |
| product | False | 22 | 0.0519 | 0.1378 |

**Grade A** for the measurement. 0.498 ± 0.004 is about as tight as a measured
quantity gets. One-sided gating does break the symmetry (0.50 → 0.05–0.10),
though with large spread (range up to 0.44).

This is a statement about the *distribution of correction directions*, not about
whether the corrections cancel in their effect on the weights. The latter is an
inference, not a measurement.

---

## B1 — the gate repair changes nothing except through `mask`

The headline I reported earlier (`sign − product = +7.1`) is **almost entirely
four runs.** Every sub-79 two-sided run in the sweep is `product` + `mask`:

| obj | gate | correction | ir% | eta |
|---:|---|---|---:|---:|
| 50.16 | product | mask | 13.83 | 0.84 |
| 54.08 | product | mask | 26.36 | 0.91 |
| 54.14 | product | mask | 26.41 | 0.97 |
| 57.95 | product | mask | 18.30 | 0.98 |

Excluding `mask` entirely:

```
sign    (blend+flip)  n=19  mean=78.904  sd=0.646
product (blend+flip)  n= 6  mean=78.973  sd=0.808
sign - product = -0.070  SE=0.362  95%CI=[-0.829, +0.690]
```

**No gate effect.** Within `mask`:

```
sign/mask     n=6  mean=79.317
product/mask  n=8  mean=66.586
diff = +12.730  SE=4.792  95%CI=[+2.668, +22.793]
```

**Grade B.** Reading: selectivity does not improve accuracy. It matters only as
protection against an aggressive correction — deleting 14–26% of gradient
entries is survivable when those entries are genuinely inconsistent (`sign`) and
fatal when ~85% of them were correctly signed (`product`). That is a coherent
mechanism, but it rests on 4 collapsed runs and a subgroup I found by looking.

## B2 — when the gate fires, accuracy drops slightly

Within `sign`/two-sided, splitting on intervention rate:

```
ir <= 1%  n= 9  mean=79.428  sd=0.635
ir >  1%  n=16  mean=78.764  sd=0.536
diff = -0.664  SE=0.251  95%CI=[-1.215, -0.113]
```

Same split in v1: `-0.177 SE=0.104`, CI `[-0.406, +0.052]` — touches zero.

**Grade B.** CI excludes zero in v2. But `ir` is an *outcome* of the sampled
hyperparameters (mostly `h`), so the two subgroups differ in `h`, `kappa` and
`beta` as well. This is the closest thing to a baseline comparison available,
and it points slightly negative.

## B3 — how large a benefit can be ruled out

```
v2 sign/two_sided  n= 25  mean=79.003  sd=0.648
v1 (all)           n=141  mean=79.006  sd=0.620
diff = -0.003  SE=0.140  95%CI=[-0.289, +0.284]
```

**Grade B**, and weaker than I presented it. Two problems:

1. v1 is not a baseline — it is C-Grad with the anti-selective gate.
2. The firing distributions differ. v1 swept `h ∈ [0.30, 0.75]` and 60% of its
   runs fired below 1%; v2 narrowed to `[0.30, 0.60]` and only 36% did. The two
   grand means average over different mixtures.

Matched on firing rate (ir ∈ [1%, 5%)), which is the honest version:

```
v2 sign - v2 product = -0.048  SE=0.419  95%CI=[-0.971, +0.875]  (n=5, 2)
v2 sign - v1 product = +0.128  SE=0.256  95%CI=[-0.435, +0.690]  (n=5, 10)
```

So: an effect larger than about **±0.3 points** is excluded by the pooled
comparison, and larger than about **±0.9 points** by the matched one. A real
gain of one or two tenths of a point is entirely compatible with this data and
would require seed replication to detect.

---

## C1 — `mask` may be the best correction mode

Within `sign`/two-sided:

```
blend  n= 8  mean=78.694  sd=0.404
mask   n= 6  mean=79.317  sd=0.599
flip   n=11  mean=79.056  sd=0.758
mask - blend = +0.623  SE=0.283  95%CI=[-0.017, +1.263]
```

**Grade C.** CI touches zero, *n* = 6, three pairwise comparisons available, no
correction applied. This estimate moved from 79.454 (n=5) to 79.317 (n=6) when a
single additional run landed during this analysis — a direct demonstration of
how unstable it is.

## C2 — best-run comparison

v1 max 80.73, top-5 mean 80.48. v2 (two-sided) max 80.25, top-5 mean 80.05.
**Grade C** — maxima over unequal *n* (141 vs 39) are not comparable without a
correction for the number of draws.

---

## Not tested

**"C-Grad provides no benefit over plain BPTT."** I stated this. The data does
not contain the comparison. It requires a `du_du` baseline arm at the same
50-epoch protocol.

**"The theory fails at every level."** Overreach. What is supported is narrower:
on this task, with this implementation, at these hyperparameter ranges, the
consistency-gate repair produced no measurable accuracy change, and the one
configuration that breaks the correction symmetry destroys training.

---

## Corrections to what I reported previously

1. **"`product` damages the network at high firing rates, rho(ir, obj) = −0.767."**
   Wrong as stated. v1's `product` arm runs at intervention rates up to **73.5%**
   with objectives of 77.5–79.8 throughout (n=20 above 20% firing, mean 78.91).
   The v2 correlation is real but is produced by the four `product`+`mask` runs;
   it is a gate×correction interaction, not a firing-rate effect.

2. **"The repaired gate produces exactly the same accuracy."** The honest form is
   a bound (±0.3 points), not an equality. And the comparison was between two
   different firing-rate distributions, not matched.

3. **"This closes the argument" / "tested at every level and fails at each."**
   Withdrawn. No baseline arm, no seed replicates, one task.

4. **Treating v1's 79.006 as a baseline.** It is not; it is C-Grad with a broken
   gate, 60% of whose runs barely fired.

---

## What would actually settle it

1. A `du_du` plain-BPTT arm, ~15 runs, same protocol. Without this nothing else
   is interpretable.
2. Seed replication: 5 seeds on 3–4 fixed configs, to separate seed noise from
   hyperparameter effect. Currently impossible.
3. `sign` × `mask` at fixed `h`, several seeds — the only cell with a plausible
   positive signal (C1).
4. Drop `two_sided=false`. Established at grade A; further runs are waste.
