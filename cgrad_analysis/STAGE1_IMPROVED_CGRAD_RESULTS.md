# Stage 1 CIFAR10-DVS Evaluation Summary

## Per-Arm Results (Mean ± SD across matched seeds)

| Arm | n (ok/nan) | val/mean_last10 | val/max | final_acc1 | eff_rate | mean_w | frac_eff_m_neg | s/ep |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A0_bptt_sig` | 3/0 | 78.39 ± 0.65 | 79.80 ± 0.61 | 78.17 | 0.00% | 0.0000 | 0.0% | 57.0s |
| `A1_impr_harmful_sig` | 2/0 | 30.53 ± 1.80 | 33.80 ± 2.40 | 30.45 | 68.00% | 0.0687 | 36.0% | 67.8s |
| `A2_impr_twosided_sig` | 3/0 | 79.11 ± 0.95 | 80.37 ± 0.55 | 79.53 | 50.79% | 0.0515 | 50.0% | 59.6s |
| `B0_bptt_gamma` | 3/0 | 79.43 ± 0.99 | 80.53 ± 0.92 | 79.53 | 0.00% | 0.0000 | 0.0% | 81.3s |
| `B1_impr_harmful_gamma` | 2/0 | 26.13 ± 6.12 | 31.95 ± 2.19 | 25.45 | 66.54% | 0.0673 | 36.2% | 81.6s |
| `B2_prior_inv_cgrad` | 3/0 | 78.63 ± 0.25 | 79.60 ± 0.10 | 78.50 | 15.13% | 0.2278 | 50.0% | 68.3s |
| `B3_impr_twosided_gamma` | 3/0 | 78.94 ± 0.72 | 80.00 ± 0.89 | 78.30 | 49.95% | 0.0512 | 50.0% | 74.9s |
| `C0_bptt_undetached` | 3/0 | 78.63 ± 1.13 | 79.73 ± 0.96 | 78.63 | 0.00% | 0.0000 | 0.0% | 61.9s |
| `C1_impr_undetached` | 2/0 | 40.31 ± 4.02 | 42.30 ± 3.82 | 40.40 | 61.80% | 0.0699 | 32.9% | 55.9s |
| `C2_impr_twosided_undetached` | 3/0 | 78.73 ± 0.58 | 79.70 ± 0.78 | 79.03 | 54.05% | 0.0499 | 50.0% | 52.6s |
| `D0_bptt_tau20_scale1` | 3/0 | 77.04 ± 0.23 | 77.97 ± 0.06 | 76.93 | 0.00% | 0.0000 | 0.0% | 50.2s |
| `D1_impr_tau20_scale1` | 2/0 | 36.88 ± 0.42 | 44.30 ± 0.14 | 38.25 | 31.35% | 0.0269 | 30.3% | 48.2s |
| `D2_impr_twosided_tau20_scale1` | 3/0 | 77.20 ± 0.75 | 78.17 ± 0.60 | 76.90 | 41.74% | 0.0386 | 50.0% | 49.0s |

## Paired Seed Contrasts (`val/mean_last10` and `val/max`)

| Contrast | Question | n_pairs | Δ mean_last10 (SE) | 95% CI | Δ val/max |
|---|---|---:|---:|---:|---:|
| `A1_impr_harmful_sig - A0_bptt_sig` | Improved C-Grad (harmful) vs Matched BPTT [sigmoid] | 2 | -48.07 (0.73) | [-57.35, -38.79] | -46.15 |
| `A2_impr_twosided_sig - A0_bptt_sig` | Improved C-Grad (two-sided) vs Matched BPTT [sigmoid] | 3 | +0.72 (0.81) | [-2.77, +4.20] | +0.57 |
| `A2_impr_twosided_sig - A1_impr_harmful_sig` | Two-sided vs Harmful-only gating [sigmoid] | 2 | +48.03 (1.23) | [+32.40, +63.66] | +46.25 |
| `B1_impr_harmful_gamma - B0_bptt_gamma` | Improved C-Grad (harmful) vs Matched BPTT [Gamma] | 2 | -52.77 (3.97) | [-103.21, -2.33] | -48.05 |
| `B3_impr_twosided_gamma - B0_bptt_gamma` | Improved C-Grad (two-sided) vs Matched BPTT [Gamma] | 3 | -0.49 (0.70) | [-3.49, +2.51] | -0.53 |
| `B3_impr_twosided_gamma - B1_impr_harmful_gamma` | Two-sided vs Harmful-only gating [Gamma] | 2 | +52.64 (4.99) | [-10.82, +116.11] | +48.20 |
| `B1_impr_harmful_gamma - B2_prior_inv_cgrad` | Improved C-Grad (harmful) vs Prior invariant_cgrad [Gamma] | 2 | -52.46 (4.08) | [-104.37, -0.56] | -47.60 |
| `B3_impr_twosided_gamma - B2_prior_inv_cgrad` | Improved C-Grad (two-sided) vs Prior invariant_cgrad [Gamma] | 3 | +0.31 (0.54) | [-2.02, +2.64] | +0.40 |
| `C0_bptt_undetached - A0_bptt_sig` | detach_reset=False vs True [BPTT, sigmoid] | 3 | +0.24 (0.60) | [-2.36, +2.84] | -0.07 |
| `C1_impr_undetached - C0_bptt_undetached` | Improved C-Grad (harmful) vs Matched BPTT [detach_reset=False] | 2 | -37.98 (1.88) | [-61.93, -14.02] | -37.35 |
| `C2_impr_twosided_undetached - C0_bptt_undetached` | Improved C-Grad (two-sided) vs Matched BPTT [detach_reset=False] | 3 | +0.10 (0.44) | [-1.80, +2.00] | -0.03 |
| `C2_impr_twosided_undetached - C1_impr_undetached` | Two-sided vs Harmful-only gating [detach_reset=False] | 2 | +38.46 (2.26) | [+9.74, +67.18] | +37.65 |
| `C2_impr_twosided_undetached - A2_impr_twosided_sig` | detach_reset=False vs True [Improved C-Grad two-sided] | 3 | -0.38 (0.66) | [-3.24, +2.48] | -0.67 |
| `D0_bptt_tau20_scale1 - C0_bptt_undetached` | tau=20/scale=1.0 vs tau=2/coupled [BPTT, detach=F] | 3 | -1.59 (0.61) | [-4.22, +1.03] | -1.77 |
| `D1_impr_tau20_scale1 - D0_bptt_tau20_scale1` | Improved C-Grad (harmful) vs Matched BPTT [tau=20, scale=1.0] | 2 | -40.24 (0.49) | [-46.47, -34.01] | -33.65 |
| `D2_impr_twosided_tau20_scale1 - D0_bptt_tau20_scale1` | Improved C-Grad (two-sided) vs Matched BPTT [tau=20, scale=1.0] | 3 | +0.16 (0.48) | [-1.92, +2.25] | +0.20 |
| `D2_impr_twosided_tau20_scale1 - D1_impr_tau20_scale1` | Two-sided vs Harmful-only gating [tau=20, scale=1.0] | 2 | +39.92 (0.53) | [+33.19, +46.65] | +33.55 |
| `D2_impr_twosided_tau20_scale1 - C2_impr_twosided_undetached` | tau=20/scale=1.0 vs tau=2/coupled [Improved C-Grad two-sided] | 3 | -1.53 (0.49) | [-3.63, +0.57] | -1.53 |
| `A0_bptt_sig - B0_bptt_gamma` | Surrogate sigmoid vs Gamma [BPTT] | 3 | -1.04 (0.90) | [-4.93, +2.85] | -0.73 |
| `A2_impr_twosided_sig - B3_impr_twosided_gamma` | Surrogate sigmoid vs Gamma [Improved C-Grad two-sided] | 3 | +0.17 (0.53) | [-2.10, +2.44] | +0.37 |

## Seed-Level Details

| Arm | Seed | Status | mean_last10 | val/max | final_acc1 | eff_rate | s/ep |
|---|---:|---|---:|---:|---:|---:|---:|
| `A0_bptt_sig` | 42 | completed | 78.06 | 79.40 | 78.10 | 0.00% | 75.2s |
| `A0_bptt_sig` | 43 | completed | 79.14 | 80.50 | 78.90 | 0.00% | 49.2s |
| `A0_bptt_sig` | 44 | completed | 77.98 | 79.50 | 77.50 | 0.00% | 46.7s |
| `A1_impr_harmful_sig` | 42 | completed | 29.26 | 32.10 | 32.10 | 68.57% | 83.4s |
| `A1_impr_harmful_sig` | 43 | completed | 31.80 | 35.50 | 28.80 | 67.43% | 52.3s |
| `A2_impr_twosided_sig` | 42 | completed | 78.52 | 80.10 | 78.60 | 50.96% | 80.1s |
| `A2_impr_twosided_sig` | 43 | completed | 78.60 | 80.00 | 79.10 | 50.75% | 51.1s |
| `A2_impr_twosided_sig` | 44 | completed | 80.21 | 81.00 | 80.90 | 50.65% | 47.5s |
| `B0_bptt_gamma` | 42 | completed | 79.26 | 80.00 | 79.60 | 0.00% | 150.4s |
| `B0_bptt_gamma` | 43 | completed | 78.54 | 80.00 | 78.80 | 0.00% | 49.4s |
| `B0_bptt_gamma` | 44 | completed | 80.50 | 81.60 | 80.20 | 0.00% | 44.0s |
| `B1_impr_harmful_gamma` | 42 | completed | 30.46 | 33.50 | 29.50 | 66.92% | 107.8s |
| `B1_impr_harmful_gamma` | 43 | completed | 21.80 | 30.40 | 21.40 | 66.16% | 55.4s |
| `B2_prior_inv_cgrad` | 42 | completed | 78.84 | 79.50 | 78.70 | 15.03% | 68.4s |
| `B2_prior_inv_cgrad` | 43 | completed | 78.35 | 79.60 | 78.00 | 14.95% | 68.3s |
| `B2_prior_inv_cgrad` | 44 | completed | 78.70 | 79.70 | 78.80 | 15.40% | 68.4s |
| `B3_impr_twosided_gamma` | 42 | completed | 78.11 | 79.30 | 77.70 | 48.83% | 115.6s |
| `B3_impr_twosided_gamma` | 43 | completed | 79.44 | 81.00 | 78.90 | 50.30% | 55.0s |
| `B3_impr_twosided_gamma` | 44 | completed | 79.27 | 79.70 | 78.30 | 50.73% | 54.0s |
| `C0_bptt_undetached` | 42 | completed | 77.33 | 78.70 | 77.30 | 0.00% | 80.1s |
| `C0_bptt_undetached` | 43 | completed | 79.24 | 80.60 | 79.20 | 0.00% | 53.4s |
| `C0_bptt_undetached` | 44 | completed | 79.33 | 79.90 | 79.40 | 0.00% | 52.2s |
| `C1_impr_undetached` | 42 | completed | 37.47 | 39.60 | 36.90 | 62.30% | 58.0s |
| `C1_impr_undetached` | 43 | completed | 43.15 | 45.00 | 43.90 | 61.31% | 53.8s |
| `C2_impr_twosided_undetached` | 42 | completed | 78.19 | 79.30 | 78.70 | 54.07% | 42.9s |
| `C2_impr_twosided_undetached` | 43 | completed | 79.35 | 80.60 | 79.80 | 54.21% | 54.0s |
| `C2_impr_twosided_undetached` | 44 | completed | 78.66 | 79.20 | 78.60 | 53.86% | 60.9s |
| `D0_bptt_tau20_scale1` | 42 | completed | 76.92 | 78.00 | 76.80 | 0.00% | 41.4s |
| `D0_bptt_tau20_scale1` | 43 | completed | 77.31 | 77.90 | 77.00 | 0.00% | 52.9s |
| `D0_bptt_tau20_scale1` | 44 | completed | 76.89 | 78.00 | 77.00 | 0.00% | 56.3s |
| `D1_impr_tau20_scale1` | 42 | completed | 37.17 | 44.40 | 42.90 | 29.24% | 40.5s |
| `D1_impr_tau20_scale1` | 43 | completed | 36.58 | 44.20 | 33.60 | 33.46% | 55.9s |
| `D2_impr_twosided_tau20_scale1` | 42 | completed | 76.56 | 77.60 | 75.80 | 40.50% | 40.0s |
| `D2_impr_twosided_tau20_scale1` | 43 | completed | 77.03 | 78.10 | 76.90 | 42.25% | 47.8s |
| `D2_impr_twosided_tau20_scale1` | 44 | completed | 78.02 | 78.80 | 78.00 | 42.48% | 59.3s |
