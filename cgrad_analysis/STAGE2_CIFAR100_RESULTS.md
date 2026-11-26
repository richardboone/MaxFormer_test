# Stage 2 CIFAR-100 MaxFormer Results (`T=4`, `400+10` Epochs)

## Per-Arm Summary (Ranked by `val/mean_last10`)

| Arm | Surr | n (ok/nan) | val/mean_last10 | val/max | final_acc1 | eff_rate | mean_w | m_neg | s/ep |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C100_A0_bptt_sig` | sigmoid | 3/0 | 81.78 ± 0.17 | 82.30 ± 0.15 | 81.52 | 0.00% | 0.0000 | 0.0% | 35.1s |
| `C100_B0_bptt_gamma` | Gamma | 3/0 | 81.78 ± 0.13 | 82.23 ± 0.10 | 81.74 | 0.00% | 0.0000 | 0.0% | 34.9s |
| `C100_A1_T8_gentle` | sigmoid | 3/0 | 81.77 ± 0.15 | 82.36 ± 0.25 | 81.60 | 2.66% | 0.0029 | 50.0% | 37.5s |
| `C100_B1_G3_flip` | Gamma | 3/0 | 81.67 ± 0.11 | 82.21 ± 0.21 | 81.65 | 0.90% | 0.0077 | 50.0% | 37.7s |

## Paired Seed Contrasts

| Contrast | Question | n_pairs | Δ mean_last10 (SE) | 95% CI | Δ val/max | wins |
|---|---|---:|---:|---:|---:|---:|
| `C100_A1_T8_gentle - C100_A0_bptt_sig` | Improved C-Grad smooth (T8) vs BPTT [sigmoid] | 3 | -0.01 (0.02) | [-0.08, +0.06] | +0.06 | 1/3 |
| `C100_B1_G3_flip - C100_B0_bptt_gamma` | Improved C-Grad gated flip (G3) vs BPTT [Gamma] | 3 | -0.11 (0.09) | [-0.51, +0.28] | -0.03 | 1/3 |
| `C100_A1_T8_gentle - C100_B0_bptt_gamma` | Improved C-Grad smooth (T8) vs Gamma BPTT | 3 | -0.01 (0.13) | [-0.57, +0.54] | +0.13 | 1/3 |
| `C100_A1_T8_gentle - C100_B1_G3_flip` | Smooth sigmoid (T8) vs Gated flip Gamma (G3) | 3 | +0.10 (0.04) | [-0.06, +0.26] | +0.15 | 3/3 |

## Seed-Level Details

| Arm | Seed | Status | mean_last10 | val/max | final_acc1 | eff_rate | mean_w | s/ep |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `C100_A0_bptt_sig` | 42 | completed | 81.66 | 82.21 | 81.54 | 0.00% | 0.0000 | 34.8s |
| `C100_A0_bptt_sig` | 43 | completed | 81.97 | 82.47 | 81.70 | 0.00% | 0.0000 | 34.9s |
| `C100_A0_bptt_sig` | 44 | completed | 81.72 | 82.21 | 81.33 | 0.00% | 0.0000 | 35.7s |
| `C100_B0_bptt_gamma` | 42 | completed | 81.68 | 82.22 | 81.60 | 0.00% | 0.0000 | 34.7s |
| `C100_B0_bptt_gamma` | 43 | completed | 81.73 | 82.14 | 81.75 | 0.00% | 0.0000 | 34.7s |
| `C100_B0_bptt_gamma` | 44 | completed | 81.93 | 82.34 | 81.86 | 0.00% | 0.0000 | 35.4s |
| `C100_A1_T8_gentle` | 42 | completed | 81.68 | 82.52 | 81.59 | 2.67% | 0.0029 | 37.3s |
| `C100_A1_T8_gentle` | 43 | completed | 81.94 | 82.49 | 81.67 | 2.64% | 0.0029 | 37.5s |
| `C100_A1_T8_gentle` | 44 | completed | 81.69 | 82.07 | 81.55 | 2.67% | 0.0029 | 37.8s |
| `C100_B1_G3_flip` | 42 | completed | 81.56 | 82.21 | 81.56 | 0.90% | 0.0077 | 37.9s |
| `C100_B1_G3_flip` | 43 | completed | 81.79 | 82.41 | 81.78 | 0.89% | 0.0076 | 37.0s |
| `C100_B1_G3_flip` | 44 | completed | 81.66 | 82.00 | 81.62 | 0.90% | 0.0077 | 38.3s |
