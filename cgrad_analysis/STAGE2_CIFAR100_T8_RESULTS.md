# Stage 2 CIFAR-100 MaxFormer Results (`T=8`, `400+10` Epochs)

## Per-Arm Summary (Ranked by `val/mean_last10`)

| Arm | Surr | n (ok/nan) | val/mean_last10 | val/max | final_acc1 | eff_rate | mean_w | m_neg | s/ep |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `C100_B0_bptt_gamma` | Gamma | 2/0 | 82.52 ± 0.37 | 83.13 ± 0.30 | 82.50 | 0.00% | 0.0000 | 0.0% | 63.5s |
| `C100_A1_T8_gentle` | sigmoid | 2/0 | 82.38 ± 0.12 | 82.83 ± 0.14 | 82.38 | 3.96% | 0.0043 | 50.0% | 69.3s |
| `C100_B1_G3_flip` | Gamma | 2/0 | 82.37 ± 0.09 | 83.07 ± 0.16 | 82.38 | 1.37% | 0.0116 | 50.0% | 69.0s |
| `C100_A0_bptt_sig` | sigmoid | 2/0 | 82.25 ± 0.22 | 82.87 ± 0.43 | 82.50 | 0.00% | 0.0000 | 0.0% | 63.8s |

## Paired Seed Contrasts

| Contrast | Question | n_pairs | Δ mean_last10 (SE) | 95% CI | Δ val/max | wins |
|---|---|---:|---:|---:|---:|---:|
| `C100_A1_T8_gentle - C100_A0_bptt_sig` | Improved C-Grad smooth (T8) vs BPTT [sigmoid] | 2 | +0.14 (0.24) | [-2.96, +3.24] | -0.03 | 1/2 |
| `C100_B1_G3_flip - C100_B0_bptt_gamma` | Improved C-Grad gated flip (G3) vs BPTT [Gamma] | 2 | -0.15 (0.32) | [-4.26, +3.97] | -0.06 | 1/2 |
| `C100_A1_T8_gentle - C100_B0_bptt_gamma` | Improved C-Grad smooth (T8) vs Gamma BPTT | 2 | -0.14 (0.18) | [-2.38, +2.11] | -0.30 | 1/2 |
| `C100_A1_T8_gentle - C100_B1_G3_flip` | Smooth sigmoid (T8) vs Gated flip Gamma (G3) | 2 | +0.01 (0.15) | [-1.86, +1.89] | -0.24 | 1/2 |

## Seed-Level Details

| Arm | Seed | Status | mean_last10 | val/max | final_acc1 | eff_rate | mean_w | s/ep |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| `C100_B0_bptt_gamma` | 42 | completed | 82.26 | 82.92 | 82.37 | 0.00% | 0.0000 | 63.4s |
| `C100_B0_bptt_gamma` | 43 | completed | 82.78 | 83.35 | 82.63 | 0.00% | 0.0000 | 63.7s |
| `C100_A1_T8_gentle` | 42 | completed | 82.30 | 82.73 | 82.43 | 3.98% | 0.0043 | 69.4s |
| `C100_A1_T8_gentle` | 43 | completed | 82.47 | 82.93 | 82.33 | 3.95% | 0.0043 | 69.2s |
| `C100_B1_G3_flip` | 42 | completed | 82.43 | 83.19 | 82.44 | 1.38% | 0.0117 | 69.3s |
| `C100_B1_G3_flip` | 43 | completed | 82.31 | 82.96 | 82.31 | 1.35% | 0.0115 | 68.6s |
| `C100_A0_bptt_sig` | 42 | completed | 82.40 | 83.17 | 82.79 | 0.00% | 0.0000 | 63.7s |
| `C100_A0_bptt_sig` | 43 | completed | 82.09 | 82.56 | 82.22 | 0.00% | 0.0000 | 63.9s |
