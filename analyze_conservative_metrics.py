
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

def analyze():
    print("Loading metrics data...")
    with open("conservative_metrics_data.pkl", "rb") as f:
        data = pickle.load(f)
    
    # Flatten all batches and layer-steps
    # Each element in data is a dict with tensors of shape [T, B, ...] or [B, ...]
    
    m_all = []
    u1_all = []
    base_all = []
    thresh_all = []
    
    for entry in data:
        # Sample 10% of the data to speed up analysis
        batch_size = entry['m'].shape[0]
        sample_size = max(1, batch_size // 10)
        idx = np.random.choice(batch_size, sample_size, replace=False)
        
        m = entry['m'][idx].flatten()
        u1 = entry['u1'][idx].flatten()
        base = entry['base_function'][idx].flatten()
        t = entry['thresh']
        
        m_all.append(m)
        u1_all.append(u1)
        base_all.append(base)
        thresh_all.append(torch.ones_like(m) * t)
        
    m_all = torch.cat(m_all).numpy()
    u1_all = torch.cat(u1_all).numpy()
    base_all = torch.cat(base_all).numpy()
    thresh_all = torch.cat(thresh_all).numpy()
    
    n_total = len(m_all)
    m_pos_mask = m_all > 0
    n_m_pos = np.sum(m_pos_mask)
    
    print(f"Total samples: {n_total}")
    print(f"Samples with m > 0: {n_m_pos} ({n_m_pos/n_total*100:.2f}%)")
    
    u1_m_pos = u1_all[m_pos_mask]
    base_m_pos = base_all[m_pos_mask]
    thresh_m_pos = thresh_all[m_pos_mask]
    delta_m_pos = u1_m_pos - thresh_m_pos
    
    print(f"Avg |base_grad| (when m>0): {np.mean(np.abs(base_m_pos)):.6f}")
    print(f"Max |base_grad| (when m>0): {np.max(np.abs(base_m_pos)):.6f}")
    print(f"Avg |delta| (when m>0): {np.mean(np.abs(delta_m_pos)):.6f}")
    print(f"Min |delta| (when m>0): {np.min(np.abs(delta_m_pos)):.6f}")
    
    # --- Metric 1: m > 0 and where u1 is ---
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    # sns.histplot(u1_all[m_pos_mask], kde=True, color='blue', bins=50)
    plt.hist(u1_all[m_pos_mask], color='blue', bins=50, alpha=0.7)
    plt.axvline(1.0, color='red', linestyle='--', label='Threshold (1.0)')
    plt.title(f"Distribution of u1 when m > 0\n({n_m_pos} samples)")
    plt.xlabel("u1 (Membrane Potential)")
    plt.ylabel("Frequency")
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hexbin(u1_all[m_pos_mask], m_all[m_pos_mask], gridsize=30, cmap='Blues', bins='log')
    plt.axvline(1.0, color='red', linestyle='--')
    plt.title("u1 vs m (for m > 0)")
    plt.xlabel("u1")
    plt.ylabel("m")
    plt.colorbar(label='log10(count)')
    
    plt.tight_layout()
    plt.savefig("metric1_m_pos_distribution.png")
    print("Saved metric1_m_pos_distribution.png")
    
    # --- Metric 2: Original method threshold crossing ---
    # We'll sweep over some learning rates to see how sensitive this is
    etas = [0.01, 0.05, 0.1, 0.2, 0.5]
    crossing_probs = []
    
    for eta in etas:
        u1_orig = u1_all[m_pos_mask]
        base_orig = base_all[m_pos_mask]
        thresh_orig = thresh_all[m_pos_mask]
        u1_new = u1_orig - eta * base_orig
        
        crossed_up = (u1_orig < thresh_orig) & (u1_new >= thresh_orig)
        crossed_down = (u1_orig >= thresh_orig) & (u1_new < thresh_orig)
        crossed = crossed_up | crossed_down
        
        prob = np.mean(crossed)
        crossing_probs.append(prob)
        print(f"Eta={eta}: Crossing Probability = {prob*100:.4f}% (Up: {np.sum(crossed_up)}, Down: {np.sum(crossed_down)})")
        
    plt.figure(figsize=(8, 6))
    plt.plot(etas, [p * 100 for p in crossing_probs], marker='o', linestyle='-', color='green')
    plt.title("Original Method Crossing Probability (when m > 0)")
    plt.xlabel("Learning Rate (eta)")
    plt.ylabel("Crossing Probability (%)")
    plt.grid(True, alpha=0.3)
    plt.savefig("metric2_crossing_probability.png")
    print("Saved metric2_crossing_probability.png")
    
    # Joint plot for a specific eta
    eta_ref = 0.1
    u1_orig = u1_all[m_pos_mask]
    base_orig = base_all[m_pos_mask]
    u1_new = u1_orig - eta_ref * base_orig
    crossed = ((u1_orig < 1.0) & (u1_new >= 1.0)) | ((u1_orig >= 1.0) & (u1_new < 1.0))
    
    plt.figure(figsize=(10, 6))
    plt.scatter(u1_orig[~crossed], (u1_new - u1_orig)[~crossed], alpha=0.1, s=5, label='No Crossing', color='gray')
    plt.scatter(u1_orig[crossed], (u1_new - u1_orig)[crossed], alpha=0.5, s=10, label='Crossing', color='red')
    plt.axvline(1.0, color='black', linestyle='--')
    plt.axhline(0, color='black', alpha=0.3)
    plt.title(f"Original Update (eta={eta_ref}) vs u1 (where m > 0)")
    plt.xlabel("u1 (start)")
    plt.ylabel("Delta u1 (-eta * base_grad)")
    plt.legend()
    plt.savefig("update_visualization.png")
    print("Saved update_visualization.png")

if __name__ == "__main__":
    analyze()
