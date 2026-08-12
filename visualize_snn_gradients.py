import numpy as np
import matplotlib.pyplot as plt

def spike_func(u, thresh=1.0):
    return (u > thresh).astype(float)

def surrogate_grad(u, thresh=1.0, gamma=1.0):
    # Triangular surrogate gradient (Gamma)
    return (1.0 / (gamma**2)) * np.maximum(0, gamma - np.abs(u - thresh))

def mem_reset_func(u, thresh=1.0):
    # Soft reset behavior: mem = u * (1 - S)
    s = spike_func(u, thresh)
    return u * (1.0 - s)

def mem_reset_grad(u, thresh=1.0, gamma=1.0, detach_reset=False):
    # dL/du = dL/dS * dS/du + dL/dV2 * dV2/du
    # Here we just look at dV2/du
    s = spike_func(u, thresh)
    ds_du = surrogate_grad(u, thresh, gamma)
    
    if detach_reset:
        # Gradient detach: just (1 - S)
        return (1.0 - s)
    else:
        # Full gradient: (1 - S) - u * dS/du
        return (1.0 - s) - u * ds_du

def main():
    u = np.linspace(0, 2, 1000)
    thresh = 1.0
    gamma = 1.0
    
    # Pre-calculate values
    s_vals = spike_func(u, thresh)
    ds_du_vals = surrogate_grad(u, thresh, gamma)
    v_reset_vals = mem_reset_func(u, thresh)
    dv_du_detach = mem_reset_grad(u, thresh, gamma, detach_reset=True)
    dv_du_full = mem_reset_grad(u, thresh, gamma, detach_reset=False)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Top Plot: Spike and Surrogate Gradient
    ax1.plot(u, s_vals, label='Spike Function $S(u)$', color='blue', linewidth=2)
    ax1.plot(u, ds_du_vals, label='Surrogate Gradient $dS/du$ (Gamma)', color='red', linestyle='--', linewidth=2)
    ax1.axvline(thresh, color='gray', linestyle=':', label='Threshold $\\theta$')
    ax1.set_title('Spike Function and its Surrogate Gradient')
    ax1.set_ylabel('Amplitude')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Bottom Plot: Membrane Potential and Reset Gradients
    ax2.plot(u, v_reset_vals, label='Membrane after Reset $V_{reset}(u)$', color='green', linewidth=2)
    ax2.plot(u, dv_du_detach, label='Grad with Detach $(1-S)$', color='orange', linestyle='--', linewidth=2)
    ax2.plot(u, dv_du_full, label='Full Grad $(1-S) - u \\cdot dS/du$', color='purple', linestyle='-.', linewidth=2)
    ax2.axvline(thresh, color='gray', linestyle=':', label='Threshold $\\theta$')
    ax2.set_title('Membrane Potential and its Gradients')
    ax2.set_xlabel('Membrane Potential $u$')
    ax2.set_ylabel('Amplitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('snn_gradients.png', dpi=300)
    print("Graph saved as snn_gradients.png")

if __name__ == "__main__":
    main()
