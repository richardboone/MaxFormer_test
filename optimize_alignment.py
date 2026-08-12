
import numpy as np
from scipy.optimize import differential_evolution

# --- Core Logic from Notebook ---
def deltaS(u1, Vth=1.0):
    # Matches notebook cell 2
    return np.where(u1 < Vth, 1.0, -1.0) # Wait, check notebook cell 2
    # Notebook Cell 2: return np.where(u1 < Vth, 1.0, -1.0)
    # Correct.

def delta_u2(u1, Vth=1.0):
    # Matches notebook cell 2
    return np.where(u1 < Vth, -u1, Vth)

def delta_L_tilde(U1, dL_du2, dL_dS=1.0, Vth=1.0):
    # Matches notebook cell 2
    return dL_dS * deltaS(U1, Vth) + dL_du2 * delta_u2(U1, Vth)

def get_dS_du1(u, thresh=1.0, gama=1.0, mode='Gamma', alpha=4.0):
    # Matches notebook cell 3
    if mode == "sigmoid":
        sgax = (u - thresh) * alpha
        sig = 1.0 / (1.0 + np.exp(-sgax))
        return (1.0 - sig) * sig * alpha
    elif mode == "Gamma":
        return (1.0 / gama**2) * np.maximum(0, gama - np.abs(u - thresh))
    return (1.0 / gama**2) * np.maximum(0, gama - np.abs(u - thresh))

def cgrad_gradient(u1, dL_dS, dL_du2, thresh=1.0, gama=1.0, mode='Gamma', alpha=4.0, 
                   epsilon=0.1, alpha_c=1.0, beta_c=1.0, p_c=1.0, detach_reset=False):
    # Matches notebook cell 3
    dS_du1 = get_dS_du1(u1, thresh, gama, mode, alpha)
    spike = (u1 >= thresh).astype(float)
    
    term_supra_threshold = (thresh * dL_du2) - dL_dS
    term_sub_threshold = dL_dS - (u1 * dL_du2)
    
    m = np.where(u1 < thresh, term_sub_threshold, term_supra_threshold)
    m_grad = np.where(u1 < thresh, m, -m)

    # NOTE: Mimicking the 'bug' in custom_neuron.py where dU2_dU1 is calculated fully
    # ignoring detach_reset for the base_function.
    # Specifically line 121 in custom_neuron.py
    # This ensures we optimize for the ACTUAL behavior of the python code.
    du2_du1_standard = (1.0 - spike) - u1 * dS_du1
    
    base_function = dL_dS * dS_du1 + dL_du2 * du2_du1_standard

    # Gates
    g_m = 1.0 / (1.0 + np.exp(-alpha_c * m))
    delta = u1 - thresh
    g_d = 1.0 / (1.0 + np.exp(-beta_c * (epsilon - np.abs(delta))))
    
    # Directionality
    g_dir = np.clip(-1 * np.sign(m_grad) * np.sign(base_function), 0, 1)
    
    # Fusion
    f = np.clip(p_c * g_m * g_d * g_dir, 0, 1)
    
    # Final Gradient
    compute_dist = np.maximum(np.abs(delta), epsilon)
    dL_du1 = f * (m_grad / compute_dist) + (1.0 - f) * base_function
    return dL_du1

def check_alignment_score(params):
    # params: [epsilon, alpha_c, beta_c, p_c]
    epsilon, alpha_c, beta_c, p_c = params
    # epsilon is now optimized

    
    Vth = 1.0
    gama = 1.0
    
    # Generate Grid
    # Focused range where alignment matters (near threshold and typical gradients)
    u1 = np.linspace(0.0, 2.0, 50)
    dLdu2 = np.linspace(-2.0, 2.0, 50)
    U1, D = np.meshgrid(u1, dLdu2)
    
    score_sum = 0
    total_points = 0
    
    # Check for dL_dS = -1 and 1
    for dL_dS in [-1.0, 1.0]:
        Z = delta_L_tilde(U1, D, dL_dS=dL_dS, Vth=Vth)
        g = cgrad_gradient(U1, dL_dS, D, thresh=Vth, gama=gama, 
                           epsilon=epsilon, alpha_c=alpha_c, beta_c=beta_c, p_c=p_c)
        
        # Alignment Logic (Cell 4)
        side_sign = np.sign(U1 - Vth)
        # We want -sign(g) to match sign(Z) * side_sign
        # Or: alignment = -sign(g) * sign(Z) * side_sign == 1
        
        # Calculate alignment value (-1 or 1 or 0)
        # Using soft sign or just standard sign
        alignment_sign = -np.sign(g) * np.sign(Z) * side_sign
        
        # Filter noise
        mask = (np.abs(g) > 1e-9) & (np.abs(Z) > 1e-9)
        
        # Only count valid points
        valid_align = alignment_sign[mask]
        
        # Score: Fraction of 1.0s (Aligned)
        # We want to maximize this.
        if len(valid_align) > 0:
            score_sum += np.sum(valid_align > 0)
            total_points += len(valid_align)
            
            # Penalize magnitude explosions explicitly? 
            # The prompt asks for consistency, which is direction.
            # But maybe we want to avoid massive gradients. 
            # For now just alignment.
            
    if total_points == 0: return 0
    
    return -1.0 * (score_sum / total_points) # Negative for minimization

def run_optimization():
    print("Starting Optimization...")
    # Bounds for epsilon, alpha_c, beta_c, p_c
    bounds = [(0.01, 0.5), (0.1, 10.0), (0.1, 10.0), (0.0, 10.0)]
    
    result = differential_evolution(check_alignment_score, bounds, seed=42, maxiter=50)
    
    print("\noptimization Complete.")
    print(f"Best Score (Misalignment): {result.fun}")
    print(f"Best Alignment %: {-result.fun * 100:.2f}%")
    print(f"Optimal Parameters:")
    print(f"  epsilon: {result.x[0]:.4f}")
    print(f"  alpha_c: {result.x[1]:.4f}")
    print(f"  beta_c:  {result.x[2]:.4f}")
    print(f"  p_c:     {result.x[3]:.4f}")
    
    # Compare with default
    default_score = check_alignment_score([0.1, 1.0, 1.0, 1.0])
    print(f"\nDefault (0.1, 1.0, 1.0, 1.0) Alignment %: {-default_score * 100:.2f}%")
    
    # Compare with smooth_cgrad defaults
    smooth_score = check_alignment_score([0.1, 1.0, 1.0, 4.0])
    print(f"Smooth Cgrad (0.1, 1.0, 1.0, 4.0) Alignment %: {-smooth_score * 100:.2f}%")

if __name__ == "__main__":
    run_optimization()
