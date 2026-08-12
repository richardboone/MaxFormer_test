#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D

# --- Global Plot Quality Settings ---
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 600
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# -----------------------------
# Core surface definition
# -----------------------------
def deltaS(u1, Vth=1.0):
    return np.where(u1 < Vth, 1.0, -1.0)

def delta_u2(u1, Vth=1.0):
    return np.where(u1 < Vth, -u1, Vth)

def delta_L_tilde(U1, dL_du2, dL_dS=1.0, Vth=1.0):
    """
    Vectorized. U1 and dL_du2 can be scalars or arrays broadcastable to same shape.
    """
    return dL_dS * deltaS(U1, Vth) + dL_du2 * delta_u2(U1, Vth)

# -----------------------------
# Surrogate Gradient Functions
# -----------------------------
def get_dS_du1(u, thresh=1.0, gama=1.0, mode='Gamma', alpha=4.0):
    if mode == "sigmoid":
        sgax = (u - thresh) * alpha
        sig = 1.0 / (1.0 + np.exp(-sgax))
        return (1.0 - sig) * sig * alpha
    elif mode == "Gamma":
        return (1.0 / gama**2) * np.maximum(0, gama - np.abs(u - thresh))
    return (1.0 / gama**2) * np.maximum(0, gama - np.abs(u - thresh))

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def standard_gradient(u1, dL_dS, dL_du2, thresh=1.0, gama=1.0, mode='Gamma', alpha=4.0, detach_reset=True):
    dS_du1 = get_dS_du1(u1, thresh, gama, mode, alpha)
    spike = (u1 >= thresh).astype(float)
    if detach_reset:
        du2_du1 = 0.0
    else:
        du2_du1 = (1.0 - spike) - u1 * dS_du1
    return dL_dS * dS_du1 + dL_du2 * du2_du1

def cgrad_gradient(u1, dL_dS, dL_du2, thresh=1.0, gama=1.0, mode='Gamma', alpha=4.0, epsilon=0.3468, alpha_c=1.1742, beta_c=0.9245, p_c=9.5334, detach_reset=False):
    dS_du1 = get_dS_du1(u1, thresh, gama, mode, alpha)
    spike = (u1 >= thresh).astype(float)
    term_supra = (thresh * dL_du2) - dL_dS
    term_sub = dL_dS - (u1 * dL_du2)
    m = np.where(u1 < thresh, term_sub, term_supra)
    m_grad = np.where(u1 < thresh, m, -m)
    if detach_reset:
        du2_du1_standard = 0.0
    else:
        du2_du1_standard = (1.0 - spike) - u1 * dS_du1
    base_function = dL_dS * dS_du1 + dL_du2 * du2_du1_standard
    g_m = 1.0 / (1.0 + np.exp(-alpha_c * m))
    delta = u1 - thresh
    g_d = 1.0 / (1.0 + np.exp(-beta_c * (epsilon - np.abs(delta))))
    g_dir = np.clip(-1 * np.sign(m_grad) * np.sign(base_function), 0, 1)
    f = np.clip(p_c * g_m * g_d * g_dir, 0, 1)
    compute_dist = np.maximum(np.abs(delta), epsilon)
    dL_du1 = f * (m_grad / compute_dist) + (1.0 - f) * base_function
    return dL_du1

def conservative_cgrad_gradient(u1, dL_dS, dL_du2, thresh=1.0, gama=1.0, mode='Gamma', alpha=4.0, 
                                intervention_threshold=0.8, epsilon=0.3, alpha_c=2.0, beta_c=2.0, 
                                corr_scale=0.5, blend_max=0.5, detach_reset=False):
    dS_du1 = get_dS_du1(u1, thresh, gama, mode, alpha)
    spike = (u1 >= thresh).astype(float)
    term_supra = (thresh * dL_du2) - dL_dS
    term_sub = dL_dS - (u1 * dL_du2)
    m = np.where(u1 < thresh, term_sub, term_supra)
    m_grad = np.where(u1 < thresh, m, -m)
    
    if detach_reset:
        du2_du1_standard = 0.0
    else:
        du2_du1_standard = (1.0 - spike) - u1 * dS_du1
    base_function = dL_dS * dS_du1 + dL_du2 * du2_du1_standard

    misalignment = -m_grad * base_function
    g_m = sigmoid(alpha_c * (np.abs(m) - 0.1))
    delta = u1 - thresh
    g_d = sigmoid(beta_c * (epsilon - np.abs(delta)))
    g_misalign = sigmoid(alpha_c * misalignment)

    intervention_signal = g_m * g_d * g_misalign
    do_intervene = (intervention_signal > intervention_threshold).astype(float)
    soft_correction = np.sign(m_grad) * np.abs(base_function) * corr_scale
    blend_factor = do_intervene * sigmoid(5 * (intervention_signal - intervention_threshold))
    dL_du1 = (1 - blend_max * blend_factor) * base_function + blend_max * blend_factor * soft_correction
    return dL_du1

def check_alignment(u1, dL_dS, dL_du2, thresh=1.0, gama=1.0, mode='Gamma', alpha=4.0, detach_reset=True, only_towards_thresh=True, gradient_mode='standard', cons_params=None):
    Z = delta_L_tilde(u1, dL_du2, dL_dS=dL_dS, Vth=thresh)
    if gradient_mode == 'C-grad':
        g = cgrad_gradient(u1, dL_dS, dL_du2, thresh, gama, mode, alpha, detach_reset=detach_reset)
    elif gradient_mode == 'conservative_cgrad':
        if cons_params is None:
            cons_params = {}
        g = conservative_cgrad_gradient(u1, dL_dS, dL_du2, thresh, gama, mode, alpha, detach_reset=detach_reset, **cons_params)
    else:
        g = standard_gradient(u1, dL_dS, dL_du2, thresh, gama, mode, alpha, detach_reset=detach_reset)

    eps = 1e-9
    side_sign = np.sign(u1 - thresh)
    alignment = -np.sign(g) * np.sign(Z) * side_sign
    
    if only_towards_thresh:
        towards = (np.sign(g) == np.sign(u1 - thresh))
        alignment[(alignment == -1) & (~towards)] = 0

    alignment[np.abs(g) < eps] = 0
    alignment[np.abs(Z) < eps] = 0
    return alignment

# -----------------------------
# Grid helpers
# -----------------------------
def make_grid(
    u1_min=0.0, u1_max=2.0, u1_n=250,
    dLdu2_min=-2.0, dLdu2_max=2.0, dLdu2_n=250
):
    u1 = np.linspace(u1_min, u1_max, u1_n)
    dLdu2 = np.linspace(dLdu2_min, dLdu2_max, dLdu2_n)
    U1, D = np.meshgrid(u1, dLdu2, indexing="xy")
    return U1, D

# -----------------------------
# 2D Alignment Grid Visualization
# -----------------------------
def plot_alignment_2d_grid(
    dL_dS_values=(-1.0, 0.0, 1.0),
    Vth=1.0,
    u1_range=(0.0, 2.0),
    dLdu2_range=(-2.0, 2.0),
    grid_n=200,
    only_towards_thresh=True,
    cons_params=None
):
    """
    Produces grid comparing alignment across BP methods and dL/dS values.
    Rows: Standard, Standard (Detach), C-grad
    Cols: dL/dS values
    """
    rows = ['Standard', 'Detach', 'C-grad']
    cols = [f"$dL/dS = {v:g}$" for v in dL_dS_values]
    
    # Adjusted figsize for a 3x3 grid that fits better in a column/page
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(10, 8.5), sharex=True, sharey=True)
    
    # Colormap for alignment: Green for Aligned (1), Red for Misaligned (-1), Gray for Neutral (0)
    align_cmap = ListedColormap(['#ff4c4c', '#cccccc', '#4cff4c']) # Red, Gray, Green
    norm_align = plt.Normalize(vmin=-1, vmax=1)
    
    U1, D = make_grid(
        u1_min=u1_range[0], u1_max=u1_range[1], u1_n=grid_n,
        dLdu2_min=dLdu2_range[0], dLdu2_max=dLdu2_range[1], dLdu2_n=grid_n
    )
    
    for r_idx, (row_label, row_name) in enumerate(zip(['Standard', 'Detach', 'Conservative C-grad'], rows)):
        for c_idx, dL_dS in enumerate(dL_dS_values):
            ax = axes[r_idx, c_idx]
            
            # Configure gradient mode and detach based on row
            if row_label == 'Standard':
                g_mode = 'standard'
                detach = False
            elif row_label == 'Detach':
                g_mode = 'standard'
                detach = True
            elif row_label == 'Conservative C-grad':
                g_mode = 'conservative_cgrad'
                detach = False 
            else:
                # Default case
                g_mode = 'standard'
                detach = False
            
            A = check_alignment(
                U1, dL_dS, D, thresh=Vth, 
                detach_reset=detach, 
                only_towards_thresh=only_towards_thresh, 
                gradient_mode=g_mode,
                cons_params=cons_params
            )
            
            im = ax.imshow(
                A, 
                extent=[u1_range[0], u1_range[1], dLdu2_range[0], dLdu2_range[1]],
                origin='lower',
                cmap=align_cmap,
                norm=norm_align,
                aspect='auto'
            )
            
            # Add Vth vertical line indicator
            ax.axvline(Vth, color='black', linestyle='--', alpha=0.3)
            
            # Add Column Labels (top row only)
            if r_idx == 0:
                ax.set_title(cols[c_idx], fontsize=16, pad=10)
            
            # Add Row Labels (left column only)
            if c_idx == 0:
                ax.set_ylabel(row_name, fontsize=16, fontweight='bold', labelpad=15)
            
            # Shared Axis labels (bottom row and left column only)
            if r_idx == len(rows) - 1:
                ax.set_xlabel(r"$u_1$")
            if c_idx == 0:
                # Append original y-axis meaning to the row label or keep separate
                # We'll just put the variable name for all subplots in the first column
                curr_ylabel = ax.get_ylabel()
                ax.set_ylabel(f"{curr_ylabel}\n" + r"$\partial L / \partial u_2$")

    # Add a global legend
    legend_elements = [
        Line2D([0], [0], marker='s', color='w', label='Consistent', markerfacecolor='#4cff4c', markersize=15),
        Line2D([0], [0], marker='s', color='w', label='Neutral', markerfacecolor='#cccccc', markersize=15),
        Line2D([0], [0], marker='s', color='w', label='Inconsistent', markerfacecolor='#ff4c4c', markersize=15)
    ]
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.02), ncol=3, frameon=True, fontsize=14)
    
    # Adjust spacing to make room for row/col headers and legend
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    output_path = 'alignment_grid_recreated.png'
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    # Balanced Conservative params as used in the notebook
    CONS_PARAMS = {
        'epsilon': 0.5, 'alpha_c': 20.0, 'beta_c': 1.0, 'intervention_threshold': 0.3,
        'corr_scale': 1.0, 'blend_max': 1.0
    }
    
    # Use optimized conservative C-grad in the final row
    plot_alignment_2d_grid(cons_params=CONS_PARAMS)
