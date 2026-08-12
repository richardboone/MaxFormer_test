import matplotlib.pyplot as plt
import numpy as np

def main():
    # Parameters
    v_th = 1.0
    epsilon = 0.05
    u1_point = v_th - epsilon
    
    # Define Loss function: u1-dependent part of the loss
    # Below thresh: L = 0.5 * (u1 - 1.5)^2 + 0.2
    # Above thresh: L = 0.5 * (u1 - 1.5)^2 + 0.6  (Jump due to S:0->1 and u2:u1->0)
    def true_loss(u):
        l_base = 0.5 * (u - 1.5)**2
        return np.where(u < v_th, l_base + 0.2, l_base + 0.6)

    u = np.linspace(0.5, 1.3, 1000)
    l_vals = true_loss(u)
    
    # Tangent at u1_point (gradient estimate)
    # L'(u) = (u - 1.5)
    slope = (u1_point - 1.5) 
    intercept = true_loss(u1_point) - slope * u1_point
    
    # Extension for dashed line
    u_tangent = np.linspace(u1_point - 0.1, v_th + 0.1, 100)
    l_tangent = slope * u_tangent + intercept
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot true loss (solid with break)
    u_below = u[u < v_th]
    l_below = l_vals[u < v_th]
    u_above = u[u >= v_th]
    l_above = l_vals[u >= v_th]
    
    ax.plot(u_below, l_below, color='blue', linewidth=2.5, label='True Loss $L(u_1)$')
    ax.plot(u_above, l_above, color='blue', linewidth=2.5)
    
    # Plot gradient prediction (dashed line)
    ax.plot(u_tangent, l_tangent, color='red', linestyle='--', linewidth=2, label='Local Linear Estimate1')
    
    # Vertical line at threshold
    ax.axvline(v_th, color='gray', linestyle=':', linewidth=1.5, label='Threshold $V_{th}$')
    
    # Annotations
    # Point of update start
    ax.scatter([u1_point], [true_loss(u1_point)], color='red', s=60, zorder=5)
    
    # Arrow for gradient prediction (lower left area, moved for clarity)
    ax.annotate(r'Gradient predicts $\downarrow$ loss', 
                xy=(v_th + 0.1, l_tangent[-1]), xytext=(v_th - 0.25, l_tangent[-1] + 0.5),
                arrowprops=dict(facecolor='red', shrink=0.05, width=2, headwidth=8, alpha=0.7),
                color='red', fontweight='bold', ha='center', fontsize=21)
    
    # Arrow for actual loss jump (upper right area)
    ax.annotate(r'Actual loss $\uparrow$', 
                xy=(v_th + 0.02, true_loss(v_th + 0.01) + 0.02), xytext=(v_th + 0.25, true_loss(v_th + 0.01) + 0.15),
                arrowprops=dict(facecolor='blue', shrink=0.05, width=2, headwidth=8),
                color='blue', fontweight='bold', ha='center', fontsize=21)
    
    # Inset for the equation (Top Left)
    textstr = r'$\Delta \tilde{L}_{jump} = \left(\frac{\partial L}{\partial S}\right)\Delta S + \left(\frac{\partial L}{\partial u_2}\right)\Delta u_2 > 0$'
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=21,
            verticalalignment='top', bbox=props)
    
    # Titles and Labels
    # ax.set_title('Inconsistent Loss Response', fontsize=16, fontweight='bold', pad=25)
    ax.set_xlabel('$u_1$', fontsize=15)
    ax.set_ylabel('Loss $L$', fontsize=15)
    ax.legend(loc='lower left', fontsize=14, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    
    # Adjust axes limits to make space
    ax.set_xlim(0.5, 1.4)
    ax.set_ylim(0.2, 1.2)
    
    # Caption (Restore and Fix)
    # caption_text = "The gradient points downhill, but the update moves the system uphill.\nThis is the emotional core of the figure."
    plt.figtext(0.5, 0.01, "", wrap=True, horizontalalignment='center', 
                fontsize=11, fontweight='bold', bbox=dict(facecolor='red', alpha=0.05))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Make room for title and caption
    plt.savefig('loss_inconsistency.pdf', format='pdf', bbox_inches='tight')
    print("Graph saved as loss_inconsistency.pdf")

if __name__ == "__main__":
    main()
