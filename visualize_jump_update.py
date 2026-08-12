import matplotlib.pyplot as plt
import numpy as np

def main():
    # Parameters
    v_th = 1.0
    epsilon = 0.05
    u1_before = v_th - epsilon
    delta_u = 2 * epsilon
    u1_after = u1_before + delta_u
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw horizontal axis (u1)
    ax.axhline(0, color='black', linewidth=1)
    
    # Draw vertical threshold line
    ax.axvline(v_th, color='gray', linestyle='--', linewidth=1.5, label='Threshold $V_{th}$')
    
    # Draw starting point (u1 = V_th - epsilon)
    ax.scatter([u1_before], [0], color='blue', s=100, zorder=5)
    ax.annotate('$u_1 = V_{th} - \epsilon$', xy=(u1_before, 0.05), xytext=(u1_before-0.2, 0.2),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3"),
                ha='center', fontsize=21)
    
    # Draw small arrow delta_u1 crossing the threshold
    ax.arrow(u1_before, 0, delta_u, 0, head_width=0.04, head_length=0.03, 
             fc='red', ec='red', length_includes_head=True, zorder=6)
    ax.text(u1_before + delta_u/2, -0.1, '$\delta u_1$', color='red', ha='center', fontweight='bold', fontsize=14)
    
    # Main Title and labels
    # ax.set_title('Minimal Jump Update', fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('$u_1$', fontsize=15)
    # ax.set_yticks([]) # Hide Y axis values
    
    # Annotations for state changes (The "Visual Trick")
    # State Before: S=0, u2=u1
    ax.text(u1_before-0.15, 0.4, 'State Before:\n$S = 0$\n$u_2 = u_1$', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="blue", alpha=0.8),
            ha='center', color='blue', fontsize=21)
    
    # State After: S=1, u2=0
    ax.text(u1_after + 0.1, 0.4, 'State After:\n$S = 1$\n$u_2 = 0$', 
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="green", alpha=0.8),
            ha='center', color='green', fontsize=21)
    
    # Emphasize the jump
    ax.annotate('', xy=(u1_after, 0), xytext=(u1_after, 0.35),
                arrowprops=dict(arrowstyle="<-", color='green', linestyle=':', linewidth=4))
    
    # Labels
    # ax.text(0.5, 0.8, '“Minimal jump-inducing update”', transform=ax.transAxes, 
            # ha='center', fontsize=12, fontstyle='italic', bbox=dict(facecolor='yellow', alpha=0.2))
    
    # Layout adjustments
    ax.set_xlim(u1_before - 0.3, u1_after + 0.3)
    ax.set_ylim(-0.2, 0.7)
    
    # Caption
    # caption = "An arbitrarily small update can cause a discrete state change.\nThis panel is where the reader should feel uneasy."
    plt.figtext(0.5, -0.05, "", wrap=True, horizontalalignment='center', 
                fontsize=11, fontweight='bold', bbox=dict(facecolor='red', alpha=0.05))
    
    plt.tight_layout()
    plt.savefig('jump_update.pdf', format='pdf', bbox_inches='tight')
    print("Graph saved as jump_update.pdf")

if __name__ == "__main__":
    main()
