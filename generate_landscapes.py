import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

# Create directory for artifacts if it doesn't exist (though the system should handle it)
output_dir = "/home/rboone/work/MaxFormer_test/figs"
os.makedirs(output_dir, exist_ok=True)

def smooth_landscape(x, y):
    return x**2 + y**2 + 0.5 * np.sin(3*x) * np.cos(3*y)

def jump_landscape(x, y):
    # Smooth part
    z = x**2 + y**2 + 0.5 * np.sin(3*x) * np.cos(3*y)
    # Sudden jump up at x > 0.5
    jump = np.where(x > 0.5, 10, 0)
    return z + jump

def plot_landscape(func, title, filename):
    x = np.linspace(-2, 2, 100)
    y = np.linspace(-2, 2, 100)
    X, Y = np.meshgrid(x, y)
    Z = func(X, Y)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
    
    # ax.set_title(title, fontsize=15)
    ax.set_xlabel('Weight 1')
    ax.set_ylabel('Weight 2')
    ax.set_zlabel('Loss', labelpad=10)
    
    # Customize the view for better visualization of the jump
    ax.view_init(elev=30, azim=135)
    ax.dist = 11 # Zoom out slightly to prevent label clipping
    
    fig.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating smooth landscape...")
    plot_landscape(smooth_landscape, "Smooth Loss Landscape", "smooth_landscape.png")
    
    print("Generating jump landscape...")
    plot_landscape(jump_landscape, "Jump Loss Landscape", "jump_landscape.png")
    
    print("Done. Visuals saved to artifacts directory.")
