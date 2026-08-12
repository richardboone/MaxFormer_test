
import torch
import torch.nn as nn
import sys
import os
import pickle

# Ensure we can import from the cifar10-100 directory
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "cifar10-100")))

import custom_neuron
from max_former import Max_Former

class MockArgs:
    def __init__(self):
        self.du_du = "conservative_cgrad"
        self.dS_du = "Gamma"
        self.snnbp_alpha = 2.0
        self.snnbp_beta = 2.0
        self.snnbp_epsilon = 0.3
        self.snnbp_intervention = 0.8
        self.snnbp_tau = 0.5
        self.use_custom_neuron = True
        self.detach_reset = True
        self.gama = 1.0

def harvest():
    print("Initializing harvesting script...")
    args = MockArgs()
    custom_neuron.set_global_args(args)
    
    metrics = []
    custom_neuron.set_metrics_collector(metrics)
    
    # Create model
    model = Max_Former(
        in_channels=3,
        num_classes=10,
        embed_dims=128, # Smaller for speed
        depths=[2, 2, 2],
        T=4
    ).cuda()
    
    model.train()
    
    print("Running forward and backward passes...")
    for i in range(5): # Run 5 batches to get a good distribution
        input_data = torch.randn(8, 3, 32, 32).cuda()
        target = torch.randint(0, 10, (8,)).cuda()
        
        output = model(input_data)
        loss = nn.functional.cross_entropy(output, target)
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer.zero_grad()
        loss.backward()
        
        print(f"Batch {batch_idx if 'batch_idx' in locals() else i+1} completed. Collected {len(metrics)} layer-steps of data.")
        
    # Save metrics
    output_file = "conservative_metrics_data.pkl"
    with open(output_file, 'wb') as f:
        pickle.dump(metrics, f)
    
    print(f"Metrics saved to {output_file}")

if __name__ == "__main__":
    harvest()
