"""Test script to debug CIFAR10-DVS conversion issue"""
import sys
import os

# Add the spikingjelly datasets module path
sys.path.insert(0, '/home/rboone/.conda/envs/maxformer/lib/python3.9/site-packages')

# First, let's try to load and convert a single file to see the actual error
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS

# Pick one aedat file to test
test_aedat = '/data/rboone/datasets/wg_dvst/extract/airplane/cifar10_airplane_0.aedat'
test_output = '/tmp/test_cifar10_airplane_0.npz'

print(f"Testing conversion of: {test_aedat}")
print(f"File exists: {os.path.exists(test_aedat)}")

try:
    print("Attempting to load origin data...")
    events = CIFAR10DVS.load_origin_data(test_aedat)
    print(f"Loaded events: t shape={events['t'].shape}, x shape={events['x'].shape}")
    
    print("Attempting to save to npz...")
    CIFAR10DVS.read_aedat_save_to_np(test_aedat, test_output)
    
    print(f"Output file exists: {os.path.exists(test_output)}")
    if os.path.exists(test_output):
        import numpy as np
        data = np.load(test_output)
        print(f"Loaded npz keys: {list(data.keys())}")
        
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
