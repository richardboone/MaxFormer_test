import os
from spikingjelly.datasets import cifar10_dvs

data_path = '/data/rboone/datasets/c10_dvs_temp'
try:
    # Use a dummy T just to initialize
    dataset = cifar10_dvs.CIFAR10DVS(root=data_path, data_type='event')
    print("Classes found:", dataset.classes)
except Exception as e:
    print("Error during initialization:", e)
