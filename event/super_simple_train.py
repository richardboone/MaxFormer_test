from spikingjelly.datasets import cifar10_dvs

origin_set = cifar10_dvs.CIFAR10DVS(root="/data/rboone/datasets/wg_dvst", data_type='frame', frames_number=16, split_by='number')