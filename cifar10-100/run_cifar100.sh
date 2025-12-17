python train.py --experiment cifar100 --config ./cifar100.yaml --model  ms_qkformer --data-path /data/datasets/cifar100/ --log-wandb --du-du LIF

# or

# python train.py --experiment cifar10 --config ./cifar10.yaml # --model max_resnet18 / ms_qkformer

# python train.py --experiment "cifar10_$(date +%Y%m%d_%H%M%S)" --config ./cifar10.yaml --data-path /data/rboone/datasets/cifar10/ --log-wandb --model ms_qkformer