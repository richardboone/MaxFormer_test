export CUDA_VISIBLE_DEVICES=0
python train.py --experiment cifar100 --config ./cifar100_gamma.yaml --model  max_former --data-path /data/datasets/cifar100/ --log-wandb --time-step 8

# or

# python train.py --experiment cifar10 --config ./cifar10.yaml # --model max_resnet18 / ms_qkformer

# python train.py --experiment "cifar10_$(date +%Y%m%d_%H%M%S)" --config ./cifar10.yaml --data-path /data/rboone/datasets/cifar10/ --log-wandb --model ms_qkformer