# python train.py --experiment cifar100 --config ./cifar100.yaml # --model max_resnet18 / ms_qkformer

# or

# python train.py --experiment cifar10 --config ./cifar10.yaml # --model max_resnet18 / ms_qkformer
export CUDA_VISIBLE_DEVICES=4
python train.py --experiment "cifar10_$(date +%Y%m%d_%H%M%S)" --config ./cifar10.yaml --data-path /data/rboone/datasets/cifar10/ --log-wandb --model max_former --dS-du Gamma --du-du smooth_cgrad