
export CUDA_VISIBLE_DEVICES=0

python train.py --experiment "cifar10_$(date +%Y%m%d_%H%M%S)" --config ./cifar10_cgrad.yaml --data-path /data/rboone/datasets/cifar10/ --log-wandb --model max_former --dS-du Gamma --du-du complex54