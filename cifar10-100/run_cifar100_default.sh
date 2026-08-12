

export DISABLE_CUPY=1
export CUDA_VISIBLE_DEVICES=2
python train.py --experiment "cifar100_$(date +%Y%m%d_%H%M%S)" --config ./cifar100.yaml --log-wandb --data-path /data/rboone/datasets/cifar100 --time-step 4 --model max_former --use-multi-epochs-loader
# or

# python train.py --experiment cifar10 --config ./cifar10.yaml # --model max_resnet18 / ms_qkformer
# export CUDA_VISIBLE_DEVICES=0
# python train.py --experiment "cifar10_$(date +%Y%m%d_%H%M%S)" --config ./cifar10.yaml --data-path /data/rboone/datasets/cifar10/ --log-wandb --model max_former --dS-du Gamma --du-du smooth_cgrad
# python train.py --experiment "cifar10_$(date +%Y%m%d_%H%M%S)" --config ./cifar10_cgrad.yaml --data-path /data/rboone/datasets/cifar10/ --log-wandb --model max_former --dS-du Gamma --du-du complex54