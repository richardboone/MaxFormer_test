# python train.py --experiment cifar100 --config ./cifar100.yaml # --model max_resnet18 / ms_qkformer

# or

export DISABLE_CUPY=1
# python train.py --experiment cifar10 --config ./cifar10.yaml # --model max_resnet18 / ms_qkformer
export CUDA_VISIBLE_DEVICES=2
# python train.py --experiment "cifar10_$(date +%Y%m%d_%H%M%S)" --config ./cifar10.yaml --data-path /data/rboone/datasets/cifar10/ --log-wandb --model max_former --dS-du Gamma --du-du smooth_cgrad
python train.py --experiment "cifar10_$(date +%Y%m%d_%H%M%S)" --config ./cifar10.yaml --data-path /data/rboone/datasets/cifar10/ --log-wandb --model max_former --time-step 4 --epochs 700 --use-multi-epochs-loader