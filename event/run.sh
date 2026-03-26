
<<<<<<< HEAD
export CUDA_VISIBLE_DEVICES=0
python train.py -c ./cifar10dvs.yaml --data-path /data/rboone/datasets/wg_dvst --log-wandb --model max_former --dS-du Gamma --du-du smooth_cgrad
=======
export CUDA_VISIBLE_DEVICES=1
python train.py -c ./cifar10dvs_cgrad.yaml --data-path /data/rboone/datasets/wg_dvst --log-wandb --model max_former --dS-du Gamma --du-du smooth_cgrad
>>>>>>> refs/remotes/origin/main
