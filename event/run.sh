
export CUDA_VISIBLE_DEVICES=2
python train.py -c ./cifar10dvs_cgrad_soft.yaml --data-path /data/rboone/datasets/wg_dvst --log-wandb --model max_former
