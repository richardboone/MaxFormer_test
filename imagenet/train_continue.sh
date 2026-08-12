export HOST_NODE_ADDR=127.0.0.1:2963
export NCCL_DEBUG=WARN
export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export NCCL_SOCKET_IFNAME=lo
# 1. Set specific GPUs (4,5,6,7)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# 2. Generate unique name to prevent FileExistsError on checkpoints
timestamp=$(date +%Y%m%d_%H%M%S)
name="train_continue_10_768_${timestamp}"
NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes=1 --nproc_per_node=4 --rdzv_endpoint=$HOST_NODE_ADDR train.py \
 --pin_mem --dist_eval -c ./conf/10_768_t4.yml --exp $name --log_dir ./log/$name --output_dir ./output/$name \
 --data_path /data/datasets/imagenet \
 --resume ./output/train_10_768_20251231_160745/last.pth.tar