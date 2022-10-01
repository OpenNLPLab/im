export NCCL_LL_THRESHOLD=0

GPUS=1
PORT=${PORT:-8889}
batch_size=2

echo $CONFIG

PROG=../im/classification/main.py
DATA=../pvc/data

for ARCH in tnn_2d_vit_tiny_rpe_v8_l1 #tnn_2d_vit_tiny_rpe_v8_l2 tnn_2d_vit_tiny_rpe_v8_l3
do
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        --use_env $PROG --data-set CIFAR --batch-size $batch_size --num_workers 1 --lr 3e-3 \
        --data-path $DATA \
        --model $ARCH \
        --fp32-resume \
        --test \
        2>&1 | tee log/${ARCH}.log
done

