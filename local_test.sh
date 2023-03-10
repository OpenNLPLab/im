export NCCL_LL_THRESHOLD=0

GPUS=1
PORT=${PORT:-8889}
batch_size=2

echo $CONFIG

PROG=../im/classification/main.py
DATA=../pvc/data

for ARCH in tnn_vit_e3g1_tiny_rpe_l1_90_prenorm tno_vit_e3g1_small_rpe_l1_90_prenorm
do
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        --use_env $PROG --data-set CIFAR --batch-size $batch_size --num_workers 1 --lr 3e-3 \
        --data-path $DATA \
        --model $ARCH \
        --fp32-resume \
        --test \
        --num_workers 0 \
        --warmup-epochs 10 \
        --broadcast_buffers \
        2>&1 | tee log/${ARCH}.log
done

