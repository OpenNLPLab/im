export NCCL_LL_THRESHOLD=0

GPUS=1
PORT=${PORT:-8889}
batch_size=2

echo $CONFIG

PROG=../im/classification/main.py
DATA=../pvc/data

# for ARCH in tnn_2d_vit_tiny_rpe_v8_l1 tnn_2d_vit_tiny_rpe_v8_l2 tnn_2d_vit_tiny_rpe_v8_l3 \
#             tnn_2d_vit_tiny_rpe_v8_l4 tnn_2d_vit_tiny_rpe_v8_l5 tnn_2d_vit_tiny_rpe_v8_l6 \
#             tnn_2d_pyr_tiny_rpe_v8_l1 tnn_2d_vit_tiny_rpe_v8_l1_prenorm
# for ARCH in tnn_2d_pyr_tiny_rpe_v8_l1 tnn_2d_vit_tiny_rpe_v8_l1
# for ARCH in norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm
# for ARCH in tnn_2d_pyr_tiny_rpe_v8_l1 tnn_2d_vit_tiny_rpe_v8_l1 norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_no_urpe norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_no_urpe
# for ARCH in norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu
# for ARCH in tnn_2d_vit_tiny_rpe_v8_l1_prenorm tnn_2d_vit_tiny_rpe_v8_l1
# for ARCH in tnn_2d_vit_tiny_rpe_v8_l1_prenorm
# for ARCH in tnn_2d_pyr_tiny_rpe_v8_l1
# for ARCH in norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu
# for ARCH in norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_nopos norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_nopos
# for ARCH in norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_nopos norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_nopos norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_maxhead norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_maxhead
# for ARCH in tnn_vit_e3g1_tiny_rpe_l1_95 tnn_vit_e3g1_tiny_rpe_l1_95_prenorm
# for ARCH in tnn_2d_vit_tiny_rpe_v8_l1_prenorm_tno_patch tnn_2d_vit_tiny_rpe_v8_l1_prenorm
# for ARCH in tnn_2d_vit_tiny_rpe_v8_l1_prenorm_tno_patch
# for ARCH in norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu
# for ARCH in norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_standard norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_standard \
#             norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu norm_vit_small_patch16_224_mix_relu_elu_rmsnorm_glu
# for ARCH in tnn_2d_pyr_tiny_rpe_v8_l1 tnn_2d_pyr_tiny_rpe_v8_l1_prenorm tnn_2d_vit_tiny_rpe_v8_l1_prenorm
# for ARCH in tno_vit_e3g1_small_rpe_l1_90_prenorm
# for ARCH in norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h1 norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_h1 \
#             norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h12 norm_vit_small_patch16_224_mix_relu_elu_rmsnorm_glu_h12 \
#             norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h1 norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h1 \
#             tnn_2d_vit_small_rpe_v8_l1_prenorm
# for ARCH in norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu \
# norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h3 norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_h3 \
# norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h1 norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_h1 \
# norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_nopos norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_nopos \
# norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_maxhead norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_maxhead
# for ARCH in tnn_2d_vit_small_rpe_v8_l1_prenorm
# for ARCH in norm_vit_tiny_patch14_224_mix_softmax_1_elu_rmsnorm_glu norm_vit_tiny_patch14_224_mix_relu_elu_rmsnorm_glu norm_vit_small_patch14_224_mix_softmax_1_elu_rmsnorm_glu_h6 norm_vit_small_patch14_224_mix_relu_elu_rmsnorm_glu_h6 norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h12 norm_vit_small_patch16_224_mix_relu_elu_rmsnorm_glu_h12
# for ARCH in tnn_vit_e3g1_tiny_rpe_l1_90_prenorm tno_vit_e3g1_small_rpe_l1_95_prenorm
# for ARCH in norm_vit_tiny_patch7_224_mix_softmax_1_elu_rmsnorm_glu norm_vit_tiny_patch7_224_mix_relu_elu_rmsnorm_glu \
#             norm_vit_tiny_overlap_patch16_224_mix_softmax_1_elu_rmsnorm_glu norm_vit_tiny_overlap_patch16_224_mix_relu_elu_rmsnorm_glu \
#             norm_vit_tiny_patch16_224_mix_softmax_1_elu_rmsnorm_glu_no_block norm_vit_tiny_patch16_224_mix_relu_elu_rmsnorm_glu_no_block
# for ARCH in tnn_2d_vit_tiny_rpe_v8_l1_prenorm_99 tnn_2d_vit_tiny_rpe_v8_l1_prenorm_95 tnn_2d_vit_tiny_rpe_v8_l1_prenorm_90
# for ARCH in norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h12_no_block norm_vit_small_patch16_224_mix_relu_elu_rmsnorm_glu_h12_no_block norm_vit_small_patch16_224_mix_relu_elu_rmsnorm_glu_h12
# for ARCH in tnn_2d_vit_small_rpe_v8_l1_prenorm tnn_2d_vit_small_rpe_v8_l1_prenorm_90 tnn_2d_vit_small_rpe_v8_l1_prenorm_95 tnn_2d_vit_small_rpe_v8_l1_prenorm_99
# for ARCH in tno_vit_e3g1_small_rpe_l1_95_prenorm
# for ARCH in tnn_2d_vit_small_rpe_v8_l1_prenorm
# for ARCH in norm_vit_tiny_overlap_patch16_224_mix_softmax_1_elu_rmsnorm_glu norm_vit_tiny_overlap_patch16_224_mix_relu_elu_rmsnorm_glu
# for ARCH in norm_vit_small_conv_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h12_no_block norm_vit_small_conv_patch16_224_mix_relu_elu_rmsnorm_glu_h12_no_block
# for ARCH in norm_vit_small_patch16_224_mix_softmax_1_elu_rmsnorm_glu_h12_no_block
# for ARCH in tnn_2d_vit_small_rpe_v8_l1_prenorm_90 tno_vit_e3g1_small_rpe_l1_90_prenorm
# for ARCH in tnn_2d_vit_tiny_rpe_v8_l1_prenorm_90 tnn_vit_e3g1_tiny_rpe_l1_90_prenorm
for ARCH in tno_vit_e3g1_big_rpe_l1_90_prenorm
do
    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        --use_env $PROG --data-set CIFAR --batch-size $batch_size --num_workers 1 --lr 3e-3 \
        --data-path $DATA \
        --model $ARCH \
        --fp32-resume \
        --test \
        --warmup-epochs 10 \
        --broadcast_buffers \
        2>&1 | tee log/${ARCH}.log
done

