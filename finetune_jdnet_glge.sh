DATASET=$1
LR=$2
BATCHSIZE=$3
WARMUP=$4
PRETRAIN_CKPT=$5

WEIGHT_DECAY=$6
# echo $WEIGHT_DECAY
ADAM_BETAS=$7
echo $ADAM_BETAS
CLIP_NORM=$8

MAX_SOURCE=$9
MAX_TARGET=$10
MAX_EPOCH=$11
MAX_KEEP_EPOCH=$12
DEVICES=$13
GPU_NUM=$14
MAX_SENTENCE=$15
UPDATE_FREQ=$(($BATCHSIZE/$MAX_SENTENCE/$GPU_NUM))

# PRETRAIN=200G_1990k
PRETRAINED_MODEL=pretrained_checkpoints/jdnet_large_pretrained_$PRETRAIN_CKPT.pt

# echo 'UPDATE_FREQ'$UPDATE_FREQ

# DATASET=coqa
DATA_DIR=./glge/data/easy/$DATASET\_data/processed_translation_prophetnet
USER_DIR=./jdnet_pretrain_v6

PRETRAIN=pretrain
# PRETRAIN=noPretrain
NET=JDNet
MODEL=Large
LOSS=L2RPLM
NGRAM=Ngram2

ARCH=ngram_transformer_prophet_large
# ARCH=ngram_transformer_prophet_base

# CRITERION=ngram_language_loss
CRITERION=ngram_language_loss_$LOSS

SAVE_DIR=glge/models/$DATASET/finetune'_checkpoints_'$PRETRAIN'_'$NGRAM'_'$NET'_'$MODEL'_'$LOSS'_lr'$LR'_trainbatchsize'$BATCHSIZE'_warm'$WARMUP'_clipNorm'$CLIP_NORM'_weightDecay'$WEIGHT_DECAY'_'$PRETRAIN_CKPT
TENSORBOARD_LOGDIR=glge/models/$DATASET/finetune'_tensorboard_'$PRETRAIN'_'$NGRAM'_'$NET'_'$MODEL'_'$LOSS'_lr'$LR'_trainbatchsize'$BATCHSIZE'_warm'$WARMUP'_clipNorm'$CLIP_NORM'_weightDecay'$WEIGHT_DECAY'_'$PRETRAIN_CKPT

# PRETRAINED_MODEL=pretrained_checkpoints/prophetnet_large_pretrained_160G_14epoch_model.pt

# sleep 150m
# CUDA_VISIBLE_DEVICES=0 fairseq-train \
CUDA_VISIBLE_DEVICES=0,1,2,3 python $USER_DIR/train.py $DATA_DIR \
--fp16 \
--user-dir $USER_DIR --task translation_prophetnet --arch $ARCH \
--optimizer adam --adam-betas $ADAM_BETAS --clip-norm $CLIP_NORM \
--lr $LR --min-lr 1e-09 \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates $WARMUP \
--dropout 0.1 --attention-dropout 0.1 --weight-decay $WEIGHT_DECAY \
--criterion $CRITERION --label-smoothing 0.1 \
--update-freq $UPDATE_FREQ  --max-sentences $MAX_SENTENCE \
--num-workers 4 \
--load-sep \
--ddp-backend=no_c10d --max-epoch $MAX_EPOCH \
--max-source-positions $MAX_SOURCE --max-target-positions $MAX_TARGET \
--skip-invalid-size-inputs-valid-test \
--seed 1 \
--truncate-source \
--save-dir $SAVE_DIR \
--keep-last-epochs $MAX_KEEP_EPOCH \
--tensorboard-logdir $TENSORBOARD_LOGDIR \
--ngram 2 \
--load-from-pretrained-model $PRETRAINED_MODEL \

# --distributed-backend nccl --distributed-world-size 8 --distributed-rank 0 \
# --distributed-init-method 'tcp://10.207.176.244:4322' --distributed-port 4322 \
# --save-interval-updates 100 --keep-interval-updates 15 \
# --reset-optimizer \
# --max-plm-update 10000 \
# --sigmoid-u 1.01 \
# --plm-decay 0.0001\
# --reset-optimizer \
# --load-from-pretrained-model $PRETRAINED_MODEL \
# --update-freq 64  --max-sentences 2 \ #large
