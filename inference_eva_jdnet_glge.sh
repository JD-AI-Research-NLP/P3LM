DATASET=$1
DATASPLIT=$2
CHECKPOINTNUM=$3
BEAM=$4 #BEAM=4
LENPEN=$5 #LENPEN=1.0
MINLEN=$6
MAXLEN=$7

MAX_SOURCE=$8
MAX_TARGET=$9
TEST_BATCHSIZE=$10
LR=$11
BATCHSIZE=$12
WARMUP=$13
PRETRAIN_CKPT=$14
WEIGHT_DECAY=$15
ADAM_BETAS=$16
echo $ADAM_BETAS
CLIP_NORM=$17
RETURNFILE=$18


USER_DIR=./jdnet_pretrain_v6
PRETRAIN=pretrain # noPretrain
NET=JDNet
MODEL=Large #Base
LOSS=L2RPLM #L2R/PLM/PLM2L2R
NGRAM=Ngram2

DATADIR=glge/data/easy/$DATASET\_data/processed_translation_prophetnet

SUFFIX='_'$PRETRAIN'_'$NGRAM'_'$NET'_'$MODEL'_'$LOSS'_'$DATASPLIT'_beam'$BEAM'_pelt'$LENPEN'_MinLen'$MINLEN'_MaxLen'$MAXLEN'_lr'$LR'_trainbatchsize'$BATCHSIZE'_warm'$WARMUP'_clipNorm'$CLIP_NORM'_weightDecay'$WEIGHT_DECAY'_'$PRETRAIN_CKPT'_epoch'$CHECKPOINTNUM

CHECK_POINT=glge/models/$DATASET/finetune'_checkpoints_'$PRETRAIN'_'$NGRAM'_'$NET'_'$MODEL'_'$LOSS'_lr'$LR'_trainbatchsize'$BATCHSIZE'_warm'$WARMUP'_clipNorm'$CLIP_NORM'_weightDecay'$WEIGHT_DECAY'_'$PRETRAIN_CKPT/checkpoint$CHECKPOINTNUM'.pt'
CHECK_POINT_BEST=glge/models/$DATASET/finetune'_checkpoints_'$PRETRAIN'_'$NGRAM'_'$NET'_'$MODEL'_'$LOSS'_lr'$LR'_trainbatchsize'$BATCHSIZE'_warm'$WARMUP'_clipNorm'$CLIP_NORM'_weightDecay'$WEIGHT_DECAY'_'$PRETRAIN_CKPT/checkpoint'_best.pt'
CHECK_POINT_LAST=glge/models/$DATASET/finetune'_checkpoints_'$PRETRAIN'_'$NGRAM'_'$NET'_'$MODEL'_'$LOSS'_lr'$LR'_trainbatchsize'$BATCHSIZE'_warm'$WARMUP'_clipNorm'$CLIP_NORM'_weightDecay'$WEIGHT_DECAY'_'$PRETRAIN_CKPT/checkpoint'_last.pt'

OUTPUT_FILE=glge/outputs/$DATASET/output$SUFFIX.txt
SORT_FILE=glge/outputs/$DATASET/sort_hypo$SUFFIX.txt
SCORE_FILE=glge/outputs/$DATASET/score$SUFFIX.txt


#========================================================================================
# # fairseq-generate $DATASET/processed --path $CHECK_POINT --user-dir prnet_pretrain --task translation_prophetnet --batch-size 32 --gen-subset test --beam $BEAM --num-workers 4 --min-len 45 --max-len-b 110 --no-repeat-ngram-size 3 --lenpen $LENPEN 2>&1 > $OUTPUT_FILE

# # python $USER_DIR/generate.py $DATASET/processed --path $CHECK_POINT --user-dir $USER_DIR --task translation_prophetnet --batch-size 32 --gen-subset test --beam $BEAM --num-workers 4 --min-len 45 --max-len-b 110 --no-repeat-ngram-size 3 --lenpen $LENPEN 2>&1 > $OUTPUT_FILE

# CUDA_VISIBLE_DEVICES=0 python3 $USER_DIR/generate.py $DATADIR --path $CHECK_POINT --user-dir $USER_DIR --task translation_prophetnet --batch-size $TEST_BATCHSIZE --truncate-source --max-source-positions $MAX_SOURCE --truncate-target --max-target-positions $MAX_TARGET --gen-subset $DATASPLIT --beam $BEAM --num-workers 4 --min-len $MINLEN --max-len-b $MAXLEN --no-repeat-ngram-size 3 --lenpen $LENPEN --num-shards 4 --shard-id 0 2>&1 > $OUTPUT_FILE.1 &

# CUDA_VISIBLE_DEVICES=1 python3 $USER_DIR/generate.py $DATADIR --path $CHECK_POINT --user-dir $USER_DIR --task translation_prophetnet --batch-size $TEST_BATCHSIZE --truncate-source --max-source-positions $MAX_SOURCE --truncate-target --max-target-positions $MAX_TARGET --gen-subset $DATASPLIT --beam $BEAM --num-workers 4 --min-len $MINLEN --max-len-b $MAXLEN --no-repeat-ngram-size 3 --lenpen $LENPEN --num-shards 4 --shard-id 1 2>&1 > $OUTPUT_FILE.2 &

# CUDA_VISIBLE_DEVICES=2 python3 $USER_DIR/generate.py $DATADIR --path $CHECK_POINT --user-dir $USER_DIR --task translation_prophetnet --batch-size $TEST_BATCHSIZE --truncate-source --max-source-positions $MAX_SOURCE --truncate-target --max-target-positions $MAX_TARGET --gen-subset $DATASPLIT --beam $BEAM --num-workers 4 --min-len $MINLEN --max-len-b $MAXLEN --no-repeat-ngram-size 3 --lenpen $LENPEN --num-shards 4 --shard-id 2 2>&1 > $OUTPUT_FILE.3 &

# CUDA_VISIBLE_DEVICES=3 python3 $USER_DIR/generate.py $DATADIR --path $CHECK_POINT --user-dir $USER_DIR --task translation_prophetnet --batch-size $TEST_BATCHSIZE --truncate-source --max-source-positions $MAX_SOURCE --truncate-target --max-target-positions $MAX_TARGET --gen-subset $DATASPLIT --beam $BEAM --num-workers 4 --min-len $MINLEN --max-len-b $MAXLEN --no-repeat-ngram-size 3 --lenpen $LENPEN --num-shards 4 --shard-id 3 2>&1 > $OUTPUT_FILE.4 &

# wait
# cat $OUTPUT_FILE.1 $OUTPUT_FILE.2 $OUTPUT_FILE.3 $OUTPUT_FILE.4 > $OUTPUT_FILE
# rm $OUTPUT_FILE.1 $OUTPUT_FILE.2 $OUTPUT_FILE.3 $OUTPUT_FILE.4

# grep ^H $OUTPUT_FILE | cut -c 3- | sort -n | cut -f3- | sed "s/ ##//g" > $SORT_FILE
#========================================================================================


# PYTHONIOENCODING=utf8 python glge/script/eval.py --version easy --dataset $DATASET --generated $SORT_FILE --split $DATASPLIT> $SCORE_FILE

PYTHONIOENCODING=utf8 python glge/script/eval.py --version easy --dataset $DATASET --generated $SORT_FILE > $SCORE_FILE


cat $SCORE_FILE | cut -f3 > $RETURNFILE



# rm $CHECK_POINT
rm $CHECK_POINT_BEST
rm $CHECK_POINT_LAST
