PREDICTION_DIR=$1
DATASET_LIST='cnndm gigaword xsum msnews squadqg msqg coqa personachat'
# DATASET_LIST='cnndm msqg squadqg coqa personachat'
# DATASET_LIST='gigaword'
# VERSION_LIST='easy medium medium+ hard'
VERSION_LIST='easy'
for DATASET in $DATASET_LIST;
do
for VERSION in $VERSION_LIST;
do
PREDICTION=$PREDICTION_DIR/$DATASET.$VERSION.dev

echo $DATASET
echo $VERSION
echo $PREDICTION

PYTHONIOENCODING=utf8 python glge/script/eval.py --version $VERSION --dataset $DATASET --generated $PREDICTION --split dev

done
done