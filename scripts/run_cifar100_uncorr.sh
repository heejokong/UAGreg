# BASE SETTINGS
gpu_id=$1
trainer=uagreg
# 
dataset=cifar100
arch=wideresnet
out=../results/${trainer}_${dataset}_$2_UNCORRELATED
# PARAMS FOR "DATASET"
num_classes=60
num_labeled=$2
mu=4
# PARAMS FOR "OPTIMIZER"
total_step=204800 # TOTAL_STEP = 1024*200
eval_step=1024
batch_size=64
lr=0.03
weight_decay=5e-4
# PARAMS FOR "SSL"
start_u=0
conf_th=0.0
lambda_u=1.0
# PARAMS FOR "OUTLIER DETECTOR"
temp_o=1.5
momentum=0.9
# PARAMS FOR "GRAPH CONTRASTIVE LEARNING"
graph_th=0.8
lambda_g=1.0
T=0.2
# RUN
CUDA_VISIBLE_DEVICES=$gpu_id python main.py --trainer $trainer --out $out --mu $mu \
                                            --dataset $dataset --num_classes $num_classes --num_labeled $num_labeled \
                                            --total_step $total_step --eval_step $eval_step --batch_size $batch_size \
                                            --arch $arch --lr $lr --weight_decay $weight_decay \
                                            --start_u $start_u --conf_th $conf_th --lambda_u $lambda_u \
                                            --temp_o $temp_o --momentum $momentum \
                                            --graph_th $graph_th --lambda_g $lambda_g --T $T \
                                            --expand_labels \
                                            # --label_biased \
                                            # --use_amp \
