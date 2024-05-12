#!/bin/bash
scp -r .cache/clip ~/.cache/
export PATH=$PATH:/home/vipuser/proj/mobilevla/mobilevla
export PYTHONPATH=$PYTHONPATH:/home/vipuser/proj/mobilevla/mobilevla

# dataset path
calvin_dataset_path='/home/vipuser/proj/calvin_data/task_ABCD_D'
# language model path
lm_path='/home/vipuser/proj/pretrain/MobileLLaMA-1.4B-Chat'
# tokenizer path
tokenizer_path='/home/vipuser/proj/pretrain/MobileLLaMA-1.4B-Chat'
#projector type
mm_projector_type='ldpnetv2'

subfix=`date "+%Y%m%d-%H%M"`
log_file="/home/vipuser/proj/logs/training_"${subfix}".log"
source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
#python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2  --master_port=6042 robot_flamingo/train/train_calvin.py \
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6042 mobilevla/train/train_calvin.py \
    --report_to_wandb \
    --co_train \
    --llm_name mobilellama-1.4b \
    --use_gripper \
    --fusion_mode post \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 \
    --batch_size_calvin 6 \
    --run_name MobileVLA-LSTM-Cotrain \
    --calvin_dataset ${calvin_dataset_path} \
    --lm_path ${lm_path} \
    --tokenizer_path ${tokenizer_path} \
    --mm_projector_type ${mm_projector_type} \
    --cross_attn_every_n_layers 4 \
    --loss_multiplier_calvin 1.0 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --from_scratch \
    --window_size 12 > ${log_file} 2>&1
