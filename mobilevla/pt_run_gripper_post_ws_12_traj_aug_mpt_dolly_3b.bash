#!/bin/bash

export PATH=$PATH:path/to/mobilevla/mobilevla
export PYTHONPATH=$PYTHONPATH:path/to/mobilevla/mobilevla

# dataset path
calvin_dataset_path='path/to/calvin_data/task_ABCD_D'
# language model path
lm_path='path/to/MobileLLaMA-1.4B-Chat'
# tokenizer path
tokenizer_path='path/to/MobileLLaMA-1.4B-Chat'
# openflamingo ckpt path
#openflamingo_checkpoint='path/to/OpenFlamingo-3B-vitl-mpt-1b-dolly/checkpoint.pt'

subfix=`date "+%Y%m%d-%H%M"`
log_file="logs/training_"${subfix}".log"
source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
#python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2  --master_port=6042 robot_flamingo/train/train_calvin.py \
torchrun --nnodes=1 --nproc_per_node=8 --master_port=6042 mobilevla/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mobilellama-1.4b \
    --use_gripper \
    --fusion_mode post \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name MobileVLADBG \
    --calvin_dataset ${calvin_dataset_path} \
    --lm_path ${lm_path} \
    --tokenizer_path ${tokenizer_path} \
    --mm_projector_type ldpnetv2 \
    --cross_attn_every_n_layers 4 \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --save_every_iter 10000 \
    --from_scratch \
    --window_size 12 > ${log_file} 2>&1
