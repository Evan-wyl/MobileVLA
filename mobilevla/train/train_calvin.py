from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
import wandb
import itertools
import numpy as np
import random
import os
import glob

from collections import OrderedDict
from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from huggingface_hub import hf_hub_download

from train_utils import get_checkpoint, train_one_epoch_calvin, \
    train_one_epoch_calvin_diff, train_one_epoch_calvin_cotrain, train_one_epoch_calvin_two_way, \
    get_ckpt_name, get_ckpt_name_pattern

from mobilevla.data.data import get_data
from mobilevla.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mobilevla.models.factory import create_model_and_transforms


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(default=None)
    lang_name: Optional[str] = field(default='MobileLLaMA-1.4B-Chat')

    vision_tower: Optional[str] = field(default=None)
    vision_tower_type: Optional[str] = field(default='clip')

    sep_adapter: Optional[bool] = field(default=True)
    mm_projector_type: Optional[str] = field(default='linear')

    sep_lm_head: Optional[bool] = field(default=False)
    use_state: Optional[bool] = field(default=False)
    use_hist: Optional[bool] = field(default=True)
    use_diff: Optional[bool] = field(default=False)
    last_action: Optional[bool] = field(default=False)
    fusion_mode: Optional[str] = field(default='')
    multi_step_action: Optional[int] = field(default=1)
    pooling: Optional[str] = field(default='max')
    residual: Optional[bool] = field(default=False)
    decoder_type: Optional[str] = field(default='lstm')
    hidden_size: Optional[int] = field(default=None)
    state_dim: Optional[int] = field(default=15)
    return_feature: Optional[bool] = field(default=False)

    freeze_adapter: Optional[bool] = field(default=False)
    unfreeze_vit: Optional[bool] = field(default=False)
    freeze_embed: Optional[bool] = field(default=False)

    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    use_gripper: Optional[bool] = field(default=False)
    pad_length: Optional[int] = field(default=-1)
    debug: Optional[bool] = field(default=False)
    real_data: Optional[bool] = field(default=False)
    co_train: Optional[bool] = field(default=False)

    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    window_size: Optional[int] = field(default=32)
    train_params: Optional[int] = field(default=-1)
    from_scratch: Optional[bool] = field(default=False)

    # wandb parameters
    report_to_wandb: Optional[bool] = field(default=False)
    wandb_project: Optional[str] = field(default=None)
    wandb_entity: Optional[str] = field(default=None)
    run_name: Optional[str] = field(default='MobileVLA')

    # diffusion model params
    n_timesteps: Optional[int] = field(default=150)
    diff_horizon: Optional[int] = field(default=32)
    predict_epsilon: Optional[bool] = field(default=True)

    learning_rate: Optional[float] = field(default=1e-4)
    batch_size_calvin: Optional[int] = field(default=1)
    train_num_samples_calvin: Optional[int] = field(default=100)
    num_epochs: Optional[int] = field(default=1)
    lr_scheduler: Optional[str] = field(default='constant')
    warmup_steps: Optional[int] = field(default=5000)

    resume_from_checkpoint: Optional[str] = field(default=None)

    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model, image_processor, tokenizer = create_model_and_transforms(model_args, data_args, training_args)

    if data_args.debug:
        calvin_dataset = get_data(data_args, image_processor, tokenizer, "debug")
    elif data_args.real_data:
        calvin_dataset = get_data(data_args, image_processor, tokenizer, "real")
    else:
        calvin_dataset = get_data(data_args, image_processor, tokenizer, "calvin")

    if training_args.co_train:
        coco_loader = get_data(data_args, image_processor, tokenizer, "coco")
        vqa_loader = get_data(data_args, image_processor, tokenizer, "vqa")
        coco_cycle_loader = itertools.cycle(coco_loader)
        vqa_cycle_loader = itertools.cycle(vqa_loader)

    if training_args.report_to_wandb:
        wandb.init(
            project=training_args.wandb_project,
            entity=training_args.wandb_entity,
            name=training_args.run_name,
            config=vars(training_args),
        )

    if model.head_type == "diffusion" and (not data_args.debug):
        normalizer = model.diffusion_model.normalizer
        all_actions = np.vstack([calvin_dataset.dataset.__getitem__((i, 1), True)["actions"] for i in range(0, 10000)])
        normalizer.fit(all_actions, last_n_dims=1, mode='limits')

    # model = model.to(device_id)
    #
    # ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

    def get_grouped_params(model):
        params_with_wd, params_without_wd = [], []

        def apply_decay(x):
            return (
                    "gated_cross_attn_layer" in x
                    and "ff_gate" not in x
                    and "attn_gate" not in x
                    and "norm" not in x
                    and "bias" not in x
            )

        for n, p in model.named_parameters():

            if apply_decay(n):
                params_with_wd.append(p)
            else:
                params_without_wd.append(p)

        return [
            {"params": [p for p in params_with_wd if p.requires_grad], "weight_decay": args.weight_decay},
            {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0},
        ]

    training_args.learning_rate = training_args.learning_rate * training_args.batch_size_calvin / 6  # adaptive lr
    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=training_args.learning_rate)

    total_training_steps = (
                                   (training_args.train_num_samples_calvin) // (training_args.batch_size_calvin * args.world_size)
                           ) * training_args.num_epochs

    if training_args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif training_args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=total_training_steps,
        )
    elif training_args.lr_scheduler == 'cosine_restart':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=training_args.warmup_steps
        )

    if os.path.exists(f"{training_args.run_name}") and training_args.resume_from_checkpoint is None:
        ckpt_name = get_ckpt_name_pattern(training_args)
        checkpoint_list = glob.glob(f"{training_args.run_name}/{ckpt_name}")
        print(ckpt_name)
        checkpoint_list = [_ for _ in checkpoint_list if "__sep" not in _ and 'iter' not in _ and 'weights' not in _]
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {training_args.run_name}.")
        else:
            training_args.resume_from_checkpoint = sorted(
                checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0])
            )[-1]
            print(
                f"Found checkpoint {training_args.resume_from_checkpoint} for run {training_args.run_name}."
            )

    resume_from_epoch = 0
    if training_args.resume_from_checkpoint is not None and training_args.from_scratch is False:
        checkpoint = torch.load(training_args.resume_from_checkpoint, map_location="cpu")

        def filter_ckpt(checkpoint, skip_keys=[]):
            new_state_dict = OrderedDict()
            for key, value in checkpoint.items():
                flag = True
                for skip_key in skip_keys:
                    if skip_key in key:
                        flag = False
                        break
                if flag:
                    new_state_dict[key] = value
            return new_state_dict

        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        if not data_args.real_data:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            resume_from_epoch = checkpoint["epoch"] + 1

    ddp_model.train()

    if data_args.real_data:
        resume_from_epoch = 0
    for epoch in range(resume_from_epoch, training_args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader

        if model.head_type == "diffusion":
            train_one_epoch_calvin_diff(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        elif model.fusion_mode == 'two_way':
            train_one_epoch_calvin_two_way(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )
        elif data_args.co_train:
            train_one_epoch_calvin_cotrain(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                coco_loader=coco_cycle_loader,
                vqa_loader=vqa_cycle_loader,
                device_id=device_id,
                wandb=wandb,
            )
        else:
            train_one_epoch_calvin(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb,
            )

        ckpt_name = get_ckpt_name(args)
        torch.save(get_checkpoint(ddp_model), f"{training_args.run_name}/{ckpt_name}")
        if training_args.report_to_wandb and training_args.save_checkpoints_to_wandb:
            wandb.save(f"{training_args.run_name}/{ckpt_name}")


if __name__ == '__main__':
    train()