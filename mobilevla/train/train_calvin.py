""" Main training script """

import argparse
import copy
import glob
import os
import random
from collections import OrderedDict
import numpy as np
import torch
import wandb
from huggingface_hub import hf_hub_download

from torch.nn.parallel import DistributedDataParallel as DDP

from mobilevla.data.data import get_data
from mobilevla.train.distributed import world_info_from_env, init_distributed_device
from train_utils import get_checkpoint, train_one_epoch_calvin, train_one_epoch_calvin_diff, train_one_epoch_calvin_cotrain, train_one_epoch_calvin_two_way, \
get_ckpt_name, get_ckpt_name_pattern
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)
from mobilevla.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from transformers import AutoTokenizer, BitsAndBytesConfig
from mobilevlm.mobilevlm.model.mobilellama import MobileLlamaForCausalLM


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


@record
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--eval_hist_size", type=int, default=-1)
    parser.add_argument("--window_size", type=int, default=32)
    parser.add_argument("--n_obs_steps", type=int, default=6)
    parser.add_argument("--tcp_rel", default=False, action="store_true")
    parser.add_argument("--clip_state", default=False, action="store_true")
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", default=False, action="store_true")
    parser.add_argument("--real_data", default=False, action="store_true")
    parser.add_argument("--local-rank", default=0, type=int)
    parser.add_argument("--report_to_wandb", default=False, action="store_true")
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--offline", default=False, action="store_true")
    parser.add_argument("--run_name", type=str, default="MobileVLA", help="used to name saving directory and wandb run")
    parser.add_argument("--precision", choices=["amp_bf16", "amp_bfloat16", "bf16", "fp16", "fp32"], default="fp32", help="Floating point precision.",)
    parser.add_argument("--head_type", type=str, default="lstm")
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--lr_scheduler", default="constant", type=str, help="constant, linear, or cosine")
    parser.add_argument("--batch_size_calvin", type=int, default=1)
    parser.add_argument("--train_num_samples_calvin", type=int, default=100)
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--warmup_steps", default=5000, type=int)
    parser.add_argument("--resume_from_checkpoint", type=str,
                        help="path to checkpoint to resume from, this should contain model, optimizer, and lr_scheduler states",
                        default=None)
    parser.add_argument("--from_scratch", default=False, action="store_true", help="whether to train the model from scratch")
    parser.add_argument("--fusion_mode", default="pre", type=str, help="pre or post to fusion vision info")
    parser.add_argument("--delete_previous_checkpoint", action="store_true", help="delete previous checkpoint when saving new checkpoint")
    parser.add_argument("--save_checkpoints_to_wandb", default=False, action="store_true", help="save checkpoints to wandb")

    args = parser.parse_args()

    if args.eval_hist_size == -1:
        args.eval_hist_size = args.window_size
        if args.head_type == "diffusion":
            args.eval_hist_size = args.n_obs_steps
    if args.tcp_rel:
        args.clip_state = True
    if args.save_checkpoints_to_wandb and not args.report_to_wandb:
        raise ValueError("save_checkpoints_to_wandb requires report_to_wandb")

    if args.offline:
        os.environ["WANDB_MODE"] = "offline"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    args.local_rank, args.rank, args.world_size = world_info_from_env()
    device_id = init_distributed_device(args)
    print("device_id: ", device_id)

    random_seed(seed=args.seed)
    kwargs = {"device_map": args.device_map}
    if args.load_8bit:
        kwargs['load_in_8bit'] = True
    elif args.load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    model = MobileLlamaForCausalLM.from_pretrained(args.model_path, low_cpu_mem_usage=True, **kwargs)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)

    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if 'v2' in getattr(model.config, "mm_projector_type", "ldpnet"):
        vision_tower.load_image_processor()
    elif not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=args.device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    if args.debug:
        calvin_dataset = get_data(args, image_processor, tokenizer, "debug")
    elif args.real_data:
        calvin_dataset = get_data(args, image_processor, tokenizer, "real")
    else:
        calvin_dataset = get_data(args, image_processor, tokenizer, "calvin")

    random_seed(args.seed, args.rank)

    if args.report_to_wandb:
        wandb.init(project=args.wandb_project,
                   entity=args.wandb_entity,
                   name=args.run_name,
                   config=vars(args))

    device_id = args.rank % torch.cuda.device_count()
    if args.precision == "bf16" or args.precision == "amp_bfloat16" or args.precision == "amp_bf16":
        model = model.bfloat16()
    elif args.precision == "fp16":
        model = model.half()
    else:
        model = model.float()
    if args.head_type == "diffusion" and (not args.debug):
        normalizer = model.diffusion_model.normalizer
        all_actions = np.vstack([calvin_dataset.dataset.__getitem__((i, 1), True)["actions"] for i in range(0, 10000)])
        normalizer.fit(all_actions, last_n_dims=1, mode="limits")

    model = model.to(device_id)

    ddp_model = DDP(model, device_ids=[device_id], find_unused_parameters=True)

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
            {"params": [p for p in params_without_wd if p.requires_grad], "weight_decay": 0.0}
        ]

    args.learning_rate = args.learning_rate * args.batch_size_calvin / 6
    optimizer = torch.optim.AdamW(get_grouped_params(ddp_model), lr=args.learning_rate)

    total_training_steps = (
        (args.train_num_sample_calvin) // (args.batch_size_calvin * args.world_size)
    ) * args.num_epochs

    if args.rank == 0:
        print(f"Total training steps: {total_training_steps}")

    if args.lr_scheduler == "linear":
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps
        )
    elif args.lr_scheduler == "cosine":
        lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=total_training_steps
        )
    elif args.lr_scheduler == "cosine_restart":
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-7)
    else:
        lr_scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps
        )

    use_diff = (args.head_type == "diffusion")

    if os.path.exists(f"{args.run_name}") and args.resume_from_checkpoint is None:
        ckpt_name = get_ckpt_name_pattern(args)
        checkpoint_list = glob.glob(f"{args.run_name}/{ckpt_name}")
        print(ckpt_name)
        checkpoint_list = [_ for _ in checkpoint_list if "_step" not in _ and "iter" not in _ and "weights" not in _]
        if len(checkpoint_list) == 0:
            print(f"Found no checkpoints for run {args.run_name}.")
        else:
            args.resume_from_checkpoint = sorted(checkpoint_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))[-1]
            print(f"Found checkpoint {args.resume_from_checkpoint} for run {args.run_name}.")

    resume_from_epoch = 0
    if args.resume_from_checkpoint is not None and not args.from_scratch:
        if args.rank == 0:
            print(f"Loading checkpoint from{args.resume_from_checkpoint}")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location="cpu")

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
        ddp_model.load_state_dict(checkpoint["model_state_dict"], False)
        if not args.real_data:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
            resume_from_epoch = checkpoint["epoch"] + 1
    ddp_model.train()
    if args.real_data:
        resume_from_epoch = 0
    for epoch in range(resume_from_epoch, args.num_epochs):
        calvin_dataset.set_epoch(epoch)
        calvin_loader = calvin_dataset.dataloader

        if args.head_type == "diffusion":
            train_one_epoch_calvin_diff(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb)
        elif args.fusion_mode == "two_way":
            train_one_epoch_calvin_two_way(
                args=args,
                model=ddp_model,
                epoch=epoch,
                tokenizer=tokenizer,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                calvin_loader=calvin_loader,
                device_id=device_id,
                wandb=wandb)
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
                wandb=wandb)

        if args.rank == 0:
            if not os.path.exists(args.run_name):
                os.makedirs(args.run_name)

            checkpoint_dict = {
                "epoch": epoch,
                "model_state_dict": get_checkpoint(ddp_model),
                "optimizer_state_dict": optimizer.state_dict(),
                "lr_scheduler_state_dict": lr_scheduler.state_dict()
            }

            ckpt_name = get_ckpt_name(args, epoch)
            ckpt_path = os.path.join(args.run_name, ckpt_name)

            print(f"Saving checkpoint to {ckpt_path}")
            torch.save(checkpoint_dict, ckpt_path)
            if args.delete_previous_checkpoint:
                if epoch > 0:
                    os.remove(ckpt_path)

        if args.rank == 0:
            if not os.path.exists(args.run_name):
                os.makedirs(args.run_name)

            ckpt_name = get_ckpt_name(args)
            torch.save(get_checkpoint(ddp_model), f"{args.run_name}/{ckpt_name}")
            if args.report_to_wandb and args.save_checkpoints_to_wandb:
                wandb.save(f"{args.run_name}/{ckpt_name}")


if __name__ == '__main__':
    main()