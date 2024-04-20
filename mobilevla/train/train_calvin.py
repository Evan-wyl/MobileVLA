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
from train_utils import get_checkpoint, train_one_epoch_calvin, train_one_epoch_calvin_diff, train_one_epoch_calvin_cotrain, train_one_epoch_calvin_two_way, \
get_ckpt_name, get_ckpt_name_pattern
from torch.distributed.elastic.multiprocessing.errors import record
from transformers import (
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

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
    parser.add_argument("--device_map", type=str, default="auto")
    parser.add_argument("--load_8bit", type=bool, default=False)
    parser.add_argument("--load_4bit", type=bool, default=False)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()

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


if __name__ == '__main__':
    main()