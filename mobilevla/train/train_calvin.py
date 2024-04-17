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

from mobilevlm.mobilevlm.model.mobilevlm import load_pretrained_model


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)


@record
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument()

    args = parser.parse_args()

    tokenizer, model, image_processor, context_len = load_pretrained_model(args.model_path, args.load_8bit, args.load_4bit)



if __name__ == '__main__':
    main()