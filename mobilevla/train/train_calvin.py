from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
import wandb

from transformers import get_constant_schedule_with_warmup, get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from transformers import LlamaTokenizer, LlamaForCausalLM, LlamaModel
from transformers import AutoTokenizer, BitsAndBytesConfig

from huggingface_hub import hf_hub_download

from mobilevlm.mobilevlm.model.mobilellama import MobileLlamaForCausalLM

from train_utils import get_checkpoint, train_one_epoch_calvin, \
    train_one_epoch_calvin_diff, train_one_epoch_calvin_cotrain, train_one_epoch_calvin_two_way, \
    get_ckpt_name, get_ckpt_name_pattern

from mobilevla.data.data import get_data
from mobilevla.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from mobilevla.models.factory import create_model_and_transforms


@dataclass
class ModelArguments:
    lang_encoder_path: Optional[str] = field(default=None)
    lang_name: Optional[str] = field(default='MobileLLaMA-1.4B-Chat')
    vision_encoder_path: Optional[str] = field(default=None)
    freeze_embed: Optional[bool] = field(default=False)
    use_state: Optional[bool] = field(default=False)
    use_hist: Optional[bool] = field(default=True)
    last_action: Optional[bool] = field(default=False)
    fusion_mode: Optional[str] = field(default='')
    sep_perceiver: Optional[bool] = field(default=True)
    freeze_perceiver: Optional[bool] = field(default=False)
    sep_lm_head: Optional[bool] = field(default=False)
    unfreeze_vit: Optional[bool] = field(default=False)
    return_feature: Optional[bool] = field(default=False)
    multi_step_action: Optional[int] = field(default=1)
    pooling: Optional[str] = field(default='max')
    residual: Optional[bool] = field(default=False)
    decoder_type: Optional[str] = field(default='lstm')
    hidden_size: Optional[int] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    use_gripper: Optional[bool] = field(default=False)
    debug: Optional[bool] = False
    pad_length: Optional[int] = field(default=-1)

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


if __name__ == '__main__':
    train()