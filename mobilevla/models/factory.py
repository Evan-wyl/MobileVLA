import torch

import transformers
from transformers import LlamaTokenizer, LlamaConfig, LlamaForCausalLM, CLIPVisionModel, AutoTokenizer, BitsAndBytesConfig

from mobilevlm.mobilevlm import conversation as conversation_lib
from mobilevlm.mobilevlm.model.mobilellama import MobileLlamaForCausalLM
from mobilevlm.mobilevlm.model.vision_encoder import build_vision_tower
from mobilevlm.mobilevlm.model.vision_projector import build_vision_projector

from mobilevla.train.train_calvin import ModelArguments, DataArguments, TrainingArguments
from mobilevla.models.mobilevlm_bc import BCMobileVLM
from mobilevla.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def create_model_and_transforms(model_args: ModelArguments, data_args: DataArguments, training_args: TrainingArguments):
    global local_rank
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else
                     (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type  # {'fp4', 'nf4'}
            )
        ))

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False)
    # adding pad_token
    tokenizer.pad_token = tokenizer.unk_token
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    lang_config = LlamaConfig.from_pretrained(model_args.model_name_or_path)
    lang_encoder = LlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args)
    lang_encoder.config = lang_config
    lang_encoder.config.use_cache = False
    lang_encoder.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(lang_encoder, "enable_input_require_grads"):
            lang_encoder.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            lang_encoder.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    vision_encoder = build_vision_tower(model_cfg=model_args)
    vision_encoder.requires_grad_(False)

    data_args.image_processor = vision_encoder.image_processor
    data_args.is_multimodal = True

    lang_config.mm_hidden_size = vision_encoder.hidden_size
    lang_config.hidden_size = lang_config.hidden_size
    mm_projector = build_vision_projector(lang_config)
    mm_projector.requires_grad_(True)

    # initialize_vision_tokenizer
    if model_args.mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        lang_encoder.resize_token_embeddings(len(tokenizer))

    if model_args.mm_use_im_start_end:
        num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        lang_encoder.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = lang_encoder.get_input_embeddings().weight.data
            output_embeddings = lang_encoder.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

        if model_args.tune_mm_mlp_adapter:
            for p in lang_encoder.get_input_embeddings().parameters():
                p.requires_grad = True
            for p in lang_encoder.get_output_embeddings().parameters():
                p.requires_grad = False

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
            assert num_new_tokens == 2
            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
    elif model_args.mm_use_im_patch_token:
        if model_args.tune_mm_mlp_adapter:
            for p in lang_encoder.get_input_embeddings().parameters():
                p.requires_grad = False
            for p in lang_encoder.get_output_embeddings().parameters():
                p.requires_grad = False

    model = BCMobileVLM(
        lang_encoder,
        vision_encoder,
        mm_projector)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

