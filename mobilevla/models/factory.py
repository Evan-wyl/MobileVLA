from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig

from mobilevla.models.mobilevlm_bc import BCMobileVLM
from mobilevlm.mobilevlm.model.vision_encoder import build_vision_tower
from mobilevla.train.train_calvin import ModelArguments, DataArguments, TrainingArguments


def create_model_and_transforms(
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments):
    text_tokenizer = LlamaTokenizer.from_pretrained(model_args.lang_encoder_path)
    text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>"]})
    if text_tokenizer.pad_token is None:
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    lang_encoder = LlamaForCausalLM.from_pretrained(model_args.lang_encoder_path)
    lang_encoder.config = LlamaConfig.from_pretrained(model_args.lang_encoder_path)
    lang_encoder.resize_token_embeddings(len(text_tokenizer))

    vision_encoder = build_vision_tower(model_cfg=model_args)
    image_processor = vision_encoder.image_processor

    lang_encoder.config.mm_vision_tower = model_args.vision_tower
    lang_encoder.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
    lang_encoder.config.mm_hidden_size = vision_encoder.hidden_size

    if 'llama' in str.lower(model_args.lang_name):
        Model_fn = BCMobileVLM
    else:
        raise NotImplementedError

    model = Model_fn(
        vision_encoder,
        lang_encoder,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        model_args,
        data_args,
        training_args
    )

    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0
    model.adapter.requires_grad_(True)

    if model_args.freeze_adapter:
        model.adapter.requires_grad_(False)
    if not model_args.freeze_embed:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
    model.lang_encoder.lm_head.requires_grad_(True)

    if model.sep_lm_head:
        model.lm_head.requires_grad_(True)
    if model.use_diff:
        model.diffusion_model.requires_grad_(True)
    if model_args.unfreeze_vit:
        model.vision_encoder.requires_grad_(True)

    return model, image_processor, text_tokenizer
