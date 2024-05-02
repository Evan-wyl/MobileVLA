import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, BitsAndBytesConfig
from mobilevla.models.mobilevlm_bc import BCMobileVLM
from mobilevlm.mobilevlm.model.mobilellama import MobileLlamaForCausalLM
from mobilevla.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


def create_model_and_transforms(
        lm_path: str,
        device_map: str,
        load_8bit: bool,
        load_4bit: bool,
        window_size: int = 32,
        freeze_embed: bool = False,
        train_params = -1,
        use_gripper=False,
        use_state=False,
        last_action=False,
        fusion_mode='',
        pad_length=-1,
        debug=False,
        sep_lm_head=False,
        unfreeze_vit=False,
        return_feature=False,
        multi_step_action=1,
        pooling='max',
        residual=False,
        tcp_rel=False,
        replan=-1,
        decoder_type='lstm',
        hidden_size=None,
        freeze_sampler=False,
        fwd_pred=False,
        fwd_pred_hand=False,
        no_image_patch=False,
        global_latent=1,
        refresh=-1):

    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(lm_path, use_fast=False)
    lang_encoder = MobileLlamaForCausalLM.from_pretrained(lm_path, low_cpu_mem_usage=True, **kwargs)

    mm_use_im_start_end = getattr(lang_encoder.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(lang_encoder.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    lang_encoder.resize_token_embeddings(len(tokenizer))

    vision_tower = lang_encoder.get_vision_tower()
    if 'v2' in getattr(lang_encoder.config, "mm_projector_type", "ldpnet"):
        vision_tower.load_image_processor()
    elif not vision_tower.is_loaded:
        vision_tower.load_model()
    image_processor = vision_tower.image_processor

    Model_fn = BCMobileVLM

    model = Model_fn(
        lang_encoder
    )

    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    if train_params == -1:
        pass
    else:
        pass

    if not freeze_embed:
        pass

    if model.sep_lm_head:
        pass
    if model.use_diff:
        pass
    if unfreeze_vit:
        pass

    print(
        f"MobileVLM model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return tokenizer, model, image_processor