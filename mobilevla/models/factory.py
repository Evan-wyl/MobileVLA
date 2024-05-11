from logging import debug
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaForCausalLM, LlamaTokenizer, LlamaConfig
import open_clip
from typing import Optional
from mobilevla.models.mobilevlm_bc import BCMobileVLM
from mobilevla.models.flamingo_mpt import MPTFlamingo


def get_transforms(
    clip_vision_encoder_path: str = "ViT-L-14",
    clip_vision_encoder_pretrained: str = "openai",
    tokenizer_path: str = "path_to/llama-7b-hf-jxu124",
    use_local_files: bool = False,
):
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )

    text_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    return image_processor, text_tokenizer


def create_model_and_transforms(
        clip_vision_encoder_path: str,
        clip_vision_encoder_pretrained: str,
        lang_encoder_path: str,
        tokenizer_path: str,
        mm_projector_type: str,
        cross_attn_every_n_layers: int = 1,
        use_local_files: bool = False,
        window_size: int = 32,
        freeze_embed: bool = False,
        train_params = -1,
        use_gripper=False,
        use_state=False,
        last_action=False,
        fusion_mode='',
        pad_length=-1,
        debug=False,
        sep_resampler=False,
        sep_lm_head=False,
        unfreeze_vit=False,
        return_feature=False,
        multi_step_action=1,
        llm_name='llama_9b',
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
        refresh=-1,
        **flamingo_kwargs,
):
    vision_encoder, _, image_processor = open_clip.create_model_and_transforms(
        clip_vision_encoder_path, pretrained=clip_vision_encoder_pretrained
    )
    # set the vision encoder to output the visual features
    vision_encoder.visual.output_tokens = True

    text_tokenizer = LlamaTokenizer.from_pretrained(
        tokenizer_path, local_files_only=use_local_files
    )
    # add Flamingo special tokens to the tokenizer
    text_tokenizer.add_special_tokens(
        {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
    )
    if text_tokenizer.pad_token is None:
        # Issue: GPT models don't have a pad token, which we use to
        # modify labels for the loss.
        text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    if debug:
        # Load the local checkpoint into a model instance.
        lang_encoder = LlamaForCausalLM.from_pretrained(lang_encoder_path, ignore_keys=["config"], trust_remote_code=True)
        # Set the `init_weights` parameter to `False` to prevent the model from loading the pretrained weights.
        lang_encoder.init_weights(False)
    else:
        print(lang_encoder_path)
        lang_encoder = LlamaForCausalLM.from_pretrained(
            lang_encoder_path, local_files_only=use_local_files, trust_remote_code=True
        )

    lang_encoder.resize_token_embeddings(len(text_tokenizer))
    # lang_encoder.config = LlamaConfig.from_pretrained(lang_encoder_path)
    
    if 'llama' in llm_name:
        Model_fn = BCMobileVLM
    elif 'mpt' in llm_name:
        Model_fn = MPTFlamingo
    else:
        raise NotImplementedError
    
    model = Model_fn(
        vision_encoder,
        lang_encoder,
        mm_projector_type,
        text_tokenizer.encode("<|endofchunk|>")[-1],
        text_tokenizer.encode("<image>")[-1],
        vis_dim=open_clip.get_model_config(clip_vision_encoder_path)["vision_cfg"][
            "width"
        ],
        cross_attn_every_n_layers=cross_attn_every_n_layers,
        window_size=window_size,
        use_gripper=use_gripper,
        use_state=use_state,
        fusion_mode=fusion_mode,
        last_action=last_action,
        pad_length=pad_length,
        sep_resampler=sep_resampler,
        sep_lm_head=sep_lm_head,
        return_feature=return_feature,
        multi_step_action=multi_step_action,
        llm=llm_name,
        pooling=pooling,
        residual=residual,
        tcp_rel=tcp_rel,
        replan=replan,
        decoder_type=decoder_type,
        hidden_size=hidden_size,
        refresh=refresh,
        fwd_pred=fwd_pred,
        fwd_pred_hand=fwd_pred_hand,
        no_image_patch=no_image_patch,
        global_latent=global_latent,
        **flamingo_kwargs,
    )

    # Freeze all parameters
    model.requires_grad_(False)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 0

    if train_params == -1:
        model.lang_encoder.gated_cross_attn_layers.requires_grad_(True)
        model.perceiver.requires_grad_(True)
    else:
        param_per_layer = 140
        layer_num = int(train_params / param_per_layer + 0.5)
        cnt = 0
        for ix in range(len(model.lang_encoder.gated_cross_attn_layers)-1, -1, -1):
            if cnt >= layer_num:
                break
            if model.lang_encoder.gated_cross_attn_layers[ix] is not None:
                model.lang_encoder.gated_cross_attn_layers[ix].requires_grad_(True)
                cnt += 1
    if freeze_sampler:
        model.perceiver.requires_grad_(False)
    if not freeze_embed:
        model.lang_encoder.get_input_embeddings().requires_grad_(True)
    model.lang_encoder.lm_head.requires_grad_(True)

    if model.sep_lm_head:
        model.lm_head.requires_grad_(True)
    if model.use_diff:
        model.diffusion_model.requires_grad_(True)
    if unfreeze_vit:
        model.vision_encoder.requires_grad_(True)
    # # Unfreeze the action head 
    # model.action_head.requires_grad_(True)

    print(
        f"Flamingo model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters"
    )

    return model, image_processor, text_tokenizer
