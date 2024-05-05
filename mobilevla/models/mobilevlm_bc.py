import torch
from torch import nn

import copy
from collections import namedtuple
from einops import rearrange, repeat

from mobilevla.models.action_head import DeterministicDecoder, DiffusionDecoder, FCDecoder, GPTDecoder
from mobilevla.train.train_calvin import ModelArguments, DataArguments, TrainingArguments


class BCMobileVLM(nn.Module):
    def __init__(
            self,
            lang_encoder: nn.Module,
            vision_encoder: nn.Module,
            mm_projector: nn.Module,
            window_size: int = 8,
            use_gripper=False,
            fusion_mode='',
            use_state=False,
            use_diff=False,
            diff_horizon=32,
            last_action=False,
            n_timesteps=150,
            state_dim=15,
            use_hist=False,
            predict_epsilon=True,
            multi_step_action=1,
            sep_lm_head=False,
            return_feature=False,
            pooling='max',
            decoder_type='lstm',
            hidden_size=None):
        super().__init__()
        self.lang_encoder = lang_encoder
        self.vision_encoder = vision_encoder
        self.mm_projector = mm_projector

        self.window_size = window_size
        self.use_gripper = use_gripper
        self.fusion_mode = fusion_mode
        self.use_state = use_state
        self.use_diff = use_diff
        self.diff_horizon = diff_horizon
        self.last_action = last_action
        self.n_timesteps = n_timesteps
        self.state_dim = state_dim
        self.use_hist = use_hist
        self.predict_epsilon = predict_epsilon
        self.multi_step_action = multi_step_action
        self.sep_lm_head = sep_lm_head
        self.return_feature = return_feature
        self.pooling = pooling
        self.decoder_type = decoder_type
        self.hidden_size = hidden_size

        self.vis_dim = self.vision_encoder.hidden_size
        self.lang_dim = self.lang_encoder.config.hidden_size

        if use_state:
            self.state_fc = nn.Linear(state_dim, self.vis_dim)
        in_features = lang_encoder.lm_head.in_features

        if decoder_type == 'lstm':
            action_head = DeterministicDecoder(in_features, window_size,
                                           use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode,
                                           use_state=use_state, return_feature=return_feature,
                                           multi_step_action=multi_step_action, pooling=pooling)
            self.lang_encoder.lm_head = action_head
        elif decoder_type == 'fc':
            action_head = FCDecoder(in_features, window_size,
                                    use_diff=use_diff, last_action=last_action,
                                    fusion_mode=fusion_mode, use_state=use_state,
                                    return_feature=return_feature,
                                    multi_step_action=multi_step_action)
            self.lang_encoder.lm_head = action_head
        elif decoder_type == 'diffusion':
            action_head = DiffusionDecoder(
                in_features,
                window_size,
                input_dim=in_features + 1,
                n_timesteps=n_timesteps,
                horizon=diff_horizon,
                predict_epsilon=predict_epsilon,
            )
            self.lang_encoder.lm_head = action_head
        elif decoder_type == 'gpt':
            action_head = GPTDecoder(in_features, window_size, use_diff=use_diff, last_action=last_action,
                                 fusion_mode=fusion_mode, multi_step_action=multi_step_action, pooling=pooling,
                                 hidden_size=hidden_size)
            self.lang_encoder.lm_head = action_head
        else:
            raise NotImplementedError

        if sep_lm_head:
            self.lm_head = self.lang_encoder.lm_head
            self.lang_encoder.lm_head = nn.Identity()

    def forward(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            past_key_values=None,
            use_cache: bool = False,
            vision_gripper=None,
            state_tensor=None,
            return_feature=False,
    ):
        images = vision_x
        if self.use_gripper:
            images = torch.concat([vision_x, vision_gripper], dim=1)

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            images=images)

        if self.sep_lm_head:
            output_llm = output.logits
            output_lm_head = self.lm_head(output_llm, state_tensor=state_tensor, return_feature=return_feature)
            output.logits = output_lm_head

        return output