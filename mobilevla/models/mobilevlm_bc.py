import torch
from einops import rearrange, repeat
from torch import nn
import copy
from mobilevla.models.action_head import DeterministicDecoder, DiffusionDecoder, FCDecoder, GPTDecoder
from collections import namedtuple


class BCMobileVLM(nn.Module):
    def __init__(
            self,
            lang_encoder: nn.Module,
            vis_dim: int,
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
            debug=False,
            predict_epsilon=True,
            pad_length=-1,
            multi_step_action=1,
            sep_lm_head=False,
            return_feature=False,
            pooling='max',
            residual=False,
            tcp_rel=False,
            replan=-1,
            decoder_type='lstm',
            hidden_size=None,
            refresh=-1
    ):
        super().__init__()
        self.use_gripper = use_gripper
        self.use_state = use_state
        self.fusion_mode = fusion_mode
        self.vis_dim = vis_dim
        self.window_size = window_size
        self.tcp_rel = tcp_rel
        self.act_step = multi_step_action
        print('window size: {}'.format(window_size))
        self.use_hist = use_hist
        self.lang_encoder = lang_encoder
        self.pad_length = pad_length
        self.replan = replan
        if self.replan != -1:
            self.replan = min(int(replan * self.window_size), 180)
        self.refresh = refresh
        self.lang_dim = lang_encoder.config.hidden_size

        self.residual = residual

        # if not debug:
        #     if 'llama' in llm:
        #         self.lang_encoder.init_flamingo(
        #             media_token_id=media_token_id,
        #             vis_hidden_size=self.vis_dim,
        #             cross_attn_every_n_layers=cross_attn_every_n_layers,
        #             use_media_placement_augmentation=self.use_media_placement_augmentation,
        #             residual=residual,
        #         )
        #     else:
        #         self.lang_encoder.init_flamingo(
        #             media_token_id=media_token_id,
        #             lang_hidden_size=self.lang_dim,
        #             vis_hidden_size=self.vis_dim,
        #             cross_attn_every_n_layers=cross_attn_every_n_layers,
        #             gradient_checkpointing=False,
        #         )

        if use_state:
            self.state_fc = nn.Linear(state_dim, self.vis_dim)
        if use_hist:
            self.frame_embs = nn.Parameter(torch.randn(self.window_size, self.vis_dim))
        # To-do: nn archiecture for actor
        in_features = lang_encoder.lm_head.in_features
        self.use_diff = use_diff
        self.decoder_type = decoder_type
        if decoder_type == 'lstm':
            lm_head = DeterministicDecoder(in_features, self.window_size,
                                           use_diff=use_diff, last_action=last_action, fusion_mode=fusion_mode,
                                           use_state=use_state, return_feature=return_feature,
                                           multi_step_action=multi_step_action, pooling=pooling)
            self.lang_encoder.lm_head = lm_head
        elif decoder_type == 'fc':
            if use_hist:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size,
                                                                         use_diff=use_diff, last_action=last_action,
                                                                         fusion_mode=fusion_mode, use_state=use_state,
                                                                         return_feature=return_feature,
                                                                         multi_step_action=multi_step_action)
            elif 'vit_concat' in fusion_mode:
                self.lang_encoder.lm_head = self.action_head = FCDecoder(in_features, self.window_size,
                                                                         use_diff=use_diff, last_action=last_action,
                                                                         fusion_mode=fusion_mode, use_state=use_state,
                                                                         return_feature=return_feature,
                                                                         multi_step_action=multi_step_action)
            else:
                raise NotImplementedError
        elif decoder_type == 'diffusion':
            if use_diff:
                self.diffusion_model = DiffusionDecoder(
                    self.action_head.hidden_size,
                    self.window_size,
                    input_dim=self.action_head.out_features + 1,
                    n_timesteps=n_timesteps,
                    horizon=diff_horizon,
                    predict_epsilon=predict_epsilon,
                )
            else:
                raise NotImplementedError
        elif decoder_type == 'gpt':
            lm_head = GPTDecoder(in_features, self.window_size, use_diff=use_diff, last_action=last_action,
                                 fusion_mode=fusion_mode, multi_step_action=multi_step_action, pooling=pooling,
                                 hidden_size=hidden_size)
            self.lang_encoder.lm_head = self.action_head = lm_head
        else:
            raise NotImplementedError

        self.sep_lm_head = sep_lm_head
        if sep_lm_head:
            self.lm_head = self.lang_encoder.lm_head
            self.lang_encoder.lm_head = nn.Identity()

    def forward(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            labels: torch.Tensor = None,
            use_cached_vision_x: bool = False,
            clear_conditioned_layers: bool = True,
            past_key_values=None,
            use_cache: bool = False,
            vision_gripper=None,
            state_tensor=None,
            return_feature=False,
            policy_mask=None
    ):
        images = torch.concat([vision_x, vision_gripper], dim=1)
        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            images=images,
            return_dict=return_dict)

        if self.sep_lm_head:
            output_llm = output.logits
            output_lm_head = self.lm_head(output_llm, state_tensor=state_tensor, return_feature=return_feature)
            output.logits = output_lm_head

        return output

    # Generate function with actor for text time adpatation
    def generate(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: torch.Tensor = None,
            num_beams=1,
            max_new_tokens=None,
            temperature=1.0,
            top_k=0,
            top_p=1.0,
            no_repeat_ngram_size=0,
            prefix_allowed_tokens_fn=None,
            length_penalty=1.0,
            num_return_sequences=1,
            do_sample=False,
            early_stopping=False,
    ):
        """
        Generate text conditioned on vision and language inputs.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                images in the same chunk are collated along T_img, and frames are collated along F
                currently only F=1 is supported (single-frame videos)
            lang_x (torch.Tensor): Language input
                shape (B, T_txt)
            max_length (int, optional): Maximum length of the output. Defaults to None.
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            num_beams (int, optional): Number of beams. Defaults to 1.
            max_new_tokens (int, optional): Maximum new tokens. Defaults to None.
            temperature (float, optional): Temperature. Defaults to 1.0.
            top_k (int, optional): Top k. Defaults to 0.
            top_p (float, optional): Top p. Defaults to 1.0.
            no_repeat_ngram_size (int, optional): No repeat ngram size. Defaults to 0.
            length_penalty (float, optional): Length penalty. Defaults to 1.0.
            num_return_sequences (int, optional): Number of return sequences. Defaults to 1.
            do_sample (bool, optional): Do sample. Defaults to False.
            early_stopping (bool, optional): Early stopping. Defaults to False.
        Returns:
            torch.Tensor: lang_x with generated tokens appended to it
        """
        if num_beams > 1:
            vision_x = vision_x.repeat_interleave(num_beams, dim=0)

        self._encode_vision_x(vision_x=vision_x)

        output = self.lang_encoder.generate(
            lang_x,
            attention_mask=attention_mask,
            eos_token_id=self.eoc_token_id,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            num_return_sequences=num_return_sequences,
            do_sample=do_sample,
            early_stopping=early_stopping,
        )

        return output