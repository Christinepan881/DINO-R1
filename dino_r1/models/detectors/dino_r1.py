# Copyright (c) Robert Bosch LLC CR/RHI1-NA. All rights reserved.

import warnings
from typing import Dict, Optional, Tuple, Union
import numpy as np
import math 

import os
import time
from mmdet.models.layers import SinePositionalEncoding

import torch
import torch.nn as nn
from mmengine.runner.amp import autocast
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import OptSampleList, SampleList
from mmdet.utils import ConfigType

from mmdet.models.layers.transformer.utils import MLP, coordinate_to_encoding
from mmdet.models.layers.transformer.grounding_dino_layers import GroundingDinoTransformerDecoder, GroundingDinoTransformerEncoder
from mmdet.models.layers.transformer.deformable_detr_layers import DeformableDetrTransformerDecoderLayer
from mmdet.models.detectors.dino import DINO
from mmdet.models.detectors.glip import create_positive_map, create_positive_map_label_to_token, run_ner
from mmdet.models.detectors.grounding_dino import clean_label_name, chunks
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh

from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmcv.ops import MultiScaleDeformableAttention
from mmcv.cnn import build_norm_layer
from mmengine.model import ModuleList


class VisualPromptEncoder(DeformableDetrTransformerDecoderLayer):

    def _init_layers(self) -> None:
        self.self_attn = MultiheadAttention(**self.self_attn_cfg)
        self.cross_attn = MultiScaleDeformableAttention(**self.cross_attn_cfg)
        self.embed_dims = self.self_attn.embed_dims
        self.ffn = FFN(**self.ffn_cfg)
        norms_list = [ build_norm_layer(self.norm_cfg, self.embed_dims)[1] for _ in range(3) ]
        self.norms = ModuleList(norms_list)
        self.visual_bbox_encoding = MLP(self.embed_dims * 2, self.embed_dims, self.embed_dims, 2)

    def forward(self,
                query: Tensor, 
                value: Tensor = None, 
                key_padding_mask: Tensor = None, 
                self_key_padding_mask=None, 
                reference_points=None, 
                spatial_shapes=None, 
                level_start_index=None, 
                valid_ratios=None,
                ) -> Tensor:

        if reference_points.shape[-1] == 4:
            reference_points_input = reference_points[:, :, None] * torch.cat( [valid_ratios, valid_ratios], -1)[:, None]
        else:
            assert reference_points.shape[-1] == 2
            reference_points_input = reference_points[:, :, None] * valid_ratios[:, None]

        query_sine_embed = coordinate_to_encoding(reference_points_input[:, :, 0, :]) 
        query_pos = self.visual_bbox_encoding(query_sine_embed) 
        query = self.cross_attn(
            query=query,
            key=None,
            value=value, 
            query_pos=query_pos,
            attn_mask=None, 
            key_padding_mask=key_padding_mask, 
            reference_points=reference_points_input, 
            spatial_shapes=spatial_shapes, 
            level_start_index=level_start_index, 
            )
        query = self.norms[1](query)

        Kmax = self_key_padding_mask.shape[-1]
        device = self_key_padding_mask.device
        dtype = self_key_padding_mask.dtype
        self_attn_mask = torch.ones(Kmax, Kmax, device=device, dtype=dtype)
        kid = torch.arange(Kmax)
        self_attn_mask[kid, kid] = False

        query = self.self_attn(
            query=query,
            key=query,
            value=query,
            query_pos=query_pos,
            key_pos=query_pos,
            attn_mask=self_attn_mask, 
            )
        query = self.norms[0](query)

        query = self.ffn(query)
        query = self.norms[2](query)

        return query 


@MODELS.register_module()
class DINO_R1(DINO):

    def __init__(self,
                 language_model,
                 *args,
                 use_autocast=False,
                 visual_prompt_encoder_cfg=None,
                 original_max_text_len=256,
                 eval_prompt_mode=None,
                 visual_prompts_pt=None,
                 contra_loss_weight=1.0,
                 grpo_alpha=1.0,
                 grpo_beta=1.0,
                 in_debug=False,
                 **kwargs) -> None:

        self.language_model_cfg = language_model
        self._special_tokens = '. '
        self.use_autocast = use_autocast

        self.visual_prompt_encoder_cfg = visual_prompt_encoder_cfg 
        self.original_max_text_len = original_max_text_len
        self.eval_prompt_mode = eval_prompt_mode
        self.visual_prompts_pt = visual_prompts_pt
        self.contra_loss_weight = contra_loss_weight
        self.grpo_alpha = grpo_alpha
        self.grpo_beta = grpo_beta

        self.in_debug = in_debug

        super().__init__(*args, **kwargs)

    def _init_layers(self) -> None:
        """Initialize layers except for backbone, neck and bbox_head."""
        self.positional_encoding = SinePositionalEncoding(
            **self.positional_encoding)
        self.encoder = GroundingDinoTransformerEncoder(**self.encoder)
        self.decoder = GroundingDinoTransformerDecoder(**self.decoder)
        self.embed_dims = self.encoder.embed_dims
        self.query_embedding = nn.Embedding(self.num_queries, self.embed_dims)
        num_feats = self.positional_encoding.num_feats
        assert num_feats * 2 == self.embed_dims, \
            f'embed_dims should be exactly 2 times of num_feats. ' \
            f'Found {self.embed_dims} and {num_feats}.'

        self.level_embed = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))
        self.memory_trans_fc = nn.Linear(self.embed_dims, self.embed_dims)
        self.memory_trans_norm = nn.LayerNorm(self.embed_dims)

        # text modules
        self.language_model = MODELS.build(self.language_model_cfg)
        self.text_feat_map = nn.Linear(
            self.language_model.language_backbone.body.language_dim,
            self.embed_dims,
            bias=True)
        
        self.visual_prompt_encoding = VisualPromptEncoder(**self.visual_prompt_encoder_cfg)
        self.contra_logit_scale = nn.Parameter(torch.ones([]) * math.log(1/0.07))
        self.visual_query_embedding = nn.Embedding(1, self.embed_dims)

        if self.eval_prompt_mode == 'v': 
            generic_visual_prompts = torch.load(self.visual_prompts_pt, map_location='cpu')
            self.generic_visual_prompts = nn.Embedding.from_pretrained(generic_visual_prompts, freeze=True)

    def init_weights(self) -> None:
        """Initialize weights for Transformer and other components."""
        super().init_weights()
        nn.init.constant_(self.text_feat_map.bias.data, 0)
        nn.init.xavier_uniform_(self.text_feat_map.weight.data)

    def to_enhance_text_prompts(self, original_caption, enhanced_text_prompts):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            if word in enhanced_text_prompts:
                enhanced_text_dict = enhanced_text_prompts[word]
                if 'prefix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['prefix']
                start_i = len(caption_string)
                if 'name' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['name']
                else:
                    caption_string += word
                end_i = len(caption_string)
                tokens_positive.append([[start_i, end_i]])

                if 'suffix' in enhanced_text_dict:
                    caption_string += enhanced_text_dict['suffix']
            else:
                tokens_positive.append(
                    [[len(caption_string),
                      len(caption_string) + len(word)]])
                caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def to_plain_text_prompts(self, original_caption):
        caption_string = ''
        tokens_positive = []
        for idx, word in enumerate(original_caption):
            tokens_positive.append(
                [[len(caption_string),
                  len(caption_string) + len(word)]])
            caption_string += word
            caption_string += self._special_tokens
        return caption_string, tokens_positive

    def get_tokens_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompts: Optional[ConfigType] = None
    ) -> Tuple[dict, str, list]:
        """Get the tokens positive and prompts for the caption."""
        if isinstance(original_caption, (list, tuple)) or custom_entities:
            if custom_entities and isinstance(original_caption, str):
                original_caption = original_caption.strip(self._special_tokens)
                original_caption = original_caption.split(self._special_tokens)
                original_caption = list(
                    filter(lambda x: len(x) > 0, original_caption))

            original_caption = [clean_label_name(i) for i in original_caption]

            if custom_entities and enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption, enhanced_text_prompts)
            else:
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption)

            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [caption_string],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            entities = original_caption
        else:
            if not original_caption.endswith('.'):
                original_caption = original_caption + self._special_tokens
            # NOTE: Tokenizer in Grounding DINO is different from
            # that in GLIP. The tokenizer in GLIP will pad the
            # caption_string to max_length, while the tokenizer
            # in Grounding DINO will not.
            tokenized = self.language_model.tokenizer(
                [original_caption],
                padding='max_length'
                if self.language_model.pad_to_max else 'longest',
                return_tensors='pt')
            tokens_positive, noun_phrases = run_ner(original_caption)
            entities = noun_phrases
            caption_string = original_caption

        return tokenized, caption_string, tokens_positive, entities

    def get_positive_map(self, tokenized, tokens_positive):
        positive_map = create_positive_map(
            tokenized,
            tokens_positive,
            max_num_entities=self.original_max_text_len,
                )
        positive_map_label_to_token = create_positive_map_label_to_token(
            positive_map, plus=1)
        return positive_map_label_to_token, positive_map

    def get_tokens_positive_and_prompts(
        self,
        original_caption: Union[str, list, tuple],
        custom_entities: bool = False,
        enhanced_text_prompt: Optional[ConfigType] = None,
        tokens_positive: Optional[list] = None,
    ) -> Tuple[dict, str, Tensor, list]:
        """Get the tokens positive and prompts for the caption.

        Args:
            original_caption (str): The original caption, e.g. 'bench . car .'
            custom_entities (bool, optional): Whether to use custom entities.
                If ``True``, the ``original_caption`` should be a list of
                strings, each of which is a word. Defaults to False.

        Returns:
            Tuple[dict, str, dict, str]: The dict is a mapping from each entity
            id, which is numbered from 1, to its positive token id.
            The str represents the prompts.
        """
        if tokens_positive is not None:
            if tokens_positive == -1:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                return None, original_caption, None, original_caption
            else:
                if not original_caption.endswith('.'):
                    original_caption = original_caption + self._special_tokens
                tokenized = self.language_model.tokenizer(
                    [original_caption],
                    padding='max_length'
                    if self.language_model.pad_to_max else 'longest',
                    return_tensors='pt')
                positive_map_label_to_token, positive_map = \
                    self.get_positive_map(tokenized, tokens_positive)

                entities = []
                for token_positive in tokens_positive:
                    instance_entities = []
                    for t in token_positive:
                        instance_entities.append(original_caption[t[0]:t[1]])
                    entities.append(' / '.join(instance_entities))
                return positive_map_label_to_token, original_caption, \
                    positive_map, entities

        chunked_size = self.test_cfg.get('chunked_size', -1)
        if not self.training and chunked_size > 0:
            assert isinstance(original_caption,
                              (list, tuple)) or custom_entities is True
            all_output = self.get_tokens_positive_and_prompts_chunked(
                original_caption, enhanced_text_prompt)
            positive_map_label_to_token, \
                caption_string, \
                positive_map, \
                entities = all_output
        else:
            tokenized, caption_string, tokens_positive, entities = \
                self.get_tokens_and_prompts(
                    original_caption, custom_entities, enhanced_text_prompt)
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)
        return positive_map_label_to_token, caption_string, \
            positive_map, entities

    def get_tokens_positive_and_prompts_chunked(
            self,
            original_caption: Union[list, tuple],
            enhanced_text_prompts: Optional[ConfigType] = None):
        chunked_size = self.test_cfg.get('chunked_size', -1) 
        original_caption = [clean_label_name(i) for i in original_caption] 

        original_caption_chunked = chunks(original_caption, chunked_size) 
        ids_chunked = chunks(
            list(range(1,
                       len(original_caption) + 1)), chunked_size) 

        positive_map_label_to_token_chunked = []
        caption_string_chunked = []
        positive_map_chunked = []
        entities_chunked = []

        for i in range(len(ids_chunked)):
            if enhanced_text_prompts is not None:
                caption_string, tokens_positive = self.to_enhance_text_prompts(
                    original_caption_chunked[i], enhanced_text_prompts)
            else: 
                caption_string, tokens_positive = self.to_plain_text_prompts(
                    original_caption_chunked[i])
            tokenized = self.language_model.tokenizer([caption_string],
                                                      return_tensors='pt')
            if tokenized.input_ids.shape[1] > self.language_model.max_tokens:
                warnings.warn('Inputting a text that is too long will result '
                              'in poor prediction performance. '
                              'Please reduce the --chunked-size.')
            positive_map_label_to_token, positive_map = self.get_positive_map(
                tokenized, tokens_positive)

            caption_string_chunked.append(caption_string)
            positive_map_label_to_token_chunked.append(
                positive_map_label_to_token)
            positive_map_chunked.append(positive_map)
            entities_chunked.append(original_caption_chunked[i])

        return positive_map_label_to_token_chunked, \
            caption_string_chunked, \
            positive_map_chunked, \
            entities_chunked

    def forward_transformer(
        self,
        encoder_inputs_dict,
        decoder_inputs_dict,
        text_dict: Dict,
        batch_data_samples: OptSampleList = None,
    ) -> Dict:
        
        encoder_outputs_dict = self.forward_encoder(
            **encoder_inputs_dict, text_dict=text_dict)

        tmp_dec_in, head_inputs_dict = self.pre_decoder(
            **encoder_outputs_dict, batch_data_samples=batch_data_samples)
        decoder_inputs_dict.update(tmp_dec_in)

        decoder_outputs_dict = self.forward_decoder(**decoder_inputs_dict)
        head_inputs_dict.update(decoder_outputs_dict)

        return head_inputs_dict

    def forward_encoder(self, feat: Tensor, feat_mask: Tensor,
                        feat_pos: Tensor, spatial_shapes: Tensor,
                        level_start_index: Tensor, valid_ratios: Tensor,
                        text_dict: Dict) -> Dict:
        text_token_mask = text_dict['text_token_mask']
        memory, memory_text = self.encoder(
            query=feat,
            query_pos=feat_pos,
            key_padding_mask=feat_mask,  # for self_attn
            spatial_shapes=spatial_shapes,
            level_start_index=level_start_index,
            valid_ratios=valid_ratios,
            # for text encoder
            memory_text=text_dict['embedded'],
            text_attention_mask=~text_token_mask,
            position_ids=text_dict['position_ids'],
            text_self_attention_masks=text_dict['masks'])
        encoder_outputs_dict = dict(
            memory=memory,
            memory_mask=feat_mask,
            spatial_shapes=spatial_shapes,
            memory_text=memory_text,
            text_token_mask=text_token_mask)
        return encoder_outputs_dict

    def pre_decoder(
        self,
        memory: Tensor,
        memory_mask: Tensor,
        spatial_shapes: Tensor,
        memory_text: Tensor,
        text_token_mask: Tensor,
        batch_data_samples: OptSampleList = None,
    ) -> Tuple[Dict]:
        bs, _, c = memory.shape

        output_memory, output_proposals = self.gen_encoder_output_proposals(memory, memory_mask, spatial_shapes)

        enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](output_memory, memory_text, text_token_mask)
        cls_out_features = enc_outputs_class.shape[-1]
        enc_outputs_coord_unact = self.bbox_head.reg_branches[self.decoder.num_layers](output_memory) + output_proposals

        # NOTE The DINO selects top-k proposals according to scores of
        # multi-class classification, while DeformDETR, where the input
        # is `enc_outputs_class[..., 0]` selects according to scores of
        # binary classification.
        topk_indices = torch.topk(enc_outputs_class.max(-1)[0], k=self.num_queries, dim=1)[1]
        topk_score = torch.gather(enc_outputs_class, 1, topk_indices.unsqueeze(-1).repeat(1, 1, cls_out_features))
        topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_indices.unsqueeze(-1).repeat(1, 1, 4))
        topk_coords = topk_coords_unact.sigmoid()
        topk_coords_unact = topk_coords_unact.detach()

        if self.training:
            select_score_per_token = enc_outputs_class.max(-1)[0] 
            select_score_per_q = torch.gather(select_score_per_token, 1, topk_indices) 

        query = self.query_embedding.weight[:, None, :]
        query = query.repeat(1, bs, 1).transpose(0, 1)
        if self.training:
            dn_label_query, dn_bbox_query, dn_mask, dn_meta = self.dn_query_generator(batch_data_samples)
            query = torch.cat([dn_label_query, query], dim=1)
            reference_points = torch.cat([dn_bbox_query, topk_coords_unact], dim=1)
        else:
            reference_points = topk_coords_unact
            dn_mask, dn_meta = None, None
        reference_points = reference_points.sigmoid()

        decoder_inputs_dict = dict(
            query=query,
            memory=memory,
            reference_points=reference_points,
            dn_mask=dn_mask,
            memory_text=memory_text,
            text_attention_mask=~text_token_mask,
        )
        # NOTE DINO calculates encoder losses on scores and coordinates
        # of selected top-k encoder queries, while DeformDETR is of all
        # encoder queries.
        head_inputs_dict = dict(
            enc_outputs_class=topk_score,
            enc_outputs_coord=topk_coords,
            dn_meta=dn_meta) if self.training else dict()
        # append text_feats to head_inputs_dict
        head_inputs_dict['memory_text'] = memory_text
        head_inputs_dict['text_token_mask'] = text_token_mask

        if self.training:
            head_inputs_dict['select_score_per_q'] = select_score_per_q 
            head_inputs_dict['select_id_per_q'] = topk_indices 
            head_inputs_dict['select_score_per_token'] = select_score_per_token.detach() 

        return decoder_inputs_dict, head_inputs_dict

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        
        with torch.no_grad():
            text_prompts = [
                data_samples.text for data_samples in batch_data_samples
            ]

            gt_labels = [
                data_samples.gt_instances.labels
                for data_samples in batch_data_samples
            ]

            if 'tokens_positive' in batch_data_samples[0]:
                tokens_positive = [
                    data_samples.tokens_positive
                    for data_samples in batch_data_samples
                ]
                positive_maps = []
                for token_positive, text_prompt, gt_label in zip(
                        tokens_positive, text_prompts, gt_labels):
                    tokenized = self.language_model.tokenizer(
                        [text_prompt],
                        padding='max_length'
                        if self.language_model.pad_to_max else 'longest',
                        return_tensors='pt')
                    new_tokens_positive = [
                        token_positive[label.item()] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                new_text_prompts = text_prompts
            else:
                new_text_prompts = []
                positive_maps = []
                if len(set(text_prompts)) == 1:
                    # All the text prompts are the same,
                    # so there is no need to calculate them multiple times.
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompts[0], True)
                    new_text_prompts = [caption_string] * len(batch_inputs)
                    for gt_label in gt_labels:
                        new_tokens_positive = [
                            tokens_positive[label] for label in gt_label
                        ]
                        _, positive_map = self.get_positive_map(
                            tokenized, new_tokens_positive)
                        positive_maps.append(positive_map)
                else:
                    for text_prompt, gt_label in zip(text_prompts, gt_labels):
                        tokenized, caption_string, tokens_positive, _ = \
                            self.get_tokens_and_prompts(
                                text_prompt, True)
                        new_tokens_positive = [
                            tokens_positive[label] for label in gt_label
                        ]
                        _, positive_map = self.get_positive_map(
                            tokenized, new_tokens_positive)
                        positive_maps.append(positive_map)
                        new_text_prompts.append(caption_string)

            text_dict = self.language_model(new_text_prompts)
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

            L = text_dict['masks'].shape[-1]
            for i, data_samples in enumerate(batch_data_samples):
                positive_map = positive_maps[i].to(
                    batch_inputs.device).bool().float()
                text_token_mask = text_dict['text_token_mask'][i]
                data_samples.gt_instances.positive_maps = positive_map[:, :L]
                data_samples.gt_instances.text_token_mask = \
                    text_token_mask.unsqueeze(0).repeat(
                        len(positive_map), 1)
        
        if self.use_autocast:
            with autocast(enabled=True):
                visual_features = self.extract_feat(batch_inputs)
        else:
            visual_features = self.extract_feat(batch_inputs)

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(visual_features, batch_data_samples)

        visual_dict, rank_oh_labels, uni_avg_visual_prompt = self.generate_visual_prompts(batch_data_samples, 
                                                    encoder_inputs_dict['feat'],
                                                    encoder_inputs_dict['feat_mask'],
                                                    encoder_inputs_dict['spatial_shapes'],
                                                    encoder_inputs_dict['level_start_index'],
                                                    encoder_inputs_dict['valid_ratios'],
                                                    ) 
        rank_v_oh_labels = rank_oh_labels 

        contra_loss = self.vl_contra_learning(text_dict['embedded'], uni_avg_visual_prompt, rank_oh_labels, batch_data_samples)
        
        B = len(batch_data_samples)

        for bi in range(B):
            M = rank_oh_labels[bi].shape[0]
            batch_data_samples[bi].gt_instances.positive_maps = rank_v_oh_labels[bi].float() 
            batch_data_samples[bi].gt_instances.text_token_mask = visual_dict['text_token_mask'][bi].repeat(M, 1) 
        
        B = len(batch_data_samples)
        L = 0 
        U = visual_dict['masks'].shape[-1]
        A = L + U
        device = visual_dict['masks'].device
        prompt_self_masks = torch.zeros(B, A, A, dtype=bool, device=device) 
        prompt_self_masks[:, :L, :L] = 0 * text_dict['masks'][:, :L, :L] 
        prompt_self_masks[:, L:, L:] = visual_dict['masks'] 
        prompt_dict = {
            'embedded': torch.cat([0 * text_dict['embedded'][:, :L], visual_dict['embedded']], dim=1), 
            'masks': prompt_self_masks.bool(), 
            'position_ids': torch.cat([0 * text_dict['position_ids'][:, :L], visual_dict['position_ids']], dim=1), 
            'text_token_mask': torch.cat([0 * text_dict['text_token_mask'][:, :L], visual_dict['text_token_mask']], dim=1).bool(), 
        }
        head_inputs_dict = self.forward_transformer(encoder_inputs_dict, 
                                                    decoder_inputs_dict, 
                                                    prompt_dict, 
                                                    batch_data_samples,
                                                    )

        grpo_meta = dict()
        grpo_meta['select_score_per_q'] = head_inputs_dict.pop('select_score_per_q')
        grpo_meta['select_id_per_q'] = head_inputs_dict.pop('select_id_per_q')
        select_score_per_token = head_inputs_dict.pop('select_score_per_token') 
        grpo_meta['ref_select_score_per_token'] = batch_data_samples[0].ref_select_score_per_token 
        if 'old_select_score_per_token' in batch_data_samples[0]:
            grpo_meta['old_select_score_per_token'] = batch_data_samples[0].old_select_score_per_token 
        else:
            grpo_meta['old_select_score_per_token'] = None

        grpo_meta['grpo_alpha'] = self.grpo_alpha
        grpo_meta['grpo_beta'] = self.grpo_beta

        losses = self.bbox_head.loss(**head_inputs_dict, batch_data_samples=batch_data_samples, grpo_meta=grpo_meta)
        losses['lv_contra_loss'] = contra_loss
        return losses , select_score_per_token 

    def generate_visual_prompts(self, 
                                batch_data_samples,
                                feat_flatten,
                                mask_flatten,
                                spatial_shapes,
                                level_start_index,
                                valid_ratios,
                                ):
        B = len(batch_data_samples)
        visual_query = [] 
        normalized_cxcywh_all = [] 
        num_ls = []
        for bi in range(B):
            img_h, img_w = batch_data_samples[bi].img_shape
            unnormalized_xyxy = batch_data_samples[bi].gt_instances.bboxes 
            M = unnormalized_xyxy.shape[0]
            num_ls.append(M)

            factor = unnormalized_xyxy.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0) 
            normalized_cxcywh = bbox_xyxy_to_cxcywh(unnormalized_xyxy / factor) 
            normalized_cxcywh_all.append(normalized_cxcywh) 

            bi_visual_query = self.visual_query_embedding.weight[:, :].repeat(M, 1) 
            visual_query.append(bi_visual_query) 
        
        C = feat_flatten.shape[-1]
        device = feat_flatten.device
        dtype = feat_flatten.dtype

        Mmax = max(num_ls)
        batch_idx = torch.cat([bi * torch.ones(num_ls[bi], dtype=torch.long) for bi in range(B)]) 
        m_idx = torch.cat([torch.arange(m, dtype=torch.long) for m in num_ls]) 

        visual_prompt = torch.zeros(B, Mmax, C, device=device, dtype=dtype) 
        visual_prompt_mask = torch.ones(B, Mmax, device=device, dtype=torch.bool) 
        reference_points = torch.tensor([0.5, 0.5, 0.99, 0.99], device=device, dtype=dtype).reshape([1, 1, 4]).repeat(B, Mmax, 1) 

        visual_prompt[batch_idx, m_idx] = torch.cat(visual_query, dim=0) 
        visual_prompt_mask[batch_idx, m_idx] = False 
        reference_points[batch_idx, m_idx] = torch.cat(normalized_cxcywh_all, dim=0) 

        visual_prompt = self.visual_prompt_encoding(
                query=visual_prompt, 
                value=feat_flatten, 
                key_padding_mask=mask_flatten, 
                self_key_padding_mask=visual_prompt_mask, 
                reference_points=reference_points, 
                spatial_shapes=spatial_shapes, 
                level_start_index=level_start_index, 
                valid_ratios=valid_ratios,
        ) 

        with torch.no_grad():
            rank_labels = torch.cat([d.gt_instances.labels for d in batch_data_samples]) 
            rank_uni_labels = torch.unique(rank_labels) 
            rank_oh_labels = rank_labels.unsqueeze(1) == rank_uni_labels.unsqueeze(0) 
            select_prob = (torch.rand(rank_oh_labels.shape, device=device) + 0.5) * rank_oh_labels 
            _, select_idx = select_prob.max(0) 

            all_instance_rank_oh_labels = rank_oh_labels 
            rank_oh_labels = torch.split(rank_oh_labels, num_ls, dim=0) 
        rank_visual_prompt = visual_prompt[batch_idx, m_idx][select_idx] 
        U = rank_visual_prompt.shape[0]
        visual_dict = {
            'embedded': rank_visual_prompt.repeat(B, 1, 1), 
            'masks': torch.eye(U, dtype=bool, device=device).repeat(B, 1, 1), 
            'position_ids': torch.zeros(B, U, dtype=torch.int64, device=device), 
            'text_token_mask': torch.ones(B, U, dtype=bool, device=device), 
        }

        all_instance_visual_prompt = visual_prompt[batch_idx, m_idx] 
        uni_avg_visual_prompt = [] 
        for ui in range(U):
            uni_avg_visual_prompt.append(all_instance_visual_prompt[all_instance_rank_oh_labels[:, ui]].mean(0)) 
        uni_avg_visual_prompt = torch.stack(uni_avg_visual_prompt, dim=0) 

        return visual_dict, rank_oh_labels, uni_avg_visual_prompt 

    def vl_contra_learning(self, text_prompt, visual_prompt, rank_oh_labels, batch_data_samples):

        B = len(batch_data_samples)
        uni_memory_l = torch.zeros_like(visual_prompt).repeat(B, 1, 1) 
        uni_label = [] 
        for bi in range(B):
            bi_labels = batch_data_samples[bi].gt_instances.labels 
            bi_pos_l = batch_data_samples[bi].gt_instances.positive_maps 
            bi_pos_v = rank_oh_labels[bi] 

            M = len(bi_labels)
            for mi in range(M):
                bi_mi_memory_l = text_prompt[bi][bi_pos_l[mi].bool()] 
                uni_memory_l[bi][bi_pos_v[mi].bool()] += bi_mi_memory_l.mean(0, keepdim=True) 

            uni_label.append(bi_pos_v.sum(0)) 

        uni_label = torch.stack(uni_label, dim=0).unsqueeze(2) 
        uni_memory_l = uni_memory_l.sum(0) / uni_label.sum(0) 

        # l2 normalize
        uni_memory_l_normed, uni_memory_v_normed = map(lambda t: F.normalize(t, p = 2, dim = -1), (uni_memory_l, visual_prompt))

        # cosine similarity as logits
        contra_logit_scale = self.contra_logit_scale.exp()
        contra_logits = contra_logit_scale * uni_memory_l_normed @ uni_memory_v_normed.t() 

        contra_labels = torch.arange(contra_logits.shape[0], device=visual_prompt.device) 
        lv_contra_loss = (F.cross_entropy(contra_logits, contra_labels) + F.cross_entropy(contra_logits.t(), contra_labels)) / 2
        return self.contra_loss_weight * lv_contra_loss

    def predict(self, batch_inputs, batch_data_samples, rescale: bool = True):
        if self.eval_prompt_mode == 'v': 
            return self.predict_v(batch_inputs, batch_data_samples, rescale)
        
        text_prompts = []
        enhanced_text_prompts = []
        tokens_positives = []
        for data_samples in batch_data_samples:
            text_prompts.append(data_samples.text)
            if 'caption_prompt' in data_samples:
                enhanced_text_prompts.append(data_samples.caption_prompt)
            else:
                enhanced_text_prompts.append(None)
            tokens_positives.append(data_samples.get('tokens_positive', None))

        assert 'custom_entities' in batch_data_samples[0]
        # Assuming that the `custom_entities` flag
        # inside a batch is always the same. For single image inference
        custom_entities = batch_data_samples[0].custom_entities

        assert len(text_prompts) == 1
        _positive_maps_and_prompts = [
            self.get_tokens_positive_and_prompts(
                text_prompts[0], custom_entities, enhanced_text_prompts[0],
                tokens_positives[0])
        ] * len(batch_inputs)

        token_positive_maps, text_prompts, _, entities = zip(
            *_positive_maps_and_prompts)

        is_rec_tasks = []
        for i, data_samples in enumerate(batch_data_samples):
            if token_positive_maps[i] is not None:
                is_rec_tasks.append(False)
            else:
                is_rec_tasks.append(True)
            data_samples.token_positive_map = token_positive_maps[i]
        
        visual_feats = self.extract_feat(batch_inputs)

        text_dict = self.language_model(list(text_prompts))
        if self.text_feat_map is not None:
            text_dict['embedded'] = self.text_feat_map(
                text_dict['embedded'])
        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(visual_feats, batch_data_samples)

        head_inputs_dict = self.forward_transformer(encoder_inputs_dict, 
                                                    decoder_inputs_dict, 
                                                    text_dict,
                                                    batch_data_samples,
                                                    )

        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples

    def predict_v(self, batch_inputs, batch_data_samples, rescale: bool = True): 

        assert len(batch_data_samples)==1
        B = 1

        visual_prompts = self.generic_visual_prompts.weight[None, :, :] 
        device = visual_prompts.device
        U = visual_prompts.shape[1]
        visual_dict = { 
            'embedded': visual_prompts, 
            'masks': torch.eye(U, dtype=bool, device=device).repeat(B, 1, 1), 
            'position_ids': torch.zeros(B, U, dtype=torch.int64, device=device), 
            'text_token_mask': torch.ones(B, U, dtype=bool, device=device), 
        }

        token_positive_map = {ci+1:[ci,] for ci in range(U)}
        batch_data_samples[0].token_positive_map = token_positive_map

        is_rec_tasks = [False,]

        entities = ([ clean_label_name(t) for t in batch_data_samples[0].text ],)

        visual_feats = self.extract_feat(batch_inputs)

        encoder_inputs_dict, decoder_inputs_dict = self.pre_transformer(visual_feats, batch_data_samples)
        head_inputs_dict = self.forward_transformer(encoder_inputs_dict, 
                                                    decoder_inputs_dict, 
                                                    visual_dict,
                                                    batch_data_samples,
                                                    )

        results_list = self.bbox_head.predict(
            **head_inputs_dict,
            rescale=rescale,
            batch_data_samples=batch_data_samples)

        for data_sample, pred_instances, entity, is_rec_task in zip(
                batch_data_samples, results_list, entities, is_rec_tasks):
            if len(pred_instances) > 0:
                label_names = []
                for labels in pred_instances.labels:
                    if is_rec_task:
                        label_names.append(entity)
                        continue
                    if labels >= len(entity):
                        warnings.warn(
                            'The unexpected output indicates an issue with '
                            'named entity recognition. You can try '
                            'setting custom_entities=True and running '
                            'again to see if it helps.')
                        label_names.append('unobject')
                    else:
                        label_names.append(entity[labels])
                # for visualization
                pred_instances.label_names = label_names
            data_sample.pred_instances = pred_instances
        return batch_data_samples

    def extract_visual_prompts_offline(self, 
                                       batch_data_samples,
                                       feat_flatten,
                                       mask_flatten,
                                       spatial_shapes,
                                       level_start_index,
                                       valid_ratios,
                                       vis_prompt_save_dir,
                                       ):

        B = len(batch_data_samples)
        visual_query = [] 
        normalized_cxcywh_all = [] 
        num_ls = []
        for bi in range(B):
            img_h, img_w = batch_data_samples[bi].img_shape
            unnormalized_xyxy = batch_data_samples[bi].gt_instances.bboxes 
            M = unnormalized_xyxy.shape[0]
            num_ls.append(M)

            factor = unnormalized_xyxy.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0) 
            normalized_cxcywh = bbox_xyxy_to_cxcywh(unnormalized_xyxy / factor) 
            normalized_cxcywh_all.append(normalized_cxcywh) 

            bi_visual_query = self.visual_query_embedding.weight[:, :].repeat(M, 1) 
            visual_query.append(bi_visual_query) 
        
        C = feat_flatten.shape[-1]
        device = feat_flatten.device
        dtype = feat_flatten.dtype

        Mmax = max(num_ls)
        batch_idx = torch.cat([bi * torch.ones(num_ls[bi], dtype=torch.long) for bi in range(B)]) 
        m_idx = torch.cat([torch.arange(m, dtype=torch.long) for m in num_ls]) 

        visual_prompt = torch.zeros(B, Mmax, C, device=device, dtype=dtype) 
        visual_prompt_mask = torch.ones(B, Mmax, device=device, dtype=torch.bool) 
        reference_points = torch.tensor([0.5, 0.5, 0.99, 0.99], device=device, dtype=dtype).reshape([1, 1, 4]).repeat(B, Mmax, 1) 

        visual_prompt[batch_idx, m_idx] = torch.cat(visual_query, dim=0) 
        visual_prompt_mask[batch_idx, m_idx] = False 
        reference_points[batch_idx, m_idx] = torch.cat(normalized_cxcywh_all, dim=0) 

        visual_prompt = self.visual_prompt_encoding(
                query=visual_prompt, 
                value=feat_flatten, 
                key_padding_mask=mask_flatten, 
                self_key_padding_mask=visual_prompt_mask, 
                reference_points=reference_points, 
                spatial_shapes=spatial_shapes, 
                level_start_index=level_start_index, 
                valid_ratios=valid_ratios,
        ) 

        assert B==1
        label_prompt = torch.cat([batch_data_samples[0].gt_instances.labels.unsqueeze(1), visual_prompt[0]], dim=-1) 
        label_prompt = label_prompt.detach().cpu()
        
        save_name = os.path.join(vis_prompt_save_dir, "{}.pt".format(time.time_ns()))
        torch.save(label_prompt, save_name)

    def only_get_select_score_per_token(self,
                                        batch_inputs: Tensor,
                                        batch_data_samples: SampleList):
        with torch.no_grad():

            text_prompts = [
                data_samples.text for data_samples in batch_data_samples
            ]

            gt_labels = [
                data_samples.gt_instances.labels
                for data_samples in batch_data_samples
            ]

            if 'tokens_positive' in batch_data_samples[0]:
                tokens_positive = [
                    data_samples.tokens_positive
                    for data_samples in batch_data_samples
                ]
                positive_maps = []
                for token_positive, text_prompt, gt_label in zip(
                        tokens_positive, text_prompts, gt_labels):
                    tokenized = self.language_model.tokenizer(
                        [text_prompt],
                        padding='max_length'
                        if self.language_model.pad_to_max else 'longest',
                        return_tensors='pt')
                    new_tokens_positive = [
                        token_positive[label.item()] for label in gt_label
                    ]
                    _, positive_map = self.get_positive_map(
                        tokenized, new_tokens_positive)
                    positive_maps.append(positive_map)
                new_text_prompts = text_prompts
            else:
                new_text_prompts = []
                positive_maps = []
                if len(set(text_prompts)) == 1:
                    # All the text prompts are the same,
                    # so there is no need to calculate them multiple times.
                    tokenized, caption_string, tokens_positive, _ = \
                        self.get_tokens_and_prompts(
                            text_prompts[0], True)
                    new_text_prompts = [caption_string] * len(batch_inputs)
                    for gt_label in gt_labels:
                        new_tokens_positive = [
                            tokens_positive[label] for label in gt_label
                        ]
                        _, positive_map = self.get_positive_map(
                            tokenized, new_tokens_positive)
                        positive_maps.append(positive_map)
                else:
                    for text_prompt, gt_label in zip(text_prompts, gt_labels):
                        tokenized, caption_string, tokens_positive, _ = \
                            self.get_tokens_and_prompts(
                                text_prompt, True)
                        new_tokens_positive = [
                            tokens_positive[label] for label in gt_label
                        ]
                        _, positive_map = self.get_positive_map(
                            tokenized, new_tokens_positive)
                        positive_maps.append(positive_map)
                        new_text_prompts.append(caption_string)

            text_dict = self.language_model(new_text_prompts)
            if self.text_feat_map is not None:
                text_dict['embedded'] = self.text_feat_map(text_dict['embedded'])

            L = text_dict['masks'].shape[-1]
            for i, data_samples in enumerate(batch_data_samples):
                positive_map = positive_maps[i].to(
                    batch_inputs.device).bool().float()
                text_token_mask = text_dict['text_token_mask'][i]
                data_samples.gt_instances.positive_maps = positive_map[:, :L]
                data_samples.gt_instances.text_token_mask = \
                    text_token_mask.unsqueeze(0).repeat(
                        len(positive_map), 1)
            
            if self.use_autocast:
                with autocast(enabled=True):
                    visual_features = self.extract_feat(batch_inputs)
            else:
                visual_features = self.extract_feat(batch_inputs)

            encoder_inputs_dict, _ = self.pre_transformer(visual_features, batch_data_samples)

            visual_dict, rank_oh_labels, _ = self.generate_visual_prompts(batch_data_samples, 
                                                        encoder_inputs_dict['feat'],
                                                        encoder_inputs_dict['feat_mask'],
                                                        encoder_inputs_dict['spatial_shapes'],
                                                        encoder_inputs_dict['level_start_index'],
                                                        encoder_inputs_dict['valid_ratios'],
                                                        ) 
            rank_v_oh_labels = rank_oh_labels 

            B = len(batch_data_samples)
            for bi in range(B):
                M = rank_oh_labels[bi].shape[0]
                batch_data_samples[bi].gt_instances.positive_maps = rank_v_oh_labels[bi].float() 
                batch_data_samples[bi].gt_instances.text_token_mask = visual_dict['text_token_mask'][bi].repeat(M, 1) 
            

            encoder_outputs_dict = self.forward_encoder(**encoder_inputs_dict, text_dict=visual_dict)
            output_memory, _ = self.gen_encoder_output_proposals(encoder_outputs_dict['memory'], encoder_outputs_dict['memory_mask'], encoder_outputs_dict['spatial_shapes'])
            enc_outputs_class = self.bbox_head.cls_branches[self.decoder.num_layers](output_memory, encoder_outputs_dict['memory_text'], encoder_outputs_dict['text_token_mask'])
            select_score_per_token = enc_outputs_class.max(-1)[0] 

            return select_score_per_token.detach() 

    def forward(self,
                inputs: torch.Tensor,
                data_samples: OptSampleList = None,
                mode: str = 'tensor'):

        if mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        elif mode == 'tensor':
            return self._forward(inputs, data_samples)
        elif mode == 'grpo':
            return self.only_get_select_score_per_token(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')





