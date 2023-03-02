import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from functools import partial
from utils.util import state_dict_data_parallel_fix
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification, BertForSequenceClassification, \
    BertTokenizer, T5EncoderModel
import argparse
import torch
import timm
import os
import numpy as np
from model.SortFormer import SortTransformer
from model.video_encoder import VisionTransformer
import math


class TVTS(BaseModel):
    def __init__(self,
                 args,
                 video_params,
                 text_params,
                 projection_dim=256,
                 load_checkpoint=None,
                 projection='minimal'):
        super().__init__()
        self.args = args
        self.video_params = video_params
        self.text_params = text_params
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()

        arch_config = video_params.get('arch_config', 'base_patch16_224')
        num_frames = video_params.get('num_frames', 16)
        if arch_config == 'base_patch16_224':
            model = VisionTransformer(
                patch_size=16, embed_dim=768, depth=12,
                num_heads=12, mlp_ratio=4, qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                num_frames=num_frames)
        else:
            raise NotImplementedError

        model.pre_logits = nn.Identity()
        ftr_dim = model.embed_dim
        if load_checkpoint in ["", None]:
            # IN-1K
            checkpoint = torch.load('./mae_pretrain_vit_base.pth',
                                    map_location='cuda:{}'.format(self.args.local_rank))
            print('ViT initialized with MAE IN-1K weights.')

            state_dict = checkpoint['model']
            for param in state_dict:
                if param == 'patch_embed.proj.weight':
                    value = state_dict[param].unsqueeze(2).repeat(1, 1, 2, 1, 1)
                    state_dict[param] = value
            # set strict=True to check carefully!
            model.load_state_dict(state_dict, strict=False)

        self.video_model = model

        if projection == 'minimal':
            txt_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, projection_dim),
            )

            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim)
            )
        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        self.n_trans = 4
        self.pred_model = SortTransformer(num_classes=self.n_trans)

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint, map_location='cuda:{}'.format(self.args.local_rank))
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            self.load_state_dict(new_state_dict, strict=True)
            print('loading checkpoint from {}'.format(load_checkpoint))

    def set_device(self, device):
        self.device = device

    def forward(self, data, return_embeds=True):

        text_data = data['text']
        video_data = data['video']
        keep_ind = data['keep_ind']
        bz = video_data.shape[0]

        text_before_embeddings, text_embeddings = self.compute_text(text_data)
        text_before_embeddings = text_before_embeddings.reshape(-1, bz, text_before_embeddings.shape[-1]).detach()
        all_embeddings = text_before_embeddings.permute(1, 0, 2)

        # average text embeddings
        # n_trans x B x D
        text_embeddings = text_embeddings.reshape(-1, bz, text_embeddings.shape[-1])
        n_trans = text_embeddings.shape[0]
        text_embeddings = torch.mean(text_embeddings, 0)

        video_before_embeddings, video_embeddings = self.compute_video(video_data, keep_ind)
        video_order_embeddings = video_before_embeddings

        if n_trans != 1:
            predict_order = self.pred_model(all_embeddings, video_order_embeddings)
        else:
            predict_order = None

        if return_embeds:
            return text_embeddings, video_embeddings, predict_order

        return sim_matrix(text_embeddings, video_embeddings)

    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_before_embeddings = self.text_model(
                text_data['input_ids'], attention_mask=text_data['attention_mask'])['pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_before_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            text_before_embeddings = self.text_model(**text_data, output_hidden_states=True,
                                                     return_dict=True).pooler_output
        text_embeddings = self.txt_proj(text_before_embeddings)
        return text_before_embeddings, text_embeddings

    def compute_video(self, video_data, keep_ind):
        video_before_embeddings = self.video_model(video_data, keep_ind)
        # [CLS] token for contrastive loss
        video_embeddings = self.vid_proj(video_before_embeddings[:, 0, :])
        return video_before_embeddings, video_embeddings


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt


if __name__ == "__main__":
    pass
