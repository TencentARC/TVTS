import torch.nn as nn
from base import BaseModel
from utils.util import state_dict_data_parallel_fix
import torch
import OpenCLIP
import json
from model.sort_transformer import SortTransformer
from model.video_encoder_ViT_H_14 import VisionTransformer, CLIPVisionCfg, \
    LayerNormFp32, LayerNorm, QuickGELU
from torch.cuda.amp import autocast, GradScaler


class TVTSv2_H_14(BaseModel):
    def __init__(self,
                 load_checkpoint=None):
        super().__init__()

        clip_model = OpenCLIP.create_model('ViT-H-14',
                                           pretrained='laion2b_s32b_b79k',
                                           cache_dir='OpenCLIP/models')

        with open('OpenCLIP/model_configs/ViT-H-14.json') as f:
            cfg = json.load(f)
        embed_dim = cfg['embed_dim']

        self.text_model = clip_model.transformer
        self.text_token_embedding = clip_model.token_embedding
        self.text_positional_embedding = clip_model.positional_embedding
        self.text_ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.text_attn_mask = clip_model.attn_mask

        vision_cfg = cfg['vision_cfg']
        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        else:
            raise NotImplementedError(f"vision_cfg is not a dict!")

        vision_heads = vision_cfg.width // vision_cfg.head_width
        act_layer = nn.GELU
        norm_layer = LayerNorm

        self.video_model = VisionTransformer(
            image_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            width=vision_cfg.width,
            layers=vision_cfg.layers,
            heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            ls_init_value=vision_cfg.ls_init_value,
            patch_dropout=vision_cfg.patch_dropout,
            input_patchnorm=vision_cfg.input_patchnorm,
            global_average_pool=vision_cfg.global_average_pool,
            attentional_pool=vision_cfg.attentional_pool,
            n_queries=vision_cfg.n_queries,
            attn_pooler_heads=vision_cfg.attn_pooler_heads,
            output_tokens=vision_cfg.output_tokens,
            output_dim=embed_dim,
            act_layer=act_layer,
            norm_layer=norm_layer,
            num_frames=12,
            mask_ratio=0.
        )

        if load_checkpoint in ["", None]:
            state_dict = clip_model.visual.state_dict()
            # convert openai weights for space-time attention
            new_state_dict = {}
            for k, v in state_dict.items():
                if 'in_proj_' in k:
                    k = k.replace('in_proj_', 'qkv.')
                if 'out_proj' in k:
                    k = k.replace('out_proj', 'proj')
                new_state_dict[k] = v
            # need to check if only miss temporal embed and head
            self.video_model.load_state_dict(new_state_dict, strict=False)
            print('ViT initialized with OpenCLIP weights.')

        if load_checkpoint not in ["", None]:
            checkpoint = torch.load(load_checkpoint)
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            self.load_state_dict(new_state_dict, strict=True)
            print('loading checkpoint from {}'.format(load_checkpoint))

    def set_device(self, device):
        self.device = device

    @autocast()
    def forward(self, data, return_embeds=True):
        text_data = data['text']
        video_data = data['video']
        keep_ind = data['keep_ind']
        bz = video_data.shape[0]

        # with torch.no_grad():
        text_before_embeddings, text_embeddings = self.compute_text(text_data)

        # average text embeddings
        # n_trans x B x D
        text_embeddings = text_embeddings.reshape(-1, bz, text_embeddings.shape[-1])
        # text_embeddings = torch.mean(text_embeddings, 0)

        video_before_embeddings, video_embeddings = self.compute_video(video_data, keep_ind)

        # print('text_embeddings.shape', text_embeddings.shape)
        # print('video_embeddings.shape', video_embeddings.shape)
        # print('video_order_embeddings.shape', video_order_embeddings.shape)
        # print('text_order_embeddings.shape', all_embeddings.shape)
        # exit(0)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)

    def compute_text(self, text_data):
        cast_dtype = self.text_model.get_cast_dtype()

        x = self.text_token_embedding(text_data).to(cast_dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.text_positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        attn_mask = self.text_attn_mask.to(x.device)
        x = self.text_model(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.text_ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        text_before_embeddings = x[torch.arange(x.shape[0]), text_data.argmax(dim=-1)] @ self.text_projection
        text_embeddings = text_before_embeddings

        return text_before_embeddings, text_embeddings

    def compute_video(self, video_data, keep_ind):
        video_embeddings, video_before_embeddings = self.video_model(video_data, keep_ind)
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
