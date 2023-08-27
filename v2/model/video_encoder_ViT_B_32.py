from collections import OrderedDict
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat


def attn(q, k, v):
    sim = einsum('b i d, b j d -> b i j', q, k)
    attn = sim.softmax(dim=-1)
    out = einsum('b i j, b j d -> b i d', attn, v)
    return out


class VarAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 initialize='random'):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if initialize == 'zeros':
            self.qkv.weight.data.fill_(0)
            self.qkv.bias.data.fill_(0)
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            self.proj.weight.data.fill_(1)
            self.proj.bias.data.fill_(0)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, einops_from, einops_to, **einops_dims):
        h = self.num_heads
        # project x to q, k, v vaalues
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        # q *= self.scale
        q = q * self.scale

        # splice out CLS token at index 1
        (cls_q, q_), (cls_k, k_), (cls_v, v_) = map(lambda t: (t[:, 0:1], t[:, 1:]), (q, k, v))

        # let CLS token attend to key / values of all patches across time and space
        cls_out = attn(cls_q, k, v)
        # rearrange across time or space
        q_, k_, v_ = map(lambda t: rearrange(t, f'{einops_from} -> {einops_to}', **einops_dims), (q_, k_, v_))

        # expand cls token keys and values across time or space and concat
        r = q_.shape[0] // cls_k.shape[0]
        cls_k, cls_v = map(lambda t: repeat(t, 'b n d -> (b r) n d', r=r), (cls_k, cls_v))

        k_ = torch.cat((cls_k, k_), dim=1)
        v_ = torch.cat((cls_v, v_), dim=1)

        # attention
        out = attn(q_, k_, v_)

        # merge back time or space
        out = rearrange(out, f'{einops_to} -> {einops_from}', **einops_dims)

        # concat back the cls token
        out = torch.cat((cls_out, out), dim=1)

        # merge back the heads
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        ## to out
        x = self.proj(out)
        x = self.proj_drop(x)
        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


# for space-time attention
class ResidualSpaceTimeAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        # new params
        self.attn = VarAttention(d_model, num_heads=n_head, qkv_bias=True)
        self.timeattn = VarAttention(d_model, num_heads=n_head, qkv_bias=True, initialize='zeros')
        self.ln_3 = LayerNorm(d_model)

        # old params
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def forward(self, x: torch.Tensor,
                einops_from_space, einops_to_space,
                einops_from_time, einops_to_time,
                time_n, space_f):
        time_output = self.timeattn(self.ln_3(x), einops_from_time, einops_to_time, n=time_n)
        time_residual = x + time_output

        space_output = self.attn(self.ln_1(time_residual), einops_from_space, einops_to_space, f=space_f)
        space_residual = x + space_output

        x = space_residual + self.mlp(self.ln_2(space_residual))
        return x


class SpaceTimeTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualSpaceTimeAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor,
                einops_from_space, einops_to_space,
                einops_from_time, einops_to_time,
                time_n, space_f):
        for blk in self.resblocks:
            x = blk(x,
                    einops_from_space, einops_to_space,
                    einops_from_time, einops_to_time,
                    time_n, space_f)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int,
                 num_frames: int = 12, mask_ratio: float = 0.):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = SpaceTimeTransformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

        # video inference
        self.patches_per_frame = (input_resolution // patch_size) ** 2
        self.temporal_embedding = nn.Parameter(scale * torch.randn(num_frames, width))
        self.einops_from_space = 'b (f n) d'
        self.einops_to_space = '(b f) n d'
        self.einops_from_time = 'b (f n) d'
        self.einops_to_time = '(b n) f d'

        # mask params
        self.mask_ratio = mask_ratio

    def forward(self, x: torch.Tensor, keep_ind: torch.Tensor):
        if len(x.shape) == 4:  # img input
            x = x.unsqueeze(1)
        B, T, C, H, W = x.shape
        x = x.view(-1, C, H, W)  # shape = [B * T, C, H, W]
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = x.reshape(B, T * x.shape[1], x.shape[2])  # shape = [B, T * grid ** 2, width]
        class_embedding = self.class_embedding.reshape(1, 1, -1).repeat(x.shape[0], 1, 1)  # shape = [*, 1, width]
        x = torch.cat([class_embedding, x], dim=1)  # shape = [*, T * grid ** 2 + 1, width]

        # [CLS] token embedding: [1, width]
        class_positional_embedding = self.positional_embedding[0:1]
        # spatial position embedding: [grid ** 2, width] => [T * grid ** 2, width]
        tile_positional_embedding = self.positional_embedding[1:].repeat(T, 1)
        # temporal position embedding: [n_cubes, width] => [T * grid ** 2, width]
        tile_temporal_embedding = self.temporal_embedding[:T].repeat_interleave(self.patches_per_frame, dim=0)
        # overall embedding: [T * grid ** 2 + 1, width]
        spatiotemporal_embedding = tile_positional_embedding + tile_temporal_embedding
        spatiotemporal_embedding = torch.cat([class_positional_embedding, spatiotemporal_embedding], dim=0)
        # add spatiotemporal embedding
        x = x + spatiotemporal_embedding

        # =====start masking=====
        keep_ind = keep_ind.unsqueeze(1).repeat(1, T, 1)

        x_cls = x[:, 0:1]
        x_patch = x[:, 1:]
        # B x T * grid ** 2 x D => B * T x grid ** 2 x D
        x_patch = x_patch.reshape(B * T, -1, x.shape[-1])

        # B x T x n_keep => B * T x n_keep
        keep_ind = keep_ind.reshape(-1, keep_ind.shape[-1])
        x_patch = x_patch[np.arange(x_patch.shape[0]).reshape(-1, 1), keep_ind]

        # B * T x n_keep x D => B x T * n_keep x D
        x_patch = x_patch.reshape(B, -1, x_patch.shape[-1])

        x = torch.cat((x_cls, x_patch), 1)
        # =====end masking=====

        x = self.ln_pre(x)

        time_n, space_f = int(self.patches_per_frame * (1 - self.mask_ratio)), T

        # print('time_n,space_f', time_n, space_f)

        x = self.transformer(x,
                             self.einops_from_space, self.einops_to_space,
                             self.einops_from_time, self.einops_to_time,
                             time_n, space_f)

        # [B, T * n_keep ** 2 + 1, D]
        x = self.ln_post(x)

        if self.proj is not None:
            x = x @ self.proj

        return x
