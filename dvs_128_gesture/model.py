import torch
from torch import einsum
import torch.nn as nn
from spikingjelly.clock_driven.neuron import MultiStepLIFNode, LIFNode, BaseNode
from spikingjelly.clock_driven.layer import MultiStepDropout
from spikingjelly.clock_driven import surrogate
from typing import Callable, overload
from timm.models.layers import to_2tuple, trunc_normal_, DropPath
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import torch.nn.functional as F
import math
from functools import partial
__all__ = ['spikformer']

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def printSpikeInfo(layertensor, name, isprint=False):
    if isprint:
        non_zero_elm = torch.count_nonzero(layertensor).item()
        spike_num = layertensor.sum().item()
        elem_num = layertensor.numel()
        spike_rate = spike_num * 1.0 / elem_num
        sparse_rate = 1 - non_zero_elm * 1.0 / elem_num
        print('%s shape:%s, elem sum: %d, elem num: %d, non zero elm: %d, fire rate: %.5f, sparse rate: %.5f, is fire:%d' % (name, layertensor.shape, spike_num, elem_num, non_zero_elm, spike_rate, sparse_rate, non_zero_elm==spike_num))
    return

class Dynamic_Threshold_LIFNode(MultiStepLIFNode):
    def __init__(self, tau: float = 2., decay_input: bool = True, v_threshold: float = 1.,
                 v_reset: float = 0., surrogate_function: Callable = surrogate.Sigmoid(),
                 detach_reset: bool = False, backend='torch'):
        super().__init__(tau, decay_input, v_threshold, v_reset, surrogate_function, detach_reset, backend)
        init_dynamic_threshold = v_threshold
        self.dynamic_threshold = nn.Parameter(torch.as_tensor(init_dynamic_threshold))

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.dynamic_threshold)

class Multi_Threshold_Acc(nn.Module):
    def __init__(self, times=1):
        super().__init__()
        self.times = times
        self.basic_threshold = 0.5
        self.range = 1.0 / self.times

        self.fire_block = nn.ModuleList()
        for j in range(self.times):
            self.fire_block.append(Dynamic_Threshold_LIFNode(tau=2.0, v_threshold = self.basic_threshold + j * self.range, detach_reset=True, backend='cupy'))

    def forward(self, x):
        x_fire = torch.zeros_like(x)

        for i in range(self.times):
            x_fire = x_fire + self.fire_block[i](x)

        x = x_fire

        return x

class Spiking_AFT_Simple(nn.Module):
    def __init__(self, dim, hidden_dim=64):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        self.to_q = nn.Linear(dim, hidden_dim)
        self.to_q_bn = nn.BatchNorm1d(hidden_dim)
        self.to_q_fire_block = Multi_Threshold_Acc()

        self.to_k = nn.Linear(dim, hidden_dim)
        self.to_k_bn = nn.BatchNorm1d(hidden_dim)
        self.to_k_fire_block = Multi_Threshold_Acc()

        self.to_v = nn.Linear(dim, hidden_dim)

        self.project = nn.Linear(hidden_dim, dim)
        self.project_bn = nn.BatchNorm1d(dim)
        self.to_project_fire_block = Multi_Threshold_Acc()

    def forward(self, x):
        T,B,N,C = x.shape

        printSpikeInfo(x, 'x in')

        x = x.flatten(0, 1)

        Q = self.to_q(x)
        K = self.to_k(x)
        V = self.to_v(x)

        K = self.to_k_bn(K.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.hidden_dim).contiguous()
        fireK = self.to_k_fire_block(K).flatten(0, 1)

        printSpikeInfo(fireK, 'fireK')

        weights = torch.mul(fireK, V).sum(dim=1, keepdim=True)

        Q = self.to_q_bn(Q.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.hidden_dim).contiguous()
        Q_sig = self.to_q_fire_block(Q).flatten(0, 1)
        printSpikeInfo(Q_sig, 'Q_sig')

        Yt = torch.mul(Q_sig, weights)

        Yt = self.project(Yt)
        Yt = self.project_bn(Yt.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.dim).contiguous()
        Yt = self.to_project_fire_block(Yt)

        printSpikeInfo(Yt, 'out')

        return Yt

class Spiking_GFNN(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        hidden_expend_dim = int(2 * hidden_dim)
        self.hidden_dim = hidden_dim
        self.dim = dim

        self.to_hidden_linear = nn.Linear(dim, hidden_expend_dim)

        self.to_hidden_v_bn = nn.BatchNorm1d(hidden_dim)
        self.to_hidden_v_lif = MultiStepLIFNode(tau=2.0, v_threshold = 1.0, detach_reset=True, backend='cupy')

        self.to_out = nn.Linear(hidden_dim, dim)
        self.to_bn = nn.BatchNorm1d(dim)
        self.to_lif = MultiStepLIFNode(tau=2.0, v_threshold = 1.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T,B,N,C = x.shape

        printSpikeInfo(x, 'gate: x in')

        x_for_input = x.flatten(0, 1)

        hidden_layer = self.to_hidden_linear(x_for_input)

        v, gate = hidden_layer.chunk(2, dim=-1)

        v = self.to_hidden_v_bn(v.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.hidden_dim).contiguous()
        v = self.to_hidden_v_lif(v).flatten(0, 1)
        printSpikeInfo(v, 'gate: v')

        out = torch.mul(gate, v)

        out = self.to_out(out)
        out = self.to_bn(out.transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, self.dim).contiguous()
        out = self.to_lif(out)

        printSpikeInfo(out, 'gate: out')

        return out

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1, block_num=0, depths=4):
        super().__init__()
        self.attn = Spiking_AFT_Simple(dim=dim)
        gate_hidden_dim = int(dim * mlp_ratio)
        self.sgfnn = Spiking_GFNN(dim=dim, hidden_dim=gate_hidden_dim)

    def forward(self, x):
        T,B,N,C = x.shape
        
        x_attn = self.attn(x)
        x = x + x_attn
        x = x + self.sgfnn(x)
        return x

class SPS(nn.Module):
    def __init__(self, img_size_h=128, img_size_w=128, patch_size=4, in_channels=2, embed_dims=256):
        super().__init__()
        self.image_size = [img_size_h, img_size_w]
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.C = in_channels
        self.H, self.W = self.image_size[0] // patch_size[0], self.image_size[1] // patch_size[1]
        self.num_patches = self.H * self.W

        self.proj_conv = nn.Conv2d(in_channels, embed_dims//8, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dims//8)
        self.proj_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.proj_conv1 = nn.Conv2d(embed_dims//8, embed_dims//4, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn1 = nn.BatchNorm2d(embed_dims//4)
        self.proj_lif1 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.proj_conv2 = nn.Conv2d(embed_dims//4, embed_dims//2, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn2 = nn.BatchNorm2d(embed_dims//2)
        self.proj_lif2 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.proj_conv3 = nn.Conv2d(embed_dims//2, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.proj_bn3 = nn.BatchNorm2d(embed_dims)
        self.proj_lif3 = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')
        self.avgpool3 = torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False)

        self.rpe_conv = nn.Conv2d(embed_dims, embed_dims, kernel_size=3, stride=1, padding=1, bias=False)
        self.rpe_bn = nn.BatchNorm2d(embed_dims)
        self.rpe_lif = MultiStepLIFNode(tau=2.0, detach_reset=True, backend='cupy')

    def forward(self, x):
        T, B, C, H, W = x.shape

        x = self.proj_conv(x.flatten(0, 1))
        x = self.proj_bn(x).reshape(T, B, -1, H, W).contiguous()
        x = self.proj_lif(x).flatten(0, 1).contiguous()
        x = self.avgpool(x)

        x = self.proj_conv1(x)
        x = self.proj_bn1(x).reshape(T, B, -1, H//2, W//2).contiguous()
        x = self.proj_lif1(x).flatten(0, 1).contiguous()
        x = self.avgpool1(x)

        x = self.proj_conv2(x)
        x = self.proj_bn2(x).reshape(T, B, -1, H//4, W//4).contiguous()
        x = self.proj_lif2(x).flatten(0, 1).contiguous()
        x = self.avgpool2(x)

        x = self.proj_conv3(x)
        x = self.proj_bn3(x).reshape(T, B, -1, H//8, W//8).contiguous()
        x = self.proj_lif3(x).flatten(0, 1).contiguous()
        x = self.avgpool3(x)

        x_feat = x.reshape(T, B, -1, H//16, W//16).contiguous()
        x = self.rpe_conv(x)
        x = self.rpe_bn(x).reshape(T, B, -1, H//16, W//16).contiguous()
        x_1 = self.rpe_lif(x)
        x = x_1 + x_feat
        
        x = x.flatten(-2).transpose(-1, -2)  # T,B,N,C
        
        x_time_acc = x[0:1, :, :, :]
        for i in range(1, T):
            x_time_acc = x_time_acc + x[i:i+1, :, :, :]
        x = x_time_acc

        return x


class Spikformer(nn.Module):
    def __init__(self,
                 img_size_h=128, img_size_w=128, patch_size=16, in_channels=2, num_classes=11,
                 embed_dims=[64, 128, 256], num_heads=[1, 2, 4], mlp_ratios=[4, 4, 4], qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[6, 8, 6], sr_ratios=[8, 4, 2]
                 ):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depths)]  # stochastic depth decay rule

        patch_embed = SPS(img_size_h=img_size_h,
                                 img_size_w=img_size_w,
                                 patch_size=patch_size,
                                 in_channels=in_channels,
                                 embed_dims=embed_dims)
        num_patches = patch_embed.num_patches
        pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims))
        block = nn.ModuleList([Block(
            dim=embed_dims, num_heads=num_heads, mlp_ratio=mlp_ratios, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[j],
            norm_layer=norm_layer, sr_ratio=sr_ratios)
            for j in range(depths)])

        setattr(self, f"patch_embed", patch_embed)
        setattr(self, f"pos_embed", pos_embed)
        setattr(self, f"block", block)

        self.head = nn.Linear(embed_dims, num_classes) if num_classes > 0 else nn.Identity()

        pos_embed = getattr(self, f"pos_embed")
        trunc_normal_(pos_embed, std=.02)
        self.apply(self._init_weights)

    @torch.jit.ignore
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        block = getattr(self, f"block")
        patch_embed = getattr(self, f"patch_embed")
        x = patch_embed(x)
        for blk in block:
            x = blk(x)
        return x.mean(2)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)  # [T, N, 2, *, *]
        x = self.forward_features(x)
        x = self.head(x.mean(0))
        return x


@register_model
def spikformer(pretrained=False, **kwargs):
    model = Spikformer(
        patch_size=16, embed_dims=256, num_heads=16, mlp_ratios=4,
        in_channels=2, num_classes=11, qkv_bias=False,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=1, sr_ratios=1,
        **kwargs
    )
    model.default_cfg = _cfg()
    return model