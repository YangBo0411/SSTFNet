#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

from torch import nn
import torch
import math
import torch.nn.functional as F
import torch.fft as fft
import pywt
import numpy as np 

from einops import einsum, repeat, rearrange, reduce
from .network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class HPDown(nn.Module):
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, kernels, groups, expand
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)
        # self.cv3 = Conv(c1 // 4, self.c // 2, 3, 2, 1)
        # self.cv4 = Conv(c1 // 4, self.c // 2, 5, 2, 2)
       
        self.cv3 = Conv(c1 // 4, self.c // 2, 5, 2, 2)
        self.cv4 = Conv(c1 // 4, self.c // 2, 7, 2, 3)

    #version 2
    def forward(self, x):
        x1, x2 = x.chunk(2, 1)

        x1 = torch.nn.functional.avg_pool2d(x1, 2, 1, 0, False, True)
        x3, x4 = x1.chunk(2, 1)
        x3 = self.cv4(x3)
        x4 = self.cv3(x4)
        x3_4 = torch.cat((x3, x4), 1)

        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)  #源码 尺寸减半的最大池化操作
        x2 = self.cv2(x2)
        return torch.cat((x3_4, x2), 1)


class MEA(nn.Module):         # MEA 3*5
    def __init__(self, c1, c2, n=1, scale=1, e=0.5):
        super(MEA, self).__init__()

        self.c = int(c2 * e)  # hidden channels
        self.mid = int(self.c * scale)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)

        # self.cv3 = RepConv(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 1)

        self.norm1 = nn.BatchNorm2d(c1)
        # ## 3*5
        # self.conv0 = nn.Conv2d(self.c, self.c, 3, padding=1, groups=self.c)          # 源码5 2
        # self.conv_spatial = nn.Conv2d(self.c, self.c, 5, stride=1, padding=2, groups=self.c, dilation=1) #源码 7 1 9 3
        ## 5*7
        self.conv0 = nn.Conv2d(self.c, self.c, 5, padding=2, groups=self.c)
        self.conv_spatial = nn.Conv2d(self.c, self.c, 7, stride=1, padding=9, groups=self.c, dilation=3)
        #
        self.conv1 = nn.Conv2d(self.c, self.c // 2, 1)
        self.conv2 = nn.Conv2d(self.c, self.c // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(self.c // 2, self.c, 1)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        attn1 = self.conv0(y[-1])
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)  #源码
        m = attn1 * sig[:, 0, :, :].unsqueeze(1)
        # attn = attn1 * sig[:, 0, :, :].unsqueeze(1) * attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        y[-1]= y[-1] * attn
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))
    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y[-1] = self.cv3(y[-1])
        y.extend(m(y[-1]) for m in self.m)
        y.extend(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

class Adaptive_global_filter(nn.Module):
    def __init__(self, ratio=10):
        super().__init__()
        self.ratio = ratio
        # 可学习的基滤波器模板（单通道，动态插值扩展）
        self.base_filter = nn.Parameter(torch.randn(1, 64, 64, 2), requires_grad=True)

    def forward(self, x):
        b, c, h, w = x.shape
        crow, ccol = h // 2, w // 2

        # --- 动态生成频域掩码 --- #
        mask_low = torch.zeros((h, w), device=x.device)
        safe_ratio = min(self.ratio, crow, ccol)  # 防止越界
        mask_low[crow - safe_ratio:crow + safe_ratio, ccol - safe_ratio:ccol + safe_ratio] = 1
        mask_high = 1 - mask_low

        # --- 动态生成滤波器权重 --- #
        filter_real = self.base_filter[..., 0]  # (1, 64, 64)
        filter_imag = self.base_filter[..., 1]
        # 插值到当前输入尺寸 (h, w)
        filter_real = F.interpolate(filter_real.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        filter_imag = F.interpolate(filter_imag.unsqueeze(0), size=(h, w), mode='bilinear', align_corners=False).squeeze(0)
        weight = torch.view_as_complex(torch.stack([filter_real, filter_imag], dim=-1))  # (1, h, w)
        weight = weight.expand(c, -1, -1)  # 扩展至输入通道数 (c, h, w)

        # --- 频域处理 --- #
        x_fre = torch.fft.fftshift(torch.fft.fft2(x, dim=(-2, -1), norm='ortho'))
        x_fre_low = torch.mul(x_fre, mask_low) * weight
        x_fre_high = torch.mul(x_fre, mask_high)
        x_fre_new = x_fre_low + x_fre_high
        x_out = torch.fft.ifft2(torch.fft.ifftshift(x_fre_new, dim=(-2, -1))).real
        return x_out  # (b, c, h, w)


class SpatialAttention(nn.Module):  # Spatial Attention Module   加上LSK结构
    def __init__(self, c1, c2, n=1, scale=1, e=0.5):
        super(SpatialAttention, self).__init__()

        self.c = int(c2 * e)  # hidden channels
        self.mid = int(self.c * scale)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)

        # self.cv3 = RepConv(self.c, self.mid, 3)
        self.m = nn.ModuleList(Conv(self.mid, self.mid, 3) for _ in range(n - 1))
        self.cv4 = Conv(self.mid, self.mid, 1)

        self.norm1 = nn.BatchNorm2d(c1)
        # ## 3*5
        # self.conv0 = nn.Conv2d(self.c, self.c, 3, padding=1, groups=self.c)          # 源码5 2
        # self.conv_spatial = nn.Conv2d(self.c, self.c, 5, stride=1, padding=2, groups=self.c, dilation=1) #源码 7 1 9 3
        ## 5*7
        self.conv0 = nn.Conv2d(self.c, self.c, 5, padding=2, groups=self.c)
        self.conv_spatial = nn.Conv2d(self.c, self.c, 7, stride=1, padding=9, groups=self.c, dilation=3)
        #
        self.conv1 = nn.Conv2d(self.c, self.c // 2, 1)
        self.conv2 = nn.Conv2d(self.c, self.c // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(self.c // 2, self.c, 1)


    def forward(self, x):
        attn1 = self.conv0(x)
        attn2 = self.conv_spatial(attn1)
        attn1 = self.conv1(attn1)
        attn2 = self.conv2(attn2)
        attn = torch.cat([attn1, attn2], dim=1)
        avg_attn = torch.mean(attn, dim=1, keepdim=True)
        max_attn, _ = torch.max(attn, dim=1, keepdim=True)
        agg = torch.cat([avg_attn, max_attn], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        attn = attn1 * sig[:, 0, :, :].unsqueeze(1) + attn2 * sig[:, 1, :, :].unsqueeze(1)  #源码
        m = attn1 * sig[:, 0, :, :].unsqueeze(1)
        # attn = attn1 * sig[:, 0, :, :].unsqueeze(1) * attn2 * sig[:, 1, :, :].unsqueeze(1)
        attn = self.conv(attn)
        x = x * attn
        return x

class FSA(nn.Module):
    def __init__(self, c1, c2, ratio=10):
        super().__init__()
        self.agf = Adaptive_global_filter(ratio=ratio)
        self.sa = SpatialAttention(c1, c2)

    def forward(self, x):
        return self.agf(x) + self.sa(x)  # (b, c, h, w)


class MSSFA(nn.Module):         # MEA 3*5  多阶段空间-频域注意力  Multi-stage spatial-frequency attention 
    def __init__(self, c1, c2, n=1, scale=1, e=0.5):
        super(MSSFA, self).__init__()

        self.c = int(c2 * e)  # hidden channels
        self.mid = int(self.c * scale)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        self.cv4 = Conv(self.mid, self.mid, 1)
        self.fsa = FSA(c1, c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.fsa(y[-1])
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

class NSA2DAdapter(nn.Module):
    """将NSA适配到2D特征图的包装器"""
    def __init__(self, in_channels, img_size=64, e=0.5, **nsa_kwargs):
        super().__init__()
        self.mid = int(in_channels * e)  # hidden channels
        self.img_size = img_size
        self.proj_in = nn.Conv2d(self.mid, self.mid, 1)
        self.nsa = SparseAttention(
            dim = self.mid,
            dim_head = 64,  # 确保最小头维度
            heads = 8,
            sliding_window_size = 8,  # 适合图像局部结构
            compress_block_size = 16,
            selection_block_size = 2,
            num_selected_blocks = 2,
            **nsa_kwargs
        )
        self.proj_out = nn.Conv2d(self.mid, self.mid, 1)

    def forward(self, x):
        """输入输出维度：(B,C,H,W)"""
        b, c, h, w = x.shape
        
        # 空间序列化
        x = self.proj_in(x)
        x_seq = rearrange(x, 'b c h w -> b (h w) c')  # (B, L, C)
        
        # NSA处理
        attn_out = self.nsa(x_seq)
        
        # 恢复空间维度
        attn_out = rearrange(attn_out, 'b (h w) c -> b c h w', h=h, w=w)
        return self.proj_out(attn_out) + x  # 残差连接

class FSA_NSA(nn.Module):
    """NSA增强的频空联合注意力"""
    def __init__(self, c1, c2, ratio=10, use_nsa=True):
        super().__init__()
        self.agf = Adaptive_global_filter(ratio=ratio)
        self.sa = SpatialAttention(c1, c2)
        self.nsa = NSA2DAdapter(c1) if use_nsa else None
        self.channel_fusion = nn.Conv2d(
            c1 * (2 + int(use_nsa)), 
            c1, 
            kernel_size=1
        )

    def forward(self, x):
        sa = self.sa(x)
        agf = self.agf(x)
        nsa_sa = self.nsa(sa)
        nsa_agf = self.nsa(agf)
        return nsa_sa + nsa_agf

        # """动态融合三种注意力路径"""
        # components = [self.agf(x), self.sa(x)]
        # if self.nsa is not None:
        #     components.append(self.nsa(x))
        # fused = torch.cat(components, dim=1)
        # return x * torch.sigmoid(self.channel_fusion(fused))
    

class MEA_FSA_LSK_NSA(nn.Module):         # MEA 3*5
    def __init__(self, c1, c2, n=1, scale=1, e=0.5):
        super(MEA_FSA_LSK_NSA, self).__init__()

        self.c = int(c2 * e)  # hidden channels
        self.mid = int(self.c * scale)

        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(self.c + self.mid * (n + 1), c2, 1)
        self.cv4 = Conv(self.mid, self.mid, 1)
        self.fsa = FSA_NSA(c1, c2)

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y[-1] = self.fsa(y[-1])
        y.append(self.cv4(y[-1]))
        return self.cv2(torch.cat(y, 1))

class Darknet(nn.Module):
    # number of blocks from dark2 to dark5.
    depth2blocks = {21: [1, 2, 2, 1], 53: [2, 8, 8, 4]}

    def __init__(
        self,
        depth,
        in_channels=3,
        stem_out_channels=32,
        out_features=("dark3", "dark4", "dark5"),
    ):
        """
        Args:
            depth (int): depth of darknet used in model, usually use [21, 53] for this param.
            in_channels (int): number of input channels, for example, use 3 for RGB image.
            stem_out_channels (int): number of output channels of darknet stem.
                It decides channels of darknet layer2 to layer5.
            out_features (Tuple[str]): desired output layer name.
        """
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        self.stem = nn.Sequential(
            BaseConv(in_channels, stem_out_channels, ksize=3, stride=1, act="lrelu"),
            *self.make_group_layer(stem_out_channels, num_blocks=1, stride=2),
        )
        in_channels = stem_out_channels * 2  # 64

        num_blocks = Darknet.depth2blocks[depth]
        # create darknet with `stem_out_channels` and `num_blocks` layers.
        # to make model structure more clear, we don't use `for` statement in python.
        self.dark2 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[0], stride=2)
        )
        in_channels *= 2  # 128
        self.dark3 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[1], stride=2)
        )
        in_channels *= 2  # 256
        self.dark4 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[2], stride=2)
        )
        in_channels *= 2  # 512

        self.dark5 = nn.Sequential(
            *self.make_group_layer(in_channels, num_blocks[3], stride=2),
            *self.make_spp_block([in_channels, in_channels * 2], in_channels * 2),
        )

    def make_group_layer(self, in_channels: int, num_blocks: int, stride: int = 1):
        "starts with conv layer then has `num_blocks` `ResLayer`"
        return [
            BaseConv(in_channels, in_channels * 2, ksize=3, stride=stride, act="lrelu"),
            *[(ResLayer(in_channels * 2)) for _ in range(num_blocks)],
        ]

    def make_spp_block(self, filters_list, in_filters):
        m = nn.Sequential(
            *[
                BaseConv(in_filters, filters_list[0], 1, stride=1, act="lrelu"),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                SPPBottleneck(
                    in_channels=filters_list[1],
                    out_channels=filters_list[0],
                    activation="lrelu",
                ),
                BaseConv(filters_list[0], filters_list[1], 3, stride=1, act="lrelu"),
                BaseConv(filters_list[1], filters_list[0], 1, stride=1, act="lrelu"),
            ]
        )
        return m

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}



class SFnet(nn.Module):     # 空间-频率
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),    # 源码 
            # HPDown(base_channels, base_channels * 2),               # yb
            MSSFA(base_channels * 2, base_channels * 2),                # yb
            # CSPLayer(                                               # 源码
            #     base_channels * 2,
            #     base_channels * 2,
            #     n=base_depth,
            #     depthwise=depthwise,
            #     act=act,
            # ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),     # 源码 
            # HPDown(base_channels * 2, base_channels * 4),                   # yb
            MSSFA(base_channels * 4, base_channels * 4),                # yb
            # CSPLayer(                                                   # 源码
            #     base_channels * 4,
            #     base_channels * 4,
            #     n=base_depth * 3,
            #     depthwise=depthwise,
            #     act=act,
            # ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),   # 源码 
            # HPDown(base_channels * 4, base_channels * 8),                   # yb
            MSSFA(base_channels * 8, base_channels * 8),                # yb
            # CSPLayer(                                                   # 源码
            #     base_channels * 8,
            #     base_channels * 8,
            #     n=base_depth * 3,
            #     depthwise=depthwise,
            #     act=act,
            # ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),   # 源码 
            # HPDown(base_channels * 8, base_channels * 16),                   # yb
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            MSSFA(base_channels * 16, base_channels * 16),                # yb
            # CSPLayer(                                                   # 源码
            #     base_channels * 16,
            #     base_channels * 16,
            #     n=base_depth,
            #     shortcut=False,
            #     depthwise=depthwise,
            #     act=act,
            # ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}

class HaarWavelet(nn.Module):
    def __init__(self, in_channels, grad=False):
        super(HaarWavelet, self).__init__()
        self.in_channels = in_channels

        self.haar_weights = torch.ones(4, 1, 2, 2)
        #h
        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1
        #v
        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1
        #d
        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.in_channels, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = grad

    def forward(self, x, rev=False):
        if not rev:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.in_channels) / 4.0
            out = out.reshape([x.shape[0], self.in_channels, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            out = x.reshape([x.shape[0], 4, self.in_channels, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.in_channels * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.in_channels)

class WFU(nn.Module):
    def __init__(self, c1, c2):
        super(WFU, self).__init__()
        dim_big, dim_small = c1, c2
        self.dim = dim_big
        self.HaarWavelet = HaarWavelet(dim_big, grad=False)
        self.InverseHaarWavelet = HaarWavelet(dim_big, grad=False)
        self.RB = nn.Sequential(
            # nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
            # nn.ReLU(),
            Conv(dim_big, dim_big, 3),
            nn.Conv2d(dim_big, dim_big, kernel_size=3, padding=1),
        )

        self.channel_tranformation = nn.Sequential(
            # nn.Conv2d(dim_big+dim_small, dim_big+dim_small // 1, kernel_size=1, padding=0),
            # nn.ReLU(),
            Conv(dim_big+dim_small, dim_big+dim_small // 1, 1),
            nn.Conv2d(dim_big+dim_small // 1, dim_big*3, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x_big, x_small = x
        haar = self.HaarWavelet(x_big, rev=False)
        a = haar.narrow(1, 0, self.dim)
        h = haar.narrow(1, self.dim, self.dim)
        v = haar.narrow(1, self.dim*2, self.dim) 
        d = haar.narrow(1, self.dim*3, self.dim)

        hvd = self.RB(h + v + d)
        a_ = self.channel_tranformation(torch.cat([x_small, a], dim=1))
        out = self.InverseHaarWavelet(torch.cat([hvd, a_], dim=1), rev=True)
        return out

class CSPDarknetP6(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5", "dark6"),
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )
        # dark6
        self.dark6 = nn.Sequential(
            Conv(base_channels * 16, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
            CSPLayer(
                base_channels * 16,
                base_channels * 16,
                n=base_depth,
                shortcut=False,
                depthwise=depthwise,
                act=act,
            ),
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        x = self.dark6(x)
        outputs["dark6"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


