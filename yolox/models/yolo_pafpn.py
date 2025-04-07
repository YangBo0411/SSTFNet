#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import SFnet, WFU, HaarWavelet
from .network_blocks import BaseConv, CSPLayer, DWConv
import torch.nn.functional as F
import numpy as np 
import math

def autopad(k, p=None, d=1): 
   
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k] 
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p

class Conv(nn.Module):
   
    default_act = nn.SiLU() 

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))



class EnhancedSpatialAttention(nn.Module):
    def __init__(self, deformable=False):
        super().__init__()
        self.deformable = deformable
        
        if deformable: 
            self.offset_conv = nn.Conv2d(2, 18, 3, padding=1) 
           
        else:          
            self.conv3 = nn.Conv2d(2, 1, 3, padding=1, padding_mode='reflect')
            self.conv5 = nn.Conv2d(2, 1, 5, padding=2, padding_mode='reflect')
            self.conv7 = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect')
            self.fusion = nn.Conv2d(3, 1, 1)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max = torch.amax(x, dim=1, keepdim=True)
        x_cat = torch.cat([x_avg, x_max], dim=1)
        
        if self.deformable:
            offset = self.offset_conv(x_cat)
           
        else:
            scale3 = self.conv3(x_cat)
            scale5 = self.conv5(x_cat)
            scale7 = self.conv7(x_cat)
            att = self.fusion(torch.cat([scale3, scale5, scale7], dim=1))
        
        return self.sigmoid(att)

class EnhancedChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8, use_std=False):
        super().__init__()
        self.use_std = use_std
        self.reduction = reduction
        
       
        self.max_path = nn.Sequential(
            nn.Conv2d(dim, dim//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(dim//reduction, dim, 1, bias=False)
        )
        self.avg_path = nn.Sequential(
            nn.Conv2d(dim, dim//reduction, 1, bias=False),
            nn.LayerNorm([dim//reduction, 1, 1]),
            nn.ReLU(),
            nn.Conv2d(dim//reduction, dim, 1, bias=False)
        )
        
        if use_std: 
            self.std_path = nn.Sequential(
                nn.Conv2d(dim, dim//reduction, 1, bias=False),
                nn.ReLU(),
                nn.Conv2d(dim//reduction, dim, 1, bias=False)
            )
        
        self.sigmoid = nn.Sigmoid()
        self.gamma = nn.Parameter(torch.zeros(1)) 

    def forward(self, x):
        max_pool = torch.amax(x, dim=(2,3), keepdim=True)
        avg_pool = torch.mean(x, dim=(2,3), keepdim=True)
        
        max_att = self.max_path(max_pool)
        avg_att = self.avg_path(avg_pool)
        
        if self.use_std:
            std = torch.std(x, dim=(2,3), keepdim=True)
            std_att = self.std_path(std)
            combined = (max_att + avg_att + std_att) / 3
        else:
            combined = (max_att + avg_att) / 2
        
        return self.sigmoid(self.gamma * combined)
    
class EnhancedHFF(nn.Module):
    def __init__(self, dim, reduction=8, deformable=False, use_std=False):
        super().__init__()
        self.sa = SpatialAttention(deformable=deformable)
        self.ca = ChannelAttention(dim, reduction, use_std=use_std)
        self.pa = PA(dim)
        
       
        self.fusion = nn.Conv2d(dim, dim, 1, padding_mode='reflect')
        nn.init.kaiming_normal_(self.fusion.weight, mode='fan_out', nonlinearity='leaky_relu')
        
       
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.beta = nn.Parameter(torch.zeros(1)) 

    def forward(self, data):
        x, y = data
        base = x + y 
        
       
        s_att = self.sa(base)
        c_att = self.ca(base)
        
       
        hybrid_att = c_att * s_att 
        pa_feature = self.pa(base, hybrid_att)
        
       
        enhanced = base + self.alpha * pa_feature
        out = self.fusion(enhanced) + self.beta * base
        return out

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.concat([x_avg, x_max], dim=1)
        output = self.sa(x2)
        output = self.sigmoid(output)
        return output

class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x): 
        max = self.maxpool(x)
        avg = self.avgpool(x)
        max_ca = self.ca(max)
        avg_ca = self.ca(avg)
        output = self.sigmoid(max_ca + avg_ca)
        return output

class PA(nn.Module):
    '''PA is pixel attention'''
    def __init__(self, nf):

        super(PA, self).__init__()
        self.conv = nn.Conv2d(nf, nf, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, att):
        x = self.conv(x)
       
        y = att
        out = torch.mul(x, y)
        return out


class CPCA_ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x

class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels,channels,kernel_size=5,padding=2,groups=channels)
        self.dconv1_7 = nn.Conv2d(channels,channels,kernel_size=(1,7),padding=(0,3),groups=channels)
        self.dconv7_1 = nn.Conv2d(channels,channels,kernel_size=(7,1),padding=(3,0),groups=channels)
        self.dconv1_11 = nn.Conv2d(channels,channels,kernel_size=(1,11),padding=(0,5),groups=channels)
        self.dconv11_1 = nn.Conv2d(channels,channels,kernel_size=(11,1),padding=(5,0),groups=channels)
        self.dconv1_21 = nn.Conv2d(channels,channels,kernel_size=(1,21),padding=(0,10),groups=channels)
        self.dconv21_1 = nn.Conv2d(channels,channels,kernel_size=(21,1),padding=(10,0),groups=channels)
        self.conv = nn.Conv2d(channels,channels,kernel_size=(1,1),padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
       
        inputs = self.conv(inputs)
        inputs = self.act(inputs)
        
        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out

class CAA(nn.Module):
    def __init__(self, ch, h_kernel_size = 11, v_kernel_size = 11) -> None:
        super().__init__()
        
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = Conv(ch, ch)
        self.h_conv = nn.Conv2d(ch, ch, (1, h_kernel_size), 1, (0, h_kernel_size // 2), 1, ch)
        self.v_conv = nn.Conv2d(ch, ch, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), 1, ch)
        self.conv2 = Conv(ch, ch)
        self.act = nn.Sigmoid()
    
    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x 

        n, c, h, w = x.size()
        x_h = self.pool_h(x)  
        x_w = self.pool_w(x).permute(0, 1, 3, 2) 

        y = torch.cat([x_h, x_w], dim=2) 
        y = self.conv1(y)      
        y = self.bn1(y)        
        y = self.act(y)        

        x_h, x_w = torch.split(y, [h, w], dim=2) 
        x_w = x_w.permute(0, 1, 3, 2) 

        a_h = self.conv_h(x_h).sigmoid() 
        a_w = self.conv_w(x_w).sigmoid() 

        out = identity * a_w * a_h 

        return out


class HFF(nn.Module):             
    def __init__(self, dim, reduction=8):
        super(HFF, self).__init__()
       
       
        
       
        self.sa = CoordAtt(dim)
        self.ca = ChannelAttention(dim, reduction)
       
        self.caa = CAA(dim)
       
       

        self.pa = PA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.beta = nn.Parameter(torch.zeros(1)) 

    def forward(self, data):
        x, y = data
        raw = x + y

       
        cattn = self.ca(raw)
        sattn = self.sa(raw)
        mix_att = sattn + cattn
        new = self.sigmoid(self.pa(raw, mix_att))
        out = raw + new * x + (1 - new) * y
        out = self.conv(out)
       

       
       
       
       

        return out
    
class CSCF(nn.Module):             
    def __init__(self, dim, reduction=8):
        super(CSCF, self).__init__()
        self.sa = CoordAtt(dim)
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()
        self.alpha = nn.Parameter(torch.ones(1)) 
        self.beta = nn.Parameter(torch.zeros(1)) 

    def forward(self, data): 
        x, y = data 
        raw = x + y 
        cattn = self.ca(raw) 
        sattn = self.sa(raw) 
        mix_att = sattn + cattn 
        new = self.sigmoid(self.pa(raw, mix_att))
        out = raw + new * x + (1 - new) * y 
        out = self.conv(out)  

        return out

class HFF_new(nn.Module):
    def __init__(self, dim, reduction=8):
        super(HFF_new, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = PA(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, data):
        x, y = data
        base = x + y 
        
       
        s_att = self.sa(base)
        c_att = self.ca(base)
        
       
        hybrid_att = c_att * s_att 
        pa_feature = self.pa(base, hybrid_att)
        
       
        enhanced = base + self.alpha * pa_feature
        out = self.fusion(enhanced) + self.beta * base
        return out


class MultiScalePCA(nn.Module):
    def __init__(self, input_channel, gamma=2, bias=1):
        super(MultiScalePCA, self).__init__()
        input_channel1, input_channel2 = input_channel
        self.input_channel1 = input_channel1
        self.input_channel2 = input_channel2

        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(1)

        kernel_size1 = int(abs((math.log(input_channel1, 2) + bias) / gamma))
        kernel_size1 = kernel_size1 if kernel_size1 % 2 else kernel_size1 + 1

        kernel_size2 = int(abs((math.log(input_channel2, 2) + bias) / gamma))
        kernel_size2 = kernel_size2 if kernel_size2 % 2 else kernel_size2 + 1

        kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))
        kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1

        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size1, padding=(kernel_size1 - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size2, padding=(kernel_size2 - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, padding=(kernel_size3 - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.up = nn.ConvTranspose2d(in_channels=input_channel2, out_channels=input_channel1, kernel_size=3, stride=2,
                                     padding=1, output_padding=1)

    def forward(self, x):
        x1, x2 = x  
        x1_ = self.avg1(x1)  
        x2_ = self.avg2(x2)  

        x1_ = self.conv1(x1_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) 
        x2_ = self.conv2(x2_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1) 

        x_middle = torch.cat((x1_, x2_), dim=1)    
        x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)  
        x_middle = self.sigmoid(x_middle)   

        x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)

        x1_out = x1 * x_1  
        x2_out = x2 * x_2  

        x2_out = self.up(x2_out)   

        result = x1_out + x2_out
        return result

class MultiScalePCA_Down(nn.Module):
    def __init__(self, input_channel, gamma=2, bias=1):
        super(MultiScalePCA_Down, self).__init__()
        input_channel1, input_channel2 = input_channel
        self.input_channel1 = input_channel1
        self.input_channel2 = input_channel2

        self.avg1 = nn.AdaptiveAvgPool2d(1)
        self.avg2 = nn.AdaptiveAvgPool2d(1)

        kernel_size1 = int(abs((math.log(input_channel1, 2) + bias) / gamma))
        kernel_size1 = kernel_size1 if kernel_size1 % 2 else kernel_size1 + 1

        kernel_size2 = int(abs((math.log(input_channel2, 2) + bias) / gamma))
        kernel_size2 = kernel_size2 if kernel_size2 % 2 else kernel_size2 + 1

        kernel_size3 = int(abs((math.log(input_channel1 + input_channel2, 2) + bias) / gamma))
        kernel_size3 = kernel_size3 if kernel_size3 % 2 else kernel_size3 + 1

        self.conv1 = nn.Conv1d(1, 1, kernel_size=kernel_size1, padding=(kernel_size1 - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size2, padding=(kernel_size2 - 1) // 2, bias=False)
        self.conv3 = nn.Conv1d(1, 1, kernel_size=kernel_size3, padding=(kernel_size3 - 1) // 2, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.down = nn.Conv2d(in_channels=input_channel2, out_channels=input_channel1, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x1, x2 = x
        x1_ = self.avg1(x1)
        x2_ = self.avg2(x2)

        x1_ = self.conv1(x1_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x2_ = self.conv2(x2_.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        x_middle = torch.cat((x1_, x2_), dim=1)
        x_middle = self.conv3(x_middle.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        x_middle = self.sigmoid(x_middle)

        x_1, x_2 = torch.split(x_middle, [self.input_channel1, self.input_channel2], dim=1)

        x1_out = x1 * x_1
        x2_out = x2 * x_2

        x2_out = self.down(x2_out)

        result = x1_out + x2_out
        return result  

class PAFPN(nn.Module):
    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = SFnet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
       
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width),     
           
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        ) 

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width), 
           
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

       
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),      
           
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

       
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),     
           
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )
        self.hff_p2 = CSCF(dim=int(in_channels[0] * width)) 
        self.hff_p1 = CSCF(dim=int(in_channels[1] * width)) 
        self.hff_p0 = CSCF(dim=int(in_channels[2] * width)) 

       
       
       

       
       
       

       
       
       
    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

       
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features                

        fpn_out0 = self.lateral_conv0(x0) 
        f_out0 = self.upsample(fpn_out0) 
        f_out0 = torch.cat([f_out0, x1], 1) 
       

        f_out0 = self.C3_p4(f_out0) 

        fpn_out1 = self.reduce_conv1(f_out0) 
        f_out1 = self.upsample(fpn_out1) 
        f_out1 = torch.cat([f_out1, x2], 1) 
       
        pan_out2 = self.C3_p3(f_out1) 

        p_out1 = self.bu_conv2(pan_out2) 
        p_out1 = torch.cat([p_out1, fpn_out1], 1) 
        pan_out1 = self.C3_n3(p_out1) 

        p_out0 = self.bu_conv1(pan_out1) 
        p_out0 = torch.cat([p_out0, fpn_out0], 1) 
        pan_out0 = self.C3_n4(p_out0) 

       
        pan_out2 = self.hff_p2([pan_out2, x2]) 
        pan_out1 = self.hff_p1([pan_out1, x1]) 
        pan_out0 = self.hff_p0([pan_out0, x0]) 

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs


from .swin_transfomer import SwinTransformer
class YOLOPAFPN_Swin(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        width=1,
        depth = 1,
        swin_width=1,
        in_features=(3,4,5),
        in_channels=[512, 1024, 2048],
        out_channels=[256, 512, 1024],
        swin_depth=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        base_dim=96,
        depthwise=False,
        act="relu",
        pretrain_img_size=224,
        ape = False,
        window_size = 7,
    ):
        super().__init__()
        self.backbone = SwinTransformer(out_indices=in_features,depths=swin_depth,num_heads=num_heads,
                                        embed_dim=base_dim,pretrain_img_size=pretrain_img_size,ape=ape,window_size=window_size)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
       
       
       
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * swin_width), int(out_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
           
            int(in_channels[1] + out_channels[1] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        ) 

        self.reduce_conv1 = BaseConv(
            int(out_channels[1] * width), int(out_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
           
            int(in_channels[0]  + out_channels[0] * width),
            int(out_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

       
        self.bu_conv2 = Conv(
            int(out_channels[0] * width), int(out_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * out_channels[0] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

       
        self.bu_conv1 = Conv(
            int(out_channels[1] * width), int(out_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * out_channels[1] * width),
            int(out_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

       
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0) 
        f_out0 = self.upsample(fpn_out0) 
        f_out0 = torch.cat([f_out0, x1], 1) 
        f_out0 = self.C3_p4(f_out0) 

        fpn_out1 = self.reduce_conv1(f_out0) 
        f_out1 = self.upsample(fpn_out1) 
        f_out1 = torch.cat([f_out1, x2], 1) 
        pan_out2 = self.C3_p3(f_out1) 

        p_out1 = self.bu_conv2(pan_out2) 
        p_out1 = torch.cat([p_out1, fpn_out1], 1) 
        pan_out1 = self.C3_n3(p_out1) 

        p_out0 = self.bu_conv1(pan_out1) 
        p_out0 = torch.cat([p_out0, fpn_out0], 1) 
        pan_out0 = self.C3_n4(p_out0) 

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

from .resnet import ResNet
class YOLOPAFPN_ResNet(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        width=1,
        depth = 1,
        resnet_depth=50,
        in_features=("stage3", "stage4", "stage5"),
        in_channels=[512, 1024, 2048],
        out_channels=[256, 512, 1024],
        depthwise=False,
        act="relu",
    ):
        super().__init__()
        self.backbone = ResNet(depth=resnet_depth,out_features=in_features)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(out_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(3 * out_channels[1] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        ) 

        self.reduce_conv1 = BaseConv(
            int(out_channels[1] * width), int(out_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(3 * out_channels[0] * width),
            int(out_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

       
        self.bu_conv2 = Conv(
            int(out_channels[0] * width), int(out_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * out_channels[0] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

       
        self.bu_conv1 = Conv(
            int(out_channels[1] * width), int(out_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * out_channels[1] * width),
            int(out_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

       
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0) 
        f_out0 = self.upsample(fpn_out0) 
        f_out0 = torch.cat([f_out0, x1], 1) 
        f_out0 = self.C3_p4(f_out0) 

        fpn_out1 = self.reduce_conv1(f_out0) 
        f_out1 = self.upsample(fpn_out1) 
        f_out1 = torch.cat([f_out1, x2], 1) 
        pan_out2 = self.C3_p3(f_out1) 

        p_out1 = self.bu_conv2(pan_out2) 
        p_out1 = torch.cat([p_out1, fpn_out1], 1) 
        pan_out1 = self.C3_n3(p_out1) 

        p_out0 = self.bu_conv1(pan_out1) 
        p_out0 = torch.cat([p_out0, fpn_out0], 1) 
        pan_out0 = self.C3_n4(p_out0) 

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

from .focal import FocalNet
class YOLOPAFPN_focal(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
            self,
            width=1,
            depth=1,
            focal_width=1,
            focal_depth=1,
            in_features=(3, 4, 5),
            in_channels=[512, 1024, 2048],
            out_channels=[256, 512, 1024],
            depths=[2, 2, 6, 2],
            focal_levels=[4, 4, 4, 4],
            focal_windows=[3, 3, 3, 3],
            use_conv_embed=True,
            use_postln=True,
            use_postln_in_modulation=False,
            use_layerscale=True,
            base_dim=96,
            depthwise=False,
            act="relu",
    ):
        super().__init__()
        self.backbone = FocalNet(embed_dim=base_dim,
                                 depths=depths,
                                 out_indices=in_features,
                                 focal_levels=focal_levels,
                                 focal_windows=focal_windows,
                                 use_conv_embed=use_conv_embed,
                                 use_postln=use_postln,
                                 use_postln_in_modulation=use_postln_in_modulation,
                                 use_layerscale=use_layerscale,
                                 )
        self.in_features = in_features

        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * focal_width), int(out_channels[1] * width), 1, 1, act=act
        )
        self.C3_p4 = CSPLayer(
            int(in_channels[1] * focal_width + out_channels[1] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        ) 

        self.reduce_conv1 = BaseConv(
            int(out_channels[1] * width), int(out_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(in_channels[0]* focal_width + out_channels[0] * width),
            int(out_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

       
        self.bu_conv2 = Conv(
            int(out_channels[0] * width), int(out_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * out_channels[0] * width),
            int(out_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

       
        self.bu_conv1 = Conv(
            int(out_channels[1] * width), int(out_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * out_channels[1] * width),
            int(out_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

       
        out_features = self.backbone(input)

        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        fpn_out0 = self.lateral_conv0(x0) 
        f_out0 = self.upsample(fpn_out0) 
        f_out0 = torch.cat([f_out0, x1], 1) 
        f_out0 = self.C3_p4(f_out0) 

        fpn_out1 = self.reduce_conv1(f_out0) 
        f_out1 = self.upsample(fpn_out1) 
        f_out1 = torch.cat([f_out1, x2], 1) 
        pan_out2 = self.C3_p3(f_out1) 

        p_out1 = self.bu_conv2(pan_out2) 
        p_out1 = torch.cat([p_out1, fpn_out1], 1) 
        pan_out1 = self.C3_n3(p_out1) 

        p_out0 = self.bu_conv1(pan_out1) 
        p_out0 = torch.cat([p_out0, fpn_out0], 1) 
        pan_out0 = self.C3_n4(p_out0) 

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs

       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       