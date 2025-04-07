#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
import copy
import time

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger

from yolox.models.post_process import postprocess,get_linking_mat
from yolox.models.post_trans import MSA_yolov, LocalAggregation
from yolox.utils import bboxes_iou
from yolox.utils.box_op import box_cxcywh_to_xyxy, generalized_box_iou
from .losses import IOUloss
from .network_blocks import BaseConv, DWConv
from typing import Tuple, Union


class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes,
            width=1.0,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False,
            heads=4,
            drop=0.0,
            use_score=True,
            defualt_p=30,
            sim_thresh=0.75,
            pre_nms=0.75,
            ave=True,
            defulat_pre=750,
            test_conf=0.001,
            use_mask=False,
            gmode=True,
            lmode=False,
            both_mode=False,
            localBlocks=1,
            **kwargs
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.Afternum = defualt_p
        self.Prenum = defulat_pre
        self.simN = defualt_p
        self.nms_thresh = pre_nms
        self.n_anchors = 1
        self.use_score = use_score
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False
        self.gmode = gmode
        self.lmode = lmode
        self.both_mode = both_mode

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.cls_convs2 = nn.ModuleList()
        
        #yb
        # self.temporal_attn = TemporalAttention(embed_dim=128, num_heads=8, dropout=0.1, max_len=32)
        self.temporal_attn = SSTA(embed_dim=128, num_heads=8, dropout=0.1, max_len=32)
        #yb


        self.width = int(256 * width)
        self.sim_thresh = sim_thresh
        self.ave = ave
        self.use_mask = use_mask

        if gmode:
            self.trans = MSA_yolov(dim=self.width, out_dim=4 * self.width, num_heads=heads, attn_drop=drop)

            # self.linear_pred = nn.Linear(int(4 * self.width), num_classes + 1)   # 源码
            self.linear_pred = nn.Linear(int(self.width), num_classes + 1)     #yb
            self.conf_pred = nn.Linear(int(self.width), 1)    #yb
        if lmode:

            if kwargs.get('reconf',False):
                self.conf_pred = nn.Linear(int(self.width), 1)
            self.LocalAggregation = LocalAggregation(dim=self.width, heads=heads, attn_drop=drop, blocks=localBlocks,
                                                     **kwargs)
            if kwargs.get('globalBlocks',0):
                self.GlobalAggregation = MSA_yolov(dim=self.width, out_dim=4 * self.width, num_heads=heads, attn_drop=drop)
                self.linear_pred = nn.Linear(int(4*self.width), num_classes + 1)
            else:
                self.linear_pred = nn.Linear(self.width,num_classes + 1)

        if both_mode:
            self.g2l = nn.Linear(int(4 * self.width), self.width)
        self.stems = nn.ModuleList()
        self.kwargs = kwargs
        Conv = DWConv if depthwise else BaseConv     # CBS

        # self.stems 一个卷积 in 256 512 1024  out 256   存储每个特征层级的“茎”卷积,用于进一步处理输入特征图，调整通道数和特征表示
        # self.cls_convs  两个卷积 in 256 out 256       存储用于分类的卷积层序列，每个特征层级有一个对应的卷积序列
        # self.reg_convs  同 self.cls_convs             存储额外的分类卷积层序列，用于生成更丰富的分类特征
        # self.cls_preds  一个卷积 in 256 out 1（类别概率） 存储用于生成类别预测的 1x1 卷积层，每个特征层级有一个对应的预测层。
        # self.reg_preds  一个卷积 in 256 out 4 （xywh）    存储用于生成边界框回归预测的 1x1 卷积层，每个特征层级有一个对应的预测层
        # self.obj_preds  一个卷积 in 256 out 1 （目标概率） 存储用于生成目标置信度预测的 1x1 卷积层，每个特征层级有一个对应的预测层
        # self.cls_convs2 同 self.cls_convs
        
        for i in range(len(in_channels)):
            self.stems.append(
                BaseConv(
                    in_channels=int(in_channels[i] * width),
                    out_channels=int(256 * width),
                    ksize=1,
                    stride=1,
                    act=act,
                )
            )
            self.cls_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_convs2.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.reg_convs.append(
                nn.Sequential(
                    *[
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                        Conv(
                            in_channels=int(256 * width),
                            out_channels=int(256 * width),
                            ksize=3,
                            stride=1,
                            act=act,
                        ),
                    ]
                )
            )
            self.cls_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * self.num_classes,  #yb 输出类别预测 1
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.reg_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=4,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )
            self.obj_preds.append(
                nn.Conv2d(
                    in_channels=int(256 * width),
                    out_channels=self.n_anchors * 1,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        for conv in self.obj_preds:
            b = conv.bias.view(self.n_anchors, -1)
            b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    # xin backbone输出的3个尺寸的特征bs16， labels 16 120 5  imgs原始图像 16 3 576 576
    def forward(self, xin, labels=None, imgs=None, nms_thresh=0.5, lframe=0, gframe=32):    
        outputs = []
        outputs_decode = []
        outputs_decode_nopostprocess = []   # yb
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        before_nms_features = []
        before_nms_regf = []

        # 循环遍历每个特征层级
        # self.cls_convs：分类卷积层列表，每个层级一个。
        # self.cls_convs2：额外的分类卷积层列表，用于更复杂的分类特征提取。
        # self.reg_convs：回归卷积层列表，每个层级一个。
        # self.strides：每个层级的步幅（stride），用于将特征图坐标映射到原始图像坐标。
        # xin：输入的特征图列表，来自不同层级的 FPN 输出。
        # k：当前循环的索引，表示特征层级。
        # x：当前层级的输入特征图，形状通常为 [batch_size, in_channels, H, W] 16 28 72 72

        for k, (cls_conv, cls_conv2, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.cls_convs2, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)        # 16 128 72 72  通过一个 1x1 卷积（BaseConv），调整特征图的通道数到统一的 256 * width，为后续的分类和回归卷积做准备
            reg_feat = reg_conv(x)      # 16 128 72 72  通过回归卷积层，提取用于边界框回归的特征
            cls_feat = cls_conv(x)      # 16 128 72 72  通过分类卷积层，提取用于类别预测的特征
            cls_feat2 = cls_conv2(x)    # 16 128 72 72  通过额外的分类卷积层，生成更丰富的分类特征 新增

            # this part should be the same as the original model
            obj_output = self.obj_preds[k](reg_feat)    # 16 1 72 72 通过目标置信度预测卷积层，生成每个锚框的置信度得分 每个特征图1个锚框
            reg_output = self.reg_preds[k](reg_feat)    # 16 4 72 72 通过边界框回归预测卷积层，生成每个锚框的回归参数
            cls_output = self.cls_preds[k](cls_feat)    # 16 1 72 72 通过类别预测卷积层，生成每个锚框的类别得分 1个类别
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)        # 16 6 72 72 将回归预测、目标置信度和类别得分沿通道维度拼接
                output_decode = torch.cat(                                         # 新增
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1    # 16 6 72 72 对目标置信度和类别得分应用sigmoid激活函数，
                )
                output, grid = self.get_output_and_grid(                        # 将输出转换为适合解码和后续处理的格式 [batch_size, anchors * H * W, channels]
                    output, k, stride_this_level, xin[0].type()                 # stride_this_level 8 
                )
                x_shifts.append(grid[:, :, 0])                          # 分别记录每个预测框在 x 和 y 方向上的网格偏移。 1 5184
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(                                # 记录当前层级的步幅（stride），用于将预测框坐标映射到原始图像坐标  1 5184   8 16 32                 
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, self.n_anchors, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())
                outputs.append(output)                      # 新增
                before_nms_features.append(cls_feat2)       # 三个不同尺度下的特征
                before_nms_regf.append(reg_feat)            # 新增
            else:
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

                # which features to choose
                before_nms_features.append(cls_feat2)
                before_nms_regf.append(reg_feat)
            outputs_decode.append(output_decode)                        # 三个列表每个列表维度为  16 6  w h （不同尺度）
            outputs_decode_nopostprocess.append(output_decode)          # yb
        # 以下前向传播代码都是yolov新增代码
        self.hw = [x.shape[-2:] for x in outputs_decode]                # 记录每个解码输出的高度和宽度
        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2    # 16  6804 6        
                                   ).permute(0, 2, 1)                       # 所有层级的解码输出在空间维度（高度和宽度）上展平，并沿通道维度拼接 
        decode_res = self.decode_outputs(outputs_decode, dtype=xin[0].type()) # 16 6804 6  模型输出的原始预测（通常是相对于特征图网格的坐标和尺度）解码为图像空间中的绝对坐标

        if self.kwargs.get('ota_mode',False) and self.training:
            ota_idxs,reg_targets = self.get_fg_idx( imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),)
        else:
            ota_idxs = None
        # bs个列表，每个列表形状为 30 8：30表示每张图片置信度最高的30个框，8表示这些框的信息，坐标，类别..
        pred_result, pred_idx = self.postpro(decode_res, num_classes=self.num_classes,      #   找出置信度最高的30个特征和索引，并进行极大值抑制
                                                     nms_thre=self.nms_thresh,
                                                     topK=self.Afternum,
                                                     ota_idxs=ota_idxs,
                                                     )

        if not self.training and imgs.shape[0] == 1:
            return self.postprocess_single_img(pred_result, self.num_classes)

        # 特征聚合与评分
        cls_feat_flatten = torch.cat(                   # 将所有层级的分类特征展平并拼接，形状为16 6084（3个尺度之和 72*72+36*36+18*18） 128 [batch_size, features, channels  
            [x.flatten(start_dim=2) for x in before_nms_features], dim=2
        ).permute(0, 2, 1)  # [b,features,channels]
        reg_feat_flatten = torch.cat(                   # 将所有层级的回归特征展平并拼接，形状为16 6084（3个尺度之和） 128 [batch_size, features, channels
            [x.flatten(start_dim=2) for x in before_nms_regf], dim=2
        ).permute(0, 2, 1)
        # 得到预测结果中得分前30的结果，bs*30 
        (features_cls, features_reg, cls_scores,            # 480 128
         fg_scores, locs, all_scores) = self.selective_feature(cls_feat_flatten,
                                                                pred_idx,
                                                                reg_feat_flatten,
                                                                imgs,
                                                                pred_result)
        
        # #---------------------origin------------------------------------------------
        # features_reg = features_reg.unsqueeze(0)  # 1 480 128 
        # features_cls = features_cls.unsqueeze(0)  # 1 480 128 [1,features,channels]

        # if not self.training:
        #     cls_scores = cls_scores.to(cls_feat_flatten.dtype)
        #     fg_scores = fg_scores.to(cls_feat_flatten.dtype)
        #     locs = locs.to(cls_feat_flatten.dtype)
        # if self.gmode:
        #     kwargs = self.kwargs
        #     kwargs.update({'lframe': lframe, 'gframe': gframe, 'afternum': self.Afternum})
        #     if self.use_score:              # 输入，筛选出的30个特征以及其得分 features_cls 480 512
        #         features_cls,fg_scores = self.trans(features_cls, features_reg, cls_scores, fg_scores, sim_thresh=self.sim_thresh,
        #                                   ave=self.ave, use_mask=self.use_mask, **kwargs)
        #     else:
        #         features_cls = self.trans(features_cls, features_reg, None, None,
        #                                   sim_thresh=self.sim_thresh, ave=self.ave, **kwargs)
        # if self.lmode:
        #     if self.both_mode:
        #         features_cls = self.g2l(features_cls).unsqueeze(0)
        #     more_args = {'width': imgs.shape[-1], 'height': imgs.shape[-2], 'fg_score': fg_scores,
        #                  'cls_score': cls_scores,'all_scores':all_scores,'lframe':lframe,
        #                  'afternum':self.Afternum,'gframe':gframe,'use_score':self.use_score}
        #     #st = time.time()
        #     features_cls,features_reg = self.LocalAggregation(features_cls[:, :lframe * self.Afternum],
        #                                                       features_reg[:, :lframe * self.Afternum],
        #                                                       locs[:lframe * self.Afternum].view(-1, self.Afternum, 4),
        #                                                       **more_args)
        #     if self.kwargs.get('globalBlocks',0):
        #         kwargs = self.kwargs
        #         kwargs.update({'lframe': lframe, 'gframe': gframe, 'afternum': self.Afternum})
        #         features_cls,fg_scores = self.GlobalAggregation(features_cls, features_reg, cls_scores, fg_scores, sim_thresh=self.sim_thresh,
        #                                   ave=self.ave, use_mask=self.use_mask, **kwargs)

        #     if self.both_mode:
        #         outputs = [o[:lframe] for o in outputs]
        #         pred_idx = pred_idx[:lframe]
        #         pred_result = pred_result[:lframe]
        # fc_output = self.linear_pred(features_cls)  # 480 2 
        # fc_output = torch.reshape(fc_output, [-1, self.Afternum, self.num_classes + 1])[:, :, :-1] # [b,afternum,cls]  # 16 30 1

        # if self.kwargs.get('reconf', False):
        #     conf_output = self.conf_pred(features_reg)
        #     conf_output = torch.reshape(conf_output, [-1, self.Afternum])
        # else:
        #     conf_output = None
        # #---------------------origin------------------------------------------------

        ##yb-------------------------------------------------------------------------
        features_cls_att = self.temporal_attn(features_cls)         # 16 30 128
        features_reg_att = self.temporal_attn(features_reg)         # 16 30 128
        fc_output = self.linear_pred(features_cls_att)  # # 16 30 2 
        fc_output = torch.reshape(fc_output, [-1, self.Afternum, self.num_classes + 1])[:, :, :-1] # [b,afternum,cls]  # 16 30 1
        conf_output = self.conf_pred(features_reg_att)  # 16 30 1
        conf_output = torch.reshape(conf_output, [-1, self.Afternum])  # 16 30
        ##yb------------------------------------------------------------------------- 
            
        if self.training:
            if self.both_mode:
                labels = labels[:lframe]
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
                refined_cls = fc_output,            # 新增
                idx=pred_idx,                       # 新增
                pred_res = pred_result,             # 新增
                conf_output = conf_output           # 新增
            )
        else:
            # ##----------yb-nopostprocess----------------------------
            # outputs = torch.cat(
            #     [x.flatten(start_dim=2) for x in outputs], dim=2
            # ).permute(0, 2, 1)
            # if self.decode_in_inference:
            #     return self.decode_outputs(outputs, dtype=xin[0].type())
            # else:
            #     return outputs
            # ##----------yb-nopostprocess----------------------------

            #--------------origin---------------------------------
            result, result_ori = postprocess(copy.deepcopy(pred_result),
                                             self.num_classes,
                                             fc_output,
                                             conf_output = conf_output,
                                             nms_thre=nms_thresh,
                                             )
            return result, result_ori  # result
            #--------------origin---------------------------------
        
    # 将卷积层的输出 (output) 转换为适合解码边界框（bounding boxes）的格式
    # 将预测的 x 和 y 坐标与网格偏移相加，并乘以步幅，将其映射回原始图像空间。
    # 对 w 和 h 使用指数函数并乘以步幅，确保宽度和高度为正数并映射回原始图像空间
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]            # self.grids 长度为3的列表

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes             # 6 坐标4+1置信度+1类别
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:  # 重新调整网格尺寸 最终网格维度 1 1 72 72 2  2表示每个位置包含xy坐标
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid       

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)  #调整输出特征形状 16 1 6 72 72  [batch_size, n_anchors, n_ch, hsize, wsize]
        output = output.permute(0, 1, 3, 4, 2).reshape(                 # 最终形状  16 5184 6  [batch_size, n_anchors * hsize * wsize, n_ch]
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)                          # 1 5184 2  与输出的预测信息进行匹配和计算
        output[..., :2] = (output[..., :2] + grid) * stride # output[..., :2] 表示预测框的 x 和 y 偏移量 形状为16 5184 2  将偏移量与网格坐标相加，得到相对于原始网格点的坐标， 原始图像上的绝对坐标
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride # output[..., 2:4] 表示预测框的宽度和高度（w, h），原始图像上的绝对宽度和高度
        return output, grid  # 调整后的预期信息 16 5184 6  调整后的网格坐标grid 1 5184 2

    # 将模型的预测转换为实际图像中的位置信息
    def decode_outputs(self, outputs, dtype, flevel=0):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(torch.full((*shape, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs
    #  从给定的特征和预测中，根据提供的索引 (idxs)，提取相关的分类特征、回归特征以及对应的得分 
    #  features 16 6804 128 
    #  idxs 16个列表，每个列表中的张量形状为30，即每张图片中找到的置信度最大的像素值的索引
    #  reg_features 16 6804 128
    #  imgs 16 3  576 576
    #  predictions 16个列表形状为 30 8 ，即每张图片所对应的信息，边界框xywh，目标置信度、类别置信度等等。
    def selective_feature(self, features, idxs, reg_features, imgs=None, predictions=None, roi_features=None): 
        features_cls = []
        features_reg = []
        cls_scores, all_scores = [], []
        fg_scores = []
        locs = []
        for i, feature in enumerate(features):                               # 16个列表，每个列表形状30 128
            features_cls.append(feature[idxs[i][:self.simN]])                # 根据索引idxs[i]中的索引 提取前 simN 个分类特征并添加到 features_cls 中
            features_reg.append(reg_features[i, idxs[i][:self.simN]])        # 根据索引，提取前 simN 个回归特征并添加到 features_reg 中
            cls_scores.append(predictions[i][:self.simN, 5])                 # 提取预测结果中类别得分部分（从第6列开始），并添加到 cls_scores 中
            fg_scores.append(predictions[i][:self.simN, 4])                  # 提取预测结果中前景得分（第5列），并添加到 fg_scores 中
            locs.append(predictions[i][:self.simN, :4])                      # 提取预测结果中边界框坐标（前4列），并添加到 locs 中
            all_scores.append(predictions[i][:self.simN, -self.num_classes:]) # 提取预测结果中所有类别的得分，并添加到 all_scores 中

        #yb----------------    
        features_cls = torch.stack(features_cls, dim=0)          # 拼接所有提取的特征和得分：16个30 128的列表 沿第0维度拼接 480 128
        features_reg = torch.stack(features_reg, dim=0)          # 同上 480 128
        cls_scores = torch.stack(cls_scores, dim=0)              # 同上 480 
        fg_scores = torch.stack(fg_scores, dim=0)                # 同上 480 
        locs = torch.stack(locs, dim=0)                          # 同上 480 4
        all_scores = torch.stack(all_scores, dim=0)              # 同上 480 4 
        #yb---------------- 

        # #-----------orgin-----------------------------------
        # features_cls = torch.cat(features_cls)          # 拼接所有提取的特征和得分：16个30 128的列表 沿第0维度拼接 480 128
        # features_reg = torch.cat(features_reg)          # 同上 480 128
        # cls_scores = torch.cat(cls_scores)              # 同上 480 
        # fg_scores = torch.cat(fg_scores)                # 同上 480 
        # locs = torch.cat(locs)                          # 同上 480 4
        # all_scores = torch.cat(all_scores)              # 同上 480 4 
        # #-----------orgin-----------------------------------
        return features_cls, features_reg, cls_scores, fg_scores, locs, all_scores

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
            refined_cls,
            idx,
            pred_res,
            conf_output=None,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets             #mixup 新增
        mixup = labels.shape[2] > 5     
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        ref_targets = []
        num_fg = 0.0
        num_gts = 0.0
        ref_masks = []          # 新增      
        conf_targets = []       # 新增  
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))         # 新增  
                conf_target = outputs.new_zeros((idx[batch_idx].shape[0], 1))                           # 新增  
                ref_target[:, -1] = 1                       

            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]  # [batch,120,class+xywh]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target_onehot = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                )
                cls_target = cls_target_onehot * pred_ious_this_matching.unsqueeze(-1)
                fg_idx = torch.where(fg_mask)[0]

                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))
                conf_target = outputs.new_zeros((idx[batch_idx].shape[0], 1))
                fg = 0

                gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))
                pred_box = pred_res[batch_idx][:, :4]
                cost_giou, iou = generalized_box_iou(pred_box, gt_xyxy)
                max_iou = torch.max(iou, dim=-1)
                for ele_idx, ele in enumerate(idx[batch_idx]):
                    loc = torch.where(fg_idx == ele)[0]

                    if len(loc):
                        ref_target[ele_idx, :self.num_classes] = cls_target[loc, :]
                        conf_target[ele_idx] = 1#obj_target[loc]
                        fg += 1
                        continue
                    if max_iou.values[ele_idx] >= 0.6:
                        if not self.kwargs.get('ota_cls', False):
                            max_idx = int(max_iou.indices[ele_idx])
                            ref_target[ele_idx, :self.num_classes] = cls_target_onehot[max_idx, :] * max_iou.values[ele_idx]
                            conf_target[ele_idx] = 1
                            fg += 1
                        if not self.kwargs.get('ota_mode', False):
                            conf_target[ele_idx] = 1#obj_target[max_idx]
                    else:
                        ref_target[ele_idx, -1] = 1 - max_iou.values[ele_idx]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            ref_targets.append(ref_target[:, :self.num_classes])
            ref_masks.append(ref_target[:, -1] == 0)
            conf_targets.append(conf_target.to(dtype))
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        ref_targets = torch.cat(ref_targets, 0)
        conf_targets = torch.cat(conf_targets, 0)

        fg_masks = torch.cat(fg_masks, 0)
        ref_masks = torch.cat(ref_masks, 0)

        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        loss_ref = (
                       self.bcewithlog_loss(
                           refined_cls.view(-1, self.num_classes)[ref_masks], ref_targets[ref_masks]
                       )
                   ).sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 3.0
        if conf_output is not None:
            loss_rconf = (
                    self.bcewithlog_loss(conf_output.view(-1, 1), conf_targets)
            ).sum() / num_fg
            if loss_rconf>20:
                #clip the loss
                loss_rconf = 1/loss_rconf*10
        else:
            loss_rconf = 0.0

        loss = reg_weight * loss_iou + loss_obj + 2 * loss_ref + loss_l1 + loss_cls + loss_rconf

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            2 * loss_ref,
            loss_l1,
            loss_rconf,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                    cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                    * obj_preds_.unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
            .unsqueeze(1)
            .repeat(1, total_num_anchors)
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        center_radius = 4.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).repeat(
            1, total_num_anchors
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )
            matching_matrix[gt_idx][pos_idx] = 1.0

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    def process_prediction(self, image_pred, num_classes, nms_thre):
        if not image_pred.size(0):
            return None, None
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        detections = torch.cat(
            (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)
        conf_score = image_pred[:, 4]
        top_pre = torch.topk(conf_score, k=self.Prenum)
        sort_idx = top_pre.indices[:self.Prenum]
        detections_temp = detections[sort_idx, :]
        nms_out_index = torchvision.ops.batched_nms(
            detections_temp[:, :4],
            detections_temp[:, 4] * detections_temp[:, 5],
            detections_temp[:, 6],
            nms_thre,
        )
        topk_idx = sort_idx[nms_out_index[:self.topK]]
        return detections[topk_idx, :], topk_idx

    def postpro(self, prediction, num_classes, nms_thre=0.75, topK=75, ota_idxs=None):
        # find topK predictions, play the same role as RPN
        '''
        边界框坐标转换：将模型预测的边界框从中心坐标和宽高表示转换为左上角和右下角坐标表示。
        处理 OTA 模式：如果在训练阶段并且启用了 OTA Optimal Transport Assignment模式,直接使用 OTA 提供的索引进行筛选。
        类别置信度计算：计算每个边界框的类别置信度，并选择置信度最高的类别。
        非极大值抑制NMS: 应用 NMS 筛选出不重叠的高置信度边界框。
        选择 Top K:从筛选后的边界框中选择前 topK 个最高置信度的边界框。

        Args:
            prediction: [batch,feature_num,5+clsnum]
            num_classes:
            conf_thre:
            conf_thre_high:
            nms_thre:

        Returns:
            [batch,topK,5+clsnum]
        '''
        self.topK = topK
        box_corner = prediction.new(prediction.shape)                       # 将xywh 转化为xyxy
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]             # 用于存储每张图像筛选后的检测框
        output_index = [None for _ in range(len(prediction))]       # 用于存储每张图像筛选后的检测框在原始预测中的索引

        for i, image_pred in enumerate(prediction):
            #take ota idxs as output in training mode
            if ota_idxs is not None and len(ota_idxs[i]) > 0:
                ota_idx = ota_idxs[i]
                topk_idx = torch.stack(ota_idx).type_as(image_pred)
                output[i] = image_pred[topk_idx, :]
                output_index[i] = topk_idx
                continue

            if not image_pred.size(0):
                continue
            # Get score and class with highest confidence  image_pred 6804 6 得出当前图片属于哪一个类别和该类别的置信度得分
            class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)  # 计算每个检测框的类别置信度 class_conf 和类别索引 class_pred，即选择置信度最高的类别

            # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
            # 得到检测框 目标置信度 obj_conf、类别置信度 class_conf、类别索引  [feature_num, 5 + 1 + 1 + clsnum]
            detections = torch.cat(
                (image_pred[:, :5], class_conf, class_pred.float(), image_pred[:, 5: 5 + num_classes]), 1)

            conf_score = image_pred[:, 4]           # 6804 每个像素的目标置信度  
            top_pre = torch.topk(conf_score, k=self.Prenum) # 选择目标置信度最高的 self.Prenum 个像素 750个
            sort_idx = top_pre.indices[:self.Prenum]        # 选择出的前 self.Prenum 个检测框的索引
            detections_temp = detections[sort_idx, :]       # 目标置信度最高的 self.Prenum 个检测框的信息
            nms_out_index = torchvision.ops.batched_nms(    # 对检测框应用 NMS
                detections_temp[:, :4],
                detections_temp[:, 4] * detections_temp[:, 5],
                detections_temp[:, 6],
                nms_thre,
            )

            topk_idx = sort_idx[nms_out_index[:self.topK]]  # 从原始检测框索引 sort_idx 中选择 NMS 筛选后的前 self.topK 个检测框的索引
            output[i] = detections[topk_idx, :]             # 存储筛选后的检测框信息
            output_index[i] = topk_idx                      # 存储筛选后的检测框在原始预测中的索引

        return output, output_index                         # bs个列表，每个列表为 30 8： 30表示每张图片置信度最高的30个框，8表示这些框的信息，坐标，类别..

    def postprocess_single_img(self, prediction, num_classes, conf_thre=0.001, nms_thre=0.5):

        output_ori = [None for _ in range(len(prediction))]
        prediction_ori = copy.deepcopy(prediction)
        for i, detections in enumerate(prediction):

            if not detections.size(0):
                continue

            detections_ori = prediction_ori[i]

            conf_mask = (detections_ori[:, 4] * detections_ori[:, 5] >= conf_thre).squeeze()
            detections_ori = detections_ori[conf_mask]
            nms_out_index = torchvision.ops.batched_nms(
                detections_ori[:, :4],
                detections_ori[:, 4] * detections_ori[:, 5],
                detections_ori[:, 6],
                nms_thre,
            )
            detections_ori = detections_ori[nms_out_index]
            output_ori[i] = detections_ori
        # print(output)
        return output_ori, output_ori


    def get_fg_idx(self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        num_gts = 0.0
        fg_ids = []
        reg_targets = []
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                fg_idx = []#torch.where(fg_mask)[0]
                reg_target = outputs.new_zeros((0, 4))

            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]  # [batch,120,class+xywh]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                fg_idx = torch.where(fg_mask)[0]
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                gt_xyxy = box_cxcywh_to_xyxy(torch.tensor(reg_target))

            fg_ids.append(fg_idx)
            reg_targets.append(reg_target)


        return fg_ids,reg_targets

#  源码
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)  # [max_len, embed_dim]
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维
        pe = pe.unsqueeze(0)  # [1, max_len, embed_dim]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embed_dim]
        Returns:
            Tensor: x + positional encoding
        """
        # print(f"x shape: {x.shape}")
        # print(f"self.pe shape: {self.pe.shape}")
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return x

# 定义时间注意力模块
class TemporalAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, max_len=16):
        super(TemporalAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.pos_encoder = PositionalEncoding(embed_dim, max_len)
    
    def forward(self, x):
        """
        Args:
            A: Tensor of shape [B, N, C] where
                B = number of frames (16)
                N = number of pixels (30)
                C = channels (128)
        
        Returns:
            Tensor of shape [B, N, C] after applying temporal attention
        """
        B, N, C = x.size()  # [16, 30, 128]
        
        # Transpose to [N, B, C] to treat N as batch size for attention
        x_transposed = x.permute(1, 0, 2)  # [30, 16, 128]
        
        # 添加位置编码
        x_transposed = self.pos_encoder(x_transposed)  # [30, 16, 128]
        
        # 应用多头自注意力
        attn_output, attn_weights = self.attn(x_transposed, x_transposed, x_transposed)  # [30, 16, 128]
        
        # 残差连接和层归一化
        attn_output = self.layer_norm1(attn_output + x_transposed)  # [30, 16, 128]
        
        # 前馈网络
        ffn_output = self.ffn(attn_output)  # [30, 16, 128]
        
        # 残差连接和层归一化
        output = self.layer_norm2(ffn_output + attn_output)  # [30, 16, 128]
        
        # 转置回 [B, N, C]
        output = output.permute(1, 0, 2)  # [16, 30, 128]
        
        return output
#  源码

class RetNetRelPos1d(nn.Module):
    """修改为处理一维时间序列的版本"""
    def __init__(self, embed_dim, num_heads, initial_value=1, heads_range=3):
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)
        
    def generate_decay_mask(self, seq_len: int):
        index = torch.arange(seq_len).to(self.decay)
        mask = index[:, None] - index[None, :]  # [seq_len, seq_len]
        mask = mask.abs()  # Absolute distance
        mask = mask * self.decay[:, None, None]  # [num_heads, seq_len, seq_len]
        return mask  # 衰减掩码
    
    def forward(self, seq_len: int):             #orgin
        # 生成相对位置的正弦/余弦编码
        index = torch.arange(seq_len).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])  # [seq_len, embed_dim]
        cos = torch.cos(index[:, None] * self.angle[None, :])  # [seq_len, embed_dim]
        # 生成衰减掩码
        mask = self.generate_decay_mask(seq_len)  # [num_heads, seq_len, seq_len]
        return (sin, cos), mask
    
    # def forward(self, seq_len: int):    # new_pos
    #     index = torch.arange(seq_len).to(self.decay)
    #     sin = torch.sin(index[:, None] * self.angle[None, :])
    #     cos = torch.cos(index[:, None] * self.angle[None, :])
    #     mask = torch.tril(torch.ones(seq_len, seq_len).to(self.decay))
    #     mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
    #     mask = torch.exp(mask * self.decay[:, None, None])
    #     mask = torch.nan_to_num(mask)
    #     mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
    #     retention_rel_pos = ((sin, cos), mask)
    #     return retention_rel_pos
    
class SSTA(nn.Module):                         # Selective Spatio-Temporal Aggregation
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1, max_len=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        # 替换位置编码模块
        self.pos_encoder = RetNetRelPos1d(
            embed_dim=embed_dim,
            num_heads=num_heads,
            initial_value=1,
            heads_range=3
        )
        # 调整MultiheadAttention为自定义实现
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        B, N, C = x.size()  # [16, 30, 128]
        x = x.permute(1, 0, 2)  # [30, 16, 128]
        
        # 生成相对位置编码与掩码
        (sin, cos), mask = self.pos_encoder(seq_len=x.size(1))  # seq_len=16
        
        # 生成Q/K/V并分头
        q = self.q_proj(x).view(x.size(0), x.size(1), self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)  # 8 30 16 16
        k = self.k_proj(x).view(x.size(0), x.size(1), self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)  # 8 30 16 16
        v = self.v_proj(x).view(x.size(0), x.size(1), self.num_heads, C // self.num_heads).permute(2, 0, 1, 3)  # 8 30 16 16
        
        # 应用相对位置编码（旋转嵌入）
        q = rotary_emb(q, sin, cos)   # 8 30 16 16
        k = rotary_emb(k, sin, cos)   # 8 30 16 16
        
        # 计算注意力分数
        scores = torch.einsum("hnqd,hnkd->hnqk", q, k)   # 8 30 16 16 # [num_heads, N, seq_len, seq_len]
        scores += mask.unsqueeze(1)  # 添加衰减掩码     # 8 30 16 16
        attn = torch.softmax(scores, dim=-1)  # 8 30 16 16
        attn = self.dropout(attn)  # 8 30 16 16
        
        # 聚合Value
        output = torch.einsum("hnqk,hnkd->hnqd", attn, v) # 8 30 16 16
        output = output.permute(1, 2, 0, 3).contiguous().view(x.size(0), x.size(1), C)  # 30 16 128
        
        # 残差连接与后续处理
        output = self.layer_norm1(output + x)  # 30 16 128
        ffn_output = self.ffn(output)   # 30 16 128
        output = self.layer_norm2(ffn_output + output)  # 30 16 128
        
        return output.permute(1, 0, 2)  # 还原为 [16, 30, 128]

def rotary_emb(x, sin, cos):
    """应用旋转位置编码（简化版）"""
    x_rot = x * cos + torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1) * sin
    return x_rot
