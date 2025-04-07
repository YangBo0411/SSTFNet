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
        self.decode_in_inference = True 
        self.gmode = gmode
        self.lmode = lmode
        self.both_mode = both_mode

        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.cls_convs2 = nn.ModuleList()

        self.temporal_attn = SSTA(embed_dim=128, num_heads=8, dropout=0.1, max_len=32)

        self.width = int(256 * width)
        self.sim_thresh = sim_thresh
        self.ave = ave
        self.use_mask = use_mask

        if gmode:
            self.trans = MSA_yolov(dim=self.width, out_dim=4 * self.width, num_heads=heads, attn_drop=drop)

           
            self.linear_pred = nn.Linear(int(self.width), num_classes + 1)    
            self.conf_pred = nn.Linear(int(self.width), 1)   
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
        Conv = DWConv if depthwise else BaseConv    

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
                    out_channels=self.n_anchors * self.num_classes, 
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

   
    def forward(self, xin, labels=None, imgs=None, nms_thresh=0.5, lframe=0, gframe=32):    
        outputs = []
        outputs_decode = []
        outputs_decode_nopostprocess = []  
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []
        before_nms_features = []
        before_nms_regf = []


        for k, (cls_conv, cls_conv2, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.cls_convs2, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)       
            reg_feat = reg_conv(x)     
            cls_feat = cls_conv(x)     
            cls_feat2 = cls_conv2(x)   

           
            obj_output = self.obj_preds[k](reg_feat)   
            reg_output = self.reg_preds[k](reg_feat)   
            cls_output = self.cls_preds[k](cls_feat)   
            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)       
                output_decode = torch.cat(                                        
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1   
                )
                output, grid = self.get_output_and_grid(                       
                    output, k, stride_this_level, xin[0].type()                
                )
                x_shifts.append(grid[:, :, 0])                         
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(                               
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
                outputs.append(output)                     
                before_nms_features.append(cls_feat2)      
                before_nms_regf.append(reg_feat)           
            else:
                output_decode = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

               
                before_nms_features.append(cls_feat2)
                before_nms_regf.append(reg_feat)
            outputs_decode.append(output_decode)                       
            outputs_decode_nopostprocess.append(output_decode)         
       
        self.hw = [x.shape[-2:] for x in outputs_decode]               
        outputs_decode = torch.cat([x.flatten(start_dim=2) for x in outputs_decode], dim=2   
                                   ).permute(0, 2, 1)                      
        decode_res = self.decode_outputs(outputs_decode, dtype=xin[0].type())

        if self.kwargs.get('ota_mode',False) and self.training:
            ota_idxs,reg_targets = self.get_fg_idx( imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),)
        else:
            ota_idxs = None
       
        pred_result, pred_idx = self.postpro(decode_res, num_classes=self.num_classes,     
                                                     nms_thre=self.nms_thresh,
                                                     topK=self.Afternum,
                                                     ota_idxs=ota_idxs,
                                                     )

        if not self.training and imgs.shape[0] == 1:
            return self.postprocess_single_img(pred_result, self.num_classes)

       
        cls_feat_flatten = torch.cat(                  
            [x.flatten(start_dim=2) for x in before_nms_features], dim=2
        ).permute(0, 2, 1) 
        reg_feat_flatten = torch.cat(                  
            [x.flatten(start_dim=2) for x in before_nms_regf], dim=2
        ).permute(0, 2, 1)
       
        (features_cls, features_reg, cls_scores,           
         fg_scores, locs, all_scores) = self.selective_feature(cls_feat_flatten,
                                                                pred_idx,
                                                                reg_feat_flatten,
                                                                imgs,
                                                                pred_result)
        features_cls_att = self.temporal_attn(features_cls)        
        features_reg_att = self.temporal_attn(features_reg)        
        fc_output = self.linear_pred(features_cls_att) 
        fc_output = torch.reshape(fc_output, [-1, self.Afternum, self.num_classes + 1])[:, :, :-1]
        conf_output = self.conf_pred(features_reg_att) 
        conf_output = torch.reshape(conf_output, [-1, self.Afternum]) 
       
            
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
                refined_cls = fc_output,           
                idx=pred_idx,                      
                pred_res = pred_result,            
                conf_output = conf_output          
            )
        else:
           
           
           
           
           
           
           
           
           

           
            result, result_ori = postprocess(copy.deepcopy(pred_result),
                                             self.num_classes,
                                             fc_output,
                                             conf_output = conf_output,
                                             nms_thre=nms_thresh,
                                             )
            return result, result_ori 
           
        
   
   
   
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]           

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes            
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]: 
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid       

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize) 
        output = output.permute(0, 1, 3, 4, 2).reshape(                
            batch_size, self.n_anchors * hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)                         
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid 

   
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
   
   
   
   
   
   
    def selective_feature(self, features, idxs, reg_features, imgs=None, predictions=None, roi_features=None): 
        features_cls = []
        features_reg = []
        cls_scores, all_scores = [], []
        fg_scores = []
        locs = []
        for i, feature in enumerate(features):                              
            features_cls.append(feature[idxs[i][:self.simN]])               
            features_reg.append(reg_features[i, idxs[i][:self.simN]])       
            cls_scores.append(predictions[i][:self.simN, 5])                
            fg_scores.append(predictions[i][:self.simN, 4])                 
            locs.append(predictions[i][:self.simN, :4])                     
            all_scores.append(predictions[i][:self.simN, -self.num_classes:])

       
        features_cls = torch.stack(features_cls, dim=0)         
        features_reg = torch.stack(features_reg, dim=0)         
        cls_scores = torch.stack(cls_scores, dim=0)             
        fg_scores = torch.stack(fg_scores, dim=0)               
        locs = torch.stack(locs, dim=0)                         
        all_scores = torch.stack(all_scores, dim=0)             
       

       
       
       
       
       
       
       
       
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
        bbox_preds = outputs[:, :, :4] 
        obj_preds = outputs[:, :, 4].unsqueeze(-1) 
        cls_preds = outputs[:, :, 5:] 

       
        mixup = labels.shape[2] > 5     
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1) 

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1) 
        y_shifts = torch.cat(y_shifts, 1) 
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
        ref_masks = []         
        conf_targets = []      
        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                ref_target = outputs.new_zeros((idx[batch_idx].shape[0], self.num_classes + 1))        
                conf_target = outputs.new_zeros((idx[batch_idx].shape[0], 1))                          
                ref_target[:, -1] = 1                       

            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0] 
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments( 
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
                    ) = self.get_assignments( 
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
        ) 
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

       
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
                is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
       
       
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
        box_corner = prediction.new(prediction.shape)                      
        box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
        box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
        box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
        box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
        prediction[:, :, :4] = box_corner[:, :, :4]

        output = [None for _ in range(len(prediction))]            
        output_index = [None for _ in range(len(prediction))]      

        for i, image_pred in enumerate(prediction):
           
            if ota_idxs is not None and len(ota_idxs[i]) > 0:
                ota_idx = ota_idxs[i]
                topk_idx = torch.stack(ota_idx).type_as(image_pred)
                output[i] = image_pred[topk_idx, :]
                output_index[i] = topk_idx
                continue

            if not image_pred.size(0):
                continue
           
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
            output[i] = detections[topk_idx, :]            
            output_index[i] = topk_idx                     

        return output, output_index                        

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
       
        return output_ori, output_ori


    def get_fg_idx(self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
    ):
        bbox_preds = outputs[:, :, :4] 
        obj_preds = outputs[:, :, 4].unsqueeze(-1) 
        cls_preds = outputs[:, :, 5:] 

       
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[..., :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(dim=2) > 0).sum(dim=1) 

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1) 
        y_shifts = torch.cat(y_shifts, 1) 
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
                gt_classes = labels[batch_idx, :num_gt, 0] 
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments( 
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
                    ) = self.get_assignments( 
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
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(max_len, embed_dim) 
        pe[:, 0::2] = torch.sin(position * div_term) 
        pe[:, 1::2] = torch.cos(position * div_term) 
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, seq_len, embed_dim]
        Returns:
            Tensor: x + positional encoding
        """
       
       
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
        B, N, C = x.size() 
        
       
        x_transposed = x.permute(1, 0, 2) 
        
       
        x_transposed = self.pos_encoder(x_transposed) 
        
       
        attn_output, attn_weights = self.attn(x_transposed, x_transposed, x_transposed) 
        
       
        attn_output = self.layer_norm1(attn_output + x_transposed) 
        
       
        ffn_output = self.ffn(attn_output) 
        
       
        output = self.layer_norm2(ffn_output + attn_output) 
        
       
        output = output.permute(1, 0, 2) 
        
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
        mask = index[:, None] - index[None, :] 
        mask = mask.abs() 
        mask = mask * self.decay[:, None, None] 
        return mask 
    
    def forward(self, seq_len: int):            
       
        index = torch.arange(seq_len).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :]) 
        cos = torch.cos(index[:, None] * self.angle[None, :]) 
       
        mask = self.generate_decay_mask(seq_len) 
        return (sin, cos), mask
    
   
   
   
   
   
   
   
   
   
   
   
    
class SSTA(nn.Module):                        
    def __init__(self, embed_dim=128, num_heads=8, dropout=0.1, max_len=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
       
        self.pos_encoder = RetNetRelPos1d(
            embed_dim=embed_dim,
            num_heads=num_heads,
            initial_value=1,
            heads_range=3
        )
       
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
        B, N, C = x.size() 
        x = x.permute(1, 0, 2) 
        
       
        (sin, cos), mask = self.pos_encoder(seq_len=x.size(1)) 
        
       
        q = self.q_proj(x).view(x.size(0), x.size(1), self.num_heads, C // self.num_heads).permute(2, 0, 1, 3) 
        k = self.k_proj(x).view(x.size(0), x.size(1), self.num_heads, C // self.num_heads).permute(2, 0, 1, 3) 
        v = self.v_proj(x).view(x.size(0), x.size(1), self.num_heads, C // self.num_heads).permute(2, 0, 1, 3) 
        
       
        q = rotary_emb(q, sin, cos)  
        k = rotary_emb(k, sin, cos)  
        
       
        scores = torch.einsum("hnqd,hnkd->hnqk", q, k)  
        scores += mask.unsqueeze(1) 
        attn = torch.softmax(scores, dim=-1) 
        attn = self.dropout(attn) 
        
       
        output = torch.einsum("hnqk,hnkd->hnqd", attn, v)
        output = output.permute(1, 2, 0, 3).contiguous().view(x.size(0), x.size(1), C) 
        
       
        output = self.layer_norm1(output + x) 
        ffn_output = self.ffn(output)  
        output = self.layer_norm2(ffn_output + output) 
        
        return output.permute(1, 0, 2) 

def rotary_emb(x, sin, cos):
    x_rot = x * cos + torch.cat([-x[..., 1::2], x[..., ::2]], dim=-1) * sin
    return x_rot
