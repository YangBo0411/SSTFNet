#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import torch.nn as nn

class YOLOX(nn.Module):
    """
    YOLOX model module. The module list is defined by create_yolov3_modules function.
    The network returns loss values from three YOLO layers during training
    and detection results during test.
    """

    def __init__(self, backbone=None, head=None):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x, targets=None,nms_thresh=0.5,lframe=0,gframe=32):    # x 16 3 576 576   target 16 120 5  120表示设定的最大标签数量值
        # fpn output content features of [dark3, dark4, dark5]
        fpn_outs = self.backbone(x)    # fpn_outs list3   16 128 72 72  16 256 36 36  16 5125 18 18
        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, rconf_loss,num_fg = self.head(
                fpn_outs, targets, x, lframe=lframe,gframe=gframe
            )
            # loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(fpn_outs, targets, x)  # 源码

            outputs = {                  # 改动
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "rconf_loss":rconf_loss,
                "num_fg": num_fg,
            }
            # outputs = {                     # 源码
            #     "total_loss": loss,
            #     "iou_loss": iou_loss,
            #     "l1_loss": l1_loss,
            #     "conf_loss": conf_loss,
            #     "cls_loss": cls_loss,
            #     "num_fg": num_fg,
            # }
        else:

            outputs = self.head(fpn_outs,targets,x,nms_thresh=nms_thresh, lframe=lframe,gframe=gframe)   # 改动
            # outputs = self.head(fpn_outs)   # 源码

        return outputs
