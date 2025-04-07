#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import os
import random

import torch
import torch.distributed as dist
import torch.nn as nn
from yolox.data.datasets import vid
from yolox.exp.base_exp import BaseExp
from yolox.data.data_augment import Vid_Val_Transform


class Exp(BaseExp):
    def __init__(self):
        super().__init__()
        self.archi_name = 'YOLOV'
        self.backbone_name = 'MCSP'
       
       
        self.num_classes = 1                               
       
        self.depth = 1.00
       
        self.width = 1.00
       
        self.act = "silu"
       
        self.pretrain_img_size = 224
        self.window_size = 7
       
        self.focal_level = 4
        self.focal_windows = 3
        self.focal_fpn_channels = [256, 512, 1024]

       
       
        self.drop_rate = 0
       
        self.head = 4
       
        self.defualt_p = 30                
       
        self.sim_thresh = 0.75          
       
        self.pre_nms = 0.75
       
        self.ave = True
       
        self.defualt_pre = 750
       
        self.use_score = True
       
        self.perspective = 0.0
       
        self.fix_bn = False
       
        self.use_aug = False
       
        self.use_mask = False
       
        self.fix_all = False
       
        self.gmode = True
       
        self.lmode = False
       
        self.both_mode = False
       
        self.lframe = 0
       
        self.lframe_val = 0
       
        self.localBlocks = 1
       
        self.gframe = 32
       
        self.gframe_val = 32
       
        self.tnum = -1
       
        self.local_stride = 1                
       
        self.iou_window = 0
       
        self.globalBlocks = 1

       
       
        self.use_ffn = True
       
        self.use_time_emd = False
       
        self.use_loc_emd = True
       
        self.loc_fuse_type = 'add'
       
        self.use_qkv = True
       
        self.local_mask = False
       
        self.local_mask_branch = ''
       
        self.pure_pos_emb = False
       
        self.loc_conf = False
       
        self.iou_base = False
       
        self.reconf = False
       
        self.ota_mode = False
       
        self.ota_cls = False
       
        self.traj_linking = False
       
        self.minimal_limit = 0
       
        self.vid_cls = True
       
        self.vid_reg = False
       
        self.conf_sim_thresh = 0.99

       
       
       
        self.data_num_workers = 12
        self.input_size = (576, 576) 
       
       
        self.multiscale_range = 5
       
       
       
        self.data_dir = '/data/yb/track/YOLOV-freq/IRDST'                       
       
        self.vid_train_path = '/data/yb/track/YOLOV-freq/IRDST/train.npy'  
        self.vid_val_path = '/data/yb/track/YOLOV-freq/IRDST/val.npy'      
       

       
       
        self.mosaic_prob = 1.0
       
        self.mixup_prob = 1.0
       
        self.hsv_prob = 1.0
       
        self.flip_prob = 0.5
       
        self.degrees = 10.0
       
        self.translate = 0.1
        self.mosaic_scale = (0.1, 2)
       
        self.enable_mixup = True
        self.mixup_scale = (0.5, 1.5)
       
        self.shear = 2.0

       

       
        self.warmup_epochs = 1
       
        self.max_epoch = 20
       
        self.warmup_lr = 0
        self.min_lr_ratio = 0.1
       
        self.basic_lr_per_img = 0.002 / 64.0
       
        self.scheduler = "yoloxwarmcos"
       
        self.no_aug_epochs = 2
       
        self.ema = True

       
        self.weight_decay = 5e-4
       
        self.momentum = 0.9
       
       
        self.print_interval = 10
       
       
        self.eval_interval = 1
       
       
        self.save_history_ckpt = True
       
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

       
       
        self.test_size = (576, 576)
       
       
        self.test_conf = 0.001
       
        self.nmsthre = 0.5

    def get_model(self):
       
        if self.backbone_name == 'MCSP':
            in_channels = [256, 512, 1024]
            from yolox.models import PAFPN
            backbone = PAFPN(self.depth, self.width, in_channels=in_channels)
        elif 'Swin' in self.backbone_name:
            from yolox.models import YOLOPAFPN_Swin

            if self.backbone_name == 'Swin_Tiny':
                in_channels = [192, 384, 768]
                out_channels = [192, 384, 768]
                backbone = YOLOPAFPN_Swin(in_channels=in_channels,
                                          out_channels=out_channels,
                                          act=self.act,
                                          in_features=(1, 2, 3))
            elif self.backbone_name == 'Swin_Base':
                in_channels = [256, 512, 1024]
                out_channels = [256, 512, 1024]
                backbone = YOLOPAFPN_Swin(in_channels=in_channels,
                                          out_channels=out_channels,
                                          act=self.act,
                                          in_features=(1, 2, 3),
                                          swin_depth=[2, 2, 18, 2],
                                          num_heads=[4, 8, 16, 32],
                                          base_dim=int(in_channels[0] / 2),
                                          pretrain_img_size=self.pretrain_img_size,
                                          window_size=self.window_size,
                                          width=self.width,
                                          depth=self.depth
                                          )
        elif 'Focal' in self.backbone_name:
            from yolox.models import YOLOPAFPN_focal
            fpn_in_channles = [96 * 4, 96 * 8, 96 * 16]
            in_channels = self.focal_fpn_channels
            backbone = YOLOPAFPN_focal(in_channels=fpn_in_channles,
                                       out_channels=in_channels,
                                       act=self.act,
                                       in_features=(1, 2, 3),
                                       depths=[2, 2, 18, 2],
                                       focal_levels=[4, 4, 4, 4],
                                       focal_windows=[3, 3, 3, 3],
                                       use_conv_embed=True,
                                       use_postln=True,
                                       use_postln_in_modulation=False,
                                       use_layerscale=True,
                                       base_dim=192, 
                                       depth=self.depth,
                                       width=self.width
                                       )


        else:
            raise NotImplementedError('backbone not support')
        from yolox.models.yolovp_msa import YOLOXHead
        from yolox.models.v_plus_head import YOLOVHead
        from yolox.models.yolov_plus import YOLOV
        from yolox.models.myolox import YOLOX

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03


        for layer in backbone.parameters():
            layer.requires_grad = False 
        
       
        more_args = {'use_ffn': self.use_ffn, 'use_time_emd': self.use_time_emd, 'use_loc_emd': self.use_loc_emd,
                     'loc_fuse_type': self.loc_fuse_type, 'use_qkv': self.use_qkv,
                     'local_mask': self.local_mask, 'local_mask_branch': self.local_mask_branch,
                     'pure_pos_emb':self.pure_pos_emb,'loc_conf':self.loc_conf,'iou_base':self.iou_base,
                     'reconf':self.reconf,'ota_mode':self.ota_mode,'ota_cls':self.ota_cls,'traj_linking':self.traj_linking,
                     'iou_window':self.iou_window,'globalBlocks':self.globalBlocks,'minimal_limit':self.minimal_limit,
                     'vid_cls':self.vid_cls,'vid_reg':self.vid_reg,'conf_sim_thresh':self.conf_sim_thresh,
                     }
        head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, heads=self.head, drop=self.drop_rate,
                         use_score=self.use_score, defualt_p=self.defualt_p, sim_thresh=self.sim_thresh,
                         pre_nms=self.pre_nms, ave=self.ave, defulat_pre=self.defualt_pre, test_conf=self.test_conf,
                         use_mask=self.use_mask,gmode=self.gmode,lmode=self.lmode,both_mode=self.both_mode,
                         localBlocks = self.localBlocks,**more_args)

       
        for layer in backbone.parameters():
            layer.requires_grad = False 

        

        self.model = YOLOX(backbone, head)


        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()

        self.model.apply(init_yolo)
        if self.fix_bn:
            self.model.apply(fix_bn)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(
            self, batch_size, is_distributed, no_aug=False, cache_img=False
    ):
        from yolox.data import TrainTransform
        from yolox.data.datasets.mosaicdetection import MosaicDetection_VID
        assert batch_size == self.lframe + self.gframe
        dataset = vid.VIDDataset(file_path=self.vid_train_path,
                                 img_size=self.input_size,
                                 preproc=TrainTransform(
                                     max_labels=50,
                                     flip_prob=self.flip_prob,
                                     hsv_prob=self.hsv_prob),
                                 lframe=self.lframe, 
                                 gframe=self.gframe,
                                 dataset_pth=self.data_dir,
                                 local_stride=self.local_stride,
                                 )
        if self.use_aug:
           
            dataset = MosaicDetection_VID(
                dataset,
                mosaic=False,
                img_size=self.input_size,
                preproc=TrainTransform(
                    max_labels=120,
                    flip_prob=self.flip_prob,
                    hsv_prob=self.hsv_prob),
                degrees=self.degrees,
                translate=self.translate,
                mosaic_scale=self.mosaic_scale,
                mixup_scale=self.mixup_scale,
                shear=self.shear,
                perspective=self.perspective,
                enable_mixup=self.enable_mixup,
                mosaic_prob=self.mosaic_prob,
                mixup_prob=self.mixup_prob,
                dataset_path=self.data_dir
            )
        dataset = vid.get_trans_loader(batch_size=batch_size, data_num_workers=4, dataset=dataset)
        return dataset

    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(
                inputs, size=tsize, mode="bilinear", align_corners=False
            )
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], [] 

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias) 
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight) 
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight) 

            optimizer = torch.optim.SGD(
                pg0, lr=lr, momentum=self.momentum, nesterov=True
            )
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            ) 
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):
        from yolox.utils import LRScheduler

        scheduler = LRScheduler(
            self.scheduler,
            lr,
            iters_per_epoch,
            self.max_epoch,
            warmup_epochs=self.warmup_epochs,
            warmup_lr_start=self.warmup_lr,
            no_aug_epochs=self.no_aug_epochs,
            min_lr_ratio=self.min_lr_ratio,
        )
        return scheduler

    def get_eval_loader(self, batch_size, tnum=None, data_num_workers=8,formal=False):
        if tnum == None:
            tnum = self.tnum
        assert batch_size == self.lframe_val+self.gframe_val
        dataset_val = vid.VIDDataset(file_path=self.vid_val_path,
                                     img_size=self.test_size, preproc=Vid_Val_Transform(), lframe=self.lframe_val,
                                     gframe=self.gframe_val, val=True, dataset_pth=self.data_dir, tnum=tnum,formal=formal,
                                     traj_linking=self.traj_linking, local_stride=self.local_stride,)
        val_loader = vid.vid_val_loader(batch_size=batch_size,
                                        data_num_workers=data_num_workers,
                                        dataset=dataset_val, )

        return val_loader

   
    def get_evaluator(self, val_loader):
        from yolox.evaluators.vid_evaluator_v2 import VIDEvaluator

       
        evaluator = VIDEvaluator(
            dataloader=val_loader,
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
            lframe=self.lframe_val,
            gframe=self.gframe_val,
            first_only = False,
        )
        return evaluator

    def get_trainer(self, args):
        from yolox.core import Trainer
        trainer = Trainer(self, args)
       
        return trainer

    def eval(self, model, evaluator, is_distributed, half=False):
        return evaluator.evaluate(model, is_distributed, half)
