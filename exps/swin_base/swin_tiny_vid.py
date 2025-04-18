import os
import torch
import torch.nn as nn

from yolox.exp import Exp as MyExp
# from yolox import YOLOX

class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.data_dir = '/home/wuren123/yb/track/other-model/YOLOV/IRDST/Data' #'/mnt/weka/scratch/datasets/coco' # #yb
        self.train_ann = "/home/wuren123/yb/track/other-model/YOLOV/IRDST/Annotations/IRDST_instances_train2017.json"        #yb
        # name of annotation file for evaluation
        self.val_ann = "/home/wuren123/yb/track/other-model/YOLOV/IRDST/Annotations/IRDST_instances_test2017.json"            #yb
        #self.val_ann = "vid_val10000_coco_fg.json"
        self.basic_lr_per_img = 0.0005 / 64.0
        self.save_history_ckpt = False
        self.max_epoch = 50
        self.input_size = (640,640)
        self.test_size = (640,640)
        self.eval_interval = 1
        self.warmup_epochs = 1
        self.no_aug_epochs = 7
        self.num_classes = 1
        self.test_conf = 0.001
        self.train_name = ''
        self.val_name = ''

        # #----yb
        # self.lframe = 0
        # #lframe for validation
        # self.lframe_val = 0
        # #local block number
        # self.localBlocks = 1
        # #global frames for training
        # self.gframe = 16
        # #globale frames for validation
        # self.gframe_val = 32
        # #----yb

    def get_model(self):
        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        in_channels = [192, 384, 768]
        out_channels = [192, 384, 768]
        from yolox.models.yolox import YOLOX
        from yolox.models.yolo_pafpn import YOLOPAFPN_Swin
        from yolox.models.yolo_head import YOLOXHead
        # from yolox.models import YOLOX, YOLOPAFPN_Swin, YOLOXHead
        backbone = YOLOPAFPN_Swin(in_channels=in_channels, out_channels=out_channels, act=self.act,in_features=(1,2,3))
        head = YOLOXHead(self.num_classes, self.width, in_channels=out_channels, act=self.act)
        self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)

        return self.model

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size
            pg0, pg1, pg2,pg3 = [], [], [], []  # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias)  # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight)  # no decay
                elif hasattr(v,'absolute_pos_embed') or hasattr(v,'relative_position_bias_table') or hasattr(v,'norm'):
                    if hasattr(v,'weight'):
                        pg3.append(v.weight)
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight)  # apply decay

            optimizer = torch.optim.AdamW(params=pg0,lr=lr,weight_decay=self.weight_decay)
            optimizer.add_param_group(
                {"params": pg1, "weight_decay": self.weight_decay}
            )  # add pg1 with weight_decay
            optimizer.add_param_group(
                {"params": pg3, "weight_decay": 0}
            )  # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})

            self.optimizer = optimizer

        return self.optimizer

    #
    # def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
    #     from yolox.evaluators import COCOEvaluator
    #
    #     val_loader = self.get_eval_loader(batch_size, is_distributed, testdev, legacy)
    #     evaluator = COCOEvaluator(
    #         dataloader=val_loader,
    #         img_size=self.test_size,
    #         confthre=self.test_conf,
    #         nmsthre=self.nmsthre,
    #         num_classes=self.num_classes,
    #         testdev=testdev,
    #         fg_AR_only=True,
    #     )
    #     return evaluator