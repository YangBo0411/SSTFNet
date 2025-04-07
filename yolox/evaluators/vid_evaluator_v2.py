#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import copy
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tqdm import tqdm
from yolox.evaluators.coco_evaluator import per_class_AR_table, per_class_AP_table
import torch
import pycocotools.coco
from pycocotools.coco import COCO
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt

from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)

vid_classes = (                             #yb
    'target'                
)


# from yolox.data.datasets.vid_classes import Arg_classes as  vid_classes

class VIDEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
            self, dataloader, img_size, confthre, nmsthre,
            num_classes, testdev=False, gl_mode=False,
            lframe=0, gframe=32,**kwargs
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size (int): image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre (float): confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre (float): IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.id = 0
        self.box_id = 0
        self.id_ori = 0
        self.box_id_ori = 0
        self.gl_mode = gl_mode
        self.lframe = lframe
        self.gframe = gframe
        self.kwargs = kwargs
        self.global_sample_count = 0  #yb 特征可视化
        self.vid_to_coco = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{"supercategorie": "", "id": 0, "name": "target"},       #yb
                           ],
            'images': [],
            'licenses': []
        }
        self.vid_to_coco_ori = {
            'info': {
                'description': 'nothing',
            },
            'annotations': [],
            'categories': [{"supercategorie": "", "id": 0, "name": "target"},       #yb
                           ],
            'images': [],
            'licenses': []
        }
        self.testdev = testdev
        self.tmp_name_ori = './ori_pred.json'
        self.tmp_name_refined = './refined_pred.json'
        self.gt_ori = './gt_ori.json'
        self.gt_refined = './gt_refined.json'

    #yb 特征可视化
    def visualize_feature_maps(self, model, imgs, img_paths, save_dir="feature_maps"):
        """
        可视化模型的主要层特征图。
        Args:
            model: 目标检测模型
            imgs: 经过预处理的输入图像
            save_dir: 保存可视化图像的路径
        """
        os.makedirs(save_dir, exist_ok=True)
        outputs_pafpn = model.backbone(imgs)
        outputs_sfnet = model.backbone.backbone(imgs)

        activation_maps = {}
        
        def hook_fn(module, input, output, layer_name):
            activation_maps[layer_name] = output
        hooks = []

        hook = model.backbone.backbone.dark2.register_forward_hook(
        lambda m, i, o, ln="dark2": hook_fn(m, i, o, ln)
    )
        hooks.append(hook)

        for i, layer in enumerate(model.head.cls_convs):
            layer_name = f"cls_conv_{i}"
            hook = layer.register_forward_hook(lambda m, i, o, ln=layer_name: hook_fn(m, i, o, ln))
            hooks.append(hook)
    
        for i, layer in enumerate(model.head.reg_convs):
            layer_name = f"reg_conv_{i}"
            hook = layer.register_forward_hook(lambda m, i, o, ln=layer_name: hook_fn(m, i, o, ln))
            hooks.append(hook)
        
        for i, layer in enumerate(model.head.cls_preds):
            layer_name = f"cls_pred_{i}"
            hook = layer.register_forward_hook(lambda m, i, o, ln=layer_name: hook_fn(m, i, o, ln))
            hooks.append(hook)
        
        for i, layer in enumerate(model.head.reg_preds):
            layer_name = f"reg_pred_{i}"
            hook = layer.register_forward_hook(lambda m, i, o, ln=layer_name: hook_fn(m, i, o, ln))
            hooks.append(hook)
        
        for i, layer in enumerate(model.head.obj_preds):
            layer_name = f"obj_pred_{i}"
            hook = layer.register_forward_hook(lambda m, i, o, ln=layer_name: hook_fn(m, i, o, ln))
            hooks.append(hook)

        # 运行前向传播触发钩子
        with torch.no_grad():
            _ = model(imgs)

        # 选择关键层，通常可以选择 backbone 的某些层
        selected_layers = {
            # Backbone层
            "dark2": activation_maps.get("dark2", None),
            "dark3": outputs_sfnet["dark3"],
            "dark4": outputs_sfnet["dark4"],
            "dark5": outputs_sfnet["dark5"],
            "pan_out2": outputs_pafpn[0],
            "pan_out1": outputs_pafpn[1],
            "pan_out0": outputs_pafpn[2],
            # Head层
            "cls_conv_0": activation_maps.get("cls_conv_0", None),
            "reg_conv_0": activation_maps.get("reg_conv_0", None),
            "cls_pred_0": activation_maps.get("cls_pred_0", None),
            "reg_pred_0": activation_maps.get("reg_pred_0", None),
            "obj_pred_0": activation_maps.get("obj_pred_0", None),
        }

        activation_maps = {}

        if img_paths:
            img_path = img_paths[0]
            img_name = os.path.basename(img_path)  # 获取图片文件名
            print(f"可视化图片: {img_name}, 路径: {img_path}")
            # for i, img_path in enumerate(img_paths):
            #     img_name = os.path.basename(img_path)  # 获取图片文件名
                # print(f"可视化图片: {img_name}, 路径: {img_path}")

                # 如果需要保存特征图：
                # feature_map = feature_maps["dark3"][i].cpu().numpy()  # 转 NumPy
                # save_path = f"./feature_maps/{img_name}_dark3.png"
                # plt.imsave(save_path, feature_map[0])  # 只保存第一通道
                # print(f"特征图已保存到: {save_path}")
            for layer_name, layer_output in selected_layers.items():
                activation = layer_output[i].cpu().numpy()  # 取 batch 中的第 i 张
                num_channels = activation.shape[0]

                fig, axes = plt.subplots(1, min(8, num_channels), figsize=(20, 5))
                if min(8, num_channels) == 1:
                    axes = [axes]
                for j in range(min(8, num_channels)):  # 可视化最多 8 个通道
                    ax = axes[j]
                    ax.imshow(activation[j], cmap='jet')
                    ax.axis('off')
                    ax.set_title(f"Channel {j}")

                plt.suptitle(f"Feature Maps - {layer_name}")
                save_path = os.path.join(save_dir, f"{img_name}_{layer_name}.png")
                plt.savefig(save_path, dpi=900)
                plt.close()
        # 移除 hooks
        for hook in hooks:
            hook.remove()
    #yb 特征可视化

    def evaluate(
            self,
            model,
            distributed=False,
            half=True,
            trt_file=None,
            decoder=None,
            test_size=None,
            img_path=None
    ):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.

        NOTE: This function will change training mode to False, please save states if needed.

        Args:
            model : model to evaluate.

        Returns:
            ap50_95 (float) : COCO AP of IoU=50:95
            ap50 (float) : COCO AP of IoU=50
            summary (sr): summary info of evaluation.
        """

        # TODO half to amp_test
        tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
        model = model.eval()
        if half:
            model = model.half()
        ids = []
        data_list = []
        labels_list = []
        ori_data_list = []
        ori_label_list = []
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        for cur_iter, (imgs, _, info_imgs, label, path, time_embedding) in enumerate(
                progress_bar(self.dataloader)
        ):
                            
            #yb 特征可视化
            batch_size = imgs.size(0)
            start_count = self.global_sample_count
            self.global_sample_count += batch_size  # 更新计数器
            #yb 特征可视化

            with torch.no_grad():
                imgs = imgs.type(tensor_type)
                # skip the the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()
                outputs, ori_res = model(imgs,
                                         lframe=self.lframe,
                                         gframe = self.gframe)

                # #yb 特征可视化
                # if self.global_sample_count // 1000 > start_count // 1000:
                #     self.visualize_feature_maps(model, imgs, path)          
                # #yb 特征可视化


                # # #---- yb ---------------  # yb  保存预测结果
                # output_dir = "results"
                # os.makedirs(output_dir, exist_ok=True)
                # confidence_threshold = 0.5
                # for i, output in enumerate(outputs):
                #     img_path = path[i]  # 获取图像路径
                #     img = cv2.imread(img_path)  # 读取原始图像
                #     if img is None:
                #         continue  # 避免图像读取失败

                #     scale = min(self.img_size[0] / float(info_imgs[i][0]), self.img_size[1] / float(info_imgs[i][1]))
                #     if output is not None and len(output) > 0:
                #         output = output.cpu()
                #         bboxes = output[:, 0:4] / scale  # 还原到原始尺寸
                #         bboxes = bboxes.numpy().astype(int)
                #         scores = (output[:, 4] * output[:, 5]).numpy()  # 置信度
                #         cls_ids = output[:, 6].numpy().astype(int)  # 类别ID
                        
                #         valid_indices = scores >= confidence_threshold
                #         if np.sum(valid_indices) == 0:
                #             # print(f"Warning: No valid detections in {img_path}, skipping...")
                #             continue  # 跳过当前帧，避免 bboxes 为空导致 IndexError
                #         bboxes = bboxes[valid_indices]
                #         scores = scores[valid_indices]
                #         cls_ids = cls_ids[valid_indices]
                #         # 绘制检测框
                #         for box, score, cls_id in zip(bboxes, scores, cls_ids):
                #             x1, y1, x2, y2 = box
                #             label_s = f"{'target'}: {score:.2f}"
                #             color = (0, 255, 0)  # 绿色边框
                #             cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                #             cv2.putText(img, label_s, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                #     subfolder_name = os.path.basename(os.path.dirname(img_path))
                #     img_name = os.path.basename(img_path) 
                #     subfolder_save_dir = os.path.join(output_dir, subfolder_name)  # 例如 'pred_results/2'
                #     os.makedirs(subfolder_save_dir, exist_ok=True)
                #     new_img_name = f"pred_{subfolder_name}_{img_name}"
                #     save_path = os.path.join(subfolder_save_dir, new_img_name)
                #     cv2.imwrite(save_path, img)  # 保存带预测框的图像
                # # #---- yb --------------- 

                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
            if self.gl_mode:
                local_num = int(imgs.shape[0] / 2)
                info_imgs = info_imgs[:local_num]
                label = label[:local_num]
            if self.kwargs.get('first_only',False):
                info_imgs = [info_imgs[0]]
                label = [label[0]]
            temp_data_list, temp_label_list = self.convert_to_coco_format(outputs, info_imgs, copy.deepcopy(label))
            data_list.extend(temp_data_list)
            labels_list.extend(temp_label_list)


        self.vid_to_coco['annotations'].extend(labels_list)
        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)

        del labels_list
        eval_results = self.evaluate_prediction(data_list, statistics)
        del data_list
        self.vid_to_coco['annotations'] = []

        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, labels):
        data_list = []
        label_list = []
        frame_now = 0

        for (output, info_img, _label) in zip(
                outputs, info_imgs, labels
        ):
            # if frame_now>=self.lframe: break
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id),
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3])
                }  # COCO json format
                self.box_id = self.box_id + 1
                label_list.append(label_pred_data)
            self.vid_to_coco['images'].append({'id': self.id})

            if output is None:
                self.id = self.id + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]
            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]

            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
            self.id = self.id + 1
            frame_now = frame_now + 1

        return data_list, label_list

    def convert_to_coco_format_ori(self, outputs, info_imgs, labels):

        data_list = []
        label_list = []
        frame_now = 0
        for (output, info_img, _label) in zip(
                outputs, info_imgs, labels
        ):
            scale = min(
                self.img_size[0] / float(info_img[0]), self.img_size[1] / float(info_img[1])
            )
            bboxes_label = _label[:, 1:]
            bboxes_label /= scale
            bboxes_label = xyxy2xywh(bboxes_label)
            cls_label = _label[:, 0]
            for ind in range(bboxes_label.shape[0]):
                label_pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": int(cls_label[ind]),
                    "bbox": bboxes_label[ind].numpy().tolist(),
                    "segmentation": [],
                    'id': self.box_id_ori,
                    "iscrowd": 0,
                    'area': int(bboxes_label[ind][2] * bboxes_label[ind][3])
                }  # COCO json format
                self.box_id_ori = self.box_id_ori + 1
                label_list.append(label_pred_data)

                # print('label:',label_pred_data)

            self.vid_to_coco_ori['images'].append({'id': self.id_ori})

            if output is None:
                self.id_ori = self.id_ori + 1
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            # print(cls.shape)
            for ind in range(bboxes.shape[0]):
                label = int(cls[ind])
                pred_data = {
                    "image_id": int(self.id_ori),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)

            self.id_ori = self.id_ori + 1
            frame_now = frame_now + 1
        return data_list, label_list

    def evaluate_prediction(self, data_dict, statistics, ori=False):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_sampler.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_sampler.batch_size)
        time_info = ", ".join(
            [
                "Average {} time: {:.2f} ms".format(k, v)
                for k, v in zip(
                ["forward", "NMS", "inference"],
                [a_infer_time, a_nms_time, (a_infer_time + a_nms_time)],
            )
            ]
        )

        info = time_info + "\n"

        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:

            _, tmp = tempfile.mkstemp()
            if ori:
                json.dump(self.vid_to_coco_ori, open(self.gt_ori, 'w'))
                json.dump(data_dict, open(self.tmp_name_ori, 'w'))
                json.dump(self.vid_to_coco_ori, open(tmp, "w"))
            else:
                json.dump(self.vid_to_coco, open(self.gt_refined, 'w'))
                json.dump(data_dict, open(self.tmp_name_refined, 'w'))
                json.dump(self.vid_to_coco, open(tmp, "w"))

            cocoGt = pycocotools.coco.COCO(tmp)
            # TODO: since pycocotools can't process dict in py36, write data to json file.

            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, "w"))
            cocoDt = cocoGt.loadRes(tmp)
            try:
                from yolox.layers import COCOeval_opt as COCOeval
            except ImportError:
                from pycocotools.cocoeval import COCOeval

                logger.warning("Use standard COCOeval.")

            cocoEval = COCOeval(cocoGt, cocoDt, annType[1])
            cocoEval.evaluate()
            cocoEval.accumulate()

            #yb
            precisions = cocoEval.eval['precision']
            precision_50 = precisions[0,:,0,0,-1]  # 第三为类别 (T,R,K,A,M)
            recalls = cocoEval.eval['recall']
            recall_50 = recalls[0,0,0,-1] # 第二为类别 (T,K,A,M)
            print("Precision: %.4f, Recall: %.4f, F1: %.4f" %(np.mean(precision_50[:int(recall_50*100)]), recall_50, 2*recall_50*np.mean(precision_50[:int(recall_50*100)])/( recall_50+np.mean(precision_50[:int(recall_50*100)]))))
            print("Get map done.")
            with open("results.txt", 'w') as f:
                for pred in precision_50:
                    f.writelines(str(pred)+'\t')
            #yb

            redirect_string = io.StringIO()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
            info += "per class AP:\n" + AP_table + "\n"

            AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
            info += "per class AR:\n" + AR_table + "\n"
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
