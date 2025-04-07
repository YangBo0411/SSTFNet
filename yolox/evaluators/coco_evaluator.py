#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import contextlib
import io
import itertools
import json
import tempfile
import time
from loguru import logger
from tabulate import tabulate
from tqdm import tqdm

import numpy as np

import torch
import os
import matplotlib.pyplot as plt
import cv2



from yolox.data.datasets import COCO_CLASSES
from yolox.models.post_process import postpro_woclass,post_threhold
from yolox.utils import (
    gather,
    is_main_process,
    postprocess,
    synchronize,
    time_synchronized,
    xyxy2xywh
)


def per_class_AR_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AR"], colums=6):
    per_class_AR = {}
    recalls = coco_eval.eval["recall"]
    # dimension of recalls: [TxKxAxM]
    # recall has dims (iou, cls, area range, max dets)
    assert len(class_names) == recalls.shape[1]

    for idx, name in enumerate(class_names):
        recall = recalls[:, idx, 0, -1]
        recall = recall[recall > -1]
        ar = np.mean(recall) if recall.size else float("nan")
        per_class_AR[name] = float(ar * 100)

    num_cols = min(colums, len(per_class_AR) * len(headers))
    result_pair = [x for pair in per_class_AR.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table


def per_class_AP_table(coco_eval, class_names=COCO_CLASSES, headers=["class", "AP"], colums=6):
    per_class_AP = {}
    precisions = coco_eval.eval["precision"]
    # dimension of precisions: [TxRxKxAxM]
    # precision has dims (iou, recall, cls, area range, max dets)
    assert len(class_names) == precisions.shape[2]

    for idx, name in enumerate(class_names):
        # area range index 0: all area ranges
        # max dets index -1: typically 100 per image
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        ap = np.mean(precision) if precision.size else float("nan")
        per_class_AP[name] = float(ap * 100)

    num_cols = min(colums, len(per_class_AP) * len(headers))
    result_pair = [x for pair in per_class_AP.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table_headers = headers * (num_cols // len(headers))
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f", headers=table_headers, numalign="left",
    )
    return table

# def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#     """
#     逆归一化图像数据，将归一化后的图像转换回可视化格式
#     """
#     print(tensor.min(), tensor.max())
#     print(torch.isnan(tensor).any())
#     mean = torch.tensor(mean).view(1, 3, 1, 1).to(tensor.device)
#     std = torch.tensor(std).view(1, 3, 1, 1).to(tensor.device)
#     tensor = tensor * std + mean  # 逆归一化
#     return tensor.clamp(0, 1)  # 限制到 [0, 1]

# def save_image(img_tensor, img_name, save_dir="feature_maps"):
#     """
#     保存原始输入图像
#     """
#     img_np = img_tensor.permute(1, 2, 0).cpu().numpy()  # 转换为 HWC 格式
#     plt.imsave(os.path.join(save_dir, f"{img_name}.png"), img_np)

def draw_boxes(img, outputs, save_path):
    """
    在原始图像上绘制预测的边界框，并保存到 `save_path`
    """
    for det in outputs:
        x1, y1, x2, y2, conf, cls = det[:6]
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = (0, 255, 0)  # 绿色
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{int(cls)}: {conf:.2f}"
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    cv2.imwrite(save_path, img)

class COCOEvaluator:
    """
    COCO AP Evaluation class.  All the data in the val2017 dataset are processed
    and evaluated by COCO API.
    """

    def __init__(
        self,
        dataloader,
        img_size: int,
        confthre: float,
        nmsthre: float,
        num_classes: int,
        testdev: bool = False,
        per_class_AP: bool = False,
        per_class_AR: bool = False,
        fg_AR_only: bool = False,
    ):
        """
        Args:
            dataloader (Dataloader): evaluate dataloader.
            img_size: image size after preprocess. images are resized
                to squares whose shape is (img_size, img_size).
            confthre: confidence threshold ranging from 0 to 1, which
                is defined in the config file.
            nmsthre: IoU threshold of non-max supression ranging from 0 to 1.
            per_class_AP: Show per class AP during evalution or not. Default to False.
            per_class_AR: Show per class AR during evalution or not. Default to False.
        """
        self.dataloader = dataloader
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.num_classes = num_classes
        self.testdev = testdev
        self.per_class_AP = per_class_AP
        self.per_class_AR = per_class_AR
        self.fg_AR_only = fg_AR_only
        self.global_sample_count = 0 #yb 特征可视化
    
    def draw_boxes(self, img, outputs, save_path):
        """
        在原始图像上绘制预测的边界框，并保存到 `save_path`
        """
        for det in outputs:
            x1, y1, x2, y2, conf, cls = det[:6]
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{int(cls)}: {conf:.2f}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imwrite(save_path, img)

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
            for i, img_path in enumerate(img_paths):                      # 保存bs中所有的特征图，仅保留1张图片时注释该句
                img_name = os.path.basename(img_path)  # 获取图片文件名    # 保存bs中所有的特征图，仅保留1张图片时注释该句
                print(f"可视化图片: {img_name}, 路径: {img_path}")          # 保存bs中所有的特征图，仅保留1张图片时注释该句

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
        half=False,
        trt_file=None,
        decoder=None,
        test_size=None,
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
        progress_bar = tqdm if is_main_process() else iter

        inference_time = 0
        nms_time = 0
        n_samples = max(len(self.dataloader) - 1, 1)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, test_size[0], test_size[1]).cuda()
            model(x)
            model = model_trt
        self.image_index_map = {}   #yb 可视化
        for cur_iter, (imgs, _, info_imgs, ids, path) in enumerate(
            progress_bar(self.dataloader)
        ):
            
            #yb 特征可视化
            batch_size = imgs.size(0)
            start_count = self.global_sample_count
            self.global_sample_count += batch_size  # 更新计数器
            #yb 特征可视化

            with torch.no_grad():
                imgs = imgs.type(tensor_type)

                # skip the last iters since batchsize might be not enough for batch inference
                is_time_record = cur_iter < len(self.dataloader) - 1
                if is_time_record:
                    start = time.time()

                outputs = model(imgs)

                #yb 特征可视化
                # if self.global_sample_count // 1000 > start_count // 1000:
                    # self.visualize_feature_maps(model, imgs, path)          
                #yb 特征可视化


                if decoder is not None:
                    outputs = decoder(outputs, dtype=outputs.type())


                if is_time_record:
                    infer_end = time_synchronized()
                    inference_time += infer_end - start
                
                if not self.fg_AR_only:
                    outputs = postprocess(
                        outputs, self.num_classes, self.confthre, self.nmsthre
                    )
                else:
                    outputs = post_threhold(
                        outputs, self.num_classes,
                    )
                if is_time_record:
                    nms_end = time_synchronized()
                    nms_time += nms_end - infer_end
                
                # #---- yb ---------------  # yb  保存预测结果    
                output_dir = "prediction-origin"
                os.makedirs(output_dir, exist_ok=True)
                confidence_threshold = 0.5
                # if len(outputs) != len(info_imgs) or len(outputs) != len(path):
                #     raise ValueError("Length of outputs, info_imgs and path must be the same.")
                # for i, (output, img_info, img_path) in enumerate(zip(outputs, info_imgs, path)):
                for output, img_info, img_path in zip(outputs, info_imgs, path):
                    # img_path = path[i]  # 获取图像路径
                    img = cv2.imread(img_path)  # 读取原始图像
                    if img is None:
                        continue  # 避免图像读取失败
                    h_orig, w_orig = img.shape[:2]
                    scale = min(self.img_size[0] / h_orig, self.img_size[1] / w_orig)
                    if output is not None and len(output) > 0:
                        output = output.cpu()
                        bboxes = output[:, 0:4] / scale  # 还原到原始尺寸
                        bboxes = bboxes.numpy().astype(int)
                        scores = (output[:, 4] * output[:, 5]).numpy()  # 置信度
                        cls_ids = output[:, 6].numpy().astype(int)  # 类别ID
                        
                        valid_indices = scores >= confidence_threshold
                        if np.sum(valid_indices) == 0:
                            print(f"Warning: No valid detections in {img_path}, skipping...")
                            continue  # 跳过当前帧，避免 bboxes 为空导致 IndexError
                        bboxes = bboxes[valid_indices]
                        scores = scores[valid_indices]
                        cls_ids = cls_ids[valid_indices]
                        # 绘制检测框
                        for box, score, cls_id in zip(bboxes, scores, cls_ids):
                            x1, y1, x2, y2 = box
                            label_s = f"{'target'}: {score:.2f}"
                            color = (0, 255, 0)  # 绿色边框
                            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(img, label_s, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    subfolder_name = os.path.basename(os.path.dirname(img_path))
                    img_name = os.path.basename(img_path) 
                    subfolder_save_dir = os.path.join(output_dir, subfolder_name)  # 例如 'pred_results/2'
                    os.makedirs(subfolder_save_dir, exist_ok=True)
                    new_img_name = f"pred_{subfolder_name}_{img_name}"
                    save_path = os.path.join(subfolder_save_dir, new_img_name)
                    cv2.imwrite(save_path, img)  # 保存带预测框的图像
            # #---- yb ---------------  # yb  保存预测结果


            data_list.extend(self.convert_to_coco_format(outputs, info_imgs, ids))

        statistics = torch.cuda.FloatTensor([inference_time, nms_time, n_samples])
        if distributed:
            data_list = gather(data_list, dst=0)
            data_list = list(itertools.chain(*data_list))
            torch.distributed.reduce(statistics, dst=0)
        #calculate average predictions per image
        if is_main_process():
            logger.info("average predictions per image: {:.2f}".format(len(data_list)/len(self.dataloader.dataset)))
        eval_results = self.evaluate_prediction(data_list, statistics)
        synchronize()
        return eval_results

    def convert_to_coco_format(self, outputs, info_imgs, ids):
        data_list = []
        for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
        ):
            if output is None:
                continue
            output = output.cpu()

            bboxes = output[:, 0:4]

            # preprocessing: resize
            scale = min(
                self.img_size[0] / float(img_h), self.img_size[1] / float(img_w)
            )
            bboxes /= scale
            bboxes = xyxy2xywh(bboxes)

            cls = output[:, 6]
            scores = output[:, 4] * output[:, 5]
            for ind in range(bboxes.shape[0]):
                label = self.dataloader.dataset.class_ids[int(cls[ind])]
                if self.fg_AR_only: # for testing the forgound class AR
                    label = 0
                pred_data = {
                    "image_id": int(img_id),
                    "category_id": label,
                    "bbox": bboxes[ind].numpy().tolist(),
                    "score": scores[ind].numpy().item(),
                    "segmentation": [],
                }  # COCO json format
                data_list.append(pred_data)
        return data_list

    def evaluate_prediction(self, data_dict, statistics):
        if not is_main_process():
            return 0, 0, None

        logger.info("Evaluate in main process...")

        annType = ["segm", "bbox", "keypoints"]

        inference_time = statistics[0].item()
        nms_time = statistics[1].item()
        n_samples = statistics[2].item()

        a_infer_time = 1000 * inference_time / (n_samples * self.dataloader.batch_size)
        a_nms_time = 1000 * nms_time / (n_samples * self.dataloader.batch_size)

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
            cocoGt = self.dataloader.dataset.coco
            # TODO: since pycocotools can't process dict in py36, write data to json file.
            if self.testdev:
                json.dump(data_dict, open("./yolox_testdev_2017.json", "w"))
                cocoDt = cocoGt.loadRes("./yolox_testdev_2017.json")
            else:
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
            redirect_string = io.StringIO()
            with contextlib.redirect_stdout(redirect_string):
                cocoEval.summarize()
            info += redirect_string.getvalue()
            cat_ids = list(cocoGt.cats.keys())
            cat_names = [cocoGt.cats[catId]['name'] for catId in sorted(cat_ids)]
            if self.per_class_AP:
                AP_table = per_class_AP_table(cocoEval, class_names=cat_names)
                info += "per class AP:\n" + AP_table + "\n"
            if self.per_class_AR:
                AR_table = per_class_AR_table(cocoEval, class_names=cat_names)
                info += "per class AR:\n" + AR_table + "\n"
            return cocoEval.stats[0], cocoEval.stats[1], info
        else:
            return 0, 0, info
