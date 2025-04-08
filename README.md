# SSTFNet: Sparse Spatio-Temporal-Frequency Feature Fusion Network for Multi-frame Infrared Small Target Detection
## The overall architecture
![image](https://github.com/YangBo0411/SSTFNet/blob/main/fig1.png)

## Usage
### Requirements
Tested on Linux , with Python 3.8, PyTorch 2.4.0, cuda 11.8.

### Installation

conda create -n sstfnet python=3.8

conda activate sstfnet

pip install -r requirements.txt

pip3 install -v -e .

### Train on Custom Datasets
The dataset format is as follows:

![image](https://github.com/YangBo0411/SSTFNet/blob/main/fig2.png)

Step 1: Train the basic detector

CUDA_VISIBLE_DEVICES=2 python tools/train.py -f /data/yb/track/SSTFNet/exps/default/yolox_s.py

Step 2: Aggregate spatio-temporal features

CUDA_VISIBLE_DEVICES=2 python tools/vid_train.py -f /data/yb/track/SSTFNet/exps/yolov/yolov_s.py -c /data/yb/track/SSTFNet/results/X-ITSDT-73-1-0.854-0.426-0.497/best_ckpt.pth

Step 3: validate

CUDA_VISIBLE_DEVICES=0 python tools/vid_eval.py  -f /data/yb/track/SSTFNet/exps/yolov/yolov_s.py -c /data/yb/track/SSTFNet/results/yolov_s_MSSFA_CSCF_SSTA-1-0.927-0.527-0.609/best_ckpt.pth

## Visualization results
![image](https://github.com/YangBo0411/SSTFNet/blob/main/fig3.png)
## Quantitative results
The model of MTMLNet weight can be downloaded from [Google Drive](https://drive.google.com/file/d/1VoFt16zfl3SRduznuSsOAV9mr2os_RFx/view?usp=drive_link) or with [BaiduYun Drive](https://pan.baidu.com/s/1GyEDbLIoc6k7RgtBYxI_1g?pwd=w57x).
![image](https://github.com/YangBo0411/SSTFNet/blob/main/fig4.png)









