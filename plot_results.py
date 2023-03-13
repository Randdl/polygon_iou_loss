import torch
from detectron2.data.transforms import ResizeTransform
from detectron2.structures import BoxMode, Instances, Boxes
from matplotlib import pyplot as plt

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer

import copy
import detectron2.data.transforms as T
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, DatasetCatalog
from detectron2.engine import DefaultPredictor, DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import CityscapesSemSegEvaluator, DatasetEvaluators, SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.data import detection_utils as utils

from data.Kitti import load_dataset_detectron2
from data.Kittidataloader import KittiDatasetMapper
from custom_roi_heads import CustomROIHeads
from custom_fastrcnn import delta_to_bases

import json

DatasetCatalog.register("Kitti_train", lambda: load_dataset_detectron2())

cfg = get_cfg()
cfg.merge_from_file("configs/base_detection_faster_rcnn.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.DATASETS.TRAIN = ("Kitti_train",)
cfg.DATALOADER.NUM_WORKERS = 0

predictor = DefaultPredictor(cfg)

checkpointer = DetectionCheckpointer(predictor.model, save_dir="model_param")
# checkpointer.load("results/predict h/model_final.pth")
# checkpointer.load("output/model_final.pth")
# checkpointer.load("results/model_final.pth")
# checkpointer.load("results/final 2/model_final.pth")
checkpointer.load("results/big final with iou/model_final.pth")
# checkpointer.load("results/big final with l1/model_final.pth")

# im = cv2.imread("images/000002.png")
im = cv2.imread("testing/000005.png")
# print(im.shape)
# plt.figure(figsize=(15, 7.5))
# plt.imshow(im[..., ::-1])
# plt.show()
outputs = predictor(im[..., ::-1])
# print(outputs["instances"])
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("Kitti_train"), scale=1)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# print(out.get_image()[..., ::-1][..., ::-1])
plt.figure(figsize=(20, 10))
plt.imshow(out.get_image()[..., ::-1][..., ::-1])
print(outputs["instances"])
bases = outputs["instances"].pred_bases.to("cpu")
top = outputs["instances"].pred_top.to("cpu")
boxes = outputs["instances"].pred_boxes.to("cpu").tensor
# print(outputs["instances"])
image_y, image_x = outputs["instances"]._image_size
scale_x = image_x / 1333
scale_y = image_y / 402
bases[:, ::2] = bases[:, ::2] * scale_x
bases[:, 1::2] = bases[:, 1::2] * scale_y
top[:, ::2] = top[:, ::2] * scale_x
top[:, 1::2] = top[:, 1::2] * scale_y
for idx in range(bases.shape[0]):
    plt.scatter(x=bases[idx, 0], y=bases[idx, 1], s=20, color="r")
    plt.scatter(x=bases[idx, 2], y=bases[idx, 3], s=20, color="r")
    plt.scatter(x=bases[idx, 4], y=bases[idx, 5], s=20, color="r")
    plt.scatter(x=bases[idx, 6], y=bases[idx, 7], s=20, color="r")
    plt.scatter(x=top[idx, 0], y=top[idx, 1], s=20, color="b")
    plt.scatter(x=top[idx, 2], y=top[idx, 3], s=20, color="b")
    plt.scatter(x=top[idx, 4], y=top[idx, 5], s=20, color="b")
    plt.scatter(x=top[idx, 6], y=top[idx, 7], s=20, color="b")

    xs = bases[idx, [0, 2, 6, 4, 0]]
    ys = bases[idx, [1, 3, 7, 5, 1]]

    lines = [(bases[idx, 0], bases[idx, 2]), (bases[idx, 1], bases[idx, 3]), 'r',
             (bases[idx, 2], bases[idx, 4]), (bases[idx, 3], bases[idx, 5]), 'r',
             (bases[idx, 4], bases[idx, 6]), (bases[idx, 5], bases[idx, 7]), 'r',
             (bases[idx, 6], bases[idx, 0]), (bases[idx, 7], bases[idx, 1]), 'r',
             (top[idx, 0], top[idx, 2]), (top[idx, 1], top[idx, 3]), 'r',
             (top[idx, 2], top[idx, 4]), (top[idx, 3], top[idx, 5]), 'r',
             (top[idx, 4], top[idx, 6]), (top[idx, 5], top[idx, 7]), 'r',
             (top[idx, 6], top[idx, 0]), (top[idx, 7], top[idx, 1]), 'r',
             (bases[idx, 0], top[idx, 0]), (bases[idx, 1], top[idx, 1]), 'r',
             (bases[idx, 2], top[idx, 2]), (bases[idx, 3], top[idx, 3]), 'r',
             (bases[idx, 4], top[idx, 4]), (bases[idx, 5], top[idx, 5]), 'r',
             (bases[idx, 6], top[idx, 6]), (bases[idx, 7], top[idx, 7]), 'r',
             ]
    plt.plot(*lines)

    # plt.plot(xs, ys, color='deepskyblue', linewidth=3)
    # plt.scatter(x=boxes[idx, 0], y=boxes[idx, 1], s=40, color="y")
    # plt.scatter(x=boxes[idx, 2], y=boxes[idx, 3], s=40, color="y")
    # break
plt.show()
