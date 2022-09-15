import torch
from detectron2.data.transforms import ResizeTransform
from detectron2.structures import BoxMode, Instances, Boxes
from matplotlib import pyplot as plt
from sklearn.metrics import PrecisionRecallDisplay, average_precision_score

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
from detectron2.evaluation import DatasetEvaluator, COCOEvaluator, inference_on_dataset, CityscapesSemSegEvaluator, \
    DatasetEvaluators, SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader

from data.Kitti import load_dataset_detectron2
from data.Kittidataloader import KittiDatasetMapper
from custom_roi_heads import CustomROIHeads
from custom_fastrcnn import delta_to_bases
from utils import mapper

import json
from polyogn_iou_loss import c_poly_loss


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


class CustomEvaluator(DatasetEvaluator):
    def __init__(self, data_loader):
        self.data_loader = data_loader

    def reset(self):
        """
        Preparation for a new round of evaluation.
        Should be called before starting a round of evaluation.
        """
        self._predictions = []

    def process(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            # print(output)
            # print(input)
            prediction = {"image_id": input["image_id"]}
            instances = output["instances"].to(torch.device("cpu"))
            # print(instances)
            prediction["pred_boxes"] = instances.pred_boxes
            prediction["pred_bases"] = instances.pred_bases
            # prediction["pred_bases"] = torch.zeros_like(instances.pred_boxes)
            prediction["scores"] = instances.scores
            prediction["pred_classes"] = instances.pred_classes
            self._predictions.append(prediction)

    def evaluate(self):
        base_ious = {}
        boxes_ious = {}
        for i in range(9):
            base_ious[i] = []
            boxes_ious[i] = []
        # for idx, inputs in enumerate(self.data_loader):
        #     instance = inputs[0]['instances'].to(torch.device("cpu"))
        #     gt_bases = instance.gt_bases
        #     gt_boxes = instance.gt_boxes.tensor
        #     gt_classes = instance.gt_classes
        #     for i in range(self._predictions[idx]['pred_bases'].shape[0]):
        #         ious = []
        #         pred_bases = self._predictions[idx]['pred_bases'][i, 0:8]
        #         pred_boxes = self._predictions[idx]['pred_boxes'].tensor[i, :]
        #         score = self._predictions[idx]['scores'][i]
        #         if score < 0.5:
        #             continue
        #         for j in range(gt_boxes.shape[0]):
        #             ious.append(bb_intersection_over_union(pred_boxes, gt_boxes[j, :]))
        #         if len(ious) < 1:
        #             continue
        #         ious = np.array(ious)
        #         max_iou = np.argmax(ious)
        #         # if ious[max_iou] < 0.5:
        #         #     # print(ious)
        #         #     continue
        #         base_iou = 1 - c_poly_loss(pred_bases.view(4, 2), gt_bases[max_iou, :].view(4, 2))
        #         base_ious[gt_classes[max_iou].item()].append(base_iou)
        #         boxes_ious[gt_classes[max_iou].item()].append(ious[max_iou])
        # for idx, inputs in enumerate(self.data_loader):
        #     instance = inputs[0]['instances'].to(torch.device("cpu"))
        #     gt_bases = instance.gt_bases
        #     gt_boxes = instance.gt_boxes.tensor
        #     gt_classes = instance.gt_classes
        #     for i in range(gt_boxes.shape[0]):
        #         ious = []
        #         pred_bases = self._predictions[idx]['pred_bases'][:, 0:8]
        #         pred_boxes = self._predictions[idx]['pred_boxes'].tensor[:, :]
        #         score = self._predictions[idx]['scores']
        #         keep = score > 0.5
        #         pred_bases = pred_bases[keep, :]
        #         pred_boxes = pred_boxes[keep, :]
        #         for j in range(pred_boxes.shape[0]):
        #             ious.append(bb_intersection_over_union(pred_boxes[j, :], gt_boxes[i, :]))
        #         ious = np.array(ious)
        #         max_iou = np.argmax(ious)
        #         if ious[max_iou] < 0.1:
        #             continue
        #         base_iou = 1 - c_poly_loss(pred_bases[max_iou, :].view(4, 2), gt_bases[i, :].view(4, 2))
        #         base_ious[gt_classes[max_iou].item()].append(base_iou)
        #         boxes_ious[gt_classes[max_iou].item()].append(ious[max_iou])
        #
        # for i in range(9):
        #     base_ious[i] = np.array(base_ious[i])
        #     boxes_ious[i] = np.array(boxes_ious[i])
        #     np.save("results2/iou {}.npy".format(i), base_ious[i])
        #     print("{} : num: {} | AP: {} | "
        #           "NoneZero: {} | NAP: {} | 0.5: {} | 0.7: {}".format(i, base_ious[i].shape, base_ious[i].mean(),
        #                                           np.count_nonzero(base_ious[i]), base_ious[i][base_ious[i] > 0].mean(),
        #                                                               np.count_nonzero(base_ious[i][base_ious[i] > 0.5]),
        #                                                               np.count_nonzero(base_ious[i][base_ious[i] > 0.7])))
        #     print("{} : boxes AP: {}".format(i, boxes_ious[i].mean()))
        kitti_evaluator = KittiEval(self.data_loader, self._predictions)
        kitti_evaluator.evaluate()
        kitti_evaluator.accumulate()
        return base_ious


class KittiEval:
    def __init__(self, gt, pt):
        self.predictions = pt
        self.ground_truth = gt
        self.sorted_scores = {}
        self.ious = {}
        self.gtms = {}
        self.ptms = {}
        self.bases_ptms = {}
        self.bases_gtms = {}
        self.gt_scores = {}

    def evaluate(self):
        threshold = 0.3
        for idx, inputs in enumerate(self.ground_truth):
            instance = inputs[0]['instances'].to(torch.device("cpu"))
            gt_bases = instance.gt_bases
            gt_boxes = instance.gt_boxes.tensor
            gt_classes = instance.gt_classes

            pred_bases = self.predictions[idx]['pred_bases'][:, 0:8]
            pred_boxes = self.predictions[idx]['pred_boxes'].tensor[:, :]
            score = self.predictions[idx]['scores']
            sort_pred = np.argsort(-score)
            pred_bases = pred_bases[sort_pred, :]
            pred_boxes = pred_boxes[sort_pred, :]
            score = score[sort_pred]
            self.sorted_scores[idx] = score

            # sort_pred = np.argsort(-score)
            # pred_bases = pred_bases[sort_pred, :]
            # pred_boxes = pred_boxes[sort_pred, :]
            # score = score[sort_pred]

            gtIds = range(gt_classes.shape[0])
            ptIds = range(score.shape[0])

            G = len(gtIds)
            D = len(ptIds)
            img_ious = np.zeros((D, G))
            for pind in ptIds:
                for gind in gtIds:
                    img_ious[pind, gind] = self.computeIoU(pred_boxes[pind, :], gt_boxes[gind, :])
            gtm = -np.ones(G)
            ptm = np.zeros(D)
            bases_ptm = np.zeros(D)
            bases_gtm = np.zeros(G)
            gt_score = np.zeros(G)

            t = threshold
            for pind in ptIds:
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind in gtIds:
                    # ground truth is already matched, skip
                    if gtm[gind] > -1:
                        continue
                    if img_ious[pind, gind] < iou:
                        continue
                    iou = img_ious[pind, gind]
                    m = gind
                if m == -1:
                    continue
                ptm[pind] = m
                gtm[m] = pind
                bases_ptm[pind] = 1 \
                    if (1 - c_poly_loss(pred_bases[pind, :].view(4, 2), gt_bases[m, :].view(4, 2))) > t \
                    else 0
                bases_gtm[m] = bases_ptm[pind]
                gt_score[m] = score[pind]
            self.ious[idx] = img_ious
            self.gtms[idx] = gtm
            self.ptms[idx] = ptm
            self.bases_ptms[idx] = bases_ptm
            self.bases_gtms[idx] = bases_gtm
            self.gt_scores[idx] = gt_score

    def computeIoU(self, pt, gt):
        return bb_intersection_over_union(pt, gt)

    def accumulate(self):
        ptm = np.concatenate([v for e, v in self.ptms.items()])
        gtm = np.concatenate([v for e, v in self.gtms.items()])
        scores = np.concatenate([v for e, v in self.gt_scores.items()])
        bases_gtm = np.concatenate([v for e, v in self.bases_gtms.items()])
        ngtm = gtm.shape[0]

        sorted_orders = np.argsort(-scores)
        print(sorted_orders)
        scores = scores[sorted_orders]
        print(scores)
        gtm = gtm[sorted_orders]
        bases_gtm = bases_gtm[sorted_orders]
        print(gtm)
        print(bases_gtm)
        gtm[gtm > -1] = 1
        gtm[gtm < 0] = 0

        tps = gtm
        bases_tps = bases_gtm
        fps = np.logical_not(tps)
        bases_fps = np.logical_not(bases_tps)
        tp_sum = np.cumsum(tps, axis=0).astype(dtype=float)
        fp_sum = np.cumsum(fps, axis=0).astype(dtype=float)
        bases_tp_sum = np.cumsum(bases_tps, axis=0).astype(dtype=float)
        bases_fp_sum = np.cumsum(bases_fps, axis=0).astype(dtype=float)
        recall = tp_sum / ngtm
        precision = tp_sum / (tp_sum + fp_sum)
        bases_recall = bases_tp_sum / ngtm
        bases_precision = bases_tp_sum / (bases_tp_sum + bases_fp_sum)

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.plot(recall, precision)
        plt.title("boxes AP curve")
        plt.show()

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        ###########################
        #   Corrupted !!!!!########
        ###########################
        plt.plot(np.flip(bases_recall), bases_precision)
        plt.title("bases AP curve")
        plt.show()

        mAP = average_precision_score(tps, scores)
        bases_mAP = average_precision_score(bases_tps, scores)
        print("boxes mAP: {} | bases mAP: {}".format(mAP, bases_mAP))
        # display = PrecisionRecallDisplay.from_predictions(tps, scores, name="boxes AP")
        # _ = display.ax_.set_title("boxes AP curve")
        # plt.show()
        # display = PrecisionRecallDisplay.from_predictions(bases_ptm, scores, name="bases AP")
        # _ = display.ax_.set_title("bases AP curve")
        # plt.show()



DatasetCatalog.register("Kitti_test", lambda: load_dataset_detectron2(train=False))

cfg = get_cfg()
cfg.merge_from_file("configs/test_faster_rcnn.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TRAIN = ("Kitti_test",)
cfg.DATALOADER.NUM_WORKERS = 0

predictor = DefaultPredictor(cfg)

model = DefaultTrainer.build_model(cfg)
checkpointer = DetectionCheckpointer(model, save_dir="model_param")
checkpointer.load("results/predict h/model_final.pth")
# checkpointer.load("output/model_final.pth")

# evaluator = COCOEvaluator("Kitti_train", output_dir="./output", tasks="bbox", distributed=False)
val_loader = build_detection_test_loader(cfg, "Kitti_test", mapper=mapper)
evaluator = CustomEvaluator(val_loader)
result = inference_on_dataset(model, val_loader, evaluator)
