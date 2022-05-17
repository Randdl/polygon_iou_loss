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
from detectron2.evaluation import DatasetEvaluator, COCOEvaluator, inference_on_dataset, CityscapesSemSegEvaluator, \
    DatasetEvaluators, SemSegEvaluator
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.data import detection_utils as utils
from detectron2.data import build_detection_test_loader

from data.Kitti import load_dataset_detectron2
from data.Kittidataloader import KittiDatasetMapper
from detectron2_custom_model import CustomROIHeads
from new_model import delta_to_bases

import json
from polyogn_iou_loss import c_poly_loss


def transform_instance_annotations(
        annotation, transforms, image_size, *, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    bbox = annotation["bbox"]
    base = np.transpose(annotation["base"])
    # base = transforms.apply_coords(base)
    # clip transformed bbox to image size
    transforms = ResizeTransform(370, 1224, 370, 1224)
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    # print(bbox)
    base = base.flatten()
    # print(base)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["base"] = base
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    return annotation


def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
    bases = [obj["base"] for obj in annos]
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)
    device = bases.device if isinstance(bases, torch.Tensor) else torch.device("cpu")
    bases = torch.as_tensor(bases, dtype=torch.float32, device=device)
    # print("bases: {}".format(bases.shape))
    if bases.numel() == 0:
        # Use reshape, so we don't end up creating a new tensor that does not depend on
        # the inputs (and consequently confuses jit)
        bases = bases.reshape((0, 4)).to(dtype=torch.float32, device=device)
    target.gt_bases = bases

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    return target


def mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
    # can use other ways to read image
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    # See "Data Augmentation" tutorial for details usage
    auginput = T.AugInput(image)
    transform = T.Resize((370, 1224))(auginput)
    image = torch.tensor(auginput.image.transpose(2, 0, 1))
    annos = [
        transform_instance_annotations(annotation, [transform], image.shape[1:])
        for annotation in dataset_dict.pop("annotations")
    ]
    return {
        # create the format that the model expects
        "image_id": dataset_dict["image_id"],
        "width": dataset_dict["width"],
        "height": dataset_dict["height"],
        "image": image,
        "instances": annotations_to_instances(annos, image.shape[1:])
    }


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
            prediction["pred_bases"] = instances.pred_bases
            prediction["pred_boxes"] = instances.pred_boxes
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
        for idx, inputs in enumerate(self.data_loader):
            instance = inputs[0]['instances'].to(torch.device("cpu"))
            gt_bases = instance.gt_bases
            gt_boxes = instance.gt_boxes.tensor
            gt_classes = instance.gt_classes
            for i in range(gt_boxes.shape[0]):
                ious = []
                pred_bases = self._predictions[idx]['pred_bases'][:, 0:8]
                pred_boxes = self._predictions[idx]['pred_boxes'].tensor[:, :]
                score = self._predictions[idx]['scores']
                keep = score > 0.5
                pred_bases = pred_bases[keep, :]
                pred_boxes = pred_boxes[keep, :]
                for j in range(pred_boxes.shape[0]):
                    ious.append(bb_intersection_over_union(pred_boxes[j, :], gt_boxes[i, :]))
                ious = np.array(ious)
                max_iou = np.argmax(ious)
                if ious[max_iou] < 0.1:
                    continue
                base_iou = 1 - c_poly_loss(pred_bases[max_iou, :].view(4, 2), gt_bases[i, :].view(4, 2))
                base_ious[gt_classes[max_iou].item()].append(base_iou)
                boxes_ious[gt_classes[max_iou].item()].append(ious[max_iou])

        for i in range(9):
            base_ious[i] = np.array(base_ious[i])
            boxes_ious[i] = np.array(boxes_ious[i])
            np.save("results2/iou {}.npy".format(i), base_ious[i])
            print("{} : num: {} | AP: {} | "
                  "NoneZero: {} | NAP: {} | 0.5: {} | 0.7: {}".format(i, base_ious[i].shape, base_ious[i].mean(),
                                                  np.count_nonzero(base_ious[i]), base_ious[i][base_ious[i] > 0].mean(),
                                                                      np.count_nonzero(base_ious[i][base_ious[i] > 0.5]),
                                                                      np.count_nonzero(base_ious[i][base_ious[i] > 0.7])))
            print("{} : boxes AP: {}".format(i, boxes_ious[i].mean()))
        return base_ious


DatasetCatalog.register("Kitti_train", lambda: load_dataset_detectron2())

cfg = get_cfg()
cfg.merge_from_file("configs/base_detection_faster_rcnn.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TRAIN = ("Kitti_train",)
cfg.DATALOADER.NUM_WORKERS = 0

predictor = DefaultPredictor(cfg)

model = DefaultTrainer.build_model(cfg)
checkpointer = DetectionCheckpointer(model, save_dir="model_param")
checkpointer.load("results/discard negative base train from scratch/model_0019999.pth")
# checkpointer.load("output/model_final.pth")

# evaluator = COCOEvaluator("Kitti_train", output_dir="./output", tasks="bbox", distributed=False)
val_loader = build_detection_test_loader(cfg, "Kitti_train", mapper=mapper)
evaluator = CustomEvaluator(val_loader)
result = inference_on_dataset(model, val_loader, evaluator)
