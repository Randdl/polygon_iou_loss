import csv
import os
import os.path
import cv2
from typing import Any, Callable, List, Optional, Tuple, Dict

from PIL import Image

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets.vision import VisionDataset
import torch

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib import colors
import numpy as np
from math import sin, cos

import detectron2.structures.boxes
import detectron2.structures.instances
from detectron2.structures import BoxMode


def computeBox3D(label, P):
    '''
    takes an object label and a projection matrix (P) and projects the 3D
    bounding box into the image plane.

    (Adapted from devkit_object/matlab/computeBox3D.m)

    Args:
      label -  object label list or array
    '''
    w = label[0]
    h = label[1]
    l = label[2]
    x = label[3]
    y = label[4]
    z = label[5]
    ry = label[6]

    # compute rotational matrix around yaw axis
    R = np.array([[+cos(ry), 0, +sin(ry)],
                  [0, 1, 0],
                  [-sin(ry), 0, +cos(ry)]])

    # 3D bounding box corners

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # --w/2

    # x_corners += -l / 2
    # y_corners += -h
    # z_corners += -w / 2
    x_corners = [i - l / 2 for i in x_corners]
    y_corners = [i - h for i in y_corners]
    z_corners = [i - w / 2 for i in z_corners]

    # bounding box in object co-ordinate
    corners_3D = np.array([x_corners, y_corners, z_corners])
    # print ( 'corners_3d', corners_3D.shape, corners_3D)

    # rotate
    corners_3D = R.dot(corners_3D)
    # print ( 'corners_3d', corners_3D.shape, corners_3D)

    # translate
    corners_3D += np.array([x, y, z]).reshape((3, 1))
    # print ( 'corners_3d', corners_3D)

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = P.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]

    # edges, lines 3d/2d bounding box in vertex index
    edges = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0], [0, 5], [1, 4], [2, 7], [3, 6]]
    lines = [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 0], [0, 5], [5, 4], [4, 1], [1, 2], [2, 7],
             [7, 6], [6, 3]]
    bb3d_lines_verts_idx = [0, 1, 2, 3, 4, 5, 6, 7, 0, 5, 4, 1, 2, 7, 6, 3]

    bb2d_lines_verts = corners_2D[:, bb3d_lines_verts_idx]  #

    corners_2D = corners_2D[:2]
    base_indices = [2, 3, 6, 7]
    base_3Dto2D = corners_2D[:, base_indices]

    return base_3Dto2D, corners_2D, corners_3D, bb2d_lines_verts[:2]


class Kitti(VisionDataset):
    """`KITTI <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark>`_ Dataset.

    It corresponds to the "left color images of object" dataset, for object detection.

    Args:
        root (string): Root directory where images are downloaded to.
            Expects the following folder structure if download=False:

            .. code::

                <root>
                    └── Kitti
                        └─ raw
                            ├── training
                            |   ├── calib
                            |   ├── image_2
                            |   └── label_2
                            └── testing
                                └── image_2
        train (bool, optional): Use ``train`` split if true, else ``test`` split.
            Defaults to ``train``.
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample
            and its target as entry and returns a transformed version.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """

    data_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
    resources = [
        "data_object_image_2.zip",
        "data_object_label_2.zip",
    ]
    image_dir_name = "image_2"
    labels_dir_name = "label_2"
    calibs_dir_name = "calib"

    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = None
    ):
        super().__init__(
            root,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )
        self.images = []
        self.targets = []
        self.calibs = []
        self.root = root
        self.train = train
        self._location = "training" if self.train else "testing"

        if not self._check_exists():
            raise RuntimeError("Dataset not found. You may use download=True to download it.")

        image_dir = os.path.join(self._raw_folder, self._location, self.image_dir_name)
        if self.train:
            labels_dir = os.path.join(self._raw_folder, self._location, self.labels_dir_name)
            calibs_dir = os.path.join(self._raw_folder, self._location, self.calibs_dir_name)
        for img_file in os.listdir(image_dir):
            self.images.append(os.path.join(image_dir, img_file))
            if self.train:
                self.targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))
                self.calibs.append(os.path.join(calibs_dir, f"{img_file.split('.')[0]}.txt"))
        self.dic = self.class_dic(root)
        print("Dataset loaded.")

    def __getitem__(self, index: int):
        """Get item at a given index.

        Args:
            index (int): Index
        Returns:
            (image, target, calib, corners)
            target is a list of dictionaries with the following keys:

            - type: str
            - truncated: float
            - occluded: int
            - alpha: float
            - bbox: float[4]
            - dimensions: float[3]
            - locations: float[3]
            - rotation_y: float[2][16]

            corners is a list of dictionaries with the following keys:

            - base_3Dto2D: float[2][4]
            - corners_2D: float[2][8]
            - corners_3D: float[3][8]
            - paths_2D:

        """
        image = Image.open(self.images[index])
        image = np.asarray(image)
        target = self._parse_target(index) if self.train else None
        # calib = self._parse_calib(index) if self.train else None
        # corners = self._parse_corners(index, target, calib) if self.train else None
        sample = {}
        sample['image'] = image
        sample['target'] = target
        # sample['corners'] = corners

        if self.transforms:
            sample = self.transforms(sample)
        return sample

    def _parse_corners(self, index, target, calib):
        corner = []
        P2_rect = calib['P2'].reshape(3, 4)
        for single in target:
            base_3Dto2D, corners_2D, corners_3D, paths_2D = computeBox3D(single, P2_rect)
            corner.append(
                {
                    # "corners_2D": corners_2D,
                    # "corners_3D": corners_3D,
                    # "paths_2D": paths_2D,
                    # "base_3Dto2D": base_3Dto2D
                    base_3Dto2D
                }
            )
        return corner

    def get_input_for_detectron2(self):
        result = []
        for i in range(len(self.images)):
            sample = self.__getitem__(i)
            image = sample['image']
            image = torch.from_numpy(image)
            image = image.permute(2, 0, 1)
            bbox = []
            classes = []
            for target in sample['target']:
                bbox.append(target['bbox'])
                classes.append(target['type'])
            bbox = np.asarray(bbox)
            bbox = torch.from_numpy(bbox)
            gt_boxes = detectron2.structures.boxes.Boxes(bbox)
            classes = torch.LongTensor(classes)
            image_size = (image.shape[1], image.shape[2])
            print(i)
            instances = detectron2.structures.instances.Instances(image_size, gt_boxes=gt_boxes, gt_classes=classes)
            result.append(
                {
                    'image': image,
                    'instances': instances
                }
            )
        return result

    def _parse_target(self, index: int) -> List:
        calib = {}
        with open(self.calibs[index]) as inp:
            for line in inp.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key] = np.array([float(x) for x in value.split()])
        P2_rect = calib['P2'].reshape(3, 4)
        target = []
        with open(self.targets[index]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                base_3Dto2D, corners_2D, corners_3D, paths_2D = computeBox3D([float(x) for x in line[8:15]], P2_rect)
                target.append(
                    {
                        "type": self.dic[line[0]],
                        # "truncated": float(line[1]),
                        # "occluded": int(line[2]),
                        # "alpha": float(line[3]),
                        "base": base_3Dto2D,
                        "corners": corners_2D,
                        "bbox": [float(x) for x in line[4:8]],
                        # "dimensions": [float(x) for x in line[8:11]],
                        # "location": [float(x) for x in line[11:14]],
                        # "rotation_y": float(line[14]),
                    }
                )
        return target

    def _parse_calib(self, index: int) -> Dict:
        calib = {}
        with open(self.calibs[index]) as inp:
            for line in inp.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key] = np.array([float(x) for x in value.split()])
        return calib

    def __len__(self) -> int:
        return len(self.images)

    @property
    def _raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _check_exists(self) -> bool:
        """Check if the data directory exists."""
        folders = [self.image_dir_name]
        if self.train:
            folders.append(self.labels_dir_name)
        return all(os.path.isdir(os.path.join(self._raw_folder, self._location, fname)) for fname in folders)

    def class_dic(self, root):
        dic = {}
        index = 0
        for dir in self.targets:
            with open(dir) as inp:
                content = csv.reader(inp, delimiter=" ")
                for line in content:
                    if not line[0] in dic:
                        dic[line[0]] = index
                        index += 1
        print("dic size: ", len(dic))
        print(dic)
        return dic

    def name_to_label(self, name):
        return self.dic[name]

    def label_to_name(self, label):
        return "not implemented"

    def num_classes(self):
        return len(self.dic)

    def plot(self, index: int):
        sample = self.__getitem__(index)
        image = sample['image']
        target = sample['target']
        for idx in range(len(target)):
            base = target[idx]['base']
            bbox = target[idx]['bbox']
            corners = target[idx]['corners']
            print(bbox)
            print(base)
            plt.scatter(x=corners[0, :], y=corners[1, :], s=40, color="w")
            plt.scatter(x=base[0, :], y=base[1, :], s=40, color="r")
            plt.scatter(x=bbox[0], y=bbox[1], s=40, color="b")
            plt.scatter(x=bbox[2], y=bbox[3], s=40, color="b")
        plt.imshow(sample['image'])
        plt.show()


def load_dataset_detectron2(root="..", train=True):
    images = []
    targets = []
    calibs = []
    image_dir_name = "image_2"
    labels_dir_name = "label_2"
    calibs_dir_name = "calib"
    _location = "training" if train else "testing"
    _raw_folder = os.path.join(root, "Kitti", "raw")
    image_dir = os.path.join(_raw_folder, _location, image_dir_name)
    if train:
        labels_dir = os.path.join(_raw_folder, _location, labels_dir_name)
        calibs_dir = os.path.join(_raw_folder, _location, calibs_dir_name)
    for img_file in os.listdir(image_dir):
        images.append(os.path.join(image_dir, img_file))
        if train:
            targets.append(os.path.join(labels_dir, f"{img_file.split('.')[0]}.txt"))
            calibs.append(os.path.join(calibs_dir, f"{img_file.split('.')[0]}.txt"))

    dic = {}
    indexx = 0
    for dir in targets:
        with open(dir) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                if not line[0] in dic:
                    dic[line[0]] = indexx
                    indexx += 1
    print("dic size: ", len(dic))

    dataset_dicts = []
    for idx in range(len(images)):
        # if idx == 200:
        #     break
        if idx % 100 == 0:
            print("{} loaded".format(idx))
        record = {}
        filename = images[idx]
        height, width = cv2.imread(filename).shape[:2]

        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width

        objs = []

        calib = {}
        with open(calibs[idx]) as inp:
            for line in inp.readlines():
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key] = np.array([float(x) for x in value.split()])
        P2_rect = calib['P2'].reshape(3, 4)

        with open(targets[idx]) as inp:
            content = csv.reader(inp, delimiter=" ")
            for line in content:
                base_3Dto2D, _, _, _ = computeBox3D([float(x) for x in line[8:15]], P2_rect)
                obj = {
                    "iscrowd": 0,
                    "bbox": [float(x) for x in line[4:8]],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "category_id": dic[line[0]],
                    "base": base_3Dto2D,
                }
                objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    # print(dataset_dicts)
    return dataset_dicts
