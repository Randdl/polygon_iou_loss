import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn

from skimage import io, transform


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, sample):
        image = sample['image']
        target = sample['target']
        h, w = image.shape[:2]
        # if h > w:
        #     new_h, new_w = self.output_size * h / w, self.output_size
        # else:
        #     new_h, new_w = self.output_size, self.output_size * w / h

        # new_h, new_w = int(new_h), int(new_w)
        new_h, new_w = self.new_h, self.new_w

        new_image = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        labels = []

        for i in target:
            i["base"][:, 0] *= [new_w / w, new_h / h]
            i["base"][:, 1] *= [new_w / w, new_h / h]
            i["base"][:, 2] *= [new_w / w, new_h / h]
            i["base"][:, 3] *= [new_w / w, new_h / h]
            new_label = np.squeeze(i["base"])
            if i["type"] == "Car":
                new_label = np.append(new_label, 0)
            else:
                new_label = np.append(new_label, 1)
            labels.append(new_label)
        labels = np.asarray(labels)

        return {'image': torch.from_numpy(new_image), 'labels': torch.from_numpy(labels)}

class Cropper(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, new_h, new_w):
        self.new_h = new_h
        self.new_w = new_w

    def __call__(self, sample):
        image = sample['image']
        target = sample['target']
        h, w = image.shape[:2]
        # if h > w:
        #     new_h, new_w = self.output_size * h / w, self.output_size
        # else:
        #     new_h, new_w = self.output_size, self.output_size * w / h

        # new_h, new_w = int(new_h), int(new_w)
        new_h, new_w = self.new_h, self.new_w

        new_image = image[0:new_h, 0:new_w, :]

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        labels = []

        for i in target:
            new_label = np.squeeze(i["base"])
            if i["type"] == "Car":
                new_label = np.append(new_label, 0)
            else:
                new_label = np.append(new_label, 1)
            labels.append(new_label)
        labels = np.asarray(labels)

        return {'image': torch.from_numpy(new_image), 'labels': torch.from_numpy(labels)}


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        return {'image': ((image / 255.0 - self.mean) / self.std), 'labels': labels}


def collater(data):
    imgs = [s['image'] for s in data]
    annots = [s['labels'] for s in data]

    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 9)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                # print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 9)) * -1

    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'image': padded_imgs, 'labels': annot_padded}

def labelToBoundingBox(ax, labeld, calibd):
    '''
    Draw 2D and 3D bpunding boxes.

    Each label  file contains the following ( copied from devkit_object/matlab/readLabels.m)
    #  % extract label, truncation, occlusion
    #  lbl = C{1}(o);                   % for converting: cell -> string
    #  objects(o).type       = lbl{1};  % 'Car', 'Pedestrian', ...
    #  objects(o).truncation = C{2}(o); % truncated pixel ratio ([0..1])
    #  objects(o).occlusion  = C{3}(o); % 0 = visible, 1 = partly occluded, 2 = fully occluded, 3 = unknown
    #  objects(o).alpha      = C{4}(o); % object observation angle ([-pi..pi])
    #
    #  % extract 2D bounding box in 0-based coordinates
    #  objects(o).x1 = C{5}(o); % left   -> in pixel
    #  objects(o).y1 = C{6}(o); % top
    #  objects(o).x2 = C{7}(o); % right
    #  objects(o).y2 = C{8}(o); % bottom
    #
    #  % extract 3D bounding box information
    #  objects(o).h    = C{9} (o); % box width    -> in object coordinate
    #  objects(o).w    = C{10}(o); % box height
    #  objects(o).l    = C{11}(o); % box length
    #  objects(o).t(1) = C{12}(o); % location (x) -> in camera coordinate
    #  objects(o).t(2) = C{13}(o); % location (y)
    #  objects(o).t(3) = C{14}(o); % location (z)
    #  objects(o).ry   = C{15}(o); % yaw angle  -> rotation aroun the y/vetical axis
    '''

    # Velodyne to/from referenece camera (0) matrix
    Tr_velo_to_cam = np.zeros((4, 4))
    Tr_velo_to_cam[3, 3] = 1
    Tr_velo_to_cam[:3, :4] = calibd['Tr_velo_to_cam'].reshape(3, 4)
    # print ('Tr_velo_to_cam', Tr_velo_to_cam)

    Tr_cam_to_velo = np.linalg.inv(Tr_velo_to_cam)
    # print ('Tr_cam_to_velo', Tr_cam_to_velo)

    #
    R0_rect = np.zeros((4, 4))
    R0_rect[:3, :3] = calibd['R0_rect'].reshape(3, 3)
    R0_rect[3, 3] = 1
    # print ('R0_rect', R0_rect)
    P2_rect = calibd['P2'].reshape(3, 4)
    # print('P2_rect', P2_rect)

    bb3d = []
    bb2d = []

    for key in labeld.keys():

        color = 'white'
        if key == 'Car':
            color = 'red'
        elif key == 'Pedestrian':
            color = 'pink'
        elif key == 'Cyclist':
            color = 'purple'
        elif key == 'DontCare':
            color = 'white'

        for o in range(labeld[key].shape[0]):

            # 2D
            left = labeld[key][o][3]
            bottom = labeld[key][o][4]
            width = labeld[key][o][5] - labeld[key][o][3]
            height = labeld[key][o][6] - labeld[key][o][4]

            p = patches.Rectangle(
                (left, bottom), width, height, fill=False, edgecolor=color, linewidth=1)
            ax.add_patch(p)

            xc = (labeld[key][o][5] + labeld[key][o][3]) / 2
            yc = (labeld[key][o][6] + labeld[key][o][4]) / 2
            bb2d.append([xc, yc])

            # 3D
            w3d = labeld[key][o][7]
            h3d = labeld[key][o][8]
            l3d = labeld[key][o][9]
            x3d = labeld[key][o][10]
            y3d = labeld[key][o][11]
            z3d = labeld[key][o][12]
            yaw3d = labeld[key][o][13]

            if key != 'DontCare':
                base_3Dto2D, corners_2D, corners_3D, paths_2D = computeBox3D(labeld[key][o], P2_rect)
                verts = paths_2D.T  # corners_2D.T
                codes = [Path.LINETO] * verts.shape[0]
                codes[0] = Path.MOVETO
                pth = Path(verts, codes)
                p = patches.PathPatch(pth, fill=False, color='purple', linewidth=2)
                ax.add_patch(p)

    # a sanity test point in velodyne co-ordinate to check  camera2 imaging plane projection
    testp = [11.3, -2.95, -1.0]
    bb3d.append(testp)

    xnd = np.array(testp + [1.0])
    # print ('bb3d xnd velodyne   ', xnd)
    # xpnd = P2.dot(R0_rect.dot(Tr_velo_to_cam.dot(xnd)))
    xpnd = Tr_velo_to_cam.dot(xnd)
    # print ('bb3d xpnd cam0      ', xpnd)
    xpnd = R0_rect.dot(xpnd)
    # print ('bb3d xpnd rect cam0 ', xpnd)
    xpnd = P2_rect.dot(xpnd)
    # print ('bb3d xpnd cam2 image', xpnd)
    # print ('bb3d xpnd cam2 image', xpnd/xpnd[2])

    p = patches.Circle((xpnd[0] / xpnd[2], xpnd[1] / xpnd[2]), fill=False, radius=3, color='red', linewidth=2)
    ax.add_patch(p)

    return np.array(bb2d), np.array(bb3d)