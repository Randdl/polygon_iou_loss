import torch
from detectron2.data.transforms import ResizeTransform
from detectron2.structures import BoxMode, Instances, Boxes
from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import plotly.graph_objects as go

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

from data.Kitti import Kitti, computeBox3D, load_dataset_detectron2, batch_computeBox3D, np_computeBox3D

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
from scipy.optimize import least_squares

DatasetCatalog.register("Kitti_train", lambda: load_dataset_detectron2())
data_loader = load_dataset_detectron2(train=False, test=False)
# data_loader = iter(data_loader)
# print(next(data_loader))

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
checkpointer.load("results/final 2/model_final.pth")
x0 = np.array([3, 1.5, 10, 0, 3, 40, 0])
bounds = np.array([0.76, 0.3, 0.2, -44, -2, -4, -3.14]), np.array([4.2, 3, 35, 40, 6, 147, 3.14])


def diff_fun(input, real_corners, P, pred_depth):
	_, corners_3D, vertices = np_computeBox3D(input, P)
	mid_point = np.mean(corners_3D, axis=1)
	depth = np.sqrt(np.sum(np.square(mid_point)))
	vertices = np.transpose(vertices)
	vertices = vertices.flatten()
	# print(real_corners)
	# print(vertices)
	diff = real_corners.numpy() - vertices
	return np.append(diff.flatten(), (depth - pred_depth.numpy()) * 4)


# iters = 0
for d in data_loader:
	# if iters >= 1:
	# 	break
	# iters += 1
	im = cv2.imread(d["file_name"])
	output_file_name = d["file_name"].replace('..\\Kitti\\raw\\training\\image_2', 'results\\l1predictions')
	output_file_name = output_file_name.replace('.png', '.txt')
	print(output_file_name)
	outputs = predictor(im[..., ::-1])
	vertices = outputs['instances'].pred_vertices.to('cpu')
	boxes = outputs["instances"].pred_boxes.to("cpu").tensor
	classes = outputs['instances'].pred_classes.to('cpu')
	scores = outputs['instances'].scores.to('cpu')
	depth = outputs['instances'].pred_depth.to('cpu')
	# print(classes)
	# print(depth)

	image_y, image_x = outputs["instances"]._image_size
	scale_x = image_x / 1333
	scale_y = image_y / 402
	vertices[:, ::2] = vertices[:, ::2] * scale_x
	vertices[:, 1::2] = vertices[:, 1::2] * scale_y

	annotations = d["annotations"]
	P2 = annotations[0]['P2']
	corners_3D = annotations[0]['corners_3D']
	# for x in annotations:
	# 	print(x['depth'])

	classes_dic = ['Car', 'Pedestrian', 'Cyclist']

	output = []
	for i in range(vertices.shape[0]):
		single_vertices = vertices[i, :]
		single_class = classes[i]
		single_box = boxes[i, :]
		single_score = scores[i]
		single_depth = depth[i]
		new_diff_fun = lambda a: diff_fun(a, single_vertices, P2, single_depth)
		res_1 = least_squares(new_diff_fun, x0, bounds=bounds)
		value_3d = res_1.x
		# print(single_vertices)
		# _, c2, new_vertices = np_computeBox3D(value_3d, P2)
		# print(new_vertices)
		# plt.scatter(x=single_vertices.reshape(8, 2)[:, 0], y=single_vertices.reshape(8, 2)[:, 1], s=20, color="r")
		# plt.scatter(x=new_vertices[0, :], y=new_vertices[1, :], s=20, color="b")
		# plt.show()

		# c1 = corners_3D
		output_list = [classes_dic[single_class], 0.00, 0, 0.00]
		output_list += single_box.tolist()
		output_list += value_3d.tolist()
		output_list.append(single_score.item())
		print(output_list)
		output_list = ' '.join([str(x) for x in output_list])
		output_list += '\n'
		output.append(output_list)
		# z = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
		# fig = go.Figure(data=[go.Scatter3d(
		# 	x=np.concatenate([c1[0], c2[0]]),
		# 	y=np.concatenate([c1[1], c2[1]]),
		# 	z=np.concatenate([c1[2], c2[2]]),
		# 	mode='markers',
		# 	marker=dict(
		# 		size=12,
		# 		color=z,  # set color to an array/list of desired values
		# 		colorscale='Viridis',  # choose a colorscale
		# 		opacity=0.8
		# 	)
		# )])
		# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
		# fig.show()

	with open(output_file_name, 'w') as fp:
		fp.writelines(output)

# gt_vertices = annotations[i]['vertices']
# gt_vertices = np.transpose(gt_vertices)
# gt_vertices = gt_vertices.flatten()

# v = Visualizer(im[:, :, ::-1], MetadataCatalog.get("Kitti_train"), scale=1)
# out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
# plt.figure(figsize=(20, 10))
# plt.imshow(out.get_image()[..., ::-1][..., ::-1])
# plt.show()
