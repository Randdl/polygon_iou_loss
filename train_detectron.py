import torch, os, json, cv2, random

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION)

# import some common detectron2 utilities
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.config import get_cfg
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_train_loader, DatasetCatalog

from data.Kitti import load_dataset_detectron2
from custom_roi_heads import CustomROIHeads
from custom_retinanet import CustomRetinaNet
from utils import Trainer


# dataset_train = load_dataset_detectron2(root='..')
# print(dataset_train[0])
d = "train"
e = ".."
DatasetCatalog.register("Kitti_" + d, lambda: load_dataset_detectron2())
DatasetCatalog.register("Kitti_test", lambda: load_dataset_detectron2(train=False))
DatasetCatalog.register("Kitti_train_test", lambda: load_dataset_detectron2(test=True))
# MetadataCatalog.get("Kitti_" + d).set(thing_classes=["balloon"])
# balloon_metadata = MetadataCatalog.get("balloon_train")

from detectron2.engine import DefaultTrainer

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
# cfg.merge_from_file("configs/base_detection_faster_rcnn.yaml")
cfg.merge_from_file("configs/test_faster_rcnn.yaml")
# cfg.merge_from_file("configs/test_retinanet.yaml")

# cfg.DATASETS.TRAIN = ("Kitti_train",)
# cfg.DATASETS.TEST = ()
# cfg.DATALOADER.NUM_WORKERS = 0
# cfg.SOLVER.IMS_PER_BATCH = 4
# cfg.SOLVER.BASE_LR = 0.0001  # pick a good LR
# cfg.SOLVER.MAX_ITER = 1500
# cfg.SOLVER.WARMUP_FACTOR = 1.0 / 100
# cfg.SOLVER.WARMUP_ITERS = 100
# cfg.SOLVER.STEPS = [500, 1000]
# cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 16
# cfg.MODEL.ROI_HEADS.NUM_CLASSES = 9


os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
# checkpointer = DetectionCheckpointer(trainer.model, save_dir="model_param")
# checkpointer.load("output/model_final.pth")
trainer.train()
# checkpointer = DetectionCheckpointer(trainer.model, save_dir="model_param")
# checkpointer.save("faster_rcnn_res50_l2_90000_iters")
