"""
Alan Naoto
Created 19.12.2019
"""
import os
import random
import numpy as np
import cv2

import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor, DefaultTrainer, SimpleTrainer
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator


def set_cfg(cfg_yaml=None):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_yaml)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    #cfg.MODEL.WEIGHTS = "../models/Faster R-CNN R50-FPN 3x.pkl"
    cfg.MODEL.WEIGHTS = ""
    # cfg.TEST.EVAL_PERIOD = 100  # Not working right now :(
    return cfg


def setup_dataset(train_json=None, imgs_dir=None):
    register_coco_instances("carla_dataset_train", {}, train_json, imgs_dir)
    #register_coco_instances("carla_dataset_val", {}, val_json, imgs_dir)
    # MetadataCatalog.get("waymo_dataset").thing_classes = ["vehicle", "pedestrian"]
    dataset_dict_train = DatasetCatalog.get("carla_dataset_train")    
    dataset_metadata_train = MetadataCatalog.get("carla_dataset_train")
    #dataset_metadata_val = MetadataCatalog.get("carla_dataset_val")    
    

    # # Debug / check dataset
    # for d in random.sample(dataset_dict, 10):
    #     img = cv2.imread(d["file_name"])
    #     visualizer = Visualizer(img[:, :, ::-1], metadata=dataset_metadata, scale=1.0)
    #     vis = visualizer.draw_dataset_dict(d)
    #     cv2.imshow('dataset_img', vis.get_image()[:, :, ::-1])
    #     cv2.waitKey(0)


def train(cfg=None, out_dir=None, resume_train=None):
    cfg.DATASETS.TRAIN = ("carla_dataset_train", )
    #cfg.DATASETS.TEST = ("carla_dataset_val", )
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025 #0.1  #0.00025
    cfg.SOLVER.MAX_ITER = 200000  # Dataset consists of around 20.000 images. I will train for 10 epochs, which is 200.000 iterations
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.OUTPUT_DIR = out_dir
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)    
    trainer.resume_or_load(resume=resume_train)
    trainer.train()


if __name__ == "__main__":
    cfg_yaml = "../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    imgs_dir = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/Waymo/skip10_dataset/overbalanced_imgs_jpg"
    train_json = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/datasets/Waymo/skip10_dataset/anns_coco/train_overbalanced.json"
    out_dir = 'fast_test'
    resume_train = False

    # Model configuration
    cfg = set_cfg(cfg_yaml)

    # Dataset setup
    setup_dataset(train_json, imgs_dir)

    # Train
    train(cfg, out_dir, resume_train=resume_train)

