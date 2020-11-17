"""
Alan Naoto
Created 19.12.2019
Updated 27.02.2020
"""
import os
import sys
import glob
import time
import datetime
import cv2

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor, DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import register_coco_instances
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, DatasetEvaluators


def set_cfg(cfg_yaml=None, weights=None, out_dir=None):
    cfg = get_cfg()
    cfg.merge_from_file(cfg_yaml)
    cfg.OUTPUT_DIR = out_dir
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # This needs to be set again or the model will try to predict on 81 classes...
    cfg.MODEL.WEIGHTS = weights
    return cfg


def predict_samples(cfg, imgs_dir, img_ext, out_dir):
    predictor = DefaultPredictor(cfg)
    classes_decodification = {x: predictor.metadata.thing_classes[x] for x in range(len(predictor.metadata.thing_classes))}
    img_paths = [os.path.join(imgs_dir, x) for x in os.listdir(imgs_dir) if x.endswith(img_ext)]
    for img_path in img_paths:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)

        outputs = predictor(img)
        bbs = outputs['instances'].pred_boxes.tensor.tolist()
        scores = outputs['instances'].scores.data.tolist()
        pred_classes = outputs['instances'].pred_classes.data.tolist()
        
        # Creates BB txt predict files for each image
        with open(os.path.join(out_dir, img_name.replace(img_ext, '.txt')), 'w') as f:
            for pred_object in range(len(bbs)):
                f.write(classes_decodification[pred_classes[pred_object]])
                f.write(' ' + str(scores[pred_object]))
                for bb_point in bbs[pred_object]:
                    f.write(' ' + str(int(bb_point)))
                f.write('\n')

                actor_name = classes_decodification[pred_classes[pred_object]]
                bbox = [int(x) for x in bbs[pred_object]]
                cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                # cv2.putText(img, actor_name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imwrite(os.path.join(out_dir, img_name), img)


if __name__ == "__main__":
    cfg_yaml = "../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    imgs_dir = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/5_Artigo/carla_waymo/new_inferences/carla_2/source/resized"
    img_ext = ".jpg"
    weights = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/9_Code/masters_results_videos/object_detect/waymo_skip10_model/model_0164999.pth"
    out_dir = "/mnt/6EFE2115FE20D75D/Naoto/UFPR/Mestrado/5_Artigo/carla_waymo/new_inferences/carla_2/detectron2"

    # Model configuration
    cfg = set_cfg(cfg_yaml, weights, out_dir)
    
    # Prediction
    os.makedirs(out_dir, exist_ok=True)

    predict_samples(cfg, imgs_dir, img_ext, out_dir)
        
# # LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64®I1
