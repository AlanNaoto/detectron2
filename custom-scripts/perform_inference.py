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


def set_cfg(cfg_yaml=None, model=None, out_dir=None):
    cfg = get_cfg()    
    cfg.merge_from_file(cfg_yaml)
    cfg.OUTPUT_DIR = out_dir
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # This needs to be set again or the model will try to predict on 81 classes...
    cfg.MODEL.WEIGHTS = model
    return cfg


def setup_dataset(cfg, ann_json, imgs_dir, split=""):
    dataset_dict = DatasetCatalog.get(f"carla_dataset_{split}")
    dataset_metadata = MetadataCatalog.get(f"carla_dataset_{split}")    
    if split == "train":
        cfg.DATASETS.TRAIN = (f"carla_dataset_{split}", )
    elif (split == "val") or (split == "test"):
        cfg.DATASETS.TEST = (f"carla_dataset_{split}", )
    return cfg, dataset_dict, dataset_metadata


def predict_samples(cfg, dataset_dict, dataset_metadata, img_ext, out_dir):
    predictor = DefaultPredictor(cfg)
    classes_decodification = {x: predictor.metadata.thing_classes[x] for x in range(len(predictor.metadata.thing_classes))}
    for img_idx, d in enumerate(dataset_dict):
        if img_idx == 0 or img_idx % 51 == 0:  # FPS calculated for a set of 50 frames
            start_time = time.time()
        img_name = os.path.basename(d['file_name'])
        img = cv2.imread(d['file_name'])

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

        if ((img_idx+1) % 50) == 0:
            FPS = 50.0/(time.time() - start_time)
            remaining_imgs = len(dataset_dict) - (img_idx + 1)
            ETA = datetime.timedelta(seconds=int(remaining_imgs/FPS))
            sys.stdout.write(f'\rDone imgs: {img_idx+1}/{len(dataset_dict)} FPS: {FPS} ETA: {ETA}')
            sys.stdout.flush()
            

if __name__ == "__main__":
    cfg_yaml = "../configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
    imgs_dir = "/media/aissrtx2060/Seagate Expansion Drive/Data/CARLA_1920x1280/imgs_jpg"
    img_ext = ".jpg"
    train_json = "/media/aissrtx2060/Seagate Expansion Drive/Data/CARLA_1920x1280/anns_coco/carla_1920x1280_train.json"
    val_json = "/media/aissrtx2060/Seagate Expansion Drive/Data/CARLA_1920x1280/anns_coco/carla_1920x1280_val.json"
    test_json = "/media/aissrtx2060/Seagate Expansion Drive/Data/CARLA_1920x1280/anns_coco/carla_1920x1280_test.json"
    dir_with_model_weights = "/media/aissrtx2060/Seagate Expansion Drive/detectron2_results/weights/carla_1920x1280_fasterrcnn_holdout"
    base_dir_results_path = "/media/aissrtx2060/Seagate Expansion Drive/detectron2_results/predictions/carla_1920x1280_fasterrcnn_holdout"
    dataset_splits_to_evaluate = ['train', "val", "test"]  # [train, val, test]

    register_coco_instances(f"carla_dataset_train", {}, train_json, imgs_dir)
    register_coco_instances(f"carla_dataset_val", {}, val_json, imgs_dir)
    register_coco_instances(f"carla_dataset_test", {}, test_json, imgs_dir)

    json_files = {"train": train_json, "val": val_json, "test": test_json}
    model_paths = [os.path.join(dir_with_model_weights, x) for x in os.listdir(dir_with_model_weights) if ".pth" in x]
    print(f"Evaluating {model_paths}")
    for model in model_paths:
        model_name = os.path.basename(model.replace('.pth', ''))
        for split in dataset_splits_to_evaluate:
            print(f'working on model {model_name}, {model_paths.index(model)+1}/{len(model_paths)} of {split}')
            out_dir = os.path.join(base_dir_results_path, split, f'{model_name}')

            # Model configuration
            cfg = set_cfg(cfg_yaml, model, out_dir)
            
            # Dataset config
            cfg, dataset_dict, dataset_metadata = setup_dataset(cfg, json_files[split], imgs_dir, split)

            # Prediction
            os.makedirs(out_dir, exist_ok=True)

            if len(glob.glob(os.path.join(out_dir, "*.txt"))) == len(dataset_dict): # Continue to next step if this whole dir has already been predicted
                print('Yay! This one is already done')
                break
            predict_samples(cfg, dataset_dict, dataset_metadata, img_ext, out_dir)
        
# # LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/usr/local/cuda/lib64®I1