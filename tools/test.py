#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import argparse
import random
import os
import time
import cv2
import tqdm
import logging
import multiprocessing as mp
from collections import OrderedDict

from detectron2.detectron2.checkpoint import DetectionCheckpointer
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.detectron2.data.datasets import load_coco_json
from detectron2.detectron2.data.detection_utils import read_image
from detectron2.detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.detectron2.utils.logger import setup_logger
from detectron2.detectron2.utils.visualizer import ColorMode
from detectron2.detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.demo.predictor import VisualizationDemo
from detectron2.detectron2.modeling import GeneralizedRCNNWithTTA

# classes
CLASS_NAMES =["pos"]
DATASET_CATEGORIES = [
    {"color": [0, 60, 100], "isthing": 1, "id": 0, "name": "pos"},]
# constants
WINDOW_NAME = "detections"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

# inference
# if want to change the visualised result, change line 62 "INPUT_IMG_PATH"
# if want to change the evaluation result, change line 215 "cfg.DATASETS.TEST"
INPUT_IMG_PATH = './dataset/val2017/'  # Your own Path
OUTPUT_IMG_PATH = './visualize_output/'

# dataset
DATASET_ROOT = './dataset/'
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train2017')
VAL_PATH = os.path.join(DATASET_ROOT, 'val2017')
TEST_PATH = os.path.join(DATASET_ROOT, 'test2017')

ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_JSON = os.path.join(ANN_ROOT, 'anno_train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'anno_valid.json')
TEST_JSON = os.path.join(ANN_ROOT, 'anno_test.json')

# announcement
PREDEFINED_SPLITS_DATASET = {
    "coco_my_train": (TRAIN_PATH, TRAIN_JSON),
    "coco_my_val": (VAL_PATH, VAL_JSON),
    "coco_my_test": (TEST_PATH, TEST_JSON),
}

########### Data Registration ################
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   metadata=get_dataset_instances_meta(),
                                   json_file=json_file,
                                   image_root=image_root)

def get_dataset_instances_meta():
    """
    purpose: get metadata of dataset from DATASET_CATEGORIES
    return: dict[metadata]
    """
    thing_ids = [k["id"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    thing_colors = [k["color"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    # assert len(thing_ids) == 2, len(thing_ids)
    thing_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(thing_ids)}
    thing_classes = [k["name"] for k in DATASET_CATEGORIES if k["isthing"] == 1]
    ret = {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes,
        "thing_colors": thing_colors,
    }
    return ret

def register_dataset_instances(name, metadata, json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_coco_json(json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadata)

########### Data Registration End ################

########### Result Evaluator ################
def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if not os.path.exists(cfg.OUTPUT_DIR):
        os.mkdir(cfg.OUTPUT_DIR)
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

########### Result Evaluator End ################

def setup_cfg(args):
    # load config from file and command-line arguments
    # the same with train.py
    cfg = get_cfg()
    args.config_file = "../checkpoints/output/config.yaml"
    cfg.merge_from_file(args.config_file)

    cfg.MODEL.WEIGHTS = '../checkpoints/output/model_0002911.pth'
    cfg.DATASETS.TEST = ("coco_my_test",)
    cfg.OUTPUT_DIR = OUTPUT_IMG_PATH

    cfg.freeze()
    return cfg


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = default_argument_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    args.output = cfg.OUTPUT_DIR
    
    # registration
    register_dataset()

    '''# visualization result
    demo = VisualizationDemo(cfg, instance_mode=ColorMode.SEGMENTATION)
    # for path in tqdm.tqdm(args.input, disable=not args.output):
    for imgfile in os.listdir(INPUT_IMG_PATH):
        # use PIL, to be consistent with evaluation
        img_fullName = os.path.join(INPUT_IMG_PATH, imgfile)
        img = read_image(img_fullName, format="BGR")
        start_time = time.time()
        predictions, visualized_output = demo.run_on_image(img)
        logger.info(
            "{}: detected {} instances in {:.2f}s".format(
                imgfile, len(predictions["instances"]), time.time() - start_time
            )
        )

        if args.output:
            if not os.path.exists(args.output):
                os.mkdir(args.output)
            if os.path.isdir(args.output):
                assert os.path.isdir(args.output), args.output
                out_filename = os.path.join(args.output, os.path.basename(imgfile))
            else:
                assert len(args.input) == 1, "Please specify a directory with args.output"
                out_filename = args.output
            visualized_output.save(out_filename)
        else:
            # no output then show it
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit'''

    # evaluation result
    model = Trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )
    res = Trainer.test(cfg, model)
    
    # write evaluation
    output_folder = os.path.join(OUTPUT_IMG_PATH, "inference")
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    eva_txt = os.path.join(output_folder, "evaluation_result.txt")
    with open(eva_txt, "w") as fp:
        fp.write(str(res))
    fp.close()
    # print(res)

