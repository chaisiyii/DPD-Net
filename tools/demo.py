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

import os
import time
import cv2
import torch
import logging
import multiprocessing as mp

from .detectron2.detectron2.checkpoint import DetectionCheckpointer
from .detectron2.detectron2.config import get_cfg
from .detectron2.detectron2.data import MetadataCatalog, DatasetCatalog
from .detectron2.detectron2.data.datasets import load_coco_json
from .detectron2.detectron2.data.detection_utils import read_image
from .detectron2.detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from .detectron2.detectron2.utils.logger import setup_logger
from .detectron2.detectron2.utils.visualizer import ColorMode, Visualizer
from .detectron2.demo.predictor import VisualizationDemo

# constants
WINDOW_NAME = "detections"
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
INPUT_IMG_PATH = 'demo/input/'
OUTPUT_IMG_PATH = 'demo/output/'

# registration
CLASS_NAMES =["abnormal cell"]
DATASET_CATEGORIES = [
    {"color": [0, 10, 255], "isthing": 1, "id": 0, "name": "abnormal cell"},]
# RGB

class MyVisualizationDemo(VisualizationDemo):
    def __init__(self, cfg, instance_mode=ColorMode.SEGMENTATION, parallel=False):
        super().__init__(cfg, instance_mode, parallel)
    
    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        vis_output = None
        predictions = self.predictor(image)
        # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        visualizer = Visualizer(image, self.metadata, instance_mode=self.instance_mode)
        if "instances" in predictions:
            instances = predictions["instances"].to(self.cpu_device)
            vis_output = visualizer.draw_instance_predictions(predictions=instances)
        return predictions, vis_output


def setup_cfg(args):
    # load config from file and command-line arguments
    # the same with train.py
    cfg = get_cfg()
    args.config_file = "checkpoints/output/config.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.WEIGHTS = "checkpoints/output/model_0002911.pth"

    cfg.freeze()
    return cfg


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = default_argument_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))
    cfg = setup_cfg(args)
    
    # visualization result
    args.output = OUTPUT_IMG_PATH
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
            assert os.path.isdir(args.output), args.output
            out_filename = os.path.join(args.output, os.path.basename(imgfile))
            visualized_output.save(out_filename)
        else:
            # no output then show it
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            cv2.imshow(WINDOW_NAME, visualized_output.get_image()[:, :, ::-1])
            if cv2.waitKey(0) == 27:
                break  # esc to quit
  