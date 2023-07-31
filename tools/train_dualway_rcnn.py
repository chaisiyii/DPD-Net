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

import logging
import os
import io
import contextlib
import torch
from collections import OrderedDict

import detectron2.detectron2.utils.comm as comm
from detectron2.detectron2.checkpoint import DetectionCheckpointer
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.data import MetadataCatalog
from detectron2.detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
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
from detectron2.detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.detectron2.data.datasets.coco import load_coco_json
from detectron2.detectron2.utils.events import TensorboardXWriter
from fvcore.common.history_buffer import HistoryBuffer
from fvcore.common.timer import Timer
from collections import defaultdict
from detectron2.detectron2.engine import HookBase, PeriodicWriter
from detectron2.detectron2.data import build_detection_train_loader
from detectron2.detectron2.structures import Boxes, BoxMode, PolygonMasks, RotatedBoxes
from detectron2.detectron2.utils.file_io import PathManager
import detectron2.detectron2.utils.comm as comm

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
VAL_LOSS_ITER = 100
NUM_GPU = 2
logger = logging.getLogger(__name__)

# classes
CLASS_NAMES =["pos"]
DATASET_CATEGORIES = [
    {"color": [0, 60, 100], "isthing": 1, "id": 0, "name": "pos"},]
# registration
# root
DATASET_ROOT = './dataset'  # Your own dataset in COCO format.
TRAIN_PATH = os.path.join(DATASET_ROOT, 'train2017')
VAL_PATH = os.path.join(DATASET_ROOT, 'val2017')
TEST_PATH = os.path.join(DATASET_ROOT, 'test2017')

ANN_ROOT = os.path.join(DATASET_ROOT, 'annotations')
TRAIN_JSON = os.path.join(ANN_ROOT, 'anno_train.json')
VAL_JSON = os.path.join(ANN_ROOT, 'anno_valid.json')
TEST_JSON = os.path.join(ANN_ROOT, 'anno_test.json')

NUC_ANN_ROOT = os.path.join(DATASET_ROOT, 'nuc-annotations')
NUC_TRAIN_JSON = os.path.join(NUC_ANN_ROOT, 'anno_train.json')
NUC_VAL_JSON = os.path.join(NUC_ANN_ROOT, 'anno_valid.json')
NUC_TEST_JSON = os.path.join(NUC_ANN_ROOT, 'anno_test.json')

# announcement
PREDEFINED_SPLITS_DATASET = {
    "coco_my_train": (TRAIN_PATH, TRAIN_JSON, NUC_TRAIN_JSON),
    "coco_my_val": (VAL_PATH, VAL_JSON, NUC_VAL_JSON),
    "coco_my_test": (TEST_PATH, TEST_JSON, NUC_TEST_JSON),
}

def load_dualway_coco_json(json_file1, json_file2, image_root, dataset_name=None):
    '''
    Load 2 json files with COCO's instances annotation format.
    Main json file for cell annotation and vice json file for nucleus annotation.

    Args:
        json_file1 (str): full path to MAIN json file for CELLS in COCO instances annotation format.
        json_file2 (str): full path to VICE json file for NUCLEUS in COCO instances annotation format.
        image_root (str or path-like): the directory where the images in this json file exists.
        dataset_name (str or None): the name of the dataset (e.g., coco_2017_train).
            When provided, this function will also do the following:

            * Put "thing_classes" into the metadata associated with this dataset.
            * Map the category ids into a contiguous range (needed by standard dataset format),
              and add "thing_dataset_id_to_contiguous_id" to the metadata associated
              with this dataset.

            This option should usually be provided, unless users need to load
            the original json content and apply more processing manually.

    Returns:
        list[dict]: a list of dicts in Detectron2 standard dataset dicts format (See
        `Using Custom Datasets </tutorials/datasets.html>`_ ) when `dataset_name` is not None.
        If `dataset_name` is None, the returned `category_ids` may be
        incontiguous and may not conform to the Detectron2 standard format.

    Notes:
        1. This function does not read the image files.
           The results do not have the "image" field.
    '''
    from pycocotools.coco import COCO

    # Read main json file (CELL)
    timer = Timer()
    cell_json_file = PathManager.get_local_path(json_file1)
    with contextlib.redirect_stdout(io.StringIO()):
        cell_coco_api = COCO(cell_json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file1, timer.seconds()))
    # Read vice json file (NUCLEUS)
    timer = Timer()
    nuc_json_file = PathManager.get_local_path(json_file2)
    with contextlib.redirect_stdout(io.StringIO()):
        nuc_coco_api = COCO(nuc_json_file)
    if timer.seconds() > 1:
        logger.info("Loading {} takes {:.2f} seconds.".format(json_file2, timer.seconds()))

    # Map category ids into contiguous ids
    id_map = None
    if dataset_name is not None:
        meta = MetadataCatalog.get(dataset_name)
        cat_ids = sorted(cell_coco_api.getCatIds())
        cats = cell_coco_api.loadCats(cat_ids)
        # The categories in a custom json file may not be sorted.
        thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
        meta.thing_classes = thing_classes

        id_map = {v: i for i, v in enumerate(cat_ids)}
        meta.thing_dataset_id_to_contiguous_id = id_map

    # Read MAIN json file for CELLS
    img_ids = sorted(cell_coco_api.imgs.keys())
    imgs = cell_coco_api.loadImgs(img_ids)
    anns = [cell_coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_anns = sum([len(x) for x in anns])
    total_num_anns = len(cell_coco_api.anns)
    if total_num_valid_anns < total_num_anns:
        logger.warning(
            f"{json_file1} contains {total_num_anns} annotations, but only "
            f"{total_num_valid_anns} of them match to images in the file."
        )

    # Read VICE json file for NUCLEUS
    nuc_anns = [nuc_coco_api.imgToAnns[img_id] for img_id in img_ids]
    total_num_valid_nuc_anns = sum([len(x) for x in nuc_anns])
    total_num_nuc_anns = len(nuc_coco_api.anns)
    if total_num_valid_nuc_anns < total_num_nuc_anns:
        logger.warning(
            f"{json_file2} contains {total_num_nuc_anns} annotations, but only "
            f"{total_num_valid_nuc_anns} of them match to images in the file."
        )

    # Read cells from anns and nucleus from nuc_anns
    imgs_anns = list(zip(imgs, anns, nuc_anns))
    logger.info("Loaded {} images in COCO format from {}".format(len(imgs_anns), json_file1))
    logger.info("Loaded {} cell annotations from {}".format(total_num_valid_anns, json_file1))
    logger.info("Loaded {} nuc annotations from {}".format(total_num_valid_nuc_anns, json_file2))

    # Final dataset annotation dict
    dataset_dicts = []
    ann_keys = ["iscrowd", "bbox", "keypoints", "category_id"]

    for (img_dict, anno_dict_list, nuc_anno_dict_list) in imgs_anns:
        record = {}
        record["file_name"] = os.path.join(image_root, img_dict["file_name"])
        record["height"] = img_dict["height"]
        record["width"] = img_dict["width"]
        image_id = record["image_id"] = img_dict["id"]

        # Load cells
        objs = []
        for anno in anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'
            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            if id_map:
                annotation_category_id = obj["category_id"]
                try:
                    obj["category_id"] = id_map[annotation_category_id]
                except KeyError as e:
                    raise KeyError(
                        f"Encountered category_id={annotation_category_id} "
                        "but this id does not exist in 'categories' of the json file."
                    ) from e
            objs.append(obj)
        record["annotations"] = objs

        # Load nucleus
        objs = []
        for anno in nuc_anno_dict_list:
            assert anno["image_id"] == image_id
            assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'
            obj = {key: anno[key] for key in ann_keys if key in anno}
            if "bbox" in obj and len(obj["bbox"]) == 0:
                raise ValueError(
                    f"One annotation of image {image_id} contains empty 'bbox' value! "
                    "This json does not have valid COCO format."
                )

            obj["bbox_mode"] = BoxMode.XYWH_ABS
            objs.append(obj)
        record["nuc_annotations"] = objs
        
        # Add into dataset dicts
        dataset_dicts.append(record)

    return dataset_dicts

def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (image_root, json_file1, json_file2) in PREDEFINED_SPLITS_DATASET.items():
        register_dataset_instances(name=key,
                                   metadata=get_dataset_instances_meta(),
                                   cell_json_file=json_file1,
                                   nuc_json_file=json_file2,
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

def register_dataset_instances(name, metadata, cell_json_file, nuc_json_file, image_root):
    """
    purpose: register dataset to DatasetCatalog,
             register metadata to MetadataCatalog and set attribute
    """
    DatasetCatalog.register(name, lambda: load_dualway_coco_json(cell_json_file, nuc_json_file, image_root, name))
    MetadataCatalog.get(name).set(json_file=cell_json_file,
                                  image_root=image_root,
                                  evaluator_type="coco",
                                  **metadata)

class ValidationLoss(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.cfg.defrost()  # make this cfg mutable.
        self.cfg.DATASETS.TRAIN = cfg.DATASETS.TEST
        self._loader = iter(build_detection_train_loader(self.cfg))
        self.num_steps = 0
        
    def after_step(self):
        self.num_steps += 1
        if self.num_steps % VAL_LOSS_ITER == 0:
            data = next(self._loader)
            with torch.no_grad():
                loss_dict = self.trainer.model(data)
            
                losses = sum(loss_dict.values())
                assert torch.isfinite(losses).all(), loss_dict

                loss_dict_reduced = {"val_" + k: v.item() for k, v in 
                                 comm.reduce_dict(loss_dict).items()}
                losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                if comm.is_main_process():
                    self.trainer.storage.put_scalars(total_val_loss=losses_reduced, 
                                                     **loss_dict_reduced)
        else:
            scalars = {}
            for k, v in self.trainer.storage._latest_scalars.items():
                if "val_" in k and "loss" in k:
                    continue
                else:
                    scalars[k] = v
            self.trainer.storage._latest_scalars = scalars
            history = defaultdict(HistoryBuffer)
            for k, v in self.trainer.storage._history.items():
                if "val_" in k and "loss" in k:
                    continue
                else:
                    history[k] = v
            self.trainer.storage._history = history


def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
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


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # faster-rcnn basic config
    args.config_file = "../detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    # adding end
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    
    ############# Adding ###############
    # cfg modify
    cfg.DATASETS.TRAIN = ("coco_my_train",) 
    cfg.DATASETS.TEST = ("coco_my_val",)
    cfg.DATALOADER.NUM_WORKERS = 1
    
    cfg.INPUT.CROP.ENABLED = True
    cfg.INPUT.MAX_SIZE_TRAIN = 1000
    cfg.INPUT.MAX_SIZE_TEST = 1000
    cfg.INPUT.MIN_SIZE_TRAIN = (800, 1200)
    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING = 'range'
    cfg.OUTPUT_DIR = "../checkpoints/output/"
    
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # because json label start from 0
    cfg.MODEL.WEIGHTS = "../checkpoints/pretrain/baseline_pretrained_dual_rcnn.pth"
    cfg.MODEL.META_ARCHITECTURE = "DualWayRCNN"
    cfg.MODEL.BACKBONE.NAME = 'build_resnet_att_fpn_backbone'
    cfg.MODEL.ROI_HEADS.NAME = 'RelationROIHeads'
    
    # solver
    cfg.SOLVER.IMS_PER_BATCH = 4*NUM_GPU*2
    ITERS_IN_ONE_EPOCH = int(3344/cfg.SOLVER.IMS_PER_BATCH)  # 209 iters/epoch
    cfg.SOLVER.MAX_ITER = (ITERS_IN_ONE_EPOCH * 25) - 1
    # params
    cfg.SOLVER.BASE_LR = 0.001*NUM_GPU*2  # base learning rate
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.WEIGHT_DECAY = 0.0001
    cfg.SOLVER.GAMMA = 0.1  # learning rate decay
    cfg.SOLVER.STEPS = (12*ITERS_IN_ONE_EPOCH, 21*ITERS_IN_ONE_EPOCH)
    cfg.SOLVER.WARMUP_FACTOR = 1.0 / 1000
    cfg.SOLVER.WARMUP_ITERS = 500
    cfg.SOLVER.WARMUP_METHOD = "linear"
    cfg.SOLVER.CHECKPOINT_PERIOD = ITERS_IN_ONE_EPOCH - 1
    # evaluation
    cfg.TEST.EVAL_PERIOD = ITERS_IN_ONE_EPOCH
    cfg.VIS_PERIOD = 100
    ############# Adding-End ###############
    
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    
    # registration
    register_dataset()

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)

    # validation loss
    val_loss = ValidationLoss(cfg)
    trainer.register_hooks([val_loss])
    # swap the order of PeriodicWriter and ValidationLoss
    # keep the PeriodicWriter the trainer._hooks[-1]
    trainer._hooks = trainer._hooks[:-2] + trainer._hooks[-2:][::-1]

    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    args.num_gpus=NUM_GPU
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
