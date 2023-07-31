# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
import torch
from torch import nn
import cv2 as cv

from detectron2.detectron2.config import configurable
from detectron2.detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.detectron2.layers import move_device_like, cat
from detectron2.detectron2.structures import ImageList, Boxes, Instances, pairwise_intersection
from detectron2.detectron2.utils.events import get_event_storage
from detectron2.detectron2.utils.logger import log_first_n
from detectron2.detectron2.utils.memory import retry_if_cuda_oom

from detectron2.detectron2.modeling.backbone import Backbone, build_backbone
from detectron2.detectron2.modeling.postprocessing import detector_postprocess
from detectron2.detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.detectron2.modeling.roi_heads import build_roi_heads, ROI_HEADS_REGISTRY
from detectron2.detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

__all__ = ["DualWayRCNN"]
logger = logging.getLogger(__name__)

def build_cell_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)

def build_nuc_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)

def pairwise_iou_on_nuc(cell_boxes: Boxes, nuc_boxes: Boxes) -> torch.Tensor:
    """
    Two lists of boxes, size N and M, computer IoU **all** N x M pairs.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        cell_boxes, nuc_boxes (Boxes): Contains N & M boxes, respectively.
    Returns:
        Tensor: IoU, sized [N,M].
    """
    area2 = nuc_boxes.area()  # [M]
    inter = pairwise_intersection(cell_boxes, nuc_boxes)  # [N,M]
    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / area2,
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

def pairwise_iou_on_cell(cell_boxes: Boxes, nuc_boxes: Boxes) -> torch.Tensor:
    """
    Two lists of boxes, size N and M, computer IoU **all** N x M pairs.
    The box order must be (xmin, ymin, xmax, ymax).
    Args:
        cell_boxes, nuc_boxes (Boxes): Contains N & M boxes, respectively.
    Returns:
        Tensor: IoU, sized [N,M].
    """
    area1 = cell_boxes.area().unsqueeze(1)  # [N,1]
    inter = pairwise_intersection(cell_boxes, nuc_boxes)  # [N,M]
    # handle empty boxes
    iou = torch.where(
        inter > 0,
        inter / area1,
        torch.zeros(1, dtype=inter.dtype, device=inter.device),
    )
    return iou

def align_from_nuc_to_cell(nuc_results, cell_results, image_sizes):
    results = []
    for nuc_result_per_image, cell_result_per_image, image_shape in zip(
        nuc_results, cell_results, image_sizes
    ):
        # Get boxes for nucleus and cells
        nuc_boxes = nuc_result_per_image.pred_boxes.clone()  # M
        cell_boxes = cell_result_per_image.pred_boxes.clone()  # N
        cell_scores = cell_result_per_image.scores.clone()
        cell_classes = cell_result_per_image.pred_classes.clone()

        # If 0 cells/nucleus be detected, early stop.
        if cell_boxes.tensor.shape[0] == 0:
            results.append(cell_result_per_image)
            continue
        if nuc_boxes.tensor.shape[0] == 0:
            results.append(cell_result_per_image)
            continue

        # Step1: filter based on iou-nuc
        # Calculate their ious based on nucleus and choose which be supported
        ious = pairwise_iou_on_nuc(cell_boxes, nuc_boxes)  # [N,M]
        has_nuc, _ = torch.max(ious, dim=-1)  # [N,1]
        support = torch.where(
            has_nuc>=1,
            torch.ones(1, dtype=ious.dtype, device=ious.device).bool(),
            torch.zeros(1, dtype=ious.dtype, device=ious.device).bool(),
        )  # If be supported, then 1; otherwise, 0.
        cell_boxes = cell_boxes[support]
        cell_scores = cell_scores[support]
        cell_classes = cell_classes[support]

        # Step2: filter based on iou-cell
        '''th_cell=0.1
        ious = pairwise_iou_on_cell(cell_boxes, nuc_boxes)  # [N,M]
        ious = torch.sqrt(ious)  # soft
        threshold = torch.tensor(th_cell, dtype=ious.dtype, device=ious.device)
        support, _ = torch.max(ious, dim=-1)  # [N,1]
        support = torch.where(
            support>=threshold,
            torch.ones(1, dtype=ious.dtype, device=ious.device).bool(),
            torch.zeros(1, dtype=ious.dtype, device=ious.device).bool(),
        )  # If be supported, then 1; otherwise, 0.
        cell_boxes = cell_boxes[support]
        cell_scores = cell_scores[support]
        cell_classes = cell_classes[support]'''

        # Filter the supported cell results
        result = Instances(image_shape)
        result.pred_boxes = cell_boxes
        result.scores = cell_scores
        result.pred_classes = cell_classes

        # Add back into results
        results.append(result)

    return results

def align_nuc(nuc_results, cell_results, image_sizes):
    """
    Different from last function.
    This function is used to filter nucleui detection result is your result
        is nucleui.
    """
    results = []
    for nuc_result_per_image, cell_result_per_image, image_shape in zip(
        nuc_results, cell_results, image_sizes
    ):
        # Get boxes for nucleus and cells
        nuc_boxes = nuc_result_per_image.pred_boxes.clone()  # M
        cell_boxes = cell_result_per_image.pred_boxes.clone()  # N
        nuc_scores = nuc_result_per_image.scores.clone()  # M
        nuc_classes = nuc_result_per_image.pred_classes.clone()  # M

        # If 0 cells/nucleus be detected, early stop.
        if cell_boxes.tensor.shape[0] == 0:
            results.append(nuc_result_per_image)
            continue
        if nuc_boxes.tensor.shape[0] == 0:
            results.append(nuc_result_per_image)
            continue

        # Step1: filter based on iou-nuc
        # Calculate their ious based on nucleus and choose which be supported
        ious = pairwise_iou_on_nuc(cell_boxes, nuc_boxes)  # [N,M]
        in_cell, _ = torch.max(ious, dim=0)  # [1,M]
        support = torch.where(
            in_cell>=1,
            torch.ones(1, dtype=ious.dtype, device=ious.device).bool(),
            torch.zeros(1, dtype=ious.dtype, device=ious.device).bool(),
        )  # If be supported, then 1; otherwise, 0.
        nuc_boxes = nuc_boxes[support]
        nuc_scores = nuc_scores[support]
        nuc_classes = nuc_classes[support]

        # Step2: filter based on iou-cell
        th_cell=0.1
        ious = pairwise_iou_on_cell(cell_boxes, nuc_boxes)  # [N,M]
        ious = torch.sqrt(ious)  # soft
        threshold = torch.tensor(th_cell, dtype=ious.dtype, device=ious.device)
        support, _ = torch.max(ious, dim=0)  # [1,M]
        support = torch.where(
            support>=threshold,
            torch.ones(1, dtype=ious.dtype, device=ious.device).bool(),
            torch.zeros(1, dtype=ious.dtype, device=ious.device).bool(),
        )  # If be supported, then 1; otherwise, 0.
        nuc_boxes = nuc_boxes[support]
        nuc_scores = nuc_scores[support]
        nuc_classes = nuc_classes[support]

        # Filter the supported cell results
        result = Instances(image_shape)
        result.pred_boxes = nuc_boxes
        result.scores = nuc_scores
        result.pred_classes = nuc_classes

        # Add back into results
        results.append(result)

    return results

def filter_nuc_results(nuc_results, image_sizes):
    """
    nucleus results that not belong to gt_classes 0
    """
    results = []
    for nuc_result_per_image, image_shape in zip(
        nuc_results, image_sizes
    ):
        # Get boxes for nucleus and cells
        nuc_boxes = nuc_result_per_image.pred_boxes.clone()  # M
        nuc_scores = nuc_result_per_image.scores.clone()  # M
        nuc_classes = nuc_result_per_image.pred_classes.clone()  # M

        # If 0 nucleus be detected, early stop.
        if nuc_boxes.tensor.shape[0] == 0:
            results.append(nuc_result_per_image)
            continue

        support = (nuc_classes == 0)  # abnormal nucleus    
        nuc_boxes = nuc_boxes[support]
        nuc_scores = nuc_scores[support]
        nuc_classes = nuc_classes[support]

        # Filter the supported cell results
        result = Instances(image_shape)
        result.pred_boxes = nuc_boxes
        result.scores = nuc_scores
        result.pred_classes = nuc_classes
        # Add back into results
        results.append(result)
    
    return results


@META_ARCH_REGISTRY.register()
class DualWayRCNN(nn.Module):
    """
    My modified dual-way R-CNN based on MyDualWayRCNN used for cell detection. 
    Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Cell RPN and Nucleui RPN
    3. Per-region feature extraction and prediction for cells and nucs
    4. Using nucleui detection to correction cell detection (Merge)
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        cell_proposal_generator: nn.Module,
        cell_roi_heads: nn.Module,
        nuc_proposal_generator: nn.Module,
        nuc_roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface.
            cell_proposal_generator: a module generates cell proposals using backbone features.
            cell_roi_heads: a ROI head performs cell's per-region computation.
            nuc_proposal_generator: a module generates nucleui proposals.
            nuc_roi_heads: a nuc ROI head preforms per-nuc-region computation.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image.
            input_format: describe the meaning of channels of input. Needed by visualization.
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super().__init__()
        self.backbone = backbone
        self.cell_proposal_generator = cell_proposal_generator
        self.cell_roi_heads = cell_roi_heads

        self.nuc_proposal_generator = nuc_proposal_generator
        self.nuc_roi_heads = nuc_roi_heads
        self._freeze()

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "cell_proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "cell_roi_heads": build_cell_roi_heads(cfg, backbone.output_shape()),
            "nuc_proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "nuc_roi_heads": build_nuc_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def _move_to_current_device(self, x):
        return move_device_like(x, self.pixel_mean)

    def _freeze(self):
        # only freeze backbone and cell detector
        for name, param in self.named_parameters():
            if "backbone" in name or "cell" in name:
                param.requires_grad = False

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 top-scoring predicted
        object proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        # inference
        if not self.training:
            return self.inference(batched_inputs)
        
        # training
        # preprocess images
        images = self.preprocess_image(batched_inputs)
        # get ground-truth instances for cells and nucleus
        assert "instances" in batched_inputs[0]
        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        assert "nuc_instances" in batched_inputs[0]
        nuc_gt_instances = [x["nuc_instances"].to(self.device) for x in batched_inputs]

        # feature extraction
        features = self.backbone(images.tensor)

        # cell detection
        assert self.cell_proposal_generator is not None
        cell_proposals, cell_proposal_loss = self.cell_proposal_generator(images, features, gt_instances)
        _, cell_detector_loss = self.cell_roi_heads(images, features, cell_proposals, gt_instances)
        # cell loss merge
        cell_proposal_losses = {}
        for k, v in cell_proposal_loss.items():
            k = "cell_"+k
            cell_proposal_losses[k] = v
        cell_detector_losses = {}
        for k, v in cell_detector_loss.items():
            k = "cell_"+k
            cell_detector_losses[k] = v
        
        # nucleus detection
        assert self.nuc_proposal_generator is not None
        nuc_proposals, nuc_proposal_loss = self.nuc_proposal_generator(images, features, nuc_gt_instances)
        _, nuc_detector_loss = self.nuc_roi_heads(images, features, nuc_proposals, nuc_gt_instances)
        # nucleui loss merge
        nuc_proposal_losses = {}
        for k, v in nuc_proposal_loss.items():
            k = "nuc_"+k
            nuc_proposal_losses[k] = v
        nuc_detector_losses = {}
        for k, v in nuc_detector_loss.items():
            k = "nuc_"+k
            nuc_detector_losses[k] = v

        if self.vis_period > 0:
            storage = get_event_storage()
            if storage.iter % self.vis_period == 0:
                self.visualize_training(batched_inputs, cell_proposals)

        losses = {}
        # merge losses
        losses.update(cell_detector_losses)
        losses.update(cell_proposal_losses)
        losses.update(nuc_detector_losses)
        losses.update(nuc_proposal_losses)
        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        assert detected_instances is None
        # Doing cell-way detection inference
        assert self.cell_proposal_generator is not None
        cell_proposals, _ = self.cell_proposal_generator(images, features, None)
        cell_roi_outs = self.cell_roi_heads(images, features, cell_proposals, None)
        if len(cell_roi_outs) == 2:
            cell_results, _ = cell_roi_outs
        else:  # len=3
            cell_results, box_features, _ = cell_roi_outs
            del box_features

        # Doing nucleus-way detection inference
        assert self.nuc_proposal_generator is not None
        nuc_proposals, _ = self.nuc_proposal_generator(images, features, None)
        nuc_roi_outs = self.nuc_roi_heads(images, features, nuc_proposals, None)
        if len(nuc_roi_outs) == 2:
            nuc_results, _ = nuc_roi_outs
        else:  # len=3
            nuc_results, box_features, _ = nuc_roi_outs
            del box_features
        
        # Doing correction/supportion/alignment, from nuc to cell.
        results = align_from_nuc_to_cell(nuc_results, cell_results, images.image_sizes)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return DualWayRCNN._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
        
    def preprocess_image(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [self._move_to_current_device(x["image"]) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        return images

    @staticmethod
    def _postprocess(instances, batched_inputs: List[Dict[str, torch.Tensor]], image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

