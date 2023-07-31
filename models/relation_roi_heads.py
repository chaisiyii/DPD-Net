# Copyright (c) Facebook, Inc. and its affiliates.
import inspect
import logging
import math
import numpy as np
from typing import Dict, List, Optional, Tuple
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
import torch.nn.functional as F

from detectron2.detectron2.config import configurable
from detectron2.detectron2.layers import ShapeSpec
from detectron2.detectron2.structures import ImageList, Instances
from detectron2.detectron2.utils.events import get_event_storage

from detectron2.detectron2.modeling.matcher import Matcher
from detectron2.detectron2.modeling.poolers import ROIPooler
from detectron2.detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.detectron2.modeling.roi_heads.roi_heads import ROIHeads
from detectron2.detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY

from utils.loss import cal_contrastive_loss, USE_CONTRASTIVE_LOSS

logger = logging.getLogger(__name__)
__all__ = ["RelationROIHeads"]


def absolute_position_encoding(position, d_model):
    '''
    Position Encoding using paper "Attention is all you need."
    PE(pos,2i) = sin[pos/(10000^2i/dmodel)]
    PE(pos,2i+1) = cos[pos/(10000^2i/dmodel)]
    Args:
        position (tensor): positions, (x,y,w,h) dim of (Ni, 4).
        d_model (int): the number of features output position encoding.
    '''
    d = d_model//4
    pe = np.array([np.exp(2*i/d * math.log(10000)) for i in range(d)])
    pe = torch.from_numpy(pe).to(position.device)  # (d, )
    pe = pe.view((1,pe.shape[0]))  # (1, d)
    
    # center x and center y
    delta = position[:,0:2]+0.5*position[:,2:4]  # x+0.5w and y+0.5h, (Ni, 2)
    delta = delta.unsqueeze(dim=-1)  # (Ni, 2, 1)
    delta = delta/pe  # (Ni, 2, d)

    # position encoding
    sin, cos = torch.sin(delta), torch.cos(delta)
    embedding = torch.cat((sin, cos), dim=-1)  # (Ni, 2, 2d)
    embedding = embedding.view((-1, d_model))  # (Ni, d_model)
    embedding = embedding.float()
    return embedding


def delta_position_encoding(delta, d_model):
    '''
    Delta position Encoding using paper "RelationNet".
    PE(pos,2i) = sin[pos/(10000^2i/dmodel)]
    PE(pos,2i+1) = cos[pos/(10000^2i/dmodel)]
    Args:
        delta (tensor): delta positions, (dx,dy,dw,dh) dim of (Ni**2, 4).
        d_model (int): the number of features position encoding.
    Returns:
        position (Tensor): (Ni**2, d_model) which indicate the similarity of pe in N**2 pairs.
    '''
    assert d_model%8 == 0
    d = d_model//8

    pe = np.array([np.exp(2*i/d * math.log(1000)) for i in range(d)])
    pe = torch.from_numpy(pe).to(delta.device)  # (d, )
    pe = pe.view((1, pe.shape[0]))  # (1, d)

    # delta
    delta = delta.unsqueeze(dim=-1)  # (Ni**2, 4, 1)
    delta = delta/pe  # (Ni**2, 4, d)=(Ni**2, 4, d_model//8)

    # sin and cos computation
    sin = torch.sin(delta)
    cos = torch.cos(delta)
    embedding = torch.cat((sin, cos), dim=-1)  # (Ni**2, 4, d_model//4)
    embedding = embedding.view((-1, d_model))  # (Ni**2, d_model)
    
    embedding = embedding.float()
    return embedding


def relative_position_encoding(position, d_model):
    '''
    Position Encoding using paper "Attention is all you need."
    PE(pos,2i) = sin[pos/(10000^2i/dmodel)]
    PE(pos,2i+1) = cos[pos/(10000^2i/dmodel)]
    Args:
        position (tensor): positions, (x,y,w,h) dim of (Ni, 4).
        d_model (int): the number of features position encoding.
    Returns:
        position (Tensor): (Ni, Ni, d_model) which indicate the similarity of pe in N**2 pairs.
    '''
    # Prepare
    Ni = position.shape[0]
    center_x = position[:,0]+0.5*position[:,2]  # (Ni,1): center_x=x+0.5w
    center_y = position[:,1]+0.5*position[:,3]  # (Ni,1): center_y=y+0.5h
    w, h = position[:,2], position[:,3]  # (Ni, 1)
    epsil = torch.tensor(1e-3).to(position.device)
    
    # get relative position
    # delta_x
    position1 = center_x.unsqueeze(dim=1)  # (Ni, 1, 1)
    position2 = center_x.unsqueeze(dim=0)  # (1, Ni, 1)
    delta_x = torch.abs(position1-position2)/w
    delta_x = torch.clamp(delta_x, min=epsil)  # (Ni, Ni, 1)
    # delta_y
    position1 = center_y.unsqueeze(dim=1)  # (Ni, 1, 1)
    position2 = center_y.unsqueeze(dim=0)  # (1, Ni, 1)
    delta_y = torch.abs(position1-position2)/h
    delta_y = torch.clamp(delta_y, min=epsil)  # (Ni, Ni, 1)
    # delta_w
    position1 = w.unsqueeze(dim=1)  # (Ni, 1, 1)
    position2 = w.unsqueeze(dim=0)  # (1, Ni, 1)
    delta_w = position1/position2  # (Ni, Ni, 1)
    # delta_h
    position1 = h.unsqueeze(dim=1)  # (Ni, 1, 1)
    position2 = h.unsqueeze(dim=0)  # (1, Ni, 1)
    delta_h = position1/position2  # (Ni, Ni, 1)
    # delta
    delta = torch.cat((delta_x, delta_y, delta_w, delta_h), dim=-1)  # (Ni, Ni, 4)
    delta = torch.log(delta)
    del position, position1, position2
    del delta_x, delta_y, delta_w, delta_h

    # get relative position's embedding
    # change
    delta = delta.view((-1, 4))  # to (Ni**2, 4)
    delta = delta_position_encoding(delta, d_model)  # (Ni**2, d_model)
    delta = delta.view((Ni, Ni, d_model))
    return delta


def relative_position_encoding_wo_wh(position, d_model):
    '''
    Position Encoding without adding delta w and delta h.
    PE(pos,2i) = sin[pos/(10000^2i/dmodel)]
    PE(pos,2i+1) = cos[pos/(10000^2i/dmodel)]
    Args:
        position (tensor): positions, (x,y,w,h) dim of (Ni, 4).
        d_model (int): the number of features position encoding.
    Returns:
        position (Tensor): (Ni, Ni, d_model) which indicate the similarity of pe in N**2 pairs.
    '''
    # Prepare
    Ni = position.shape[0]
    center_x = position[:,0]+0.5*position[:,2]  # (Ni,1): center_x=x+0.5w
    center_y = position[:,1]+0.5*position[:,3]  # (Ni,1): center_y=y+0.5h
    w, h = position[:,2], position[:,3]  # (Ni, 1)
    epsil = torch.tensor(1e-3).to(position.device)
    
    # get relative position
    # delta_x
    position1 = center_x.unsqueeze(dim=1)  # (Ni, 1, 1)
    position2 = center_x.unsqueeze(dim=0)  # (1, Ni, 1)
    delta_x = torch.abs(position1-position2)/w
    delta_x = torch.clamp(delta_x, min=epsil)  # (Ni, Ni, 1)
    # delta_y
    position1 = center_y.unsqueeze(dim=1)  # (Ni, 1, 1)
    position2 = center_y.unsqueeze(dim=0)  # (1, Ni, 1)
    delta_y = torch.abs(position1-position2)/h
    delta_y = torch.clamp(delta_y, min=epsil)  # (Ni, Ni, 1)
    # delta
    delta = torch.cat((delta_x, delta_y), dim=-1)  # (Ni, Ni, 2)
    delta = torch.log(delta)
    del position, position1, position2
    del delta_x, delta_y

    # get relative position's embedding
    # change
    delta = delta.view((-1, 2))  # to (Ni**2, 2)
    
    # get delta's embedding
    assert d_model%4 == 0
    d = d_model//4
    pe = np.array([np.exp(2*i/d * math.log(1000)) for i in range(d)])
    pe = torch.from_numpy(pe).to(delta.device)  # (d, )
    pe = pe.view((1, pe.shape[0]))  # (1, d)

    # delta
    delta = delta.unsqueeze(dim=-1)  # (Ni**2, 2, 1)
    delta = delta/pe  # (Ni**2, 2, d)=(Ni**2, 2, d_model//4)

    # sin and cos computation
    sin, cos = torch.sin(delta), torch.cos(delta)
    embedding = torch.cat((sin, cos), dim=-1)  # (Ni**2, 2, d_model//2)
    embedding = embedding.view((-1, d_model))  # (Ni**2, d_model)
    embedding = embedding.float()
    
    # change back
    embedding = embedding.view((Ni, Ni, d_model))
    return embedding


def relative_position_encoding_wo_fc_layer(position):
    '''
    Position Encoding without adding delta w and delta h.
    Directly model relative position into a single value without using fc layer lately.
    Args:
        position (Tensor): positions, (x,y,w,h) dim of (Ni, 4).
    Returns:
        delta (Tensor): (Ni, Ni) which indicate relative position of N**2 pairs.
    '''
    # Prepare
    center_x = position[:,0]+0.5*position[:,2]  # (Ni,1): center_x=x+0.5w
    center_y = position[:,1]+0.5*position[:,3]  # (Ni,1): center_y=y+0.5h
    w, h = position[:,2], position[:,3]  # (Ni, 1)
    
    # get relative position
    # delta_x
    position1 = center_x.unsqueeze(dim=1)  # (Ni, 1, 1)
    position2 = center_x.unsqueeze(dim=0)  # (1, Ni, 1)
    delta_x = torch.abs(position1-position2)/w  # (Ni, Ni, 1)
    # delta_y
    position1 = center_y.unsqueeze(dim=1)  # (Ni, 1, 1)
    position2 = center_y.unsqueeze(dim=0)  # (1, Ni, 1)
    delta_y = torch.abs(position1-position2)/h  # (Ni, Ni, 1)
    # delta
    delta = torch.cat((delta_x, delta_y), dim=-1)  # (Ni, Ni, 2)
    delta, _ = torch.max(delta, dim=-1)
    delta = delta.squeeze(dim=-1)  # (Ni, Ni)
    return delta


class RelationModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        num_features,
        fuse_location=True,
        is_relative_location=True,
        ):
        """
        Args:
            in_channels (int): the dim of input(as well as output) features.
            out_channels (int): the dim of output(as well as output) features.
            num_features (int): the dim of transition features in relation module.
            fuse_location (bool): fuse location information into transition features or not.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_features = num_features
        self.fuse_location = fuse_location
        self.is_relative_location = is_relative_location

        self.trans_norm = nn.BatchNorm1d(in_channels)
        self.query = nn.Linear(in_channels, num_features)
        self.key = nn.Linear(in_channels, num_features)
        self.value = nn.Linear(in_channels, out_channels)
        weight_init.c2_xavier_fill(self.query)
        weight_init.c2_xavier_fill(self.key)
        weight_init.c2_xavier_fill(self.value)
        self.dropout_q = nn.Dropout(p = 0.5)
        self.dropout_k = nn.Dropout(p = 0.5)
        self.dropout_v = nn.Dropout(p = 0.5)
        self.out_norm = nn.BatchNorm1d(out_channels)

        # using relative location brings a different network architecture
        self.loc_info = min(128, self.out_channels)
        if fuse_location and is_relative_location:
            self.relative = nn.Linear(self.loc_info, 1)
            weight_init.c2_xavier_fill(self.relative)
            self.dropout_r = nn.Dropout(p = 0.5)


    def forward(self, box_features, proposals):
        """
        Args:
            box_features (tensor): (M, box_head.output_shape) for M boxes detection.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        """
        # 1. prepare relation features (batchnorm the feature first)
        num_prop_per_image = [len(p) for p in proposals]
        box_features = self.trans_norm(box_features)  # (M, in_channels)
        # list[tensor: (Ni, num_features)]
        box_features = box_features.split(num_prop_per_image)

        # 2. do relation per image
        relation_features = []
        for feature, proposals_per_image in zip(box_features, proposals):
            keep_feature = feature
            # add absolute location
            if self.fuse_location and not self.is_relative_location:
                location = proposals_per_image.proposal_boxes.tensor  # dim(Ni, 4)
                location = absolute_position_encoding(location, self.in_channels)
                feature = 0.2*location + feature  # dim(Ni, in_channels)
            # add relative location
            if self.fuse_location and self.is_relative_location:
                location = proposals_per_image.proposal_boxes.tensor  # dim(Ni, 4)
                rela_loc = relative_position_encoding(location, self.loc_info)  # (Ni, Ni, loc_info)
                rela_loc = self.dropout_r(self.relative(rela_loc))  # (Ni, Ni, 1)
                rela_loc = rela_loc.squeeze(dim=2)  # (Ni, Ni)
            
            # relation
            query = self.dropout_q(self.query(feature))  # (Ni, in_channels) -> (Ni, num_features)
            key = self.dropout_k(self.key(feature).permute(1,0))  # (num_features, Ni)
            value = self.dropout_v(self.value(keep_feature))  # (Ni, out_channels) using keep_feature

            # attention score        
            att_score = torch.mm(query, key)  # (Ni, Ni)
            dk = torch.sqrt(torch.tensor(self.num_features))
            att_score = att_score/dk
            
            # add relative location information
            if self.fuse_location and self.is_relative_location:
                # rela_weight = F.softmax(rela_loc, dim=-1)  # (Ni, Ni)
                # rela_weight = torch.clamp(rela_loc, min=5e-4)  # 5e-4 ok, plus relu
                # rela_weight = torch.log(rela_weight)  # (Ni, Ni)
                rela_weight = F.log_softmax(rela_loc, dim=-1)  # log_softmax good
                att_score = att_score + rela_weight  # (Ni, Ni)
                
            # transform score to probability
            att_prob = F.softmax(att_score, dim=-1)  # (Ni, Ni)
            att_value = torch.mm(att_prob, value)  # (Ni, out_channels)
            relation_features.append(att_value)
        
        # 3. collect batch feature back
        relation_features = torch.cat(relation_features, dim=0)  # (M, out_channels)
        relation_features = self.out_norm(relation_features)  # (M, out_channels)
        
        return relation_features


@ROI_HEADS_REGISTRY.register()
class RelationROIHeads(ROIHeads):
    """
    It's relation-based ROI head including ROI feature relation module.
    There is ROI feature sharing between tasks after pooler and head, before predictor.
    After pooler-head, relation module interacts specific ROI features, then each head 
    independently processes the features by each head's own predictor.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_relations: List[nn.Module],
        box_predictor: nn.Module,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head.
            box_head (nn.Module): transform features to make box predictions.
            box_relations (List[nn.Module]): coarse classify and interact features.
                Especially enhance positive instances' features.
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_relations = box_relations
        self.box_predictor = box_predictor
        for i in range(len(self.box_relations)):
            relation = self.box_relations[i]
            self.add_module("relation_module{}".format(i), relation)

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If RelationROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg, ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution)
        )

        # relation module
        num_relations = 4
        box_out_channels = box_head.output_shape.channels
        relation_out_channels = box_out_channels//num_relations
        box_relations = []
        for _ in range(num_relations):
            relation = RelationModule(
                in_channels=box_out_channels, 
                out_channels=relation_out_channels,
                num_features=relation_out_channels,
                fuse_location=True,
                is_relative_location=True,
            )
            box_relations.append(relation)
            
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_relations": box_relations,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features, proposals)
            return proposals, losses
        else:
            pred_instances, box_features_mem = self._forward_box(features, proposals)
            return pred_instances, box_features_mem, {}

    def _forward_box(self, features: Dict[str, torch.Tensor], proposals: List[Instances]):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)

        # contrastive loss
        if USE_CONTRASTIVE_LOSS:
            loss_contrastive = cal_contrastive_loss(box_features, proposals)

        # doing relations using relation modules
        relation_box_features = []
        box_features_mem = [box_features]
        for i in range(len(self.box_relations)):
            box_relation = self.box_relations[i]
            relation_feature = box_relation(box_features, proposals)
            relation_box_features.append(relation_feature) #(M, out_channels)
        relation_features = torch.cat(relation_box_features,dim=1)  #(M, in_channels)
        
        # shortcut connection
        box_features = box_features + relation_features
        predictions = self.box_predictor(box_features)
        box_features_mem.append(box_features)
        del box_features

        if self.training:
            losses = self.box_predictor.losses(predictions, proposals)
            if USE_CONTRASTIVE_LOSS:
                losses['loss_contrastive'] = loss_contrastive
            return losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances, box_features_mem

    