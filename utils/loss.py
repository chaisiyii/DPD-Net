import logging
import torch
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)
USE_CONTRASTIVE_LOSS = True
USE_CONTRASTIVE_LOSS_ROI_HEADS = False

def cal_contrastive_loss(features, proposals, temperature=5, sigma=0.001):
    '''
    Calculate cantrastive loss for scores.
    Args:
    features (tensor): (M, f), features for batches.
    proposals (List[Instances]): proposals for batches. Each has fields 
    "proposal_boxes", and "objectness_logits", "gt_classes", "gt_boxes".
    subsample: whether subsample the background part.
    Notes:
    In final model the params: t=10, sigma=0.001
    '''
    # per-image operation
    num_prop_per_image = [len(p) for p in proposals]
    # from tensor (M, num_features) to list[tensor: (Ni, num_features)]
    features = features.split(num_prop_per_image)
    contrastive_loss = torch.tensor(0).to(features[0].device)  # accumulate for each image

    for features_per_image, proposals_per_image in zip(features, proposals):
        if not proposals_per_image.has("gt_classes"):
            continue
        # find foreground and background proposals
        gt_classes = proposals_per_image.gt_classes
        bg_class = torch.max(gt_classes)
        # gt_class == bg_class means in background
        choice = (gt_classes == 0)  # abnormal nuc/cell
        # choice = (gt_classes>=0) & (gt_classes<bg_class)  #tp(fg) mask OLD_VERSION BEFORE NEG NUC
        if torch.sum(choice.float())==0:  # no tp(fg) predicted
            continue
        fg_feature = features_per_image[choice]  # foreground proposal features
        bg_feature = features_per_image[~choice]  # background proposal features
        fg, bg = fg_feature.shape[0], bg_feature.shape[0]
        feature = torch.cat((fg_feature, bg_feature), dim=0)  # (n,f)
        gt_classes = torch.zeros((fg+bg, 1)).to(feature.device)  # (n,1)
        gt_classes[0:fg] = 1

        # prepare pair-wise mask (n,n)
        mask = torch.eq(gt_classes, gt_classes.T).float()  # who have same gt will get 1
        logits_mask = torch.ones_like(mask)-torch.eye(fg+bg).to(mask.device)  # diag 0 other 1
        positive_mask = mask * logits_mask  # positive pair mask except self
        negative_mask = 1. - mask  # negative pair mask from different classes
        positive_num = 2*torch.sum(positive_mask, dim=1)-1  # (n,1) for 2*N_yi-1 for each i
        
        # calculate similarity
        logits = torch.matmul(feature, feature.T)/temperature  # (n,n)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)  # (n,1)
        logits = logits - logits_max.detach()  # for numerical stability  # (n,n)
        exp_logits = torch.exp(logits)  # (n,n)

        # calculate denominators
        denominator = exp_logits*positive_mask + exp_logits*negative_mask  # (n,n)
        denominator = torch.sum(denominator, dim=1)  # (n,1)
        denominator = torch.clamp(denominator, min=1e-13)  # (n,1)

        # loss
        log_probs = logits - torch.log(denominator)  # (n,n) equals log(exp_logits/denominator)
        if torch.any(torch.isnan(log_probs)):
            raise ValueError("Log_prob has nan!")
        log_probs = torch.sum(log_probs*positive_mask , dim=1)/positive_num  # (n,1)
        loss = -log_probs*sigma
        loss = torch.mean(loss)  # (n,1) -> (1,1)
        contrastive_loss = contrastive_loss + loss

    return contrastive_loss

