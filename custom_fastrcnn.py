# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Dict, List, Tuple, Union
import torch
from fvcore.nn import giou_loss, smooth_l1_loss
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Linear, ShapeSpec, batched_nms, cat, nonzero_tuple
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.structures import Boxes, Instances
from detectron2.utils.events import get_event_storage

from polyogn_iou_loss import c_poly_loss

__all__ = ["fast_rcnn_inference", "FastRCNNOutputLayers"]

logger = logging.getLogger(__name__)

"""
Shape shorthand in this module:

    N: number of images in the minibatch
    R: number of ROIs, combined over all images, in the minibatch
    Ri: number of ROIs in image i
    K: number of foreground classes. E.g.,there are 80 foreground classes in COCO.

Naming convention:

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`).

    pred_class_logits: predicted class scores in [-inf, +inf]; use
        softmax(pred_class_logits) to estimate P(class).

    gt_classes: ground-truth classification labels in [0, K], where [0, K) represent
        foreground object classes and K represents the background class.

    pred_proposal_deltas: predicted box2box transform deltas for transforming proposals
        to detection box predictions.

    gt_proposal_deltas: ground-truth box2box transform deltas
"""


def fast_rcnn_inference(
        boxes: List[torch.Tensor],
        bases: List[torch.Tensor],
        scores: List[torch.Tensor],
        image_shapes: List[Tuple[int, int]],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        fast_rcnn_inference_single_image(
            boxes_per_image, scores_per_image, bases_per_image, image_shape, score_thresh, nms_thresh, topk_per_image
        )
        for scores_per_image, boxes_per_image, bases_per_image, image_shape in zip(scores, boxes, bases, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def delta_to_h(delta_h, boxes):
    y1 = boxes[::, 1]
    y2 = boxes[::, 3]
    dy = y2 - y1
    return dy + delta_h * dy


def delta_to_bases(bases, boxes):
    midx = bases[::, 0]
    midy = bases[::, 1]
    firstx = bases[::, 2]
    firsty = bases[::, 3]
    secondx = bases[::, 4]
    secondy = bases[::, 5]

    x1 = boxes[::, 0]
    y1 = boxes[::, 1]
    x2 = boxes[::, 2]
    y2 = boxes[::, 3]
    dx = x2 - x1
    dy = y2 - y1
    midx = (x1 + x2) / 2 + bases[::, 0] * dx
    midy = (y1 + y2) / 2 + bases[::, 1] * dy
    x1 = midx + firstx * dx
    y1 = midy + firsty * dy
    x2 = midx + secondx * dx
    y2 = midy + secondy * dy
    x3 = midx - secondx * dx
    y3 = midy - secondy * dy
    x4 = midx - firstx * dx
    y4 = midy - firsty * dy

    return torch.stack((x1, y1, x2, y2, x3, y3, x4, y4, midx, midy), dim=-1)


def fast_rcnn_inference_single_image(
        boxes,
        scores,
        bases,
        image_shape: Tuple[int, int],
        score_thresh: float,
        nms_thresh: float,
        topk_per_image: int,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        bases = bases[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # print("num_bbox_reg_classes: ", num_bbox_reg_classes)
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    # print("image_shape: ", image_shape)
    # print(boxes.tensor.shape)
    # print(bases.shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4
    # print(boxes.shape)
    bases = bases.view(-1, num_bbox_reg_classes, 7)
    # print(bases.shape)

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # print(filter_mask.shape)
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    bases = bases[filter_mask]
    scores = scores[filter_mask]

    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, bases, scores, filter_inds = boxes[keep], bases[keep], scores[keep], filter_inds[keep]

    h = bases[:, 6]
    bases = bases[:, 0:6]
    bases = delta_to_bases(bases, boxes)
    h = delta_to_h(h, boxes)
    # print(mid.shape)
    # print(mid)

    result = Instances(image_shape)
    result.pred_bases = bases
    result.pred_h = h
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]
    # print(image_shape)
    # print(result)
    return result, filter_inds[:, 0]


class FastRCNNOutputs:
    """
    An internal implementation that stores information about outputs of a Fast R-CNN head,
    and provides methods that are used to decode the outputs of a Fast R-CNN head.
    """

    def __init__(
            self,
            box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            smooth_l1_beta=0.0,
            box_reg_loss_type="smooth_l1",
    ):
        """
        Args:
            box2box_transform (Box2BoxTransform/Box2BoxTransformRotated):
                box2box transform instance for proposal-to-detection transformations.
            pred_class_logits (Tensor): A tensor of shape (R, K + 1) storing the predicted class
                logits for all R predicted object instances.
                Each row corresponds to a predicted object instance.
            pred_proposal_deltas (Tensor): A tensor of shape (R, K * B) or (R, B) for
                class-specific or class-agnostic regression. It stores the predicted deltas that
                transform proposals into final box detections.
                B is the box dimension (4 or 5).
                When B is 4, each row is [dx, dy, dw, dh (, ....)].
                When B is 5, each row is [dx, dy, dw, dh, da (, ....)].
            proposals (list[Instances]): A list of N Instances, where Instances i stores the
                proposals for image i, in the field "proposal_boxes".
                When training, each Instances must have ground-truth labels
                stored in the field "gt_classes" and "gt_boxes".
                The total number of all instances must be equal to R.
            smooth_l1_beta (float): The transition point between L1 and L2 loss in
                the smooth L1 loss function. When set to 0, the loss becomes L1. When
                set to +inf, the loss becomes constant 0.
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
        """
        self.box2box_transform = box2box_transform
        self.num_preds_per_image = [len(p) for p in proposals]
        self.pred_class_logits = pred_class_logits
        self.pred_proposal_deltas, self.pred_bases = pred_proposal_deltas
        self.smooth_l1_beta = smooth_l1_beta
        self.box_reg_loss_type = box_reg_loss_type

        self.image_shapes = [x.image_size for x in proposals]

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            self.proposals = box_type.cat([p.proposal_boxes for p in proposals])
            assert (
                not self.proposals.tensor.requires_grad
            ), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                self.gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                self.gt_classes = cat([p.gt_classes for p in proposals], dim=0)
                assert proposals[0].has("gt_h")
                self.gt_h = cat([p.gt_h for p in proposals], dim=0)
                assert proposals[0].has("gt_bases")
                self.gt_bases = cat([p.gt_bases for p in proposals], dim=0)
        else:
            self.proposals = Boxes(torch.zeros(0, 4, device=self.pred_proposal_deltas.device))
        self._no_instances = len(self.proposals) == 0  # no instances found

    def _log_accuracy(self):
        """
        Log the accuracy metrics to EventStorage.
        """
        num_instances = self.gt_classes.numel()
        pred_classes = self.pred_class_logits.argmax(dim=1)
        bg_class_ind = self.pred_class_logits.shape[1] - 1

        fg_inds = (self.gt_classes >= 0) & (self.gt_classes < bg_class_ind)
        num_fg = fg_inds.nonzero().numel()
        fg_gt_classes = self.gt_classes[fg_inds]
        fg_pred_classes = pred_classes[fg_inds]

        num_false_negative = (fg_pred_classes == bg_class_ind).nonzero().numel()
        num_accurate = (pred_classes == self.gt_classes).nonzero().numel()
        fg_num_accurate = (fg_pred_classes == fg_gt_classes).nonzero().numel()

        storage = get_event_storage()
        if num_instances > 0:
            storage.put_scalar("fast_rcnn/cls_accuracy", num_accurate / num_instances)
            if num_fg > 0:
                storage.put_scalar("fast_rcnn/fg_cls_accuracy", fg_num_accurate / num_fg)
                storage.put_scalar("fast_rcnn/false_negative", num_false_negative / num_fg)

    def softmax_cross_entropy_loss(self):
        """
        Compute the softmax cross entropy loss for box classification.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_class_logits.sum()
        else:
            self._log_accuracy()
            return F.cross_entropy(self.pred_class_logits, self.gt_classes, reduction="mean")

    def box_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_proposal_deltas.sum()

        box_dim = self.gt_boxes.tensor.size(1)  # 4 or 5
        cls_agnostic_bbox_reg = self.pred_proposal_deltas.size(1) == box_dim
        device = self.pred_proposal_deltas.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        if cls_agnostic_bbox_reg:
            # pred_proposal_deltas only corresponds to foreground class for agnostic
            gt_class_cols = torch.arange(box_dim, device=device)
        else:
            fg_gt_classes = self.gt_classes[fg_inds]
            # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
            # where b is the dimension of box representation (4 or 5)
            # Note that compared to Detectron1,
            # we do not perform bounding box regression for background classes.
            gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)

        if self.box_reg_loss_type == "smooth_l1":
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                self.proposals.tensor, self.gt_boxes.tensor
            )
            # print(self.gt_boxes.tensor.shape)
            # print(gt_proposal_deltas.shape)
            # print("boxes: ", gt_class_cols)
            # print(fg_inds[:, None])
            # print(self.pred_proposal_deltas)
            # print(self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols])
            loss_box_reg = smooth_l1_loss(
                self.pred_proposal_deltas[fg_inds[:, None], gt_class_cols],
                gt_proposal_deltas[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        elif self.box_reg_loss_type == "giou":
            loss_box_reg = giou_loss(
                self._predict_boxes()[fg_inds[:, None], gt_class_cols],
                self.gt_boxes.tensor[fg_inds],
                reduction="sum",
            )
        else:
            raise ValueError(f"Invalid bbox reg loss type '{self.box_reg_loss_type}'")

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        loss_box_reg = loss_box_reg / self.gt_classes.numel()
        return loss_box_reg

    def base_reg_loss(self):
        """
        Compute the smooth L1 loss for box regression.

        Returns:
            scalar Tensor
        """
        if self._no_instances:
            return 0.0 * self.pred_bases.sum()

        # print(self.gt_bases.shape)
        # box_dim = self.gt_bases.size(1)  # 4 or 5
        box_dim = 7  # 4 or 5
        device = self.pred_bases.device

        bg_class_ind = self.pred_class_logits.shape[1] - 1

        # Box delta loss is only computed between the prediction for the gt class k
        # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
        # for non-gt classes and background.
        # Empty fg_inds produces a valid loss of zero as long as the size_average
        # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
        # and would produce a nan loss).
        fg_inds = nonzero_tuple((self.gt_classes >= 0) & (self.gt_classes < bg_class_ind))[0]
        fg_gt_classes = self.gt_classes[fg_inds]
        # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
        # where b is the dimension of box representation (4 or 5)
        # Note that compared to Detectron1,
        # we do not perform bounding box regression for background classes.
        gt_class_cols = box_dim * fg_gt_classes[:, None] + torch.arange(box_dim, device=device)
        # print(self.proposals.tensor[fg_inds].shape)
        # print(self.pred_bases[fg_inds[:, None], gt_class_cols].shape)

        # proposal_bases = self.pred_bases[fg_inds[:, None], gt_class_cols]
        # midx = proposal_bases[:, 0]
        # midy = proposal_bases[:, 0]
        # firstx = proposal_bases[:, 2]
        # firsty = proposal_bases[:, 3]
        # secondx = proposal_bases[:, 4]
        # secondy = proposal_bases[:, 5]
        #
        # x1 = self.proposals.tensor[fg_inds][:, 0]
        # y1 = self.proposals.tensor[fg_inds][:, 1]
        # x2 = self.proposals.tensor[fg_inds][:, 2]
        # y2 = self.proposals.tensor[fg_inds][:, 3]
        # proposal_midx = (x1 + x2) / 2
        # proposal_midy = y1 + y2
        # dx = x2 - x1
        # dy = y2 - y1
        # midx = (x1 + x2) / 2 + midx * dx
        # midy = y2 - midy * dy
        # mid = torch.stack((midx, midy), dim=1)
        #
        # firstx = firstx * dx
        # firsty = firsty * dy
        # secondx = secondx * dx
        # secondy = secondy * dy
        # first = torch.stack((firstx, firsty), dim=1)
        # second = torch.stack((secondx, secondy), dim=1)
        # # print(mid.shape)
        # # print(mid)
        # bases_transformed = torch.cat((mid - first, mid + second, mid - second, mid + first), dim=1)
        # # bases_transformed = torch.stack((x1, x2, x2, y1, x1, y2, x2, y2), dim=1)
        # # bases_transformed = bases_transformed * (1 + self.pred_bases[fg_inds[:, None], gt_class_cols])
        # # print(1 + self.pred_bases[fg_inds[:, None], gt_class_cols])
        #
        # # print(bases_transformed)
        # # print(self.gt_bases[fg_inds])

        gt_class_cols_boxes = torch.arange(4, device=device)
        pred_boxes = self.box2box_transform.apply_deltas(self.pred_proposal_deltas[:, gt_class_cols_boxes], self.proposals.tensor)

        # print(delta_to_bases(self.pred_bases[fg_inds[:, None], gt_class_cols], pred_boxes[fg_inds]))
        # print(self.gt_bases)
        # print(pred_boxes.shape)
        # print(self.gt_boxes.tensor.shape)
        # print(pred_boxes)
        # print(self.gt_boxes.tensor)

        # x1 = self.proposals.tensor[:, 0]
        # y1 = self.proposals.tensor[:, 1]
        # x2 = self.proposals.tensor[:, 2]
        # y2 = self.proposals.tensor[:, 3]
        x1 = pred_boxes[:, 0]
        y1 = pred_boxes[:, 1]
        x2 = pred_boxes[:, 2]
        y2 = pred_boxes[:, 3]
        proposal_midx = (x1 + x2) / 2
        proposal_midy = (y1 + y2) / 2
        dx = x2 - x1
        dy = y2 - y1
        gt_bases_midx = (self.gt_bases[:, 0] + self.gt_bases[:, 2] + self.gt_bases[:, 4] + self.gt_bases[:, 6]) / 4
        gt_bases_midy = (self.gt_bases[:, 1] + self.gt_bases[:, 3] + self.gt_bases[:, 5] + self.gt_bases[:, 7]) / 4
        gt_bases_midx1 = self.gt_bases[:, 0]
        gt_bases_midy1 = self.gt_bases[:, 1]
        gt_bases_midx2 = self.gt_bases[:, 2]
        gt_bases_midy2 = self.gt_bases[:, 3]

        gt_bases_midx1 = gt_bases_midx1 - gt_bases_midx
        gt_bases_midy1 = gt_bases_midy1 - gt_bases_midy
        gt_bases_midx2 = gt_bases_midx2 - gt_bases_midx
        gt_bases_midy2 = gt_bases_midy2 - gt_bases_midy

        gt_bases_midx = gt_bases_midx - proposal_midx
        gt_bases_midy = gt_bases_midy - proposal_midy
        gt_bases_midx = gt_bases_midx / dx
        gt_bases_midy = gt_bases_midy / dy
        gt_bases_mid = torch.stack((gt_bases_midx, gt_bases_midy), dim=1)

        gt_bases_midx1 = gt_bases_midx1 / dx
        gt_bases_midy1 = gt_bases_midy1 / dy
        gt_bases_midx2 = gt_bases_midx2 / dx
        gt_bases_midy2 = gt_bases_midy2 / dy

        gt_h_delta = (self.gt_h - dy) / dy

        gt_bases_delta = torch.stack((gt_bases_midx, gt_bases_midy, gt_bases_midx1, gt_bases_midy1,
                                      gt_bases_midx2, gt_bases_midy2, gt_h_delta), dim=1)
        # print(gt_bases_delta[fg_inds][0, :])
        # print(self.pred_bases[fg_inds[:, None], gt_class_cols][0, :])

        POLY = False
        if not POLY:
            # print("bases: ", gt_class_cols)
            # loss_base_reg = 1e-4 * smooth_l1_loss(
            #     self.pred_bases[fg_inds[:, None], gt_class_cols],
            #     self.gt_bases[fg_inds],
            #     self.smooth_l1_beta,
            #     reduction="sum",
            # )
            # print(bases_transformed)
            # print(self.gt_bases[fg_inds])
            # loss_base_reg = 1e-3 * smooth_l1_loss(
            #     bases_transformed,
            #     self.gt_bases[fg_inds],
            #     self.smooth_l1_beta,
            #     reduction="sum",
            # )
            loss_base_reg = smooth_l1_loss(
                self.pred_bases[fg_inds[:, None], gt_class_cols],
                gt_bases_delta[fg_inds],
                self.smooth_l1_beta,
                reduction="sum",
            )
        else:
            # preds = self.pred_bases[fg_inds[:, None], gt_class_cols]
            # print(preds.shape)
            gts = self.gt_bases[fg_inds]
            loss_base_reg = 0
            for idx in range(bases_transformed.shape[0]):
                poly_loss = c_poly_loss(bases_transformed[idx, :].view(4, 2), gts[idx, :].view(4, 2))
                if (1 - poly_loss) < 1e-5:
                    loss_base_reg += 1e-4 * smooth_l1_loss(
                        bases_transformed[idx, :],
                        gts[idx, :],
                        self.smooth_l1_beta,
                        reduction="sum", )
                else:
                    print(poly_loss)
                    loss_base_reg += poly_loss
                # print(loss_base_reg)

            # print(loss_base_reg)
            # print(loss_base_reg)
            # print(self.gt_classes.numel())

        # The loss is normalized using the total number of regions (R), not the number
        # of foreground regions even though the box regression loss is only defined on
        # foreground regions. Why? Because doing so gives equal training influence to
        # each foreground example. To see how, consider two different minibatches:
        #  (1) Contains a single foreground region
        #  (2) Contains 100 foreground regions
        # If we normalize by the number of foreground regions, the single example in
        # minibatch (1) will be given 100 times as much influence as each foreground
        # example in minibatch (2). Normalizing by the total number of regions, R,
        # means that the single example in minibatch (1) and each of the 100 examples
        # in minibatch (2) are given equal influence.
        # print(loss_base_reg)
        loss_base_reg = loss_base_reg / self.gt_classes.numel()
        # print(loss_base_reg)
        return loss_base_reg * 10

    def _predict_boxes(self):
        """
        Returns:
            Tensor: A Tensors of predicted class-specific or class-agnostic boxes
                for all images in a batch. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        return self.box2box_transform.apply_deltas(self.pred_proposal_deltas, self.proposals.tensor)

    """
    A subclass is expected to have the following methods because
    they are used to query information about the head predictions.
    """

    def losses(self):
        """
        Compute the default losses for box head in Fast(er) R-CNN,
        with softmax cross entropy loss and smooth L1 loss.

        Returns:
            A dict of losses (scalar tensors) containing keys "loss_cls" and "loss_box_reg".
        """
        return {"loss_cls": self.softmax_cross_entropy_loss(), "loss_box_reg": self.box_reg_loss(),
                "loss_base_reg": self.base_reg_loss()}

    def predict_boxes(self):
        """
        Deprecated
        """
        return self._predict_boxes().split(self.num_preds_per_image, dim=0)

    def predict_probs(self):
        """
        Deprecated
        """
        probs = F.softmax(self.pred_class_logits, dim=-1)
        return probs.split(self.num_preds_per_image, dim=0)

    def inference(self, score_thresh, nms_thresh, topk_per_image):
        """
        Deprecated
        """
        boxes = self.predict_boxes()
        scores = self.predict_probs()
        image_shapes = self.image_shapes
        return fast_rcnn_inference(
            boxes, scores, image_shapes, score_thresh, nms_thresh, topk_per_image
        )


class NewFastRCNNOutputLayers(nn.Module):
    """
    Two linear layers for predicting Fast R-CNN outputs:

    1. proposal-to-detection box regression deltas
    2. classification scores
    """

    @configurable
    def __init__(
            self,
            input_shape: ShapeSpec,
            *,
            box2box_transform,
            num_classes: int,
            test_score_thresh: float = 0.0,
            test_nms_thresh: float = 0.5,
            test_topk_per_image: int = 100,
            cls_agnostic_bbox_reg: bool = False,
            smooth_l1_beta: float = 0.0,
            box_reg_loss_type: str = "smooth_l1",
            loss_weight: Union[float, Dict[str, float]] = 1.0,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss. Only used if
                `box_reg_loss_type` is "smooth_l1"
            box_reg_loss_type (str): Box regression loss type. One of: "smooth_l1", "giou"
            loss_weight (float|dict): weights to use for losses. Can be single float for weighting
                all losses, or a dict of individual weightings. Valid dict keys are:
                    * "loss_cls": applied to classification loss
                    * "loss_box_reg": applied to box regression loss
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * (input_shape.width or 1) * (input_shape.height or 1)
        # prediction layer for num_classes foreground classes and one background class (hence + 1)
        self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)
        # modified
        self.base_pred = Linear(input_size, num_bbox_reg_classes * 7)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        # nn.init.normal_(self.base_pred.weight, mean=0.6, std=0.6)
        nn.init.normal_(self.base_pred.weight, std=0.003)
        # print(self.base_pred.weight.shape)
        # for idx in range(num_bbox_reg_classes):
        #     nn.init.normal_(self.base_pred.weight[idx*6, :], mean=0.6, std=0.01)
        #     nn.init.normal_(self.base_pred.weight[idx * 6 + 1, :], mean=0.205, std=0.01)
        #     nn.init.normal_(self.base_pred.weight[idx * 6 + 2, :], mean=0.575, std=0.01)
        #     nn.init.normal_(self.base_pred.weight[idx * 6 + 3, :], mean=0.3525, std=0.01)
        #     nn.init.normal_(self.base_pred.weight[idx * 6 + 4, :], mean=0.575, std=0.01)
        #     nn.init.normal_(self.base_pred.weight[idx * 6 + 5, :], mean=-0.3525, std=0.01)
        # self.base_pred.weight[:, idx * 6 + 2:idx * 6 + 4] = 0.6
        # self.base_pred.weight[:, idx * 6 + 4] = 0.6
        # self.base_pred.weight[:, idx * 6 + 5] = 0.6
        for l in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(l.bias, 0)

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image
        self.box_reg_loss_type = box_reg_loss_type
        if isinstance(loss_weight, float):
            loss_weight = {"loss_cls": loss_weight, "loss_box_reg": loss_weight, "loss_base_reg": loss_weight}
        self.loss_weight = loss_weight
        # modified
        # for param in self.parameters():
        #     param.requires_grad = True

    @classmethod
    def from_config(cls, cfg, input_shape):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "box_reg_loss_type": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE,
            "loss_weight": {"loss_box_reg": cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_WEIGHT},
            # fmt: on
        }

    def forward(self, x):
        """
        Args:
            x: per-region features of shape (N, ...) for N bounding boxes to predict.

        Returns:
            (Tensor, Tensor):
            First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.

            Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)
        proposal_bases = self.base_pred(x)

        # proposal_bases = proposal_bases.view(-1, 9, 6)
        # mid = proposal_bases[:, :, 0:2]
        # first = proposal_bases[:, :, 2:4]
        # second = proposal_bases[:, :, 4:6]
        # proposal_bases = torch.cat((mid + first, mid + second, mid - first, mid - second), dim=2)
        # proposal_bases = proposal_bases.view(-1, 72)
        # print(proposal_bases[0,:,:])
        # print(proposal_bases.view(-1, 72).shape)
        return (scores, proposal_deltas), proposal_bases

    # TODO: move the implementation to this class.
    def losses(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_boxes``,
                ``gt_classes`` are expected.

        Returns:
            Dict[str, Tensor]: dict of losses
        """
        # for param in self.parameters():
        #     print(param)
        predictions, pred_bases = predictions
        scores, proposal_deltas = predictions
        # print(pred_bases)
        # print(proposal_deltas)
        losses = FastRCNNOutputs(
            self.box2box_transform,
            scores,
            # modified
            (proposal_deltas, pred_bases),
            proposals,
            self.smooth_l1_beta,
            self.box_reg_loss_type,
        ).losses()
        # for param in self.parameters():
        #     print(param)
        # return {}
        return {k: v * self.loss_weight.get(k, 1.0) for k, v in losses.items()}

    def inference(self, predictions: Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor],
                  proposals: List[Instances]):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        predictions, pred_bases = predictions

        # _, proposal_deltas = predictions
        # proposal_boxes = [p.proposal_boxes for p in proposals]
        # proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        # predict_boxes = self.box2box_transform.apply_deltas(proposal_deltas, proposal_boxes)
        # print("weights: ", self.box2box_transform.weights)
        #
        # pred_bases = delta_to_bases(pred_bases, predict_boxes)
        # print(predict_boxes)
        # print(pred_bases)
        # print(predict_boxes.shape)
        # print(pred_bases.shape)

        boxes = self.predict_boxes(predictions, proposals)
        bases = self.predict_bases(pred_bases, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]
        return fast_rcnn_inference(
            boxes,
            bases,
            scores,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were used
                to compute predictions. The fields ``proposal_boxes``, ``gt_classes`` are expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[
                torch.arange(N, dtype=torch.long, device=predict_boxes.device), gt_classes
            ]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions. The ``proposal_boxes`` field is expected.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of proposals for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            # ensure fp32 for decoding precision
            proposal_deltas,
            proposal_boxes,
        )  # Nx(KxB)
        # print("boxes:", predict_boxes.shape)
        return predict_boxes.split(num_prop_per_image)

    def predict_bases(
            self, pred_bases: torch.Tensor, proposals: List[Instances]
    ):
        num_inst_per_image = [len(p) for p in proposals]
        # print("bases:", pred_bases.shape)
        return pred_bases.split(num_inst_per_image, dim=0)

    def predict_probs(
            self, predictions: Tuple[torch.Tensor, torch.Tensor], proposals: List[Instances]
    ):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features that were
                used to compute predictions.

        Returns:
            list[Tensor]:
                A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of proposals for image i.
        """
        scores, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)
