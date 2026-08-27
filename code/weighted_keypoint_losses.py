"""Weighted YOLO pose losses used by the training workflow.

These losses operate on YOLO's coordinate predictions, not segmentation
heatmaps. Keeping them in an importable module also makes checkpoints portable.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from ultralytics.models.yolo.pose.train import PoseTrainer
from ultralytics.nn.tasks import PoseModel
from ultralytics.utils import RANK
from ultralytics.utils.loss import v8PoseLoss


def _weighted_mean(loss: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Mean over visible keypoints, normalized so weights do not alter global scale."""
    active_weights = mask.to(loss.dtype) * weights.view(1, -1)
    return ((loss * active_weights).sum(1) / active_weights.sum(1).clamp_min(1e-9)).mean()


class WeightedOKSKeypointLoss(nn.Module):
    """RECOMMENDED: native YOLO/OKS location error with per-keypoint importance."""

    def __init__(self, sigmas: torch.Tensor, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("sigmas", sigmas)
        self.register_buffer("weights", weights)

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        squared_distance = (pred_kpts[..., :2] - gt_kpts[..., :2]).square().sum(-1)
        # Same OKS-shaped error as Ultralytics KeypointLoss; only its reduction is weighted.
        error = squared_distance / ((2 * self.sigmas).square() * (area + 1e-9) * 2)
        return _weighted_mean(1 - torch.exp(-error), kpt_mask, self.weights)


class WeightedSmoothL1KeypointLoss(nn.Module):
    """Robust alternative: area-normalized Smooth-L1 coordinate error; easy to tune but not OKS-aligned."""

    def __init__(self, weights: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("weights", weights)

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        scale = area.sqrt().clamp_min(1e-6).unsqueeze(-1)
        normalized_delta = (pred_kpts[..., :2] - gt_kpts[..., :2]) / scale
        point_loss = F.smooth_l1_loss(normalized_delta, torch.zeros_like(normalized_delta), reduction="none").sum(-1)
        return _weighted_mean(point_loss, kpt_mask, self.weights)


class DynamicWeightedOKSLoss(WeightedOKSKeypointLoss):
    """Experimental: OKS loss whose weights track hard keypoints via an error EMA (best on one GPU)."""

    def __init__(self, sigmas, weights, momentum: float = 0.95, limit: float = 2.0) -> None:
        super().__init__(sigmas, weights)
        self.register_buffer("base_weights", weights.clone())
        self.register_buffer("error_ema", torch.ones_like(weights))
        self.momentum = momentum
        self.limit = limit

    def forward(self, pred_kpts, gt_kpts, kpt_mask, area):
        squared_distance = (pred_kpts[..., :2] - gt_kpts[..., :2]).square().sum(-1)
        error = 1 - torch.exp(-squared_distance / ((2 * self.sigmas).square() * (area + 1e-9) * 2))
        with torch.no_grad():
            visible = kpt_mask.sum(0).clamp_min(1)
            batch_error = (error.detach() * kpt_mask).sum(0) / visible
            present = kpt_mask.any(0)
            self.error_ema[present] = torch.lerp(
                self.error_ema[present], batch_error[present], 1 - self.momentum
            )
            relative = self.error_ema / self.error_ema.mean().clamp_min(1e-9)
            self.weights.copy_(self.base_weights * relative.clamp(1 / self.limit, self.limit))
            self.weights.div_(self.weights.mean().clamp_min(1e-9))
        return _weighted_mean(error, kpt_mask, self.weights)


class WeightedPoseLoss(v8PoseLoss):
    """Ultralytics pose criterion with a selectable location-loss reduction."""

    def __init__(self, model, variant: str, weights: Sequence[float]) -> None:
        super().__init__(model)
        tensor_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        if tensor_weights.numel() != self.kpt_shape[0]:
            raise ValueError(f"Configured {tensor_weights.numel()} weights but model has {self.kpt_shape[0]} keypoints")
        if not torch.isfinite(tensor_weights).all() or (tensor_weights <= 0).any():
            raise ValueError("All keypoint weights must be finite and greater than zero")
        if variant == "weighted_oks":
            self.keypoint_loss = WeightedOKSKeypointLoss(self.keypoint_loss.sigmas, tensor_weights)
        elif variant == "weighted_smooth_l1":
            self.keypoint_loss = WeightedSmoothL1KeypointLoss(tensor_weights)
        elif variant == "dynamic_oks":
            self.keypoint_loss = DynamicWeightedOKSLoss(self.keypoint_loss.sigmas, tensor_weights)
        else:
            raise ValueError(f"Unknown weighted pose loss: {variant}")


class WeightedPoseModel(PoseModel):
    """Importable model class, so its training checkpoints remain loadable."""

    loss_variant = "weighted_oks"
    keypoint_weights: tuple[float, ...] = ()

    def init_criterion(self):
        return WeightedPoseLoss(self, self.loss_variant, self.keypoint_weights)


def make_weighted_pose_trainer(variant: str, weights: Sequence[float], keypoint_names: Sequence[str]):
    """Create the trainer class expected by ``YOLO.train(trainer=...)``."""
    configured_weights = tuple(float(value) for value in weights)
    configured_names = tuple(keypoint_names)
    WeightedPoseModel.loss_variant = variant
    WeightedPoseModel.keypoint_weights = configured_weights

    class WeightedPoseTrainer(PoseTrainer):
        """PoseTrainer that builds WeightedPoseModel and loads pretrained weights."""

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = WeightedPoseModel(cfg, nc=self.data["nc"], data_kpt_shape=self.data["kpt_shape"], verbose=verbose and RANK == -1)
            if model.model[-1].kpt_shape[0] != len(configured_names):
                raise ValueError(
                    f"Dataset/model defines {model.model[-1].kpt_shape[0]} keypoints, but KEYPOINT_NAMES has "
                    f"{len(configured_names)}. Make their order and length match dataset.yaml."
                )
            if weights:
                model.load(weights)
            return model

    return WeightedPoseTrainer
