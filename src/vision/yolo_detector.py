"""
Unit IV: YOLO-Style Multi-Task Hand Detector and Gesture Classifier.
Predicts bounding box coordinates, objectness presence, and sign class logits simultaneously.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SignYOLODetector(nn.Module):
    """
    YOLO-style Anchor-Free Multi-Task Vision Detector for Sign Language:
    Simultaneously performs:
    1. Gesture classification across 98 categories.
    2. Hand bounding box localization [x_center, y_center, width, height] for left and right hands.
    3. Objectness confidence (hand detection certainty).
    """

    def __init__(self, in_features: int = 126, num_classes: int = 98, dropout: float = 0.25):
        super().__init__()
        self.in_features = in_features
        self.num_classes = num_classes

        # Multi-scale CNN backbone
        self.conv1 = nn.Conv1d(in_features, 64, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(128)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

        # Multi-task heads
        self.cls_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        # Bounding box head: [x_min, y_min, x_max, y_max] for left & right hands (8 coordinates)
        self.bbox_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 8),
            nn.Sigmoid(),  # Coordinates normalized to [0, 1]
        )

        # Objectness confidence head: left hand confidence, right hand confidence
        self.obj_head = nn.Sequential(
            nn.Linear(256, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: shape (Batch, 25, 126) or (Batch, 126, 25)
        Returns:
            cls_logits: (Batch, num_classes)
            bboxes: (Batch, 8) normalized hand boxes
            objectness: (Batch, 2) detection confidences
        """
        if x.dim() == 3 and x.shape[1] == 25 and x.shape[2] == self.in_features:
            x = x.permute(0, 2, 1)

        feat = F.relu(self.bn1(self.conv1(x)))
        feat = F.relu(self.bn2(self.conv2(feat)))
        feat = F.relu(self.bn3(self.conv3(feat)))

        pooled = self.pool(feat).squeeze(-1)
        pooled = self.dropout(pooled)

        cls_logits = self.cls_head(pooled)
        bboxes = self.bbox_head(pooled)
        objectness = self.obj_head(pooled)

        return cls_logits, bboxes, objectness

    def compute_multitask_loss(
        self,
        cls_logits: torch.Tensor,
        bboxes: torch.Tensor,
        objectness: torch.Tensor,
        y_true_cls: torch.Tensor,
        target_bboxes: torch.Tensor = None,
        target_obj: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Multi-task Loss = L_cls + lambda_box * L_box + lambda_obj * L_obj
        """
        loss_cls = F.cross_entropy(cls_logits, y_true_cls)

        if target_bboxes is None:
            # Generate pseudo ground truth boxes from keypoint bounds if not provided
            target_bboxes = torch.full_like(bboxes, 0.5)
        if target_obj is None:
            target_obj = torch.ones_like(objectness)

        loss_box = F.smooth_l1_loss(bboxes, target_bboxes)
        loss_obj = F.binary_cross_entropy(objectness, target_obj)

        total_loss = loss_cls + 1.5 * loss_box + 0.5 * loss_obj

        losses_dict = {
            "total_loss": float(total_loss.item()),
            "loss_cls": float(loss_cls.item()),
            "loss_box": float(loss_box.item()),
            "loss_obj": float(loss_obj.item()),
        }
        return total_loss, losses_dict

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
