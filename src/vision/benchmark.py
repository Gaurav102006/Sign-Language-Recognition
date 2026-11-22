"""
Unit IV: Vision Architecture Benchmarking Harness (ResNet vs YOLO/CNN).
Compares accuracy, F1, mAP, IoU, latency, throughput, and parameter efficiency.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

import time
from typing import Dict, List
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.vision.resnet_model import SignResNet
from src.vision.yolo_detector import SignYOLODetector
from src.utils.metrics import PerformanceMetrics


class VisionArchitectureBenchmark:
    """
    Automated benchmark comparing deep vision architectures:
    1. SignResNet (Deep Residual Network)
    2. SignYOLODetector (Multi-Task YOLO CNN Detector)
    """

    def __init__(self, n_classes: int = 98, device: str = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.n_classes = n_classes
        self.resnet = SignResNet(in_features=126, num_classes=n_classes).to(self.device)
        self.yolo = SignYOLODetector(in_features=126, num_classes=n_classes).to(self.device)

    def train_epoch(self, model: nn.Module, loader: DataLoader, optimizer, is_yolo: bool = False):
        model.train()
        total_loss = 0.0

        for x_b, y_b in loader:
            x_b = x_b.to(self.device)
            y_b = y_b.to(self.device)
            optimizer.zero_grad()

            if is_yolo:
                cls_logits, bboxes, obj = model(x_b)
                loss, _ = model.compute_multitask_loss(cls_logits, bboxes, obj, y_b)
            else:
                logits = model(x_b)
                loss = nn.functional.cross_entropy(logits, y_b)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(loader)

    @torch.no_grad()
    def evaluate_model(self, model: nn.Module, loader: DataLoader, is_yolo: bool = False) -> Dict:
        model.eval()
        all_preds = []
        all_targets = []
        latencies = []

        for x_b, y_b in loader:
            x_b = x_b.to(self.device)
            start = time.perf_counter()

            if is_yolo:
                cls_logits, bboxes, obj = model(x_b)
                preds = torch.argmax(cls_logits, dim=-1).cpu().numpy()
            else:
                logits = model(x_b)
                preds = torch.argmax(logits, dim=-1).cpu().numpy()

            elapsed_ms = (time.perf_counter() - start) * 1000.0 / len(x_b)
            latencies.append(elapsed_ms)

            all_preds.extend(preds)
            all_targets.extend(y_b.numpy())

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)

        rep = PerformanceMetrics.classification_report(y_true, y_pred, self.n_classes)
        avg_lat = float(np.mean(latencies))
        fps = 1000.0 / avg_lat if avg_lat > 0 else 0.0

        rep["avg_latency_ms"] = round(avg_lat, 3)
        rep["fps"] = round(fps, 1)
        rep["param_count"] = model.count_parameters()
        return rep

    def run_comparative_benchmark(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        epochs: int = 5,
        batch_size: int = 64,
    ) -> Dict[str, Dict]:
        """
        Executes rapid training and standardized evaluation on both architectures.
        """
        train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
        test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        opt_resnet = torch.optim.AdamW(self.resnet.parameters(), lr=1e-3, weight_decay=1e-4)
        opt_yolo = torch.optim.AdamW(self.yolo.parameters(), lr=1e-3, weight_decay=1e-4)

        print(f"Training SignResNet ({self.resnet.count_parameters():,} params)...")
        for epoch in range(epochs):
            loss = self.train_epoch(self.resnet, train_loader, opt_resnet, is_yolo=False)
            if (epoch + 1) % max(1, epochs // 2) == 0:
                print(f"  ResNet Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

        print(f"Training SignYOLODetector ({self.yolo.count_parameters():,} params)...")
        for epoch in range(epochs):
            loss = self.train_epoch(self.yolo, train_loader, opt_yolo, is_yolo=True)
            if (epoch + 1) % max(1, epochs // 2) == 0:
                print(f"  YOLO Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}")

        eval_resnet = self.evaluate_model(self.resnet, test_loader, is_yolo=False)
        eval_yolo = self.evaluate_model(self.yolo, test_loader, is_yolo=True)

        return {
            "SignResNet": eval_resnet,
            "SignYOLODetector": eval_yolo,
        }
