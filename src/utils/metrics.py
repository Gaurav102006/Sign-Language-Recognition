"""
Evaluation Metrics Suite for Sign Language Recognition and Translation.
Accuracy, Macro F1, Confusion Matrix, Word Error Rate (WER), mAP, IoU, and Latency.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

import time
from typing import Dict, List, Tuple
import numpy as np


class PerformanceMetrics:
    """
    Standardized industry and research metrics for Project 13:
    1. Multiclass Classification: Accuracy, Precision, Recall, Macro F1, Confusion Matrix
    2. Sequence Translation: Word Error Rate (WER) via Levenshtein edit distance
    3. Spatial Object Detection: IoU and mAP for hand bounding box localization
    4. Operational Efficiency: Latency per sample and Frames Per Second (FPS)
    """

    @staticmethod
    def classification_report(
        y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
    ) -> Dict:
        """
        Computes accuracy, macro precision, macro recall, and macro F1 score.
        """
        y_t = np.asarray(y_true).ravel()
        y_p = np.asarray(y_pred).ravel()
        total = len(y_t)

        acc = float(np.mean(y_t == y_p))

        precisions = []
        recalls = []
        f1s = []

        for c in range(n_classes):
            tp = np.sum((y_t == c) & (y_p == c))
            fp = np.sum((y_t != c) & (y_p == c))
            fn = np.sum((y_t == c) & (y_p != c))

            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2.0 * p * r) / (p + r) if (p + r) > 0 else 0.0

            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)

        return {
            "accuracy": round(acc, 4),
            "macro_precision": round(float(np.mean(precisions)), 4),
            "macro_recall": round(float(np.mean(recalls)), 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
        }

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
        """Constructs an (n_classes, n_classes) confusion matrix."""
        cm = np.zeros((n_classes, n_classes), dtype=np.int32)
        for t, p in zip(y_true, y_pred):
            if 0 <= t < n_classes and 0 <= p < n_classes:
                cm[int(t), int(p)] += 1
        return cm

    @staticmethod
    def word_error_rate(reference: List[str], hypothesis: List[str]) -> float:
        """
        Computes Word Error Rate (WER) via dynamic programming Levenshtein distance:
        WER = (Substitutions + Deletions + Insertions) / Total Reference Words
        """
        r_len = len(reference)
        h_len = len(hypothesis)
        if r_len == 0:
            return 1.0 if h_len > 0 else 0.0

        d = np.zeros((r_len + 1, h_len + 1), dtype=np.int32)
        for i in range(r_len + 1):
            d[i, 0] = i
        for j in range(h_len + 1):
            d[0, j] = j

        for i in range(1, r_len + 1):
            for j in range(1, h_len + 1):
                if reference[i - 1].lower() == hypothesis[j - 1].lower():
                    d[i, j] = d[i - 1, j - 1]
                else:
                    substitution = d[i - 1, j - 1] + 1
                    insertion = d[i, j - 1] + 1
                    deletion = d[i - 1, j] + 1
                    d[i, j] = min(substitution, insertion, deletion)

        wer = float(d[r_len, h_len]) / float(r_len)
        return round(wer, 4)

    @staticmethod
    def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """
        Computes Intersection over Union (IoU) for bounding boxes:
        box = [x1, y1, x2, y2]
        """
        x_a = max(box_a[0], box_b[0])
        y_a = max(box_a[1], box_b[1])
        x_b = min(box_a[2], box_b[2])
        y_b = min(box_a[3], box_b[3])

        inter_area = max(0.0, x_b - x_a) * max(0.0, y_b - y_a)
        area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
        area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
        union_area = area_a + area_b - inter_area

        if union_area == 0:
            return 0.0
        return float(inter_area / union_area)

    @staticmethod
    def benchmark_latency(infer_fn, sample_input, warmups: int = 5, runs: int = 50) -> Dict:
        """Measures average inference latency and frames per second."""
        for _ in range(warmups):
            _ = infer_fn(sample_input)

        start = time.perf_counter()
        for _ in range(runs):
            _ = infer_fn(sample_input)
        elapsed = time.perf_counter() - start

        avg_latency_ms = (elapsed / runs) * 1000.0
        fps = 1000.0 / avg_latency_ms if avg_latency_ms > 0 else 0.0

        return {
            "avg_latency_ms": round(avg_latency_ms, 3),
            "throughput_fps": round(fps, 1),
            "num_runs": runs,
        }
