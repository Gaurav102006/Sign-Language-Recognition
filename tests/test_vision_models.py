"""
Unit Test: Unit IV Vision Models (SignResNet and SignYOLODetector).
"""

import unittest
import torch
from src.vision.resnet_model import SignResNet
from src.vision.yolo_detector import SignYOLODetector


class TestVisionModels(unittest.TestCase):
    def test_resnet_forward(self):
        model = SignResNet(in_features=126, num_classes=98)
        # Input shape: (Batch=4, Time=25, Features=126)
        x = torch.randn(4, 25, 126)
        logits = model(x)
        self.assertEqual(logits.shape, (4, 98))

    def test_yolo_detector_forward_and_loss(self):
        detector = SignYOLODetector(in_features=126, num_classes=98)
        x = torch.randn(4, 25, 126)
        logits, bboxes, obj = detector(x)

        self.assertEqual(logits.shape, (4, 98))
        self.assertEqual(bboxes.shape, (4, 8))
        self.assertEqual(obj.shape, (4, 2))

        # Test multi-task loss calculation
        y_true = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        total_loss, losses = detector.compute_multitask_loss(logits, bboxes, obj, y_true)
        self.assertGreater(total_loss.item(), 0.0)
        self.assertIn("loss_cls", losses)
        self.assertIn("loss_box", losses)
        self.assertIn("loss_obj", losses)


if __name__ == "__main__":
    unittest.main()
