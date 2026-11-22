"""
Unit tests for Machine Learning Classifier, Multilingual Translator, and Vision Inference Engine.
"""

import unittest
import numpy as np
from src.models.ml_classifier import SignMLClassifier
from src.utils.translator import MultilingualSignTranslator
from src.vision.inference_engine import SignInferenceEngine


class TestMLAndTranslation(unittest.TestCase):
    """Test suite covering ML classifier, Translation, and Inference Engine."""

    def setUp(self):
        self.ml_classifier = SignMLClassifier()
        self.translator = MultilingualSignTranslator()
        self.inference_engine = SignInferenceEngine()
        self.dummy_sequence = np.random.randn(25, 126).astype(np.float32)

    def test_ml_feature_extraction(self):
        """Verify that 630 spatio-temporal features are extracted correctly."""
        feats = SignMLClassifier.extract_features(self.dummy_sequence)
        self.assertEqual(feats.shape, (630,))
        self.assertFalse(np.isnan(feats).any())

    def test_ml_batch_feature_extraction(self):
        """Verify batch feature extraction of shape (N, 630)."""
        batch = np.random.randn(4, 25, 126).astype(np.float32)
        feats = SignMLClassifier.extract_features(batch)
        self.assertEqual(feats.shape, (4, 630))

    def test_ml_prediction(self):
        """Verify ML prediction returns a valid class string and confidence in [0, 1]."""
        pred_class, conf = self.ml_classifier.predict(self.dummy_sequence)
        self.assertIsInstance(pred_class, str)
        self.assertGreater(len(pred_class), 0)
        self.assertTrue(0.0 <= conf <= 1.0)

    def test_ml_top_k(self):
        """Verify top-k candidate ranking."""
        top5 = self.ml_classifier.get_top_k(self.dummy_sequence, k=5)
        self.assertEqual(len(top5), 5)
        # Check sorting in descending order of probability
        probs = [p for _, p in top5]
        self.assertEqual(probs, sorted(probs, reverse=True))

    def test_multilingual_word_translation(self):
        """Verify translation of individual words via offline dictionary."""
        hi_help = self.translator.translate_word("help", "hi")
        self.assertEqual(hi_help, "मदद")

        es_water = self.translator.translate_word("water", "es")
        self.assertEqual(es_water, "agua")

        fr_doctor = self.translator.translate_word("doctor", "fr")
        self.assertEqual(fr_doctor, "docteur")

    def test_sentence_synthesis(self):
        """Verify natural sentence synthesis from word tokens."""
        s = self.translator.synthesize_sentence_from_words(["help", "doctor"])
        self.assertIn("doctor", s.lower())
        self.assertTrue(s.endswith("."))

    def test_inference_engine_models(self):
        """Verify inference engine works with ResNet, Random Forest, and Ensemble."""
        for model_name in [
            "ResNet (Deep Learning)",
            "Random Forest (Machine Learning)",
            "Ensemble (DL + ML)",
        ]:
            pred_class, conf, top5 = self.inference_engine.predict_sequence(
                self.dummy_sequence, model_name
            )
            self.assertIsInstance(pred_class, str)
            self.assertTrue(0.0 <= conf <= 1.0)
            self.assertEqual(len(top5), 5)


if __name__ == "__main__":
    unittest.main()
