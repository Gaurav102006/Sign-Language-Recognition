"""
Unit Test: Unit II Tensor Mathematics Sandbox.
"""

import unittest
import numpy as np
from src.core.tensor_math import TensorMathSandbox


class TestTensorMath(unittest.TestCase):
    def test_broadcasting(self):
        res = TensorMathSandbox.demonstrate_broadcasting(batch_size=2, seq_len=10, feature_dim=126)
        self.assertEqual(res["sequence_shape"], [2, 10, 126])
        self.assertEqual(res["broadcast_offset_shape"], [2, 10, 126])
        self.assertEqual(res["broadcast_temporal_shape"], [2, 10, 126])

    def test_inner_products(self):
        a = np.ones((10, 126))
        b = np.ones((10, 126))
        res = TensorMathSandbox.inner_products_demo(a, b)
        self.assertIn("vector_dot", res)
        self.assertEqual(res["matrix_mult_shape"], [10, 10])
        self.assertEqual(res["batched_einsum_shape"], [4, 25, 8])

    def test_reshaping(self):
        t = np.random.randn(4, 25, 126)
        shapes = TensorMathSandbox.reshape_and_permute(t)
        self.assertEqual(shapes["flat_mlp_shape"], (4, 3150))
        self.assertEqual(shapes["per_frame_shape"], (100, 126))
        self.assertEqual(shapes["transposed_conv_shape"], (4, 126, 25))

    def test_pairwise_distances(self):
        pts = np.zeros((21, 3))
        pts[1] = [1.0, 0.0, 0.0]
        dists = TensorMathSandbox.pairwise_landmark_distances(pts)
        self.assertEqual(dists.shape, (21, 21))
        self.assertAlmostEqual(dists[0, 1], 1.0)
        self.assertAlmostEqual(dists[0, 0], 0.0)


if __name__ == "__main__":
    unittest.main()
