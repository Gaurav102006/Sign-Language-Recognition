"""
Unit II: Matrix Computation, Tensor Operations, Broadcasting, and Inner Products.
Programmatic tensor sandbox demonstrating core tensor algebra on Sign Language data.
Aligned to CSAI3027P Deep Learning - Project 13.
"""

from typing import Dict, Tuple
import numpy as np


class TensorMathSandbox:
    """
    Mathematical demonstration of Unit II tensor operations:
    1. Multidimensional tensor creation and shape/stride introspection.
    2. Broadcasting across batch, temporal (frame), and spatial (landmark) dimensions.
    3. Inner products: vector dot, matrix multiplication, and batched contractions.
    4. Tensor reshaping, spatial-temporal transposition, and flattening.
    5. Geometric landmark operations: pairwise Euclidean distances and joint cosine angles.
    """

    @staticmethod
    def inspect_tensor(t: np.ndarray, name: str = "Tensor") -> Dict:
        """Inspects dimensions, strides, memory layout, and norm."""
        arr = np.asarray(t)
        return {
            "name": name,
            "shape": list(arr.shape),
            "ndim": arr.ndim,
            "dtype": str(arr.dtype),
            "strides": list(arr.strides),
            "c_contiguous": arr.flags["C_CONTIGUOUS"],
            "f_contiguous": arr.flags["F_CONTIGUOUS"],
            "frobenius_norm": float(np.linalg.norm(arr)),
        }

    @staticmethod
    def demonstrate_broadcasting(batch_size: int = 4, seq_len: int = 25, feature_dim: int = 126) -> Dict:
        """
        Demonstrates numpy broadcasting across:
        Base landmark sequence: (batch_size, seq_len, feature_dim)
        Center offset vector: (feature_dim,)
        Frame temporal weights: (seq_len, 1)
        """
        rng = np.random.default_rng(42)
        sequence_tensor = rng.normal(0, 1, size=(batch_size, seq_len, feature_dim))
        offset_vector = rng.normal(0, 0.1, size=(feature_dim,))
        temporal_weights = np.linspace(0.5, 1.5, seq_len).reshape(seq_len, 1)

        # 1. Broadcasting (feature_dim,) into (batch_size, seq_len, feature_dim)
        broadcast_offset = sequence_tensor + offset_vector

        # 2. Broadcasting (seq_len, 1) across batch and features
        broadcast_temporal = sequence_tensor * temporal_weights

        return {
            "sequence_shape": list(sequence_tensor.shape),
            "offset_shape": list(offset_vector.shape),
            "temporal_weights_shape": list(temporal_weights.shape),
            "broadcast_offset_shape": list(broadcast_offset.shape),
            "broadcast_temporal_shape": list(broadcast_temporal.shape),
        }

    @staticmethod
    def inner_products_demo(a: np.ndarray, b: np.ndarray) -> Dict:
        """
        Demonstrates various inner products:
        - Vector dot product
        - Matrix multiplication
        - Batched matrix multiply (BMM)
        """
        results = {}
        # 1. 1D Vector dot
        v1 = a.ravel()[:126]
        v2 = b.ravel()[:126]
        results["vector_dot"] = float(np.dot(v1, v2))

        # 2. 2D Matrix multiplication
        m1 = a.reshape(-1, 126)[:10, :]
        m2 = b.reshape(-1, 126)[:10, :].T
        results["matrix_mult_shape"] = list(np.matmul(m1, m2).shape)

        # 3. Batched Einstein summation: B_ijk * B_ikl -> B_ijl
        t1 = np.ones((4, 25, 10))
        t2 = np.ones((4, 10, 8))
        einsum_res = np.einsum("bij,bjk->bik", t1, t2)
        results["batched_einsum_shape"] = list(einsum_res.shape)

        return results

    @staticmethod
    def reshape_and_permute(x: np.ndarray) -> Dict[str, Tuple]:
        """
        Demonstrates tensor reshaping essential for Sign Language models:
        (B, T, D) -> (B, T*D) [Fully-connected / MLP temporal flattening]
        (B, T, D) -> (B*T, D) [Per-frame landmark processing]
        (B, T, D) -> (B, D, T) [1D Conv / Temporal conv format]
        """
        arr = np.asarray(x)
        if arr.ndim != 3:
            raise ValueError("Expected 3D tensor of shape (Batch, Time, Dims)")

        b, t, d = arr.shape
        flat_mlp = arr.reshape(b, t * d)
        per_frame = arr.reshape(b * t, d)
        transposed_conv = np.transpose(arr, (0, 2, 1))

        return {
            "original_shape": arr.shape,
            "flat_mlp_shape": flat_mlp.shape,
            "per_frame_shape": per_frame.shape,
            "transposed_conv_shape": transposed_conv.shape,
        }

    @staticmethod
    def pairwise_landmark_distances(hand_landmarks_21x3: np.ndarray) -> np.ndarray:
        """
        Computes the 21x21 pairwise Euclidean distance matrix for hand keypoints
        using efficient tensor broadcasting:
        D_{ij} = ||p_i - p_j||_2 = sqrt(sum_k (p_ik - p_jk)^2)
        """
        pts = np.asarray(hand_landmarks_21x3)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (21, 21, 3)
        dist_matrix = np.sqrt(np.sum(diff**2, axis=-1))  # (21, 21)
        return dist_matrix
