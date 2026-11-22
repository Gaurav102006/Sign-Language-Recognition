"""
Unit III: 2D and 3D Convolution Implemented from Scratch with Visualizations.
Direct spatial filtering (Sobel, Edge, Gaussian) and Spatio-Temporal 3D dynamic gesture filtering.
Aligned to CSAI3027P Deep Learning - Project 13: Sign Language Assistant.
"""

from typing import Dict, Tuple
import numpy as np


class ScratchConvolution:
    """
    Direct mathematical implementation of 2D and 3D convolution operations
    without high-level neural network black boxes.
    """

    @staticmethod
    def pad2d(img: np.ndarray, pad: int) -> np.ndarray:
        """Pads 2D image (H, W) or (C, H, W) with zeros."""
        if pad == 0:
            return img
        if img.ndim == 2:
            return np.pad(img, ((pad, pad), (pad, pad)), mode="constant")
        elif img.ndim == 3:
            return np.pad(img, ((0, 0), (pad, pad), (pad, pad)), mode="constant")
        raise ValueError("Expected 2D or 3D input for pad2d")

    @classmethod
    def conv2d(
        cls,
        image: np.ndarray,
        kernel: np.ndarray,
        stride: int = 1,
        padding: int = 0,
    ) -> np.ndarray:
        """
        Computes 2D spatial convolution:
        Output[i, j] = sum_{u, v} Image[i*s + u, j*s + v] * Kernel[u, v]
        
        Args:
            image: shape (H, W) or (C_in, H, W)
            kernel: shape (K_h, K_w) or (C_in, K_h, K_w)
            stride: step size S
            padding: zero padding P
        """
        img = np.asarray(image, dtype=np.float64)
        kern = np.asarray(kernel, dtype=np.float64)

        # Handle 2D single channel
        if img.ndim == 2:
            img = img[np.newaxis, :, :]
        if kern.ndim == 2:
            kern = kern[np.newaxis, :, :]

        c_in, h_in, w_in = img.shape
        c_k, k_h, k_w = kern.shape
        assert c_in == c_k, f"Channel mismatch: img has {c_in}, kernel has {c_k}"

        padded_img = cls.pad2d(img, padding)
        _, h_pad, w_pad = padded_img.shape

        h_out = (h_pad - k_h) // stride + 1
        w_out = (w_pad - k_w) // stride + 1
        output = np.zeros((h_out, w_out), dtype=np.float64)

        for i in range(h_out):
            h_start = i * stride
            h_end = h_start + k_h
            for j in range(w_out):
                w_start = j * stride
                w_end = w_start + k_w
                receptive_field = padded_img[:, h_start:h_end, w_start:w_end]
                output[i, j] = np.sum(receptive_field * kern)

        return output

    @classmethod
    def conv3d_spatiotemporal(
        cls,
        video: np.ndarray,
        kernel3d: np.ndarray,
        stride_t: int = 1,
        stride_s: int = 1,
        padding_s: int = 0,
    ) -> np.ndarray:
        """
        Computes 3D Spatio-Temporal Convolution for dynamic gesture action recognition:
        Output[t, h, w] = sum_{tau, u, v} Video[t*s_t + tau, h*s_s + u, w*s_s + v] * Kernel3D[tau, u, v]
        
        Args:
            video: shape (T, H, W) temporal sequence of gesture frames
            kernel3d: shape (K_t, K_h, K_w) spatio-temporal filter
        """
        vid = np.asarray(video, dtype=np.float64)
        k3d = np.asarray(kernel3d, dtype=np.float64)
        t_in, h_in, w_in = vid.shape
        k_t, k_h, k_w = k3d.shape

        # Spatial pad each frame
        if padding_s > 0:
            padded_frames = [cls.pad2d(vid[t], padding_s) for t in range(t_in)]
            vid_padded = np.stack(padded_frames, axis=0)
        else:
            vid_padded = vid

        _, h_pad, w_pad = vid_padded.shape

        t_out = (t_in - k_t) // stride_t + 1
        h_out = (h_pad - k_h) // stride_s + 1
        w_out = (w_pad - k_w) // stride_s + 1

        output = np.zeros((t_out, h_out, w_out), dtype=np.float64)

        for t in range(t_out):
            t_start = t * stride_t
            t_end = t_start + k_t
            for i in range(h_out):
                h_start = i * stride_s
                h_end = h_start + k_h
                for j in range(w_out):
                    w_start = j * stride_s
                    w_end = w_start + k_w
                    vol = vid_padded[t_start:t_end, h_start:h_end, w_start:w_end]
                    output[t, i, j] = np.sum(vol * k3d)

        return output

    @staticmethod
    def get_standard_filters() -> Dict[str, np.ndarray]:
        """Hand edge detection and contour enhancement filters."""
        sobel_h = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        sobel_v = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        gaussian = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float64) / 16.0

        # Spatio-temporal motion difference kernel (K_t=3, K_h=3, K_w=3)
        motion_filter = np.zeros((3, 3, 3), dtype=np.float64)
        motion_filter[0] = -0.5 * sobel_h
        motion_filter[1] = 0.0
        motion_filter[2] = 0.5 * sobel_h

        return {
            "Sobel_Horizontal": sobel_h,
            "Sobel_Vertical": sobel_v,
            "Laplacian": laplacian,
            "Gaussian_Blur": gaussian,
            "SpatioTemporal_Motion": motion_filter,
        }
