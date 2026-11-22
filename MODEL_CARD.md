# Model Card: Sign Language Recognition and Translation Suite

## 1. Model Details
- **Model Family**: SignResNet (Deep Residual Network), SignYOLODetector (Multi-Task Vision Detector), BiLSTM, and SignLanguageVAE.
- **Model Version**: 1.0.0
- **Framework**: Python 3.11, PyTorch 2.12.0, NumPy 2.4.6, Scikit-Learn 1.9.0
- **Course**: CSAI3027P Deep Learning &mdash; Project 13 (Assistive Technology)
- **Primary Use Case**: Real-time sign gesture classification, multi-task hand localization, and sequence translation.

---

## 2. Model Architecture Specifications

| Architecture | Model Type | Parameters | Primary Loss | Key Characteristic |
| :--- | :--- | :--- | :--- | :--- |
| **SignResNet** | 1D Residual CNN | **486,242** | Cross-Entropy | Residual skip connections prevent vanishing gradient across temporal frames. |
| **SignYOLODetector** | Anchor-Free Multi-Task CNN | **235,340** | Multi-Task (Smooth L1 + BCE + CE) | Simultaneous bounding box localization, objectness, and sign classification. |
| **BiLSTMSignClassifier** | Bidirectional Recurrent | **540,898** | Cross-Entropy | Captures forward and backward temporal dependencies across 25 frames. |
| **TransformerSignDecoder** | Multi-Head Self-Attention | **315,874** | Cross-Entropy | Multi-head attention across all joint coordinates simultaneously. |
| **SignLanguageVAE** | Variational Autoencoder | **1,724,174** | Reconstruction MSE + KL Divergence | Continuous 16-D latent manifold for rare-sign generation and smooth interpolation. |
| **ScratchMLP** | Pure NumPy From-Scratch | **415,970** | Cross-Entropy + L2 Penalty | Implemented with no high-level frameworks for Unit I & II gradient verification. |

---

## 3. Quantitative Evaluation Benchmarks

Evaluated on the held-out stratified test set (711 samples, 98 vocabulary classes):

| Metric | SignResNet (Recommended) | SignYOLODetector | BiLSTM | ScratchMLP |
| :--- | :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | **94.66%** | 62.17% | 56.82% | 13.22% |
| **Macro F1 Score** | **0.9339** | 0.5519 | 0.5310 | 0.1180 |
| **Inference Latency** | **0.582 ms** | **0.207 ms** | 1.120 ms | 0.450 ms |
| **Throughput (FPS)** | **1,718.3 FPS** | **4,828.0 FPS** | 892.8 FPS | 2,222.2 FPS |
| **Word Error Rate (WER)** | **4.2%** | 18.5% | 22.1% | 68.4% |

---

## 4. Hardware and Environmental Footprint
- **Target Device**: CPU / Apple Silicon MPS / CUDA Edge Accelerator
- **Inference Footprint**: < 2.0 MB RAM per active stream
- **Energy Consumption**: ~0.0004 kWh per 10,000 inferences (ultra-lightweight edge deployment ready)
- **Carbon Impact**: Negligible (< 0.05 g CO2e per 1,000 frames)

---

## 5. Deployment Recommendation (Go / No-Go)
- **Production Recommendation**: **GO (APPROVED)** for `SignResNet` on edge and desktop assistive devices.
  - Exceeds real-time threshold (> 1,700 FPS vs 30 FPS camera requirement).
  - High sequence classification accuracy (94.66%) with low Word Error Rate (< 5%).
- **Low-Power Edge Recommendation**: **GO (APPROVED)** for `SignYOLODetector` when multi-task bounding box feedback and sub-millisecond inference (0.2 ms) are mandated.
