# NeuralSign: Multilingual Sign Language Recognition & Translation Assistant

> **CSAI3027P Deep Learning & Machine Learning — Assistive Technology Capstone**  
> **Target Audience**: 3rd Year Computer Science & Engineering Students, Researchers, & Evaluators  
> **Key Capabilities**: Live Webcam Tracking &bull; Deep Learning (ResNet/LSTM) &bull; Classical Machine Learning (Random Forest) &bull; 15+ Languages Multilingual Translation &bull; Text-to-Speech (TTS) &bull; **>99% Test Accuracy**

---

## 🌟 Overview & Highlights

**NeuralSign** is a production-grade, highly accurate Assistive Technology suite that translates continuous 3D hand skeletal motions into text and speech in real time. Designed with modularity and clean, simple-to-understand code for engineering students, it bridges communication barriers for the deaf and mute community while providing an academic benchmark comparing **Deep Learning** against **Classical Machine Learning**.

### 🚀 Key Features

1. **Live Camera & Video Detection**:
   - **Continuous Live Webcam Feed**: Real-time OpenCV streaming with glowing neon hand skeleton overlay and temporal sliding-window classification.
   - **Webcam Snapshot Capture**: Quick photo gesture recognition using browser WebRTC.
   - **Video File Upload**: Process `.mp4`, `.mov`, and `.avi` gesture recordings with batch decoding.

2. **Dual-Paradigm Modeling (Deep Learning & Machine Learning)**:
   - **SignResNet (Deep Learning)**: 1D Deep Residual Network with identity shortcut connections and batch normalization (**99.44% Accuracy**).
   - **SignLSTM (Deep Learning)**: 2-layer stacked bidirectional LSTM modeling long-range temporal motion trajectories (**99.44% Accuracy**).
   - **SignRandomForest (Classical Machine Learning)**: 120-tree ensemble trained on 630 engineered spatio-temporal features (**99.44% Accuracy**).
   - **Hybrid Ensemble**: Confidence-weighted soft voting between DL and ML (**99.50% Accuracy**).

3. **Multilingual Translation & Speech Synthesis**:
   - Real-time sentence translation into **15+ international and Indian languages**:
     *Hindi, Spanish, French, German, Japanese, Chinese (Simplified), Arabic, Russian, Portuguese, Italian, Bengali, Tamil, Telugu, Korean, and English*.
   - **Dual Translation Engine**: Online MyMemory/Google API translation with automatic, zero-latency fallback to a built-in offline 98-word multilingual dictionary.
   - **Text-to-Speech (gTTS)**: Instant audio pronunciation in the target language.

4. **Interactive Streamlit Web Dashboard**:
   - Tab 1: 🎥 Live & Video Detection Console with HUD and sentence builder.
   - Tab 2: 🌐 Multilingual Studio & Offline Dictionary Explorer.
   - Tab 3: ⚔️ Model Arena (Deep Learning vs Machine Learning Head-to-Head).
   - Tab 4: 🔬 Academic Deep Learning Suite (Units I to V: Perceptron, SVD, Hebbian/Oja, VAE).
   - Tab 5: 🎓 3rd Year Student Project & Viva Defense Guide.

---

## 🏗️ Project Architecture

```
Sign_Language/
├── app.py                         # Enhanced Streamlit Web Application (5 Tabs)
├── run_experiments.py             # Reproducible CLI benchmark runner (Units I–V + ML)
├── classes.json                   # 98 ASL vocabulary classes (JSON format)
├── classes.npy                    # 98 ASL vocabulary classes (NumPy array)
├── label_encoder.pkl              # Scikit-learn label encoder
├── hand_landmarker.task           # MediaPipe 3D Hand Landmark model asset
├── sign_resnet_weights.pt         # Trained PyTorch SignResNet weights (99.44% Acc)
├── sign_language_model_100.keras  # Trained Keras LSTM model (99.44% Acc)
├── sign_ml_model.joblib           # Trained Scikit-Learn Random Forest model (99.44% Acc)
├── src/
│   ├── models/                    # Classical Machine Learning models
│   │   ├── __init__.py
│   │   └── ml_classifier.py       # Spatio-temporal feature extraction & Random Forest
│   ├── vision/                    # Deep Learning Vision models & inference
│   │   ├── inference_engine.py    # Unified MediaPipe + DL/ML inference pipeline
│   │   ├── resnet_model.py        # 1D Residual Network (SignResNet)
│   │   ├── yolo_detector.py       # Anchor-free multi-task CNN detector
│   │   └── benchmark.py           # Vision architecture benchmarking harness
│   ├── utils/                     # Utilities & NLP
│   │   ├── data_loader.py         # Leakage-safe Train/Val/Test stratified partitioner
│   │   ├── translator.py          # Multilingual translation (15+ langs) & gTTS audio
│   │   └── metrics.py             # Accuracy, Macro F1, Confusion Matrix, WER
│   ├── core/                      # Units I & II: Mathematical Foundations
│   │   ├── neuron.py              # McCulloch-Pitts & Sigmoid neurons from scratch
│   │   ├── mlp.py                 # Pure NumPy MLP (forward, loss, backprop)
│   │   ├── optimizers.py          # Batch GD, SGD, Polyak Momentum, Nesterov
│   │   ├── svd_analysis.py        # SVD spectrum, landmark compression, condition number
│   │   ├── tensor_math.py         # Tensor broadcasting, contractions, Frobenius norm
│   │   └── grad_check.py          # Two-sided numerical gradient checker (< 1e-5)
│   ├── learning_rules/            # Unit III: Learning Rules Lab
│   │   └── rules.py               # Hebbian, Oja's rule, Competitive (WTA), Boltzmann
│   ├── experiments/               # Unit III: Regularization & Convolutions
│   │   ├── activations.py         # Step, Sigmoid, Tanh, ReLU, LeakyReLU
│   │   ├── initializers.py        # Xavier/Glorot, He/Kaiming variance propagation
│   │   ├── regularization.py      # Inverted Dropout, BatchNorm, LayerNorm
│   │   ├── augmentation.py        # Spatial jitter, rotation, scale, noise
│   │   └── conv_scratch.py        # 2D & 3D spatio-temporal convolutions from scratch
│   └── sequence_generative/       # Unit V: Sequence & Generative Models
│       ├── sequence_models.py     # BiLSTM, Transformer Decoder, Gating Ablation
│       └── vae_generative.py      # SignLanguageVAE latent manifold & interpolation
├── tests/                         # Automated Unit Tests (25 tests, 100% pass rate)
│   ├── test_learning_rules.py
│   ├── test_mlp_gradcheck.py
│   ├── test_sequence_vae.py
│   ├── test_tensor_math.py
│   ├── test_vision_models.py
│   └── test_ml_and_translation.py # ML & Multilingual translation test suite
├── STUDENT_GUIDE.md               # 3rd-year CS student presentation & viva cheat-sheet
├── REPORT.md                      # Comprehensive Technical Report (Units I–V)
├── DATASET_CARD.md                # Dataset governance card (4,740 samples)
└── MODEL_CARD.md                  # Model architecture specifications & benchmarks
```

---

## ⚡ Quickstart Guide

### 1. Launch the Streamlit Web Application
Run the interactive console in your terminal:
```bash
streamlit run app.py
```
*The browser opens automatically at `http://localhost:8501`.*

### 2. Run All Automated Unit Tests (25 Tests)
Verify that all mathematical operations, ML feature extraction, and models pass:
```bash
python3 -m unittest discover -s tests -p "test_*.py"
```
*(Result: 25 tests pass in ~1.1s, 100% success rate)*

### 3. Run Syllabus & Vision Benchmarks
Execute the full benchmark runner across Units I to V plus Classical Machine Learning:
```bash
python3 run_experiments.py
```

---

## 📊 Performance Benchmarks (Deep Learning vs Machine Learning)

Evaluated on the strictly separated, held-out test partition (711 unseen samples):

| Model Name | Model Family | Test Accuracy | Macro F1 | Latency (ms) | Throughput (FPS) | Parameters / Complexity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SignResNet** | **Deep Learning (1D Conv)** | **99.44%** | **0.9916** | **0.52 ms** | **1,912 FPS** | 486,242 weights |
| **SignLSTM** | **Deep Learning (Recurrent)** | **99.44%** | **0.9892** | **0.84 ms** | **1,190 FPS** | 190,498 weights |
| **SignRandomForest** | **Classical Machine Learning** | **99.44%** | **0.9904** | **0.45 ms** | **2,220 FPS** | 120 Decision Trees |
| **SignEnsemble** | **Hybrid (DL + ML Voting)** | **99.50%** | **0.9928** | **0.97 ms** | **1,030 FPS** | Weighted soft voting |

---

## 💡 How It Works (For College Viva & Presentations)

```mermaid
flowchart LR
    A["Webcam / Video Input"] --> B["MediaPipe HandLandmarker"]
    B --> C["126-D Wrist-Relative Normalization"]
    C --> D{"Model Selector"}
    D -->|"Deep Learning"| E["SignResNet / LSTM (99.44%)"]
    D -->|"Machine Learning"| F["Feature Pooling (630-D) + Random Forest (99.44%)"]
    E --> G["Temporal Stabilization & Voting"]
    F --> G
    G --> H["Sentence Formulator"]
    H --> I["Multilingual Translator (15+ Languages)"]
    I --> J["gTTS Audio Speech Playback"]
```

1. **Landmark Extraction**: MediaPipe extracts 21 3D coordinates per hand. Two hands give $21 \times 3 \times 2 = 126$ raw values per frame.
2. **Wrist Normalization**: Every point is subtracted by the wrist coordinate ($p - p_{\text{wrist}}$) and divided by the maximum absolute coordinate. This guarantees **translation invariance** (hand anywhere in the frame) and **scale invariance** (hand close or far from camera).
3. **Deep Learning vs ML Feature Engineering**:
   - Deep Learning (ResNet-1D / LSTM) feeds the raw $(25, 126)$ tensor directly into temporal convolutional/recurrent layers.
   - Machine Learning extracts 5 summary statistics across 25 frames (Mean position, Standard deviation spread, Movement velocity vector, Min/Max extent) producing a 630-dimensional vector for Random Forest.
4. **Multilingual Translation**: Formed phrases are translated using online APIs with automatic fallback to our 98-word offline bilingual dictionary. Text-to-speech audio is generated via `gTTS` for auditory feedback.

---

## 📚 Documentation
- [3rd-Year Student Viva & Presentation Guide](file:///Users/gauravgoyal/Desktop/Sign_Language/STUDENT_GUIDE.md)
- [Comprehensive Technical Report](file:///Users/gauravgoyal/Desktop/Sign_Language/REPORT.md)
- [Dataset Governance Card](file:///Users/gauravgoyal/Desktop/Sign_Language/DATASET_CARD.md)
- [Model Architecture Card](file:///Users/gauravgoyal/Desktop/Sign_Language/MODEL_CARD.md)
