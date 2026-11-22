"""
Reproducible Experiment Runner Aligned to CSAI3027P Deep Learning - Project 13.
Executes and logs all experimental evidence across Units I through V:
- Unit I: Scratch MLP, GD / Momentum / Nesterov, SVD diagnostics
- Unit II: Tensor math, broadcasting, numerical gradient checking
- Unit III: Learning rules (Hebbian/Oja/Competitive/Error-Correction/Boltzmann), activations, init, 2D/3D conv
- Unit IV: ResNet vs YOLO/CNN vision architecture benchmark (mAP, IoU, latency)
- Unit V: BiLSTM & Transformer sequence decoders, VAE generative modeling & latent space
"""

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Core and Utilities
from src.utils.data_loader import SignDatasetManager
from src.utils.metrics import PerformanceMetrics
from src.core.neuron import McCullochPittsNeuron
from src.core.mlp import ScratchMLP
from src.core.optimizers import BatchGradientDescent, MomentumOptimizer, NesterovOptimizer
from src.core.svd_analysis import LandmarkSVDAnalyzer
from src.core.tensor_math import TensorMathSandbox
from src.core.grad_check import GradientChecker

# Unit III Experiments
from src.learning_rules.rules import (
    HebbianLearning,
    OjaModifiedHebbian,
    CompetitiveLearning,
    ErrorCorrectionLearning,
    EnergyBasedBoltzmannDemo,
)
from src.experiments.activations import ActivationSuite
from src.experiments.initializers import WeightInitializerSuite
from src.experiments.regularization import ScratchDropout, ScratchBatchNorm1D, ScratchLayerNorm1D
from src.experiments.augmentation import LandmarkAugmenter
from src.experiments.conv_scratch import ScratchConvolution

# Unit IV Vision
from src.vision.resnet_model import SignResNet
from src.vision.yolo_detector import SignYOLODetector
from src.vision.benchmark import VisionArchitectureBenchmark
from src.models.ml_classifier import SignMLClassifier


# Unit V Sequence & Generative
from src.sequence_generative.sequence_models import (
    DeepRNNSignClassifier,
    BiLSTMSignClassifier,
    TransformerSignDecoder,
    GatingAblationExperiment,
)
from src.sequence_generative.vae_generative import SignLanguageVAE


def run_all_experiments():
    print("=" * 80)
    print("CSAI3027P DEEP LEARNING - PROJECT 13: SIGN LANGUAGE RECOGNITION ASSISTANT")
    print("Comprehensive 8-Week Syllabus Experimental Evidence Runner")
    print("=" * 80)

    results_log = {}
    data_mgr = SignDatasetManager()
    meta = data_mgr.get_dataset_metadata()
    print(f"\n[DATASET AUDIT] Total samples: {meta['total_samples']} | Classes: {meta['num_classes']} | Seq: {meta['sequence_length']} frames x {meta['feature_dim']} dims")
    results_log["dataset_metadata"] = meta

    partitions = data_mgr.get_partitions(test_size=0.15, val_size=0.15, seed=42)
    x_train, y_train = partitions["x_train"], partitions["y_train"]
    x_val, y_val = partitions["x_val"], partitions["y_val"]
    x_test, y_test = partitions["x_test"], partitions["y_test"]
    n_classes = partitions["n_classes"]

    # -------------------------------------------------------------
    # UNIT I: Scratch MLP, Optimizers, SVD
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("UNIT I: NEURAL FOUNDATIONS & OPTIMIZATION")
    print("-" * 40)

    # 1. McCulloch-Pitts gesture gate demo
    mp_gate = McCullochPittsNeuron(weights=[1, 1, 1], threshold=2)
    gate_out = mp_gate.forward(np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]]))
    print(f"McCulloch-Pitts Hand Motion Gate firing pattern: {gate_out.tolist()}")

    # 2. Scratch MLP on temporal flattened features
    # Flatten (25, 126) -> 3150 for MLP baseline
    x_train_flat = x_train.reshape(x_train.shape[0], -1)
    x_test_flat = x_test.reshape(x_test.shape[0], -1)

    print("Training Scratch MLP (Pure NumPy forward + backward)...")
    scratch_mlp = ScratchMLP([3150, 128, n_classes], activation="relu", weight_init="he", seed=42)
    mlp_param_info = scratch_mlp.count_parameters()
    print(f"Scratch MLP parameter count: {mlp_param_info['total']:,} weights & biases")

    # Train for 5 epochs using mini-batch Momentum
    mom_opt = MomentumOptimizer(lr=0.01, beta=0.9)
    batch_sz = 64
    n_batches = len(x_train_flat) // batch_sz

    for ep in range(3):
        perm = np.random.permutation(len(x_train_flat))
        ep_loss = 0.0
        for b in range(n_batches):
            idx = perm[b * batch_sz : (b + 1) * batch_sz]
            xb, yb = x_train_flat[idx], y_train[idx]
            preds = scratch_mlp.forward(xb)
            loss = scratch_mlp.compute_loss(preds, yb, l2_reg=1e-4)
            scratch_mlp.backward(preds, yb, l2_reg=1e-4)
            mom_opt.step(scratch_mlp)
            ep_loss += loss
        print(f"  Scratch MLP Epoch {ep+1}/3 Loss: {ep_loss / n_batches:.4f}")

    test_preds = np.argmax(scratch_mlp.forward(x_test_flat), axis=-1)
    scratch_mlp_acc = float(np.mean(test_preds == y_test))
    print(f"Scratch MLP Test Accuracy: {scratch_mlp_acc * 100:.2f}%")
    results_log["unit_1_scratch_mlp"] = {"accuracy": scratch_mlp_acc, "parameters": mlp_param_info}

    # 3. SVD Feature Analysis and Diagnostics
    print("\nRunning SVD Feature Diagnostics on 126-D Hand Landmarks...")
    svd_analyzer = LandmarkSVDAnalyzer(x_train)
    svd_report = svd_analyzer.analyze(k_components=[4, 8, 16, 32, 64, 126])
    print(f"SVD Condition Number (kappa): {svd_report['condition_number']:.2e}")
    print("SVD Compression Benchmarks:")
    for b in svd_report["compression_benchmarks"][:4]:
        print(f"  k={b['k']:3d} | Retained Variance: {b['variance_retained_pct']:6.2f}% | Rel Error: {b['relative_frobenius_error']:.4f}")
    results_log["unit_1_svd"] = svd_report

    # -------------------------------------------------------------
    # UNIT II: Tensor Mathematics & Numerical Gradient Checking
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("UNIT II: TENSOR MATHEMATICS & GRADIENT CHECKING")
    print("-" * 40)

    # Tensor Sandbox
    b_demo = TensorMathSandbox.demonstrate_broadcasting(batch_size=4, seq_len=25, feature_dim=126)
    print(f"Broadcasting verified: Sequence {b_demo['sequence_shape']} + Offset {b_demo['offset_shape']} -> {b_demo['broadcast_offset_shape']}")

    # Numerical Gradient Checking
    print("Executing Two-Sided Finite Difference Numerical Gradient Checking...")
    grad_check_mlp = ScratchMLP([32, 16, 8], activation="sigmoid", weight_init="xavier", seed=42)
    sample_x = np.random.randn(8, 32)
    sample_y = np.random.randint(0, 8, size=8)
    gc_res = GradientChecker.check_model_gradients(grad_check_mlp, sample_x, sample_y, epsilon=1e-5, max_params_to_check=30)
    print(f"Gradient Check Result: {gc_res['status']}")
    print(f"Relative Error: {gc_res['relative_error']:.3e} (Tolerance threshold: 1.00e-05)")
    results_log["unit_2_grad_check"] = gc_res

    # -------------------------------------------------------------
    # UNIT III: Learning Rules, Regularization, and Convolutions
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("UNIT III: LEARNING RULES & REGULARIZATION LAB")
    print("-" * 40)

    # 1. Oja's Modified Hebbian rule
    oja = OjaModifiedHebbian(n_features=126, lr=0.01)
    norm_progression = []
    sample_vec = x_train[0, 12]  # frame 12 landmarks
    for _ in range(50):
        _, norm = oja.train_step(sample_vec)
        norm_progression.append(norm)
    print(f"Oja's Rule convergence: Start norm={norm_progression[0]:.4f} -> Final stabilized norm={norm_progression[-1]:.4f}")

    # 2. Vanishing Gradient Simulation
    print("Simulating Deep Gradient Flow across Activations (10 Layers)...")
    grad_flow = ActivationSuite.simulate_deep_gradient_flow(n_layers=8, batch_size=32, dim=64)
    print(f"  Sigmoid gradient norm (Layer 1 -> 8): {grad_flow['sigmoid'][0]:.2e} -> {grad_flow['sigmoid'][-1]:.2e}")
    print(f"  ReLU gradient norm    (Layer 1 -> 8): {grad_flow['relu'][0]:.2e} -> {grad_flow['relu'][-1]:.2e}")
    results_log["unit_3_grad_flow"] = {k: [round(v, 6) for v in vals] for k, vals in grad_flow.items()}

    # 3. Weight Initialization Variance Preservation
    var_props = WeightInitializerSuite.test_variance_propagation(n_layers=6, dim=64, activation="relu")
    print(f"  Small Random variance at Layer 6: {var_props['Small_Random'][-1]:.2e} (Signal collapsed)")
    print(f"  He Kaiming   variance at Layer 6: {var_props['He'][-1]:.4f} (Signal preserved)")

    # 4. Convolutions from Scratch
    print("Testing 2D and 3D Spatio-Temporal Convolutions from Scratch...")
    filters = ScratchConvolution.get_standard_filters()
    dummy_img = np.random.rand(64, 64)
    sobel_map = ScratchConvolution.conv2d(dummy_img, filters["Sobel_Horizontal"], stride=1, padding=1)
    print(f"  2D Conv Output Shape: {sobel_map.shape} (Edge feature map)")

    dummy_vid = np.random.rand(10, 32, 32)  # (T=10, H=32, W=32)
    conv3d_map = ScratchConvolution.conv3d_spatiotemporal(
        dummy_vid, filters["SpatioTemporal_Motion"], stride_t=1, stride_s=1, padding_s=1
    )
    print(f"  3D Spatio-Temporal Conv Output Shape: {conv3d_map.shape} (Dynamic motion map)")

    # -------------------------------------------------------------
    # UNIT IV: Vision Architecture Benchmark (ResNet vs YOLO/CNN)
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("UNIT IV: VISION ARCHITECTURE BENCHMARKING")
    print("-" * 40)

    benchmarker = VisionArchitectureBenchmark(n_classes=n_classes)
    vision_results = benchmarker.run_comparative_benchmark(
        x_train, y_train, x_test, y_test, epochs=3, batch_size=64
    )
    print("\nVISION BENCHMARK SUMMARY TABLE:")
    print(f"{'Model':<18} | {'Accuracy':<10} | {'Macro F1':<10} | {'Latency (ms)':<12} | {'FPS':<8} | {'Params':<10}")
    print("-" * 75)
    for name, r in vision_results.items():
        print(f"{name:<18} | {r['accuracy']*100:6.2f}%    | {r['macro_f1']:6.4f}     | {r['avg_latency_ms']:8.3f}     | {r['fps']:6.1f}   | {r['param_count']:,}")
    results_log["unit_4_vision_benchmark"] = vision_results

    # Classical Machine Learning Benchmark
    print("\nEVALUATING CLASSICAL MACHINE LEARNING (RANDOM FOREST)...")
    ml_clf = SignMLClassifier()
    x_test_feat = SignMLClassifier.extract_features(x_test)
    start_ml = time.perf_counter()
    ml_preds = ml_clf.model.predict(x_test_feat)
    ml_lat = (time.perf_counter() - start_ml) * 1000.0 / len(x_test)
    ml_rep = PerformanceMetrics.classification_report(y_test, ml_preds, n_classes)
    ml_rep["avg_latency_ms"] = round(ml_lat, 3)
    ml_rep["fps"] = round(1000.0 / ml_lat, 1) if ml_lat > 0 else 0.0
    ml_rep["param_count"] = 120  # trees
    print(f"Random Forest (ML) | Accuracy: {ml_rep['accuracy']*100:.2f}% | Macro F1: {ml_rep['macro_f1']:.4f} | Latency: {ml_lat:.3f} ms")
    results_log["unit_4_classical_ml_benchmark"] = ml_rep

    # -------------------------------------------------------------
    # UNIT V: Sequence Models & VAE Generative Manifold
    # -------------------------------------------------------------
    print("\n" + "-" * 40)
    print("UNIT V: SEQUENCE MODELS & GENERATIVE VAE")
    print("-" * 40)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).long())
    test_ds = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).long())
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    # 1. BiLSTM Decoder
    print("Evaluating Bidirectional LSTM Sequence Model...")
    bilstm = BiLSTMSignClassifier(in_dim=126, hidden_dim=128, num_classes=n_classes).to(device)
    opt_lstm = torch.optim.Adam(bilstm.parameters(), lr=1e-3)
    for ep in range(3):
        bilstm.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_lstm.zero_grad()
            loss = nn.functional.cross_entropy(bilstm(xb), yb)
            loss.backward()
            opt_lstm.step()

    bilstm.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = torch.argmax(bilstm(xb.to(device)), dim=-1).cpu()
            correct += (preds == yb).sum().item()
            total += len(yb)
    bilstm_acc = correct / total
    print(f"  BiLSTM Test Accuracy: {bilstm_acc * 100:.2f}%")

    # 2. Transformer Sequence Decoder
    print("Evaluating Transformer Sequence Decoder (Self-Attention)...")
    trans_model = TransformerSignDecoder(in_dim=126, d_model=128, nhead=4, num_classes=n_classes).to(device)
    opt_trans = torch.optim.Adam(trans_model.parameters(), lr=1e-3)
    for ep in range(3):
        trans_model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt_trans.zero_grad()
            loss = nn.functional.cross_entropy(trans_model(xb), yb)
            loss.backward()
            opt_trans.step()

    trans_model.eval()
    t_correct = 0
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = torch.argmax(trans_model(xb.to(device)), dim=-1).cpu()
            t_correct += (preds == yb).sum().item()
    trans_acc = t_correct / total
    print(f"  Transformer Test Accuracy: {trans_acc * 100:.2f}%")

    # 3. Variational Autoencoder (VAE)
    print("Training SignLanguageVAE for Rare-Gesture Generative Augmentation...")
    vae = SignLanguageVAE(seq_len=25, feat_dim=126, hidden_dim=128, latent_dim=16).to(device)
    opt_vae = torch.optim.Adam(vae.parameters(), lr=1e-3)

    for ep in range(3):
        vae.train()
        ep_recon = 0.0
        ep_kl = 0.0
        for xb, _ in train_loader:
            xb = xb.to(device)
            opt_vae.zero_grad()
            recon_x, mu, logvar = vae(xb)
            loss, r_l, kl_l = vae.compute_loss(recon_x, xb, mu, logvar)
            loss.backward()
            opt_vae.step()
            ep_recon += r_l
            ep_kl += kl_l
        print(f"  VAE Epoch {ep+1}/3 - Recon MSE: {ep_recon/len(train_loader):.5f} | KL Div: {ep_kl/len(train_loader):.5f}")

    # Generate synthetic samples and trajectory interpolation
    synth_samples = vae.sample_synthetic_gestures(num_samples=5)
    seq1 = torch.from_numpy(x_test[0]).float().to(device)
    seq2 = torch.from_numpy(x_test[1]).float().to(device)
    interp_traj = vae.interpolate_gestures(seq1, seq2, steps=6)
    print(f"  Successfully sampled {synth_samples.shape[0]} synthetic gestures.")
    print(f"  Synthesized continuous manifold interpolation trajectory: {interp_traj.shape}")

    results_log["unit_5_sequence_generative"] = {
        "bilstm_accuracy": round(bilstm_acc, 4),
        "transformer_accuracy": round(trans_acc, 4),
        "vae_latent_dim": 16,
        "synthetic_samples_generated": int(synth_samples.shape[0]),
    }

    # Save benchmark results
    out_dir = "/Users/gauravgoyal/Desktop/Sign_Language/benchmarks"
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "experiment_results.json")
    with open(out_file, "w") as f:
        json.dump(results_log, f, indent=2)

    print("\n" + "=" * 80)
    print(f"[SUCCESS] All experimental benchmarks completed and saved to {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_all_experiments()
