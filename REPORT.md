# CSAI3027P Deep Learning &mdash; Technical Project Report
## Project 13: Sign Language Recognition and Translation Assistant
**Industry Domain**: Assistive Technology & Human-Computer Interaction  
**Author**: Deep Learning Engineering Research Cohort  
**Submission Framework**: Python 3.11 / PyTorch / NumPy / Scikit-Learn  
**Repository Location**: [`/Users/gauravgoyal/Desktop/Sign_Language`](file:///Users/gauravgoyal/Desktop/Sign_Language)

---

## Executive Summary
This report details the design, mathematical formulation, from-scratch implementation, experimental benchmarking, and industrial deployment readiness of **Project 13: Sign Language Recognition and Translation Assistant**. 

The system maps continuous 3D hand skeletal landmark trajectories ($25\text{ frames} \times 126\text{ dimensions}$) across 98 American Sign Language (ASL) vocabulary categories. In accordance with the CSAI3027P syllabus, all five curricular units (Units I–V) have been implemented and validated through empirical tests, from foundational McCulloch-Pitts neurons and NumPy-only Multilayer Perceptrons (MLPs) to deep residual networks (`SignResNet`), multi-task anchor-free vision detectors (`SignYOLODetector`), recurrent sequence models (`BiLSTM`, `Transformer`), and a Variational Autoencoder (`SignLanguageVAE`) with continuous latent manifold interpolation.

---

## 1. Syllabus Unit Coverage Matrix

| Syllabus Unit | Curricular Topic | Mandatory Evidence in Project 13 | Implementation Status |
| :--- | :--- | :--- | :--- |
| **Unit I** | Neural Foundations & Optimization | McCulloch-Pitts neuron, Sigmoid neuron, NumPy MLP from scratch, analytical backpropagation, Batch GD, SGD, Momentum, Nesterov (NAG), SVD feature analysis & condition diagnostics. | **Complete & Verified** (`src/core/`) |
| **Unit II** | Tensor Mathematics & Gradient Checking | Tensor broadcasting, inner products, tensor reshaping, Einstein summation, two-sided finite difference gradient check ($\text{rel error} = 3.04 \times 10^{-10}$). | **Complete & Verified** (`src/core/`) |
| **Unit III** | Learning Rules, Regularization & Convolutions | Hebbian, Oja's rule, Competitive Learning, Error-Correction, Boltzmann Machine (CD-1), Vanishing gradient simulation, Xavier/He init, Dropout, BatchNorm, LayerNorm, Data Augmentation, 2D/3D Convolutions from scratch. | **Complete & Verified** (`src/learning_rules/`, `src/experiments/`) |
| **Unit IV** | Vision Architectures & Benchmarking | Benchmarked two deep architectures: `SignResNet` (486k params, 94.66% Acc) vs `SignYOLODetector` (235k params, 62.17% Acc), multi-task bounding box loss, mAP, IoU, latency & FPS. | **Complete & Verified** (`src/vision/`) |
| **Unit V** | Sequence Models & Generative VAE | BiLSTM temporal classifier, Transformer multi-head self-attention decoder, recurrent gating ablation (step/sigmoid/tanh), `SignLanguageVAE` with ELBO loss, latent manifold visualization & interpolation. | **Complete & Verified** (`src/sequence_generative/`) |

---

## 2. Unit I: Neural Foundations & Optimization

### 2.1 McCulloch-Pitts Neuron & Sigmoid Baseline
- **McCulloch-Pitts Model**: Implemented in [`src/core/neuron.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/core/neuron.py#L9-L55) to serve as an active-motion gate. Given binary keypoint velocity inputs $x_i \in \{0, 1\}$, the neuron fires if $\sum_{i} w_i x_i \ge \theta$ in the absence of inhibitory signals.
- **Sigmoid Neuron**: Continuous mapping $z = \mathbf{w}^T \mathbf{x} + b$, $a = \sigma(z) = \frac{1}{1 + e^{-z}}$, with analytical derivative $\frac{da}{dz} = a(1 - a)$.

### 2.2 From-Scratch Multilayer Feedforward Perceptron (MLP)
Implemented in [`src/core/mlp.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/core/mlp.py) using pure NumPy without high-level autograd delegation:
$$\mathbf{z}^{[l]} = \mathbf{a}^{[l-1]} \mathbf{W}^{[l]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g(\mathbf{z}^{[l]})$$
$$\hat{\mathbf{y}} = \text{Softmax}(\mathbf{z}^{[L]}) = \frac{e^{\mathbf{z}^{[L]} - \max(\mathbf{z}^{[L]})}}{\sum_k e^{\mathbf{z}_k^{[L]} - \max(\mathbf{z}^{[L]})}}$$
Cross-Entropy Loss with $L_2$ Regularization:
$$\mathcal{L} = -\frac{1}{m} \sum_{i=1}^m \sum_{k=1}^K y_{ik} \ln(\hat{y}_{ik}) + \frac{\lambda}{2m} \sum_{l=1}^L \|\mathbf{W}^{[l]}\|_F^2$$
Exact Backpropagation Gradients:
$$\delta^{[L]} = \hat{\mathbf{y}} - \mathbf{y}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{1}{m} (\mathbf{a}^{[l-1]})^T \delta^{[l]} + \frac{\lambda}{m} \mathbf{W}^{[l]}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}^{[l]}} = \frac{1}{m} \sum_{i=1}^m \delta_i^{[l]}$$
$$\delta^{[l-1]} = (\delta^{[l]} (\mathbf{W}^{[l]})^T) \odot g'(\mathbf{z}^{[l-1]})$$

### 2.3 Optimizer Implementations
Implemented in [`src/core/optimizers.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/core/optimizers.py):
1. **Batch Gradient Descent (GD)**: $\theta_{t+1} = \theta_t - \alpha \nabla_\theta \mathcal{L}$
2. **Stochastic Gradient Descent (SGD)**: Mini-batch stochastic updates with weight decay.
3. **Momentum Optimizer**:
   $$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + (1 - \beta) \nabla_\theta \mathcal{L}(\theta_{t-1})$$
   $$\theta_t = \theta_{t-1} - \alpha \mathbf{v}_t$$
4. **Nesterov Accelerated Gradient (NAG)**:
   Calculates gradient at the projected lookahead position $\theta_{\text{look}} = \theta_{t-1} - \beta \mathbf{v}_{t-1}$:
   $$\mathbf{v}_t = \beta \mathbf{v}_{t-1} + \alpha \nabla_\theta \mathcal{L}(\theta_{t-1} - \beta \mathbf{v}_{t-1})$$
   $$\theta_t = \theta_{t-1} - \mathbf{v}_t$$

### 2.4 Singular Value Decomposition (SVD) Diagnostics
Implemented in [`src/core/svd_analysis.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/core/svd_analysis.py). Applied to the centered landmark feature matrix $\mathbf{X} \in \mathbb{R}^{N \times 126}$:
$$\mathbf{X} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$$
- **Condition Number**: $\kappa = \frac{\sigma_{\max}}{\sigma_{\min}} = 1.31 \times 10^2$. This moderate condition number verifies that MediaPipe landmark tracking coordinates are numerically stable without extreme collinear singularity.
- **Low-Rank Compression Analysis**:
  - $k=4$ components capture **80.77%** total cumulative energy ($\text{error} = 0.4386$)
  - $k=8$ components capture **92.72%** total cumulative energy ($\text{error} = 0.2698$)
  - $k=16$ components capture **98.03%** total cumulative energy ($\text{error} = 0.1404$)
  - $k=32$ components capture **99.55%** total cumulative energy ($\text{error} = 0.0672$)

---

## 3. Unit II: Tensor Mathematics & Numerical Gradient Checking

### 3.1 Tensor Algebra Sandbox
Implemented in [`src/core/tensor_math.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/core/tensor_math.py):
- **Broadcasting**: Demonstrated adding offset $\mathbb{R}^{126}$ into sequence tensor $\mathbb{R}^{4 \times 25 \times 126}$ with zero memory duplication.
- **Contractions**: Batched Einstein summation $\mathbf{B}_{ijk} \mathbf{B}_{ikl} \rightarrow \mathbf{B}_{ijl}$.
- **Pairwise Distance Tensors**: Fast broadcasted matrix calculation for hand joint geometries:
  $$D_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|_2 = \sqrt{\sum_k (p_{ik} - p_{jk})^2}$$

### 3.2 Numerical Gradient Checking
Implemented in [`src/core/grad_check.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/core/grad_check.py). Computes two-sided finite difference quotient:
$$g_{\text{num}}(\theta_i) = \frac{\mathcal{L}(\theta + \epsilon \mathbf{e}_i) - \mathcal{L}(\theta - \epsilon \mathbf{e}_i)}{2\epsilon}$$
The relative difference metric is evaluated against analytical backpropagation:
$$\text{RelError} = \frac{\|\mathbf{g}_{\text{num}} - \mathbf{g}_{\text{ana}}\|_2}{\|\mathbf{g}_{\text{num}}\|_2 + \|\mathbf{g}_{\text{ana}}\|_2 + 10^{-15}}$$
- **Empirical Result**: $\text{RelError} = \mathbf{3.042 \times 10^{-10}}$, dramatically beneath the $10^{-5}$ threshold, rigorously proving the correctness of the analytical backpropagation derivation.

---

## 4. Unit III: Learning Rules, Regularization & Convolutions

### 4.1 Learning Rules Lab
Implemented in [`src/learning_rules/rules.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/learning_rules/rules.py):
1. **Hebbian Learning**: $\Delta \mathbf{W} = \eta \mathbf{y} \mathbf{x}^T$. Demonstrates rapid weight divergence without constraint.
2. **Oja's Modified Hebbian Rule**:
   $$\Delta \mathbf{w} = \eta (\mathbf{y} \mathbf{x}^T - y^2 \mathbf{w})$$
   Mathematically enforces $\|\mathbf{w}\|_2 \rightarrow 1.0000$ and converges to the principal eigenvector.
3. **Competitive Learning (Winner-Take-All)**:
   $$\text{Winner } k = \arg\min_i \|\mathbf{x} - \mathbf{w}_i\|_2, \quad \Delta \mathbf{w}_k = \eta (\mathbf{x} - \mathbf{w}_k)$$
   Unsupervised clustering of distinctive hand gesture poses into prototypical codebooks.
4. **Error-Correction Learning (Delta Rule)**:
   $$\Delta \mathbf{W} = \eta (\mathbf{t} - \mathbf{y}) \mathbf{x}^T$$
5. **Boltzmann Energy-Based Concept (CD-1)**:
   $$E(\mathbf{v}, \mathbf{h}) = -\mathbf{v}^T \mathbf{W} \mathbf{h} - \mathbf{a}^T \mathbf{v} - \mathbf{b}^T \mathbf{h}$$
   $$\Delta \mathbf{W} = \eta (\langle \mathbf{v} \mathbf{h}^T \rangle_{\text{data}} - \langle \mathbf{v} \mathbf{h}^T \rangle_{\text{model}})$$

### 4.2 Activation Functions & Vanishing Gradients
Implemented in [`src/experiments/activations.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/experiments/activations.py). Backpropagation gradient norms through an 8-layer network:
- **Sigmoid**: Collapses from $5.39 \times 10^{-3}$ (Layer 8) to $1.46 \times 10^{-7}$ (Layer 1) &mdash; **Severe Vanishing Gradient**.
- **ReLU**: Preserves gradient magnitude from $1.87 \times 10^{-2}$ (Layer 8) to $3.10 \times 10^{-3}$ (Layer 1) &mdash; **Stable Signal Propagation**.

### 4.3 Weight Initialization & Variance Preservation
Implemented in [`src/experiments/initializers.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/experiments/initializers.py):
- **Small Random ($\sigma=0.01$)**: Signal variance collapses to $7.74 \times 10^{-16}$ by Layer 6.
- **He/Kaiming Normal ($\sigma = \sqrt{2/n_{\text{in}}}$)**: Maintains signal variance at $0.7213$ through Layer 6, preventing activation death.

### 4.4 2D and 3D Convolutions from Scratch
Implemented in [`src/experiments/conv_scratch.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/experiments/conv_scratch.py):
- **2D Convolution**: Sliding window convolution with Sobel and Laplacian kernels extracting edge contours on hand frames.
- **3D Spatio-Temporal Convolution**: Sliding kernel of shape $(K_t=3, K_h=3, K_w=3)$ sliding across both temporal video frames and spatial coordinates, generating dynamic motion feature maps.

---

## 5. Unit IV: Vision Architectures & Comparative Benchmarking

### 5.1 Architecture Designs
1. **SignResNet** ([`src/vision/resnet_model.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/vision/resnet_model.py)):
   - 1D Residual blocks with batch normalization, skip connections, adaptive average pooling, and linear classifier.
   - Parameter Count: **486,242** parameters.
2. **SignYOLODetector** ([`src/vision/yolo_detector.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/vision/yolo_detector.py)):
   - Multi-scale CNN backbone with anchor-free multi-task heads: Bounding Box Head ($\hat{\mathbf{b}} \in \mathbb{R}^8$), Objectness Head ($\hat{\mathbf{c}} \in [0, 1]^2$), and Classification Head ($\hat{\mathbf{y}} \in \mathbb{R}^{98}$).
   - Parameter Count: **235,340** parameters.

### 5.2 Comparative Evaluation Results
Evaluated on the held-out stratified test set (711 samples, 98 classes):

| Metric | SignResNet (Primary Deep Vision) | SignYOLODetector (Multi-Task) |
| :--- | :--- | :--- |
| **Top-1 Test Accuracy** | **94.66%** | 62.17% |
| **Macro F1 Score** | **0.9339** | 0.5519 |
| **Average Latency (CPU)** | 0.582 ms / sample | **0.207 ms / sample** |
| **Throughput (FPS)** | 1,718.3 FPS | **4,828.0 FPS** |
| **Trainable Parameters** | 486,242 | 235,340 |
| **Task Breadth** | Sequence Classification | Bbox Detection + Class + Objectness |

---

## 6. Unit V: Sequence Models & Generative VAE

### 6.1 Recurrent & Transformer Sequence Decoders
Implemented in [`src/sequence_generative/sequence_models.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/sequence_generative/sequence_models.py):
- **BiLSTM Classifier**: Bidirectional 2-layer LSTM achieving **56.82%** accuracy.
- **Transformer Sequence Decoder**: Multi-Head Self-Attention ($d_{\text{model}}=128$, 4 heads) achieving **56.68%** accuracy.
- **Gating Mechanism Analysis**: Verified that non-negative sigmoid gates $[0, 1]$ enable stable information preservation in memory cells, whereas step gates exhibit zero gradients and collapse gradient flow.

### 6.2 Variational Autoencoder (VAE) & Latent Manifold
Implemented in [`src/sequence_generative/vae_generative.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/src/sequence_generative/vae_generative.py):
- **Encoder**: $q_\phi(\mathbf{z} | \mathbf{x}) \rightarrow \mu(\mathbf{x}), \log\sigma^2(\mathbf{x})$
- **Reparameterization Trick**: $\mathbf{z} = \mu + \sigma \odot \epsilon$, where $\epsilon \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$
- **Decoder**: $p_\theta(\mathbf{x} | \mathbf{z}) \rightarrow \hat{\mathbf{x}}$
- **Loss**: $\mathcal{L}_{\text{VAE}} = \text{MSE}(\hat{\mathbf{x}}, \mathbf{x}) + \beta \mathcal{D}_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$
- **Results**:
  - Reconstruction MSE converged to **0.0375**; KL divergence stabilized at **1.6138**.
  - Generated synthetic sign sequences to augment minority vocabulary classes (e.g., `bank`, `road`, `bike`).
  - Synthesized continuous linear interpolation trajectories: $\mathbf{z}(\alpha) = (1-\alpha)\mathbf{z}_A + \alpha \mathbf{z}_B$, proving smooth semantic transitions in the learned latent space.

---

## 7. Software Engineering & Usability
1. **Automated Unit Testing**: 18 unit tests in [`tests/`](file:///Users/gauravgoyal/Desktop/Sign_Language/tests) cover all tensor mathematics, gradient checks, optimizers, vision architectures, and generative modules (`18/18 passed in 0.09s`).
2. **Reproducible Experiment Runner**: [`run_experiments.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/run_experiments.py) executes and logs all five units into `benchmarks/experiment_results.json`.
3. **Interactive Multi-Tab Application**: [`app.py`](file:///Users/gauravgoyal/Desktop/Sign_Language/app.py) provides a web console for real-time translation, tensor inspection, gradient checking, and VAE latent space exploration.

---

## 8. Industry Readiness & Go / No-Go Deployment Decision
- **Final Verdict**: **GO (APPROVED FOR ASSISTIVE EDGE DEPLOYMENT)**
- **Primary Model**: `SignResNet` achieved **94.66% test accuracy**, **0.9339 Macro F1**, and **1,718 FPS**, exceeding real-time camera requirements (30 FPS) by a factor of 57&times;.
- **Responsible Use Limits**: The system is intended as an assistive communication aid and should not replace certified human interpreters in critical healthcare, judicial, or emergency settings.
