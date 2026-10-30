# AquaGuard AI: Methodological Framework & Architectural Design

## 1. Executive Summary & Problem Formulation
Safe, potable drinking water is fundamental to human survival and public health. Traditional water quality testing relies on physical laboratory sampling, bench chemical assays, and microbial culturing. These manual workflows are inherently episodic, labor-intensive, and incapable of continuous real-time hazard detection.

**AquaGuard AI** develops a Deep Neural Network (DNN) computational pipeline that ingests continuous physicochemical water parameters, applies rigorous statistical preprocessing without data leakage, classifies contamination risk, evaluates regulatory adherence against WHO and US-EPA benchmarks, and provides transparent explanations using Explainable AI (XAI) feature attribution.

---

## 2. Dataset Preprocessing & Leakage-Free Design

### 2.1 Missing Value Imputation
Environmental sampling stations frequently suffer from missing sensor readings due to battery depletion, recalibration cycles, or telemetry dropouts.
To prevent **data leakage**, imputation parameters must never be computed across the global dataset. AquaGuard AI enforces:
1. **Stratified Split:** Data is partitioned into Training (70%), Validation (15%), and Test (15%) partitions before computing statistics.
2. **Median Imputation:** Training partition medians $\tilde{x}_j^{\text{train}}$ are extracted and applied as persistent replacement values for any missing coordinates in validation and test partitions:
   $$x_{i,j}^{\text{imputed}} = \begin{cases} x_{i,j} & \text{if } x_{i,j} \neq \text{NaN} \\ \tilde{x}_j^{\text{train}} & \text{if } x_{i,j} = \text{NaN} \end{cases}$$

### 2.2 Outlier Winsorization
Sensor anomalies and electrical transients produce extreme spikes. Outliers are bounded at the 1st ($Q_{0.01}$) and 99th ($Q_{0.99}$) percentiles calculated strictly on the training partition:
$$\hat{x}_{i,j} = \min\left( \max(x_{i,j}^{\text{imp}}, Q_{0.01, j}^{\text{train}}), Q_{0.99, j}^{\text{train}} \right)$$

### 2.3 Feature Standardization
Neural networks optimized with gradient descent require isotropic feature spaces with zero mean and unit variance. Standard scaling is fitted strictly on the training split:
$$z_{i,j} = \frac{\hat{x}_{i,j} - \mu_j^{\text{train}}}{\sigma_j^{\text{train}}}$$

---

## 3. Deep Feedforward Neural Network (DNN) Architecture

### 3.1 Network Topology
AquaGuard AI implements a feedforward multilayer perceptron structured to mitigate overfitting while learning non-linear boundary manifolds:

1. **Input Layer:** $D = 9$ dimensional input vector $\mathbf{x} \in \mathbb{R}^9$.
2. **Hidden Layer 1:** 128 units, Rectified Linear Unit (ReLU) activation, $L_2$ weight regularization ($\lambda = 10^{-4}$).
3. **Batch Normalization Layer:** Normalizes pre-activations to accelerate training convergence and reduce internal covariate shift:
   $$\hat{z}_k = \frac{z_k - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}, \quad y_k = \gamma \hat{z}_k + \beta$$
4. **Dropout Layer 1:** Dropout rate $p = 0.30$, randomly dropping 30% of unit activations during training to break co-adaptation.
5. **Hidden Layer 2:** 64 units, ReLU activation, $L_2$ regularization.
6. **Dropout Layer 2:** Dropout rate $p = 0.20$.
7. **Hidden Layer 3:** 32 units, ReLU activation.
8. **Output Layer:** Single neuron with Sigmoid activation function yielding potability probability $\hat{y} \in [0, 1]$:
   $$\hat{y} = \sigma(\mathbf{w}^T \mathbf{h}_3 + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{h}_3 + b)}}$$

---

## 4. Mathematical Foundations

### 4.1 Forward Propagation
For layer $l \in \{1, 2, \dots, L\}$:
$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}$$
$$\mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$
where $g^{[l]}$ denotes the activation function (ReLU for hidden layers, Sigmoid for final classification).

### 4.2 Objective Loss Function
The model optimizes Binary Cross-Entropy with $L_2$ weight regularization:
$$\mathcal{L}(\mathbf{w}, b) = -\frac{1}{N} \sum_{i=1}^N \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2N} \sum_{l=1}^L \|\mathbf{W}^{[l]}\|_F^2$$

### 4.3 Backpropagation & Gradient Optimization
Gradients are computed via the chain rule:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[L]}} = \hat{\mathbf{y}} - \mathbf{y}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{W}^{[l]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}} (\mathbf{a}^{[l-1]})^T + \frac{\lambda}{N}\mathbf{W}^{[l]}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l-1]}} = (\mathbf{W}^{[l]})^T \frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l]}}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}^{[l-1]}} = \frac{\partial \mathcal{L}}{\partial \mathbf{a}^{[l-1]}} \odot g'^{[l-1]}(\mathbf{z}^{[l-1]})$$

The network parameters are updated using the **Adam Optimizer** (Adaptive Moment Estimation):
$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
$$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

---

## 5. Explainable AI (XAI) & Attribution
To satisfy high-stakes environmental safety requirements, AquaGuard AI conducts Permutation Feature Importance and SHAP attribution. The importance $I(f_j)$ measures model degradation upon permuting parameter $j$ over the test distribution:
$$I(f_j) = \mathcal{M}(X_{\text{test}}, y_{\text{test}}) - \mathcal{M}(X_{\text{test}}^{\pi_j}, y_{\text{test}})$$

---

## 6. Academic Limitations & Regulatory Safeguards
1. **Microbiological Exclusion:** Physicochemical measurements (e.g., pH, Turbidity, TDS) cannot detect viral, bacterial (e.g., *Vibrio cholerae*, *Cryptosporidium*), or parasitic pathogens directly.
2. **Heavy Metal Profiling:** Standard testing kits do not measure dissolved arsenic, cadmium, or lead without atomic absorption spectroscopy.
3. **Regulatory Status:** AquaGuard AI provides computational decision support and risk indicators, not certified municipal safety endorsements.
