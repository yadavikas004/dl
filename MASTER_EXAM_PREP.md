# 🏆 Deep Learning: Ultimate Exam Master Revision File

This document consolidates all Unit Notes, Question Paper (QP1-QP5) solutions, and mandatory Keywords into one guide. **Memorize this to write your exam successfully.**

---

## 🔑 Part 1: High-Priority Keywords (Must use in answers)
*   **Linear Algebra:** Eigen-decomposition, Singular Matrix, Orthogonal, Hadamard Product, Tensors.
*   **Numerical:** Overflow, Underflow, Condition Number, Saddle Point, Convexity.
*   **Networks:** Multi-Layer Perceptron (MLP), XOR separation, Activation (ReLU/Softmax), Hyperparameters.
*   **Regularization:** Overfitting/Underfitting, Sparsity (L1), Weight Decay (L2), Co-adaptation (Dropout), Early Stopping.
*   **CNN:** Filter/Kernel, Stride, Padding (Same/Valid), Feature Map, Receptive Field, Weight Sharing.
*   **RNN:** Sequential data, Hidden state, BPTT (Backprop Through Time), Vanishing Gradient, Unfolding.
*   **Advanced:** Latent Variables, Manifold, Zero-Sum (GANs), Stochastic, Posterior ($P(\theta|D)$), KL Divergence.

---

## 🏛️ Part 2: Detailed Unit-Wise Revision & QP Solutions

### Unit I: Applied Math & Numerical Optimization
**1. Matrix Operations (QP3):**
*   **Product:** $A \cdot B$ (Row $\times$ Column). Result size ($m \times p$) from ($m \times n$) and ($n \times p$).
*   **Hadamard ($A \odot B$):** Element-wise multiplication. Must be same shape.
*   **Dot Product:** Sum of element-wise products. Result is a scalar.

**2. Eigenvalues & Eigenvectors (QP2, QP3):**
*   **Definition:** Vectors $v$ whose direction doesn't change when transformed by matrix $A$.
*   **Equation:** $A v = \lambda v$.
*   **Role:** Used in PCA for dimensionality reduction and understanding matrix properties.

**3. Poor Conditioning (QP1, QP2):**
*   **Definition:** When tiny changes in input cause massive changes in output.
*   **Formula:** Condition Number $= \frac{|\lambda_{max}|}{|\lambda_{min}|}$. High number = Poorly conditioned (unstable).

**4. Optimization Challenges (QP1, QP3):**
*   **Saddle Point:** Gradient is zero, but it's a min in one axis and max in another. High-dimensional networks have more saddle points than local minima.
*   **Lagrangian:** Technique to solve optimization with constraints (e.g., keeping weights below a threshold).

---

### Unit II: Deep Networks & Regularization
**1. Why Hidden Layers? (The XOR Problem - QP1):**
*   XOR cannot be solved by a linear classifier (single neuron).
*   **Solution:** Hidden layers project inputs into a feature space where they become linearly separable.

**2. Activation Functions (QP1):**
*   **Sigmoid:** Good for probability but causes vanishing gradients.
*   **ReLU:** $max(0, x)$. Standard for deep networks; solves vanishing gradient.
*   **Softmax:** Used in output layers for multi-class classification (sum of outputs = 1).

**3. Regularization Techniques (QP1, QP3, QP5):**
*   **L1 (Lasso):** $\|w\|_1$. Adds sum of absolute weights. Results in **Sparsity**.
*   **L2 (Weight Decay):** $\|w\|_2^2$. Adds sum of squared weights. Keeps weights small.
*   **Dropout:** Randomly deactivates neurons ($p=0.5$). Prevents **co-adaptation**.
*   **Batch Normalization:** Normalizes input to each layer. Speeds up training and stabilizes gradients.

**4. Hyperparameter Tuning (QP1, QP3):**
*   **Definition:** Manual selection of parameters *not* learned by the model (e.g., Learning rate, Depth).
*   **Grid Search vs Random Search:** Systematically trying combinations vs randomly picking.

---

### Unit III: CNNs & Sequence Modeling
**1. CNN Components (QP2, QP3, QP4):**
*   **Convolution Layer:** Extracts spatial features using filters.
*   **Pooling (Max/Avg):** Reduces dimension, provides translation invariance, and reduces computation.
*   **Stride/Padding:** Stride (step size), Padding (adding 0s to edges).

**2. Object Localization & Vision (QP1, QP2, QP3):**
*   **Sliding Window:** Checking small overlapping crops of an image for objects.
*   **Inception Net:** Uses parallel convolutions (1x1, 3x3, 5x5) to capture different feature sizes.
*   **Transfer Learning:** Taking weights from a pre-trained model (ResNet/VGG) and applying to new data.

**3. RNN & Sequence Modeling (QP2, QP3):**
*   **Computational Unfolding:** Copying the same cell for each step in a sequence.
*   **LSTM/GRU (QP3):** Solves the vanishing gradient problem using "Gates" to keep long-term memory.
*   **Word Embeddings (QP4):** Mapping words to dense vectors (Word2Vec) so similar words (Apple/Orange) are mathematically close.

---

### Unit IV: Advanced Research & Generative Models
**1. Linear Factor Models (QP4):**
*   **Probabilistic PCA:** Finds hidden factors that explain data variance, assuming Gaussian noise.
*   **Factor Analysis:** Unique noise for each dimension (unlike PCA).
*   **ICA (Independent Component Analysis):** Separates signals (e.g., "Cocktail Party Problem") assuming non-Gaussian distribution.

**2. Autoencoders (QP1, QP2, QP5):**
*   **Goal:** Learn a compressed representation (Code).
*   **Denoising AE:** Reconstructs inputs from "noisy" versions. Excellent for robust features.
*   **Undercomplete AE:** Hidden layer is smaller than input, forcing compression.

**3. GANs - Generative Adversarial Networks (QP1, QP4):**
*   **Minimax Game:** Generator vs Discriminator.
*   **Generator:** Learns to create "Fake" data to fool the Discriminator.
*   **Discriminator:** Learns to distinguish between "Real" (from dataset) and "Fake".
*   **Ideally:** Discriminator probability becomes 0.5 (cannot tell real from fake).

**4. Approximate Inference (QP2, QP3):**
*   **Why?** Solving exact probability formulas (integrals) in deep models is computationally impossible.
*   **Expectation Maximization (EM):** Two-step iterative loop to find latent variables.
*   **MAP (Maximum A Posteriori):** Bayesian method using prior information to estimate parameters.

---

## ✍️ Part 3: Math Revision Summary (Memorize these Formulas)

1.  **Inverse Matrix ($A^{-1}$):** $A^{-1} = \frac{1}{|A|} adj(A)$
2.  **L2 Norm:** $\|x\|_2 = \sqrt{x_1^2 + x_2^2 + ... + x_n^2}$
3.  **L1 Norm:** $\|x\|_1 = |x_1| + |x_2| + ... + |x_n|$
4.  **Gradient Descent Update:** $\theta = \theta - \eta \cdot \nabla J(\theta)$
5.  **Momentum Update:** $v = \alpha v - \eta \nabla$, and $\theta = \theta + v$
6.  **ELBO (Evidence Lower Bound):** Used to optimize approximate inference.

---

### 💡 Final Tip for the Exam Room
*   **Diagrams matter:** If you're explaining CNN, draw a 3-layer diagram (Conv-Pool-FC).
*   **Bold Keywords:** Highlight terms like **Feature Map**, **Backpropagation**, and **Non-convex**.
*   **Time Management:** Solve valid convolution numericals quickly; they are reliable marks.
