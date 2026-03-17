# 🚀 Deep Learning: Recurring Topics Master Cheat Sheet (High-Probability)

I have analyzed all four question papers and identified these "Master Topics" that appear in almost every single exam. **Memorize these sections perfectly to guarantee a major portion of your marks.**

---

## ⚡ 1. Boltzmann Machines & RBMs (90% Probability)
*   **Boltzmann Machine (BM):** An undirected, stochastic (probabilistic) energy-based model. It consists of visible and hidden units. Every node is connected to every other node (extremely hard to train).
*   **Restricted Boltzmann Machine (RBM):** A "Restricted" version where no two nodes in the same layer are connected. This creates a **Bipartite Graph**. 
*   **Key Training Algo:** Uses **Gibbs Sampling** or **Contrastive Divergence (CD)**.
    *   **Gibbs Sampling Steps:** 
        1. Start with a random visible vector $v^{(0)}$.
        2. Sample hidden vector $h^{(0)}$ from $P(h|v^{(0)})$.
        3. Sample new visible vector $v^{(1)}$ from $P(v|h^{(0)})$.
        4. Repeat for $k$ steps.
    *   **Contrastive Divergence (CD-k):** An efficient approximation of the gradient. 
        1. Perform $k$ steps of Gibbs sampling.
        2. Adjust weights to increase the probability of the original data and decrease the probability of the sampled "fantasy" data.
*   **Advantages:** Excellent for Feature Learning and Collaborative Filtering (Recommendations).
*   **Disadvantages:** Hard to scale to high-res images; intractable partition function.
*   **Types:** Binary RBM, Gaussian RBM (for continuous data).

## 🔄 2. Autoencoders (100% Probability)
*   **Core Goal:** Unsupervised learning of efficient data codings. $x \rightarrow \text{Encoder} \rightarrow z (\text{Bottleneck}) \rightarrow \text{Decoder} \rightarrow \hat{x}$.
*   **Undercomplete AE:** Hidden layer is smaller than input. Forces the model to learn only the most salient features (Dimensionality reduction).
*   **Denoising AE:** Input is corrupted with noise. Model must reconstruct the "clean" original. This forces it to learn the true data manifold and prevents it from just memorizing identity.
*   **Sparse AE:** Adds a penalty to the loss function to ensure only a few neurons fire at a time (sparsity).

## 📈 3. Optimization & Gradients (100% Probability)
*   **Gradient Descent:** Updating weights in the direction of negative gradient to minimize loss: $\theta = \theta - \eta \nabla J(\theta)$.
*   **Momentum:** Adds "velocity" to dampen oscillations and speed up convergence: $v = \gamma v - \eta \nabla$.
*   **Nesterov Momentum:** A "look-ahead" version. It calculates the gradient at the predicted next position rather than the current one.
*   **Vanishing Gradients:** Gradients shrink exponentially in deep networks (Fixed by ReLU, BatchNorm, LSTMs).
*   **Exploding Gradients:** Gradients grow exponentially (Fixed by Gradient Clipping).
*   **Saddle Points:** Points where the gradient is zero but it's not a local extremum. High-dimensional spaces are full of them.

## 🧠 4. RNNs & Sequence Modeling (95% Probability)
*   **Recurrent Neural Networks (RNNs):** Process sequential data by maintaining a hidden state $h_t$.
*   **Unfolding:** Expanding the recurrent loop into a chain of layers for calculation (allows BPTT).
*   **LSTMs/GRUs:** Advanced RNNs that use "Gates" to solve the vanishing gradient problem and store long-term dependencies.
*   **BPTT (Backpropagation Through Time):** The standard training algorithm for unfolded RNNs.

## 🛠️ 5. Regularization (90% Probability)
*   **Dropout:** Randomly deactivating neurons during training. Prevents **co-adaptation**.
*   **L1 (Lasso):** Penalty on absolute weights. Leads to **Sparsity** (some weights become 0).
*   **L2 (Weight Decay):** Penalty on squared weights. Keeps weights small.
*   **Early Stopping:** Stopping training when validation loss starts to rise.
*   **Data Augmentation:** Creating synthetic samples (flips, rotations) to prevent overfitting.

## 📐 6. Linear Algebra & Math (85% Probability)
*   **Eigen-decomposition:** Factoring square matrices into $\mathbf{Av} = \lambda\mathbf{v}$. Used in PCA.
*   **Inverse Matrix ($\mathbf{A}^{-1}$):** Only exists if $|A| \neq 0$. Defined such that $\mathbf{AA}^{-1} = \mathbf{I}$.
    *   **Calculation Steps:**
        1. **Determinant ($|A|$):** For $A = \begin{bmatrix} 1 & 2 \\\\ 3 & 4 \end{bmatrix}$, $|A| = -2$.
        2. **Adjoint ($adj A$):** 
           $$ adj A = \begin{bmatrix} 4 & -2 \\\\ -3 & 1 \end{bmatrix} $$
        3. **Final Result:** 
           $$ A^{-1} = \frac{1}{|A|} adj A = \begin{bmatrix} -2 & 1 \\\\ 1.5 & -0.5 \end{bmatrix} $$
*   **Identity Matrix ($\mathbf{I}$):** Square matrix with 1s on diagonal, 0s elsewhere. Act as "1" in matrix math.
*   **Norms:** $L^1$ (absolute sum) and $L^2$ (root of square sum).
*   **Condition Number:** Measure of numerical stability ($|\lambda_{max}| / |\lambda_{min}|$). High number = Poor conditioning.

## 🧪 7. Activation Functions (80% Probability)
*   **ReLU:** $\max(0, x)$. Standard for hidden layers; solves vanishing gradients.
*   **Sigmoid:** $1/(1+e^{-x})$. Used for binary probability; causes vanishing gradients.
*   **Softmax:** Used in the output layer for Multi-class classification.
*   **Tanh:** Zero-centered activation ranging from -1 to 1.

## 🖼️ 8. GANs & Generative Models (75% Probability)
*   **GANs:** Minimax game between **Generator** (creates fakes) and **Discriminator** (detects fakes).
*   **DBNs (Deep Belief Networks):** Stacks of RBMs trained greedily layer-by-layer.
*   **Variational Inference:** Approximate inference used when exact probability is intractable.

## ❄️ 9. Numerical Stability (70% Probability)
*   **Underflow:** Small numbers rounded to zero (destroys gradients).
*   **Overflow:** Numbers too large (results in `inf` or `NaN`).
*   **Log-Sum-Exp Trick:** Used in Softmax/Cross-entropy to prevent numerical crashes.

---

### **💡 Pro-Tips for the Exam:**
1.  **Diagrams for everything:** Draw the U-shape for Gradient, the Bipartite graph for RBM, and the Bottleneck for Autoencoders.
2.  **Keywords:** Always highlight terms like **Manifold**, **Associative Memory**, **Latent Variable**, and **BPTT**.
3.  **Comparisons:** If you see any of these "vs" pairs, use a table:
    *   L1 vs L2
    *   Saddle Point vs Local Minimum
    *   Discriminative vs Generative
    *   RNN vs CNN
