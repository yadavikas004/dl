# Deep Learning: Keywords Cheat Sheet

Include these technical terms in your answers to impress the examiner and secure higher marks.

---

## 🟢 Unit I: Math Fundamentals
*   **Orthogonal Matrix:** A square matrix whose transpose equals its inverse ($A^T = A^{-1}$).
*   **Eigen-decomposition:** Factoring a matrix into eigenvalues and eigenvectors.
*   **Condition Number:** Measure of numerical stability ($λ_{max} / λ_{min}$).
*   **Overflow/Underflow:** Exceeding memory limits / rounding small values to zero.
*   **Saddle Point:** Zero gradient but not a local extremum.
*   **Lagrangian Formulation:** For solving constrained optimization problems.
*   **Jacobian Matrix:** Matrix of all first-order partial derivatives (essential for Backprop).
*   **Hessian Matrix:** Matrix of second-order derivatives (used to find saddle points).
*   **Frobenius Norm:** The generalization of the $L^2$ norm for matrices ($\sqrt{\sum A_{ij}^2}$).
*   **Basis & Span:** Set of linearly independent vectors that "reach" any point in a space.
*   **Singular Value Decomposition (SVD):** Generalization of eigen-decomposition for non-square matrices.
*   **Moore-Penrose Pseudoinverse:** Used to find "best-fit" solutions for non-invertible systems.
*   **Trace Operation:** Sum of diagonal elements (invariant under basis change).

---

## 🟢 Unit II: Training & Regularization
*   **Hyperparameter Tuning:** Adjusting settings like learning rate or batch size.
*   **Sparsity:** When many weights are zero (induced by L1 regularization).
*   **Weight Decay:** Explicitly using L2 regularization to keep weights small.
*   **Co-adaptation:** When neurons depend too much on each other (prevented by Dropout).
*   **Non-convex Loss Surface:** Complex error landscapes with many local minima.
*   **Inception Module:** Architecture that uses parallel filters of different sizes.
*   **Early Stopping:** Monitoring validation error and stopping training before overfitting starts.
*   **Data Augmentation:** Creating synthetic samples (flips, rotations) to improve generalization.
*   **Batch Normalization:** Normalizing layer activations to stabilize and speed up training.
*   **Gradient Clipping:** Capping gradients to prevent numerical explosions in RNNs.
*   **Empirical Risk Minimization (ERM):** Minimizing training error in hopes of generalizing well.
*   **Adversarial Training:** Training on corrupted/adversarial inputs to improve robustness.
*   **Label Smoothing:** Preventing the model from becoming over-confident (regularization).
*   **Cross-Entropy Loss:** Standard objective for classification ($-\sum y \log \hat{y}$).

---

## 🟢 Unit III: CNNs & Sequence Modeling
*   **Weight Sharing:** Using the same filter across the whole image (reduces parameters).
*   **Translation Invariance:** Network identifying objects regardless of their position.
*   **Sliding Window:** Standard approach for object localization.
*   **Vanishing Gradients:** Gradients becoming zero in deep chains (fixed by LSTM/GRU).
*   **Computational Unfolding:** Converting a loop (RNN) into a chain of layers.
*   **Gated Logic:** Used in LSTMs (Input, Forget, Output gates) for long-term memory.
*   **BPTT (Backpropagation Through Time):** The standard training algorithm for RNNs.
*   **Teacher Forcing:** Using ground-truth inputs during training of sequential models.
*   **Attention Mechanism:** Mechanism allowing the model to focus on specific parts of input.
*   **Residual Connection:** "Skip connections" that bypass layers to fix vanishing gradients (ResNet).
*   **Dilated (Atrous) Convolution:** Convolution with gaps to increase receptive field without pooling.
*   **Max/Average Pooling:** Downsampling feature maps while providing translation invariance.
*   **GRU (Gated Recurrent Unit):** Simpler version of LSTM with only Reset and Update gates.
*   **Hidden State ($h_t$):** Vector that carries "memory" across time steps in an RNN.

---

## 🟢 Unit IV: Advanced Research
*   **Manifold:** A low-dimensional surface where high-dimensional data actually lives.
*   **Latent Variable:** A hidden factor that influences observed data.
*   **Zero-Sum Game:** The adversarial nature of GANs (Generator vs Discriminator).
*   **Energy-Based Model:** Stochastic networks like Boltzmann Machines.
*   **Posterior Distribution:** The probability of parameters given the data ($P(\theta|D)$).
*   **Monte Carlo Sampling:** Randomly sampling data to estimate properties (MCMC).
*   **Exploration/Exploitation:** The fundamental trade-off in Reinforcement Learning.
*   **KL Divergence:** A measure of how one probability distribution differs from another.
*   **ELBO (Evidence Lower Bound):** A objective function used to approximate the log-likelihood in VAEs.
*   **Contrastive Divergence:** An efficient algorithm for training Restricted Boltzmann Machines (RBMs).
*   **Gibbs Sampling:** A MCMC algorithm used to generate samples from complex distributions.
*   **Deep Belief Network (DBN):** Generative model consisting of multiple layers of RBMs (greedy training).
*   **Factor Analysis:** Identifying hidden (latent) factors that explain data variance.
*   **ICA (Independent Component Analysis):** Separating mixed signals (The Cocktail Party Problem).
*   **Variational Autoencoder (VAE):** Autoencoder that learns a continuous distribution (mean/variance).
*   **Denoising Autoencoder:** Learning features by reconstructing original data from noisy input.
*   **Manifold Learning:** Extracting the low-dimensional structure (manifold) from high-dimensional data.
*   **MCMC (Markov Chain Monte Carlo):** Sampling class used to approximate complex probabilities.

---

### 🔥 Top Tips for Paper Writing:
1.  **Bold the keywords:** When you use a term from this list, underline or write it clearly.
2.  **Define it first:** Start a brief answer with the definition including the keyword.
3.  **Support with Formulas:** E.g., for L1 regularization, write $\|w\|_1$.
