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

---

## 🟢 Unit II: Training & Regularization
*   **Hyperparameter Tuning:** Adjusting settings like learning rate or batch size.
*   **Sparsity:** When many weights are zero (induced by L1 regularization).
*   **Weight Decay:** Explicitly using L2 regularization to keep weights small.
*   **Co-adaptation:** When neurons depend too much on each other (prevented by Dropout).
*   **Non-convex Loss Surface:** Complex error landscapes with many local minima.
*   **Inception Module:** Architecture that uses parallel filters of different sizes.

---

## 🟢 Unit III: CNNs & Sequence Modeling
*   **Weight Sharing:** Using the same filter across the whole image (reduces parameters).
*   **Translation Invariance:** Network identifying objects regardless of their position.
*   **Sliding Window:** Standard approach for object localization.
*   **Vanishing Gradients:** Gradients becoming zero in deep chains (fixed by LSTM/GRU).
*   **Computational Unfolding:** Converting a loop (RNN) into a chain of layers.
*   **Gated Logic:** Used in LSTMs (Input, Forget, Output gates) for long-term memory.

---

## 🟢 Unit IV: Advanced Research
*   **Manifold:** A low-dimensional surface where high-dimensional data actually lives.
*   **Latent Variable:** A hidden factor that influences observed data.
*   **Zero-Sum Game:** The adversarial nature of GANs (Generator vs Discriminator).
*   **Energy-Based Model:** Stochastic networks like Boltzmann Machines.
*   **Posterior Distribution:** The probability of parameters given the data ($P(\theta|D)$).
*   **Monte Carlo Sampling:** Randomly sampling data to estimate properties (MCMC).
*   **Exploration/Exploitation:** The fundamental trade-off in Reinforcement Learning.

---

### 🔥 Top Tips for Paper Writing:
1.  **Bold the keywords:** When you use a term from this list, underline or write it clearly.
2.  **Define it first:** Start a brief answer with the definition including the keyword.
3.  **Support with Formulas:** E.g., for L1 regularization, write $\|w\|_1$.
