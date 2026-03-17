# Deep Learning Exam: High-Priority Practice Set

This set contains solved mathematical problems and frequent long-form questions found in your question papers.

---

## 1. Mathematical Problems (Must Practice)

### Q: Convolute the input $[2, 3, 5, 6, 7, 9]$ with kernel $[2, 3]$.
**Solution:**
Convolution is a sliding product sum (assume Valid padding):
1.  $(2 \times 2) + (3 \times 3) = 4 + 9 = \mathbf{13}$
2.  $(3 \times 2) + (5 \times 3) = 6 + 15 = \mathbf{21}$
3.  $(5 \times 2) + (6 \times 3) = 10 + 18 = \mathbf{28}$
4.  $(6 \times 2) + (7 \times 3) = 12 + 21 = \mathbf{33}$
5.  $(7 \times 2) + (9 \times 3) = 14 + 27 = \mathbf{41}$
**Result:** $[13, 21, 28, 33, 41]$

### Q: Calculate the $L^1$ and $L^2$ norms for vector $x = [3, -4]$.
*   **$L^1$ Norm (Manhattan):** $\|x\|_1 = |3| + |-4| = 3 + 4 = \mathbf{7}$
*   **$L^2$ Norm (Euclidean):** $\|x\|_2 = \sqrt{3^2 + (-4)^2} = \sqrt{9 + 16} = \sqrt{25} = \mathbf{5}$

### Q: Demonstrate the calculation of an Inverse Matrix for $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$.
*   **Determinant ($|A|$):** $(1 \times 4) - (2 \times 3) = 4 - 6 = -2$.
*   **Adjoint ($adj A$):** Swap diagonal elements, negate others $\rightarrow \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix}$.
*   **Inverse ($A^{-1}$):** $\frac{1}{|A|} adj A = \frac{1}{-2} \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix} = \mathbf{\begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix}}$.

---

## 2. High-Frequency Brief Answers

### Q: Discuss in detail learning the XOR function.
*   **Problem:** XOR is not linearly separable. A single-layer perceptron (linear model) cannot solve it because it can't draw a single line to separate $\{0,1\}$ from $\{0,0\}$ and $\{1,1\}$.
*   **Solution:** Use a Multi-Layer Perceptron (MLP).
    *   **Architecture:** Input layer (2), Hidden layer (2 with ReLU/Sigmoid), Output layer (1).
    *   **Mechanism:** The hidden layer projects the input into a new space where the points become linearly separable.

### Q: Explain the mechanism of RNN Unfolding.
*   RNNs process sequences by applying the same weights to each timestamp.
*   **Unfolding** means expanding the recurrent loop into a chain of layers.
*   If we have inputs $x_1, x_2, x_3$, unfolding creates 3 computational steps where hidden state $h_1$ is passed to the next step to calculate $h_2$, and so on.
*   **Advantage:** Allows the use of standard backpropagation (Backpropagation Through Time).

### Q: Compare Boltzmann Machines (BM) and Restricted Boltzmann Machines (RBM).
| Feature | Boltzmann Machine (BM) | Restricted Boltzmann Machine (RBM) |
| :--- | :--- | :--- |
| **Connections** | Any node can connect to any other | Bipartite graph (Visible ↔ Hidden only) |
| **Intra-layer** | Connections allowed within layers | No connections within a layer |
| **Complexity** | High (difficult to train) | Simpler and more efficient |
| **Usage** | General optimization | Dimensionality reduction, DBN building blocks |

---

## 3. Key Algorithms to Memorize
*   **Backpropagation:** Uses the **Chain Rule** to update weights. $\frac{\partial Loss}{\partial W} = \frac{\partial Loss}{\partial Activation} \cdot \frac{\partial Activation}{\partial Preactivation} \cdot \frac{\partial Preactivation}{\partial W}$.
*   **Expectation Maximization (EM):**
    1.  **E-Step:** Estimate the values of hidden/latent variables.
    2.  **M-Step:** Maximize the likelihood of the parameters given the estimated values.
*   **MAP (Maximum A Posteriori):** Finds the point estimate of parameters $\theta$ that maximizes the posterior: $\theta_{MAP} = \text{argmax } P(D|\theta)P(\theta)$.
