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

### Q: Demonstrate the calculation of an Inverse Matrix for A:
$$ A = \begin{bmatrix} 1 & 2 \\\\ 3 & 4 \end{bmatrix} $$
*   **Determinant ($|A|$):** $(1 \times 4) - (2 \times 3) = 4 - 6 = -2$.
*   **Adjoint ($adj A$):**
    $$ adj A = \begin{bmatrix} 4 & -2 \\\\ -3 & 1 \end{bmatrix} $$
*   **Inverse ($A^{-1}$):**
    $$ A^{-1} = \frac{1}{|A|} adj A = \frac{1}{-2} \begin{bmatrix} 4 & -2 \\\\ -3 & 1 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\\\ 1.5 & -0.5 \end{bmatrix} $$

### Q: Compute Matrix Multiplication $C = A \cdot B$
$$ A = \begin{bmatrix} 1 & 2 \\\\ 3 & 4 \end{bmatrix}, B = \begin{bmatrix} 5 & 6 \\\\ 7 & 8 \end{bmatrix} $$
**Solution:**
$$ C = \begin{bmatrix} (1 \cdot 5 + 2 \cdot 7) & (1 \cdot 6 + 2 \cdot 8) \\\\ (3 \cdot 5 + 4 \cdot 7) & (3 \cdot 6 + 4 \cdot 8) \end{bmatrix} = \begin{bmatrix} (5 + 14) & (6 + 16) \\\\ (15 + 28) & (18 + 32) \end{bmatrix} = \mathbf{\begin{bmatrix} 19 & 22 \\\\ 43 & 50 \end{bmatrix}} $$

### Q: Calculate Max Pooling (2x2 filter, Stride 2)
**Input Matrix:**
$$ \begin{bmatrix} 12 & 20 & 30 & 0 \\\\ 8 & 12 & 2 & 0 \\\\ 34 & 70 & 37 & 4 \\\\ 112 & 100 & 25 & 12 \end{bmatrix} $$
**Solution:**
Divide the matrix into 2x2 blocks:
1.  Top-Left: $\max(12, 20, 8, 12) = \mathbf{20}$
2.  Top-Right: $\max(30, 0, 2, 0) = \mathbf{30}$
3.  Bottom-Left: $\max(34, 70, 112, 100) = \mathbf{112}$
4.  Bottom-Right: $\max(37, 4, 25, 12) = \mathbf{37}$
**Output:** $\begin{bmatrix} 20 & 30 \\\\ 112 & 37 \end{bmatrix}$

### Q: Calculate CNN Output Size
**Parameters:** Input $I = 32 \times 32$, Filter $F = 5 \times 5$, Stride $S = 1$, Padding $P = 0$.
**Formula:** $O = \lfloor \frac{I - F + 2P}{S} \rfloor + 1$
**Solution:**
$O = \lfloor \frac{32 - 5 + 2(0)}{1} \rfloor + 1 = 27 + 1 = \mathbf{28 \times 28}$

### Q: Compute Mean Squared Error (MSE)
**Predicted:** $\hat{y} = [2.5, 0.0, 2.1]$, **Actual:** $y = [3.0, -0.5, 2.0]$
**Formula:** $MSE = \frac{1}{n} \sum (\hat{y}_i - y_i)^2$
**Solution:**
1.  $(2.5 - 3.0)^2 = (-0.5)^2 = 0.25$
2.  $(0.0 - (-0.5))^2 = (0.5)^2 = 0.25$
3.  $(2.1 - 2.0)^2 = (0.1)^2 = 0.01$
$MSE = \frac{0.25 + 0.25 + 0.01}{3} = \frac{0.51}{3} = \mathbf{0.17}$

### Q: Softmax Calculation for vector $z = [1, 2, 3]$
**Formula:** $\sigma(z)_i = \frac{e^{z_i}}{\sum e^{z_j}}$
**Solution:**
1.  $e^1 \approx 2.718, e^2 \approx 7.389, e^3 \approx 20.086$
2.  Sum $= 2.718 + 7.389 + 20.086 = 30.193$
3.  $\sigma(z)_1 = 2.718 / 30.193 \approx \mathbf{0.09}$
4.  $\sigma(z)_2 = 7.389 / 30.193 \approx \mathbf{0.24}$
5.  $\sigma(z)_3 = 20.086 / 30.193 \approx \mathbf{0.67}$
**Result:** $[0.09, 0.24, 0.67]$ (Sums to 1.0)

### Q: Compute Gradient Descent Step
**Initial Weight:** $w = 2.0$, **Learning Rate:** $\eta = 0.1$, **Loss Function Gradient:** $\frac{\partial J}{\partial w} = 4w$.
**Formula:** $w_{next} = w - \eta \frac{\partial J}{\partial w}$
**Solution:**
1.  Calculate gradient at $w=2.0$: $4(2.0) = \mathbf{8.0}$
2.  Update weight: $w = 2.0 - (0.1 \times 8.0) = 2.0 - 0.8 = \mathbf{1.2}$

### Q: Activation Function Numerical
**Input:** $x = -2.5, y = 1.0$
1.  **ReLU Output:** $\max(0, -2.5) = \mathbf{0}$
2.  **Sigmoid Output (for y):** $\frac{1}{1 + e^{-1.0}} = \frac{1}{1 + 0.367} = \frac{1}{1.367} \approx \mathbf{0.73}$

### Q: Matrix Trace Calculation
**Matrix A:**
$$ \begin{bmatrix} 5 & 2 & 1 \\\\ 3 & 10 & 4 \\\\ 1 & 8 & -2 \end{bmatrix} $$
**Formula:** $Tr(A) = \sum A_{ii}$ (Sum of diagonal elements)
**Solution:**
$Tr(A) = 5 + 10 + (-2) = \mathbf{13}$

### Q: Chain Rule (Numerical Example)
**Suppose:** $z = 3x^2$ and $y = \sin(z)$. Find $\frac{dy}{dx}$ at $x = 1$.
**Formula:** $\frac{dy}{dx} = \frac{dy}{dz} \times \frac{dz}{dx}$
**Solution:**
1.  $\frac{dz}{dx} = 6x$. At $x=1$, $\frac{dz}{dx} = \mathbf{6}$.
2.  $\frac{dy}{dz} = \cos(z)$. At $x=1$, $z = 3(1)^2 = 3$. So $\frac{dy}{dz} = \cos(3)$.
3.  $\frac{dy}{dx} = \cos(3) \times 6 \approx -0.99 \times 6 \approx \mathbf{-5.94}$

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

### Q: Perceptron Update Rule
**Inputs:** x = [1, 0, 1], **Weights:** w = [0.2, -0.5, 0.4], **Bias:** b = 0.1, **Target:** t = 1, **Learning Rate:** η = 0.5.
**Step 1: Calculate Net Input (z)**
z = (1 * 0.2) + (0 * -0.5) + (1 * 0.4) + 0.1 = 0.2 + 0 + 0.4 + 0.1 = **0.7**
**Step 2: Activation (Threshold at 0)**
Since z > 0, Output y_hat = **1**.
**Step 3: Update Weights**
Since y_hat = t, no update is needed (w_new = w_old). If y_hat was 0, then:
Δw = η(t - y_hat)x = 0.5(1 - 0)[1, 0, 1] = [0.5, 0, 0.5].

### Q: Kullback-Leibler (KL) Divergence
**Given P:** [0.1, 0.9], **Given Q:** [0.5, 0.5]
**Formula:** D_KL(P||Q) = Σ P(i) log(P(i)/Q(i)) (using log base 2)
**Solution:**
1.  Term 1: 0.1 * log2(0.1 / 0.5) = 0.1 * log2(0.2) = 0.1 * (-2.32) ≈ -0.232
2.  Term 2: 0.9 * log2(0.9 / 0.5) = 0.9 * log2(1.8) = 0.9 * (0.848) ≈ 0.763
3.  D_KL = -0.232 + 0.763 = **0.531**

### Q: AdaGrad Optimization Step
**Current Weight:** w = 1.0, **Accumulated Square Gradient:** G = 0.1, **Current Gradient:** g = 0.5, **Learning Rate:** η = 0.01.
**Formula:** w = w - [η / sqrt(G + ε)] * g (Assume ε = 10^-8)
**Solution:**
1.  Update G: G_new = G + g^2 = 0.1 + (0.5)^2 = 0.1 + 0.25 = **0.35**
2.  Update w: w = 1.0 - [0.01 / sqrt(0.35)] * 0.5 = 1.0 - [0.005 / 0.5916] ≈ 1.0 - 0.00845 = **0.99155**

### Q: Information Gain / Entropy
**Data:** 4 samples (2 Positive, 2 Negative). **Split:** Feature A splits into [Pos, Pos] and [Neg, Neg].
**Step 1: Parent Entropy**
H(S) = -(0.5 * log2(0.5) + 0.5 * log2(0.5)) = -(-0.5 - 0.5) = **1.0**.
**Step 2: Child Entropy**
Both children are pure (H = 0).
**Step 3: Information Gain**
IG = 1.0 - (0.5 * 0 + 0.5 * 0) = **1.0** (Perfect split).
