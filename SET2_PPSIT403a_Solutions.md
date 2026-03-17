# 📄 Advanced Deep Learning (PPSIT403a) - 6-Mark Comprehensive Solutions: SET 2

This document provides detailed model answers for the PPSIT403a Question Paper (AI Track). Each answer is structured to maximize marks in a 6-mark format.

---

## 🏛️ Q.I (Answer Any Two)

### **a: Explain Identity and Inverse matrix with an example.**
*   **Identity Matrix ($\mathbf{I}$):** A square matrix with 1s on the main diagonal and 0s elsewhere. It is the multiplicative identity in matrix algebra. $\mathbf{A} \cdot \mathbf{I} = \mathbf{A}$.
*   **Inverse Matrix ($\mathbf{A}^{-1}$):** A matrix such that $\mathbf{A} \cdot \mathbf{A}^{-1} = \mathbf{I}$. Only exists if $\mathbf{A}$ is square and $|A| \neq 0$.
*   **Example:** For 

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

, $|A| = -2$.
    $$ \mathbf{A}^{-1} = \frac{1}{-2} \begin{bmatrix} 4 & -2 \\\\ -3 & 1 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\\\ 1.5 & -0.5 \end{bmatrix} $$

### **b: Write a short note on Norms.**
*   **Definition:** A function that measures the magnitude/length of a vector.
*   **L1 Norm:** $\sum |x_i|$. Used to induce **Sparsity** (making parameters zero).
*   **L2 Norm:** $\sqrt{\sum x_i^2}$. Most common norm, used for **Weight Decay** to keep weights small.
*   **Properties:** Non-negativity, Triangle inequality ($||a+b|| \le ||a||+||b||$), and Homogeneity.

### **c: Write a note on Poor Conditioning.**
*   **Concept:** A matrix/system is poorly conditioned if small changes in the input lead to massive changes in the output.
*   **Condition Number:** $\kappa = \frac{|\lambda_{max}|}{|\lambda_{min}|}$. A high number indicates high sensitivity.
*   **Impact:** Causes numerical instability in matrix inversion and makes gradient descent training very slow and unstable (oscillations).

### **d: Define Local minimum with Minimum, Maximum, and Saddle point.**
*   **Local Minimum:** A point where the loss is lower than all surrounding points.
*   **Local Maximum:** A point where the loss is higher than all surrounding points.
*   **Saddle Point:** A point where the gradient is zero, but it is a minimum in one direction and a maximum in another. These are the primary obstacles in deep learning optimization.

---

## 🏛️ Q.II (Answer Any Two)

### **a: What is a Simple Deep Neural Network? Explain with Example.**
*   **Definition:** An artificial neural network with an input layer, an output layer, and **at least two** hidden layers.
*   **Structure:** Inputs $\rightarrow$ Hidden 1 (Edges) $\rightarrow$ Hidden 2 (Shapes) $\rightarrow$ Output (Classification).
*   **Example (OCR):** Input (Pixels) $\rightarrow$ Hidden 1 (128 units) $\rightarrow$ Hidden 2 (64 units) $\rightarrow$ Output (10 digits).

### **b: Write a short Note on bias-variance tradeoff.**
*   **Bias:** Error from simple assumptions (leads to **Underfitting**).
*   **Variance:** Error from high sensitivity to training noise (leads to **Overfitting**).
*   **The Tradeoff:** Simple models have high bias; complex models have high variance. Goal: Find total error minimum by balancing complexity.

### **c: Define and explain Data Augmentation.**
*   Creating synthetic training data by transforming existing images (flipping, rotating, scaling).
*   **Purpose:** To increase dataset size and force the model to be **Invariant** to orientation and scale, preventing overfitting.

### **d: Explain Momentum in detail.**
*   An extension to Gradient Descent that adds a fraction of the previous weight update to the current one.
*   **Formula:** $v \leftarrow \beta v + \eta \nabla$; $W \leftarrow W - v$.
*   **Benefit:** Helps the optimizer "roll through" local minima/saddle points and reduces oscillations in steep ravines.

---

## 🏛️ Q.III (Answer Any Two)

### **a: What is a convolutional neural network (CNN)?**
*   A specialized network for processing grid data (images).
*   Uses **Filters** to extract features, **Pooling** to reduce dimensions, and **Fully Connected** layers for final labels.
*   Features: Local connectivity and Weight sharing.

### **b: Write a note on Auto-completion.**
*   A sequence prediction task where the model predicts the next word/token based on prior context.
*   Uses LSTMs or Transformers to learn the probability $P(w_n | w_{1..n-1})$.

### **c: Explain different types of RNN.**
1.  **One-to-Many:** One input (image) $\rightarrow$ Sequence output (caption).
2.  **Many-to-One:** Sequence input (review) $\rightarrow$ Label (sentiment).
3.  **Many-to-Many:** Sequence input $\rightarrow$ Sequence output (translation).

### **d: Write a note on computer vision.**
*   Field enabling computers to "see" and interpret visual data.
*   Tasks: Classification, Object localization, and Semantic Segmentation. Uses CNNs as the core engine.

---

## 🏛️ Q.IV (Answer Any Two)

### **a: Explain the concept of Probabilistic PCA.**
*   A latent variable model that assumes observed data $x$ is generated from hidden factors $z$ plus isotropic Gaussian noise.
*   It provides a likelihood framework for PCA, enabling handling of missing data.

### **b: Write a short note on Slow Feature Analysis (SFA).**
*   Unsupervised learning technique that extracts features from a sequence that vary as slowly as possible over time while still carrying information. Useful for learning invariant features from video.

### **c: What is the significance of Denoising Autoencoders?**
*   Significance: Forces the model to learn the intrinsic structural properties of data (manifold) by reconstructing clean data from noisy input. It ensures a robust representation.

### **d: List applications of Autoencoders.**
1.  Dimensionality Reduction.
2.  Image Denoising.
3.  Anomaly Detection in high-dimensional logs.
4.  Feature Learning as pre-training.

---

## 🏛️ Q.V (Answer Any Two)

### **a: Define posterior probability in Bayesian statistics.**
*   The probability of a parameter $\theta$ given the observed data $D$.
*   Calculated using Bayes' Theorem: $P(\theta|D) = \frac{P(D|\theta)P(\theta)}{P(D)}$.

### **b: Compare Discriminative vs. Generative Modeling.**
*   **Discriminative:** $P(y|x)$ - Learns the decision boundary (e.g., Logistic Regression, SVM).
*   **Generative:** $P(x|y)$ or $P(x,y)$ - Learns data distribution (e.g., GANs, VAEs, Naive Bayes).

### **c: Write a short note on Boltzmann Machines.**
*   Energy-based stochastic networks with undirected connections.
*   Consists of visible units (data) and hidden units (latent features).
*   RBMs are the restricted version that allows efficient training.

### **d: Explain Deep Belief Networks with its Algorithm.**
*   **Architecture:** Stacked RBMs trained greedily layer-by-layer.
*   **Algorithm:** 
    1.  Train first RBM on data.
    2.  Use hidden states as input for next RBM.
    3.  Train all layers.
    4.  Supervised fine-tuning for classification.
