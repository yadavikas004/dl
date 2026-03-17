# 📄 Advanced Deep Learning (PSIT4P3a) - 6-Mark Comprehensive Solutions: QP4

This document covers the fourth and final question paper (QP4). It uses high-detail explanations and professional formatting to ensure you are prepared for the M.Sc. IT (AI Track) exam.

---

## 🏛️ Section I: Foundations & Numerical Issues

### **a. What is Deep Learning? Explain scalar and vector in detail.**
*   **Deep Learning:** A subfield of Machine Learning based on Artificial Neural Networks with multiple layers (reasons for "Deep"). It focuses on learning hierarchical representations of data automatically, moving from simple features in early layers to abstract concepts in deeper layers.
*   **Scalar ($x$):**
    *   A single numerical value representing magnitude. It is a 0-dimensional tensor.
    *   Example: $y = 3.14$, where $y \in \mathbb{R}$.
*   **Vector ($\mathbf{x}$):**
    *   A 1-dimensional array of numbers. It represents both magnitude and direction.
    *   Mathematical Form: $\mathbf{x} = [x_1, x_2, ..., x_n]^T$.
    *   It lives in an $n$-dimensional space $\mathbb{R}^n$.
    *   Example: A 2D velocity vector $\mathbf{v} = [2, -5]^T$.

### **b. Write a Note on Multiplying Matrices and Vectors with example.**
*   **Concept:** Multiplying a matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ by a vector $\mathbf{x} \in \mathbb{R}^n$ results in a new vector $\mathbf{y} \in \mathbb{R}^m$.
*   **Operation:** The $i^{th}$ element of $\mathbf{y}$ is the dot product of the $i^{th}$ row of $\mathbf{A}$ with the vector $\mathbf{x}$.
*   **Example:**
    Let $\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $\mathbf{x} = \begin{bmatrix} 5 \\ 6 \end{bmatrix}$.
    $\mathbf{y} = \mathbf{A}\mathbf{x} = \begin{bmatrix} (1 \times 5) + (2 \times 6) \\ (3 \times 5) + (4 \times 6) \end{bmatrix} = \begin{bmatrix} 17 \\ 39 \end{bmatrix}$.
*   **Significance:** In Deep Learning, every layer transformation is essentially $\mathbf{y} = \sigma(\mathbf{W}\mathbf{x} + \mathbf{b})$.

### **c. Compare Overflow and Underflow.**
*   **The Problem:** Computers use a finite number of bits to represent real numbers, leading to precision errors.
*   **Underflow:** Occurs when a number is so close to zero that it is rounded to zero. Since we often divide by these numbers or take their logs, underflow can cause "Division by Zero" or `-inf` errors.
*   **Overflow:** Occurs when a number is so large it exceeds the maximum representable limit (evaluates to `inf`). Recursive multiplications in deep networks often lead to this.
*   **Deep Learning Fix:** Use the **log-sum-exp** trick for Softmax and Cross-Entropy calculation to maintain numerical stability.

### **d. What is a need of Gradient Optimization? Explain in detail.**
*   Neural networks are essentially high-dimensional mathematical functions with millions of parameters ($W$).
*   **Why we need it:** We cannot find the optimal $W$ using simple algebra because the Loss Function $J(W)$ is a non-convex, extremely complex landscape.
*   **The Process:** Gradient Descent "walks" down the hill. By calculating the gradient $\nabla J(W)$ (the slope), we know which direction makes the error go up. We move in the **opposite direction** to reach the bottom (the minimum error).
*   **Importance:** Without it, training a multi-layer network would be impossible as there is no other way to coordinate the updates of millions of interdependent weights.

---

## 🏛️ Section II: Regularization & Advanced Methods

### **a. What is Regularization? Explain Underfitting and Overfitting.**
*   **Regularization:** Any technique intended to reduce the generalization error (test error) of a model, even if it slightly increases the training error.
*   **Overfitting:** The model learns the noise and details of the training data too well. Result: High training accuracy, but very low test accuracy.
*   **Underfitting:** The model is too simple to learn the underlying pattern. Result: Poor performance on both training and test data.
*   **Regularization Role:** It adds a "penalty" for complexity (like L2 norm), forcing the model to stay simple and generalize better.

### **b. Write a note on Multi-Task learning.**
*   **Definition:** A subfield of machine learning in which multiple learning tasks are solved at the same time, while exploiting commonalities and differences across tasks.
*   **Mechanism:** Multiple output "heads" are attached to a shared hidden representation. 
*   **Internal Regularization:** Task A acts as a regularizer for Task B. Since the model must perform well on both, it is less likely to overfit the noise of just one task.
*   **Example:** A car vision system that simultaneously detects "Pedestrians," "Traffic Lights," and "Lane Markings" using the same shared backbone CNN.

### **c. Define: Tangent Distance, Tangent Prop, and Manifold Tangent Classifier.**
These are techniques to make models **invariant** to specific local transformations:
1.  **Tangent Distance:** A distance metric that is invariant to small transformations (like rotation/scaling) by measuring the distance between "tangent planes" of the data manifolds.
2.  **Tangent Prop:** A regularization technique that adds a penalty to the loss function to ensure the model's output doesn't change when the input is transformed along a known "tangent vector."
3.  **Manifold Tangent Classifier:** Uses an autoencoder to learn the manifold structure first, then uses the learned "tangents" to train a classifier that is robust to variations along those tangent directions.

---

## 🏛️ Section III: Convolution & Networks

### **a. What is Pooling? What are the different types of Pooling?**
*   **Definition:** A downsampling operation in CNNs that reduces the spatial dimensions (Height/Width) of the feature maps.
*   **Purpose:** 1. Reduces computational load. 2. Provides "Translation Invariance" (the feature is detected even if it shifts slightly). 3. Prevents overfitting.
*   **Types:**
    1.  **Max Pooling:** Picks the maximum value from a window (e.g., 2x2). Most common as it captures the most "prominent" features.
    2.  **Average Pooling:** Takes the average of the window. Leads to smoother features but can wash out details.
    3.  **Global Pooling:** Reduces the entire feature map to a single value.

### **b. What are the variants of the basic Convolution Function?**
1.  **Strided Convolution:** Instead of moving 1 pixel at a time, the filter jumps $S$ pixels. Reduces output size.
2.  **Dilated (Atrous) Convolution:** Introduces spaces (zeros) between filter elements to increase the "Receptive Field" without adding parameters. Used in segmentation.
3.  **Transposed Convolution:** Used to "upscale" images (e.g., in GANs or Autoencoders).
4.  **Depthwise Separable Convolution:** Splits standard convolution into two steps to save 90% of the computation (used in MobileNet).

### **c. What is Recurrent Neural Network? Explain in detail.**
*   **Architecture:** Unlike Feedforward nets, RNNs have a **loop**. The output of a neuron at time $t$ is fed back alongside the input at time $t+1$.
*   **Memory:** The "Hidden State" $h_t$ acts as the network's memory, storing information about previous inputs in the sequence.
*   **Equation:** $h_t = \sigma(W_{hh}h_{t-1} + W_{xh}x_t + b)$.
*   **BPTT:** Trained using Backpropagation Through Time by "unfolding" the loop across timestamps.

---

## 🏛️ Section IV: ICA, Coding & Representation

### **a. Explain Independent Component Analysis (ICA).**
*   **Goal:** To separate a multivariate signal into additive subcomponents that are maximally independent and **non-Gaussian**.
*   **The Cocktail Party Problem:** If two people speak simultaneously and two microphones record them, ICA can recover the two original, distinct voices.
*   **Difference from PCA:** PCA looks for orthogonal directions of max variance (uncorrelated). ICA looks for directions that are statistically independent. ICA is better at finding the "source" signals.

### **b. What is sparse coding? What are the disadvantages?**
*   **Definition:** Overcomplete representation where data is modeled as a linear combination of a small number of "atoms" from a large dictionary. Most coefficients are **zero**.
*   **Objective:** To find the most efficient features for representing the data.
*   **Disadvantages:**
    1.  **Computational Cost:** Inferring the sparse code for a new input requires solving an optimization problem (slow).
    2.  **Non-differentiable:** The standard L1 penalty is hard to optimize with standard backprop.

### **c. How do we obtain Undercomplete & Denoising Autoencoders?**
1.  **Undercomplete:** Set the number of neurons in the hidden (bottleneck) layer to be **less** than the input layer. This forces the model to compress information.
2.  **Denoising:** 
    *   Take input $X$.
    *   Add noise to get $\tilde{X}$ (stochastic corruption).
    *   Train the autoencoder to map $\tilde{X} \rightarrow X$ (the original clean data).

---

## 🏛️ Section V: Inference & Boltzmann Machines

### **a. Write a note on Inference and Approximate Inference.**
*   **Inference:** The process of using a trained model to calculate the probability of hidden variables given the observed data $P(h|x)$.
*   **Exact Inference:** Possible in simple models but "Intractable" in deep models because calculating the denominator (the Evidence) involves an infinite integral.
*   **Approximate Inference:**
    1.  **Variational Inference (VI):** Converts the problem into optimization (finding a simple $q$ that looks like $P$).
    2.  **Monte Carlo (MCMC):** Uses random sampling to estimate the probability.

### **b. Explain Maximum a Posteriori (MAP) algorithm.**
*   **Goal:** To find the most likely value of parameters $\theta$ given the data $D$.
*   **Formula:** $\theta_{MAP} = \text{argmax}_\theta P(D|\theta)P(\theta)$.
*   **Mechanism:** It combines the **Likelihood** (how well the model fits the data) with a **Prior** (our existing knowledge about $\theta$).
*   **Difference from MLE:** MLE only considers data. MAP adds the "Prior," which acts as a regularizer, preventing extreme parameter values.

### **c. Define Boltzmann Machines and explain types.**
*   **Definition:** A stochastic, undirected energy-based neural network.
*   **Mechanism:** It learns the internal representation of data by reaching a state of thermal equilibrium (minimized energy).
*   **Types:**
    1.  **General Boltzmann Machine:** Any node can connect to any other node. (Very hard to train).
    2.  **Restricted Boltzmann Machine (RBM):** Bipartite graph—no connections within a layer. 
    3.  **Deep Boltzmann Machine (DBM):** A stack of multi-layer RBMs with undirected connections throughout.

### **d. Advantages and Disadvantages of RBM.**
*   **Advantages:**
    1.  Great for **Unsupervised Feature Learning**.
    2.  Provides a fast algorithm for "Greedy Layer-wise Pre-training."
    3.  Excellent for Collaborative Filtering (Recommendations).
*   **Disadvantages:**
    1.  The "Partition Function" is intractable, making it hard to evaluate the true likelihood.
    2.  Training relies on Gibbs Sampling (Contrastive Divergence), which can be slow to converge.
    3.  Hard to scale to very high-resolution images compared to CNNs.
