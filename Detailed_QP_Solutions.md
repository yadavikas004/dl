# Detailed Solutions: Question Papers 1 - 5 (Deep Learning)

This document provides structured answers to the questions found in your past papers (`QP1` to `QP5`).

---

## Unit I: Linear Algebra & Numerical Optimization

### Q: Differentiate between Matrix Product and Hadamard Product.
*   **Matrix Product:** The standard "row by column" multiplication. Result is a set of dot products. Notation: $A \cdot B$.
*   **Hadamard Product:** Element-wise multiplication. Matrices must have the same shape. Notation: $A \odot B$. Example: $[1, 2] \odot [3, 4] = [3, 8]$.

### Q: Explain Constraint Optimization (Lagrangian Method).
*   In deep learning, we often need to minimize a loss function $f(x)$ subject to constraints $g(x) = 0$.
*   **Approach:** We create the **Lagrangian function**: $L(x, \lambda) = f(x) + \lambda g(x)$.
*   By taking the partial derivative of $L$ with respect to $x$ and $\lambda$ and setting them to zero, we find the constrained minimum.

### Q: Why is Gradient Optimization needed?
*   Neural networks are "black boxes" with millions of weights. Analytical solutions (solving for $x$ directly) are impossible.
*   **Gradient Optimization** allows the model to "feel" the direction of the minimum error iteratively, updating weights to reduce loss in small steps.

---

## Unit II: Deep Networks & Training

### Q: What is Hyperparameter Tuning?
*   Parameters (like Weights) are learned by the model. **Hyperparameters** are settings you choose *before* training.
*   **Examples:** Learning rate, Batch size, Number of layers, Activation functions.
*   **Tuning:** The process of trying different combinations to find the one that gives the best accuracy.

### Q: Write a short note on Inception Networks.
*   **Concept:** Instead of choosing one filter size (3x3 or 5x5), Inception modules use *all* of them (1x1, 3x3, 5x5) in parallel.
*   **Benefit:** Allows the network to learn features at different scales simultaneously. Uses 1x1 convolutions to reduce computational cost (bottleneck layers).

### Q: Explain Vanishing and Exploding Gradients.
*   **Vanishing Gradient:** Moving backward through many layers, gradients become very small (close to 0), so earlier layers stop learning. (Common with Sigmoid activation).
*   **Exploding Gradient:** Gradients grow exponentially large, causing weights to oscillate or become NaN. (Common in RNNs).
*   **Solution:** Use ReLU activation, Batch Normalization, or Gradient Clipping.

---

## Unit III: CNN & Sequence Modeling

### Q: Explain the Sliding Window Approach for Object Localization.
*   **Mechanism:** A window of a fixed size slides over the entire image. Each "crop" is passed into a CNN to check if a specific object exists.
*   **Limitation:** Computationally very slow because it checks every possible position. 
*   **Modern fix:** Use "Region Proposal" or "YOLO" (You Only Look Once) architectures.

### Q: What is Transfer Learning in Image Classification?
*   Instead of training a model from scratch, you take a model already trained on a huge dataset (like ImageNet) and "fine-tune" it on your specific task.
*   **Benefit:** Saves time, hardware resources, and works well even with small datasets.

### Q: Compare Different Types of RNNs.
1.  **Simple RNN:** Suffers from short-term memory (vanishing gradient).
2.  **LSTM (Long Short-Term Memory):** Uses "Gates" (input, forget, output) to store information for long durations.
3.  **GRU (Gated Recurrent Unit):** A simpler version of LSTM with fewer gates, faster to train.

---

## Unit IV: Research & Advanced Deep Learning

### Q: Explain the "Manifold Interpretation" of PCA.
*   It assumes that even if data has thousands of dimensions (like pixels in an image), the actual meaningful variations lie on a much lower-dimensional "surface" or **Manifold**.
*   PCA attempts to find the linear directions that best approximate this manifold to reduce data complexity.

### Q: What is Slow Feature Analysis (SFA)?
*   **Principle:** In high-speed data (like video), meaningful objects change slowly (e.g., a person walking), while raw pixels change rapidly.
*   SFA extracts features that vary the slowest over time, effectively capturing the "essence" of the scene.

### Q: How to balance Exploration vs. Exploitation? (Reinforcement Learning)
*   **Exploitation:** Using what you already know to get a reward (e.g., picking the best-performing move).
*   **Exploration:** Trying new things to see if they might be better in the long run.
*   **Balance:** Often achieved using $\epsilon$-greedy strategy, where the agent explores with probability $\epsilon$ and exploits otherwise.

### Q: Brief note on Boltzmann Machines.
*   A type of "Energy-Based Model." Nodes are stochastic (probabilistic).
*   Unlike Feedforward networks, they are undirected.
*   **Problem:** Hard to train because every node is connected to every other node. RBMs solve this by removing connections within layers.
