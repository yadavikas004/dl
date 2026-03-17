# 📄 Advanced Deep Learning (PPSCSMAJM401) - FULL & Detailed Model Solutions: QP3

This is the fully completed document for QP3, including all "OR" options (P and Q) for every section. I have added detailed mathematical notation for scalars, vectors, and matrices as requested, and filled in the missing answers for Section III (P and Q).

---

## 🏛️ Section I: Applied Math & Optimization

### **A. Differentiate between Scalar, Vector, and Matrix.**
*   **Scalar ($x$):**
    *   **Definition:** A single numerical value. It has 0 dimensions ($0^{th}$ order tensor).
    *   **Mathematical Notation:** Usually denoted by a lowercase, italicized letter like $x \in \mathbb{R}$.
    *   **Example:** Let $x = 5$.
*   **Vector ($\mathbf{x}$):**
    *   **Definition:** A one-dimensional array of numbers. It has both magnitude and direction ($1^{st}$ order tensor).
    *   **Mathematical Notation:** Usually denoted by a lowercase, boldface letter. An $n$-dimensional vector is $\mathbf{x} = [x_1, x_2, ..., x_n]^T \in \mathbb{R}^n$.
    *   **Example:** $\mathbf{x} = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$.
*   **Matrix ($\mathbf{A}$):**
    *   **Definition:** A two-dimensional array of numbers arranged in rows and columns ($2^{nd}$ order tensor).
    *   **Mathematical Notation:** Usually denoted by an uppercase, boldface letter. A matrix with $m$ rows and $n$ columns is $\mathbf{A} \in \mathbb{R}^{m \times n}$.
    *   **Example:** $\mathbf{A} = \begin{bmatrix} a_{11} & a_{12} \\ a_{21} & a_{22} \end{bmatrix}$.

**Comparison Table:**
| Feature | Scalar ($x$) | Vector ($\mathbf{x}$) | Matrix ($\mathbf{A}$) |
| :--- | :--- | :--- | :--- |
| **Rank/Order** | 0 | 1 | 2 |
| **Dimensions** | Point | Line ($n \times 1$) | Grid ($m \times n$) |
| **Elements** | Just $x$ | $x_i$ | $A_{i,j}$ |

*   **[Diagram Instruction]:** Sketch a dot for Scalar, a straight line with an arrow for Vector, and a 3x3 grid of cells for Matrix.

---

### **B. What is the need of Gradient Optimization? Explain in detail.**
*   **The Problem:** Deep neural networks have millions of learnable parameters ($W, b$). The loss function $J(W, b)$ is a complex, high-dimensional surface. We cannot analytically solve for the minimum because the equations are non-linear and coupled.
*   **The Solution:** We use an iterative approach. At each step, we calculate the gradient $\nabla J(\theta)$, which points in the direction of the steepest **increase**.
*   **Mechanism:** By moving in the **opposite direction** ($-\nabla J(\theta)$), we slowly descend the error hill towards the minimum.
*   **Update Rule:** $\theta_{t+1} = \theta_{t} - \eta \nabla_\theta J(\theta_t)$, where $\eta$ is the learning rate.
*   **Importance:** It allows the network to "learn" from its errors by adjusting weights in small steps to reduce total loss.

---

### **P. Explain Eigenvectors and Eigenvalues.**
*   **Definition:** For a square matrix $\mathbf{A}$, a non-zero vector $\mathbf{v}$ is an **eigenvector** if multiplying $\mathbf{A}$ by $\mathbf{v}$ only scales $\mathbf{v}$ by a factor $\lambda$ (the **eigenvalue**).
*   **Fundamental Equation:** $\mathbf{A}\mathbf{v} = \lambda\mathbf{v}$.
*   **Geometric Meaning:** The vector $\mathbf{v}$ does not change its direction under the transformation $\mathbf{A}$; only its length changes by factor $\lambda$.
*   **Application in DL:** Used in **Principal Component Analysis (PCA)** to find the directions of maximum variance in data and in understanding the stability of recurrent networks.

---

### **Q. Explain Matrix Product, Hadamard Product, and Dot Product.**
1.  **Matrix Product ($\mathbf{A} \mathbf{B}$):** 
    *   The standard way of multiplying matrices where the $(i,j)^{th}$ element is the dot product of the $i^{th}$ row of $\mathbf{A}$ and $j^{th}$ column of $\mathbf{B}$.
2.  **Hadamard Product ($\mathbf{A} \odot \mathbf{B}$):** 
    *   Element-wise multiplication. Matrices must have identical dimensions. $(\mathbf{A} \odot \mathbf{B})_{ij} = A_{ij} \times B_{ij}$.
3.  **Dot Product ($\mathbf{a} \cdot \mathbf{b}$):** 
    *   Sum of products of corresponding entries of two vectors. Result is a **scalar**. $\mathbf{a} \cdot \mathbf{b} = \sum a_i b_i$.

---

## 🏛️ Section II: Training & Regularization

### **A. Describe Deep feedforward network with its types.**
*   **Definition:** A neural network where information travels strictly from the input layer, through hidden layers, to the output layer with no feedback loops. 
*   **Objective:** To approximate some function $f^*(x)$.
*   **Types:**
    1.  **Multilayer Perceptron (MLP):** The most basic dense network.
    2.  **CNN:** Captures spatial dependencies using convolutional layers.
    3.  **Radial Basis Function (RBF) Networks:** Uses RBF kernels as activations.

### **B. Compare and contrast different optimization algorithms.**
*   **SGD:** Simple, noisy updates, sensitive to learning rate.
*   **Momentum:** Adds "speed" to updates; helps escape plateau and shallow minima.
*   **RMSProp:** Adapts learning rate by dividing the gradient by a running average of its magnitude.
*   **Adam:** The "Gold Standard"—combines Momentum and RMSProp properties.

---

### **P. Discuss Early Stopping, Dropout, and Batch Normalization.** [Repeat from QP2 Logic - Expanded]
*   **Early Stopping:** Monitoring validation error and stopping the training process before the model starts fitting noise (overfitting).
*   **Dropout:** Randomly deactivating a fraction of neurons ($p$) during each training step. It forces the network to learn redundant representations and prevents **co-adaptation**.
*   **Batch Normalization:** Re-centers and re-scales layer inputs. It stabilizes the distribution of inputs (Internal Covariate Shift), speeds up training, and acts as a minor regularizer.

---

### **Q. Define and Explain Data Augmentation.**
*   **Definition:** Artificially expanding a training dataset by applying transformations to existing data. 
*   **Logic:** It forces the model to be invariant to certain changes (e.g., a car is still a car even if flipped horizontally).
*   **Techniques:** Flipping, Rotation, Scaling, Cropping, and Color Shifting.

---

## 🏛️ Section III: CNN, RNN & Sequence Modelling [REVISED & COMPLETED]

### **A. Difference between CNN and RNN.**
*   **CNN (Convolutional Neural Network):** 
    *   Architecture: Convolutional layers $\rightarrow$ Pooling layers.
    *   Focus: Capturing **spatial** patterns (e.g., edges, shapes in images).
    *   Memory: Memoryless (treats each input independently).
*   **RNN (Recurrent Neural Network):**
    *   Architecture: Recurrent units with a hidden state $h_t$ that loops back.
    *   Focus: Capturing **temporal/sequential** patterns (e.g., words in a sentence).
    *   Memory: Has a "hidden state" that maintains information from previous time steps.

### **B. How Transfer Learning improves CNNs for Image Classification?**
*   **Mechanism:** We take a model pre-trained on a massive dataset (like ImageNet) and "transfer" its learned features to a smaller, niche dataset.
*   **Why it works:** Early layers of CNNs learn generic features (edges, corners) that are useful for *any* vision task. 
*   **Process:** 
    1.  Load pre-trained weights.
    2.  Freezing: Keep early layers fixed.
    3.  Replacing: Swap the final fully-connected (FC) layer with a new one matching the current classes.
    4.  Fine-tuning: Optionally train the whole network with a very small learning rate.

### **P. What is Sequence Modelling? What are different types of networks?**
*   **Sequence Modelling:** The task of learning patterns in data where the **order** matters (e.g., speech, text, time-series).
*   **Types of Networks:**
    1.  **Recurrent Neural Networks (RNNs):** Basic sequence handlers using hidden state loops.
    2.  **Long Short-Term Memory (LSTM):** Advanced RNNs with "Gates" to store long-term memory.
    3.  **Gated Recurrent Units (GRU):** Efficient version of LSTM with fewer gates.
    4.  **Transformers:** (Modern) Use "Self-Attention" to process entire sequences in parallel without recurrence.
*   **[Diagram Instruction]:** Draw a sequence of inputs $x_1, x_2, x_3$ feeding into a chain of connected blocks $h_1 \rightarrow h_2 \rightarrow h_3$.

### **Q. Write a note on computer vision.**
*   **Definition:** An interdisciplinary field that deals with how computers can gain high-level understanding from digital images or videos.
*   **Key Operations:**
    1.  **Convolution:** For feature extraction.
    2.  **Pooling:** For computational efficiency and translation invariance.
    3.  **Non-linearity (ReLU):** For learning complex patterns.
*   **Applications:** Autonomous driving, facial recognition, medical imaging (tumor detection), and object counting.

---

## 🏛️ Section IV: Latent Models & Inference

### **A. Linear Factor Models & Probabilistic PCA.**
*   **Linear Factor Model:** $x = Wh + \epsilon$ (mapping latent $h$ to observed $x$).
*   **PPCA:** Adds a probabilistic framework to PCA. It assumes the noise $\epsilon$ and the latent factors $h$ follow a Gaussian distribution. This allows using maximum likelihood estimation.

### **B. Significance of Denoising Autoencoders & Applications.**
*   **Significance:** Prevents the network from learning the trivial identity function. By trying to recover clean data from noisy inputs, the model is forced to learn the "Manifold" or the true structure of the data.
*   **Applications:** Image restoration, unsupervised feature pre-training, and anomaly detection.

### **P. What is Manifold Interpretation of PCA?**
*   **Concept:** PCA assumes that high-dimensional data actually lies on or near a lower-dimensional linear subspace (the manifold). 
*   **Goal:** PCA finds the "Principal Components" (hyperplane) that capture the maximum variance of the data, effectively finding the flat manifold that best represents the data points.

### **Q. Explain how approximate Inference works.**
*   **The Problem:** In deep probabilistic models, the **posterior probability** $P(h|x)$ is often intractable because the denominator (evidence) requires an impossible integral over all hidden configurations.
*   **The Solution (Approximate Inference):** We approximate the true posterior $P$ with a simpler target distribution $q$ (e.g., Gaussian).
*   **Method:** We optimize the parameters of $q$ to make it as similar as possible to $P$ (by minimizing the KL Divergence). This is known as **Variational Inference**.

---

## 🏛️ Section V: Formalisms & Sequential Logic

### **A. Short Note on Norms.**
*   Focus on the $L^1$ norm ($\sum |x_i|$) leading to **sparsity** and the $L^2$ norm ($\sqrt{\sum x_i^2}$) leading to **small weights**. State the 4 properties: Non-negativity, Identity, Triangle Inequality, and Scalability.

### **B. Mathematical formulation of SGD with Nesterov momentum.**
*   **Velocity:** $v_t = \gamma v_{t-1} - \eta \nabla J(\theta_{t-1} + \gamma v_{t-1})$
*   **Parameters:** $\theta_t = \theta_{t-1} + v_t$
*   **Key Idea:** We take the gradient at the "look-ahead" position rather than the current position.

### **P. How is unfolding used in training RNNs?**
*   **Concept:** To train an RNN using standard gradient descent, we "unfold" the recurrent connection across time steps.
*   **Result:** The network looks like a deep, finite directed acyclic graph where each "layer" represents a timestamp. This allows the use of **Backpropagation Through Time (BPTT)**.

### **Q. How to implement Slow Feature Analysis (SFA).**
*   SFA aims to extract features that vary slowly over time from rapidly changing pixel inputs.
*   **Steps:** 1. Normalize data $\rightarrow$ 2. Non-linear Polynomial expansion $\rightarrow$ 3. Solve Generalized Eigenvalue problem for the covariance of the derivatives.
