# 📄 Advanced Deep Learning (PPSITMAJM402) - Complete Comprehensive Solutions: SET 1

This document provides high-scoring, 6-mark level answers for **EVERY** sub-question (A, B, P, and Q) in the PPSITMAJM402 dataset. 

---

## 🏛️ Q. 1: Linear Algebra & Numerical Optimization

### **A: Definitions, properties, and applications of scalars, vectors, and matrices.**
*   **Scalar ($s$):** A 0-dimensional tensor (point). 
    *   **Property:** Commutative ($a \times b = b \times a$).
    *   **Application:** Represents learning rates, loss values, or hyperparameters.
    *   **Example:** $\eta = 0.001$.
*   **Vector ($\mathbf{x}$):** A 1-dimensional array of numbers representing magnitude and direction.
    *   **Property:** Member of an $n$-dimensional space $\mathbb{R}^n$.
    *   **Application:** Represents a single feature vector or a bias term in a layer.
    *   **Example:** $\mathbf{x} = [2.1, -1.0, 0.5]^T$.
*   **Matrix ($\mathbf{A}$):** A 2-dimensional array of numbers.
    *   **Property:** Represents a linear transformation. Addition is commutative, but multiplication is **non-commutative** ($\mathbf{AB} \neq \mathbf{BA}$).
    *   **Application:** Represents the weight parameters ($\mathbf{W}$) in a neural network layer.
    *   **Example:** 

$$
\mathbf{A} = \begin{bmatrix} 1 & 3 \\ 2 & 4 \end{bmatrix}

$$ .

### **B: Identity and inverse matrices with a calculation example.**
*   **Identity ($\mathbf{I}$):** A square matrix with 1s on the diagonal and 0s elsewhere. It acts as the multiplicative identity.
*   **Inverse ($\mathbf{A}^{-1}$):** A matrix such that $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$. Only exists for non-singular square matrices ($|A| \neq 0$).
*   **Example Case:** Let 

$$
\mathbf{A} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}

$$ . 
    1.  **Determinant ($|A|$):** $(1 \times 4) - (2 \times 3) = -2$. 
    2.  **Adjugate:** Swap diagonal, negate others 

$$
\rightarrow \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix}

$$ .
    3.  **Inverse:** 

$$
\frac{1}{-2} \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix}

$$ .

### **P: Define Minimum, Maximum, and Saddle point.**
*   **Local Minimum:** A point where the function value is lower than its immediate neighbors. In deep learning, we seek this to minimize loss.
*   **Local Maximum:** A point where the function value is higher than its immediate neighbors. 
*   **Saddle Point:**
    *   **Definition:** A critical point where the gradient is zero, but it is a minimum along one direction and a maximum along another.
    *   **Significance:** In high-dimensional optimization, saddle points are more common than local minima and cause the gradient descent to slow down or stall.

### **Q: Write a note on Poor Conditioning.**
*   **Definition:** A system is poorly conditioned if small changes in the input cause massive, unstable changes in the output.
*   **Metric:** Measured by the **Condition Number** $\kappa = \frac{\lambda_{max}}{\lambda_{min}}$.
*   **Impact on DL:**
    1.  **Inversion Error:** Matrix inversion becomes numerically inaccurate.
    2.  **Optimization:** Gradient descent becomes highly sensitive to the learning rate. It causes "zig-zagging" behavior in the loss surface, making convergence extremely difficult.
*   **Solution:** Use techniques like Batch Normalization or second-order optimizers.

---

## 🏛️ Q. 2: Training & Regularization

### **A: Discuss in detail learning the XOR function.**
*   **The Conflict:** XOR is not "linearly separable." A single-layer network cannot draw a line to separate $[0,1], [1,0]$ (Positive) from $[0,0], [1,1]$ (Negative).
*   **The Hidden Layer:** To solve XOR, we must add a hidden layer with at least 2 neurons and a non-linear activation function (like ReLU).
*   **Workflow:** The input is mapped to a hidden space where the points are transformed. In this new space, the final output neuron can easily draw a single linear decision boundary to classify the classes correctly.

### **B: Compare and contrast L2 regularization with L1 regularization.**
| Feature | L1 (Lasso) | L2 (Ridge) |
| :--- | :--- | :--- |
| **Penalty** | Adds sum of absolute weights ($\alpha \|w\|_1$) | Adds sum of squared weights ($\alpha \|w\|_2^2$) |
| **Weight Effect** | Shrinks coefficients exactly to zero. | Shrinks coefficients towards zero, but never zero. |
| **Benefit** | Automatically performs **Feature Selection**. | Prevents any single weight from becoming too large. |
| **Sparsity** | High (Produces Sparse Models) | Low (All weights remain, just smaller) |

### **P: Explain the bagging technique and contrast with other ensemble methods.**
*   **Bagging:** Training multiple versions of the same model on different random subsets (Bootstrap samples) of the data and averaging results. Reduces **Variance**.
*   **Ensemble vs. Individual:** Individual models often overfit the training data. Ensembles average out these "noises."
*   **Contrast:**
    *   **Boosting:** Models are trained sequentially (one after another) to correct the mistakes of previous models. (e.g., AdaBoost). Reduces **Bias**.
    *   **Stacking:** Uses a meta-learner to combine different types of models (e.g., SVM + CNN).

### **Q: Define plateaus and saddle points and their difference from local extrema.**
*   **Plateau:** A flat region on the loss surface where the gradient remains near zero for a long time. Leads to slow learning.
*   **Saddle Point:** A point where the gradient is zero but it isn't a maximum or minimum (it's "mountain pass" shape).
*   **Difference:** Unlike **Minima** (downward in all axes) or **Maxima** (upward in all axes), **Saddle Points** have gradients that point up in some axes and down in others. In deep learning, saddle points are the main cause of optimization "stalls."

---

## 🏛️ Q. 3: CNNs & Sequential Data

### **A: Explain the mechanism of a convolutional neural network (CNN).**
*   **Convolutional Layer:** Uses kernels/filters that slide over the image to detect features (edges, textures). Uses **Weight Sharing** to reduce parameters.
*   **Non-Linearity (ReLU):** Applies activation to introduce complexity.
*   **Pooling Layer:** Downsamples the map (Max/Avg) to provide **Translation Invariance** (still finds the object if it moves) and reduces computational load.
*   **Flattening & FC:** The last layers convert the 2D maps into 1D vectors for final classification.

### **B: Explain different types of data types in DL.**
1.  **Numerical (Continuous):** Standard pixel values or sensor data.
2.  **Categorical (Nominal):** Classes like "Cat", "Dog" (usually processed via **One-Hot Encoding**).
3.  **Sequential:** Data where order matters (Text, Time-series).
4.  **Spatial:** Grid-based data like Images or Videos.
5.  **Multi-modal:** Combining text, audio, and visual data in a single model.

### **P: What is sequence modelling? State its applications.**
*   **Sequence Modelling:** Modelling data where the position of an element depends on previous elements.
*   **Applications:**
    1.  **Speech Recognition:** Converting audio sequences to text.
    2.  **Machine Translation:** Translating sentences (sequence of words).
    3.  **Sentiment Analysis:** Understanding context in sentences.
    4.  **Genomics:** Analyzing DNA base-pair sequences.

### **Q: How is unfolding used in training recurrent neural networks (RNNs)?**
*   **Unfolding:** Mapping a recurrent network (which has a loop) into a computational graph with $T$ layers, where each layer represents a time step.
*   **Mechanism:** The shared weights $W$ are replicated across all time steps.
*   **Significance:** It allows us to apply standard **Backpropagation** to a recurrent model. This specific algorithm is called **Backpropagation Through Time (BPTT)**.

---

## 🏛️ Q. 4: Factor Models & Latent Variables

### **A: Compare and contrast Probabilistic PCA and Factor Analysis.**
*   **Similarity:** Both assume that observed data $x$ comes from hidden latent variables $z$ via a linear transform: $x = Wz + \mu + \epsilon$.
*   **Difference:**
    *   **PPCA:** Assumes the noise $\epsilon$ is "isotropic" (same variance in all directions: $\sigma^2 I$).
    *   **Factor Analysis:** Assumes each observed variable can have its own independent noise variance. This makes Factor Analysis better at modeling unique noise for separate features.

### **B: Discuss the role of Denoising Autoencoders.**
*   **Role:** To force the autoencoder to learn "robust" features rather than just memorizing input.
*   **Mechanism:** It adds noise (e.g., Gaussian) to the input. The network must attempt to output the **un-corrupted** version.
*   **Impact:** This prevents the network from learning the identity function and ensures it discovers the true high-level manifold of the data.

### **P: Explain how approximate inference works in machine learning.**
*   **The Problem:** In complex models, calculating the exact probability $P(h|x)$ is "intractable" because the math is too hard for computers to solve exactly.
*   **Working:** We pick a simpler distribution $q$ (like a Gaussian) and adjust its parameters until it looks as much like our true (hard) distribution $P$ as possible.
*   **Key Metric:** We minimize the **Kullback-Leibler (KL) Divergence** between $q$ and $P$.

### **Q: Compare Boltzmann Machines and Restricted Boltzmann Machines.**
*   **Boltzmann Machine:** A general undirected stochastic model where *any* node can connect to *any* other node. Too complex to train at scale.
*   **RBM:** A "Restricted" version that allows connections only between the **Visible** layer and the **Hidden** layer. Interactions within a layer are forbidden. This restriction makes training efficient via **Contrastive Divergence**.

---

## 🏛️ Q. 5: Advanced Topics

### **A: Write a short note on "Eigen Decompositions."**
*   **Definition:** Factoring a square matrix $\mathbf{A}$ into its eigenvalues ($\lambda$) and eigenvectors ($\mathbf{v}$).
*   **Math:** $\mathbf{Av} = \lambda\mathbf{v}$.
*   **Significance:** It reveals the "natural coordinates" of a matrix. In Deep Learning, it is essential for PCA and for analyzing the stability of gradients in deep networks.

### **B: Explain Back-propagation and compare with other differentiation algorithms.**
*   **Backpropagation:** Efficiently calculates gradients by starting at the output and moving backward using the **Chain Rule**.
*   **Vs. Finite Difference:** Finite difference perturbs each weight one by one and measures change. It is extremely slow. **Vs. Forward-Mode AD:** Forward-mode is efficient only when there are few inputs and many outputs. Backprop is best for Neural Networks because it calculates all weight gradients in one backward pass.

### **P: Write a note on large-scale deep learning.**
*   **Complexity:** Dealing with hundreds of layers and billions of parameters (e.g., GPT models).
*   **Enabling Technologies:**
    1.  **Distributed Training:** Using clusters of hundreds of GPUs.
    2.  **Data Parallelism:** Processing different "batches" of data on different GPUs simultaneously.
    3.  **Quantization:** Reducing weight precision (from 32-bit to 8-bit) to save memory and speed up computation.

### **Q: Explain Deep Belief Networks with its working.**
*   **Structure:** A stack of multiple Restricted Boltzmann Machines (RBMs).
*   **Working (Greedy Learning):**
    1.  Train the bottom RBM on raw data.
    2.  Use the hidden states of RBM 1 as "input data" for the next RBM.
    3.  Repeat for all layers.
    4.  Finally, perform a "Supervised Fine-tuning" pass on the whole stack to optimize for a specific task like classification.
