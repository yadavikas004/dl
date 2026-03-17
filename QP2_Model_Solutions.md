# 📄 Advanced Deep Learning (PPSIT403a) - 6-Mark Comprehensive Solutions: QP2

This document provides high-scoring, detailed model answers for **EVERY** question in QP2. Each answer is expanded to meet the standard for a **6-mark university question**.

---

## 🏛️ Section I: Linear Algebra & Numerical Optimization

### **a. Explain Identity and Inverse matrix with an example.**
*   **Identity Matrix ($I$):**
    *   **Definition:** A square matrix in which all the elements of the principal diagonal are ones and all other elements are zeros. 
    *   **Significance:** It acts like the number '1' in matrix algebra. Multiplying any matrix $A$ by $I$ results in $A$ ($A \cdot I = I \cdot A = A$).
    *   **Example (2x2):** 

$$
I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}

$$ .
*   **Inverse Matrix ($A^{-1}$):**
    *   **Definition:** For a square matrix $A$, if there exists a matrix $B$ such that $A \cdot B = B \cdot A = I$, then $B$ is the inverse of $A$ ($A^{-1}$).
    *   **Condition:** A matrix has an inverse ONLY if it is square and its determinant is non-zero ($|A| \neq 0$). If $|A|=0$, it is "singular."
    *   **Formula:** $A^{-1} = \frac{1}{|A|} \text{adj}(A)$.
    *   **Example:** For 

$$
A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
$$

, $|A| = (1 \times 4) - (2 \times 3) = -2$. The inverse is 

$$
A^{-1} = \frac{1}{-2} \begin{bmatrix} 4 & -2 \\ -3 & 1 \end{bmatrix} = \begin{bmatrix} -2 & 1 \\ 1.5 & -0.5 \end{bmatrix}

$$ .

### **b. Write a short note on Norms.**
*   **Definition:** A norm is a function that maps a vector (or matrix) to a non-negative real value, representing its "size" or magnitude.
*   **Properties:** A valid norm must satisfy: 1. Non-negativity ($\|x\| \ge 0$), 2. Zero vector property ($\|x\|=0 \iff x=0$), 3. Triangle inequality ($\|x+y\| \le \|x\| + \|y\|$), and 4. Homogeneity ($\|\alpha x\| = |\alpha| \cdot \|x\|$).
*   **Common Norms in DL:**
    1.  **$L^1$ Norm (Manhattan):** Sum of absolute values ($||x||_1 = \sum |x_i|$). Used in Lasso regularization to induce **sparsity**.
    2.  **$L^2$ Norm (Euclidean):** Square root of the sum of squares ($||x||_2 = \sqrt{\sum x_i^2}$). Most common in DL for weight decay (L2 regularization).
    3.  **$L^\infty$ Norm (Max Norm):** Absolute value of the largest element.
    4.  **Frobenius Norm:** Used for matrices; it is the $L^2$ norm equivalent for matrix entries.
*   **Importance:** Norms are used as "penalty terms" in loss functions to prevent overfitting by keeping weights small.

### **c. Write a note on Poor Conditioning.**
*   **Definition:** A function or matrix is "poorly conditioned" if tiny changes in the input lead to massive, unstable changes in the output.
*   **Condition Number:** It is quantified by the **Condition Number** ($\kappa$), which for a matrix is the ratio of the largest to smallest eigenvalue: $\kappa(A) = \frac{|\lambda_{max}|}{|\lambda_{min}|}$.
*   **Impact on Deep Learning:**
    1.  **Numerical Instability:** Poor conditioning makes matrix inversion very difficult and prone to rounding errors.
    2.  **Gradient Descent Issues:** If the loss surface is poorly conditioned (e.g., very steep in one direction and very flat in another), standard gradient descent will "jitter" or oscillate wildly, making convergence extremely slow.
*   **Solution:** Second-order optimization methods (like Newton's method) or proper weight initialization and Batch Normalization help mitigate poor conditioning.

### **d. Define Local minimum with Minimum, Maximum and Saddle point.**
During optimization, we look for critical points where the gradient is zero ($\nabla f(x) = 0$):
1.  **Local Minimum:** A point where the function value is lower than all surrounding points. In DL, we aim for this to minimize error.
2.  **Global Minimum:** The absolute lowest point on the entire loss surface.
3.  **Local Maximum:** A point where the function value is higher than all surrounding points.
4.  **Saddle Point:**
    *   **Definition:** A point where the gradient is zero, but it is a minimum along one axis and a maximum along another.
    *   **DL Context:** In extremely high-dimensional spaces (like neural networks), true local minima are rare. **Saddle points are the biggest challenge** because the gradient becomes zero, causing training to stop or "stall" even though we haven't reached a minimum.
5.  **Visual Example:** A "U" shape (Minimum), an inverted "U" (Maximum), and a horse saddle shape (Saddle point).

---

## 🏛️ Section II: Training & Neural Network Basics

### **a. What is a Simple Deep Neural Network? Explain with Example.**
*   **Definition:** A neural network with an input layer, an output layer, and **at least two or more** hidden layers. "Deep" refers to the depth (number of layers).
*   **Components:**
    1.  **Input Layer:** Receives raw features (e.g., pixel values).
    2.  **Hidden Layers:** Perform intermediate feature extraction using non-linear activations (ReLU).
    3.  **Output Layer:** Provides the final prediction (Softmax/Sigmoid).
*   **Example (Digit Classifier):** 
    *   **Input:** 784 nodes (for 28x28 image).
    *   **Hidden 1:** 128 neurons (learns simple edges).
    *   **Hidden 2:** 64 neurons (learns shapes/loops).
    *   **Output:** 10 nodes (representing digits 0-9).
*   **Significance:** Multiple layers allow the network to learn a "hierarchy" of features, moving from simple edges in early layers to complex objects in later layers.

### **b. Write a short note on bias-variance tradeoff.**
*   **Bias:** Error due to overly simple assumptions. High bias leads to **Underfitting** (the model is too simple to capture the pattern).
*   **Variance:** Error due to high sensitivity to small fluctuations in training data. High variance leads to **Overfitting** (the model captures noise as if it were a pattern).
*   **The Tradeoff:**
    *   As model complexity increases, **Bias decreases** but **Variance increases**.
    *   The goal is to find the "Sweet Spot" where total error (Bias + Variance) is minimized.
*   **Solution:** Regularization, Dropout, and gathering more data are ways to find the optimal balance and ensure the model generalizes well to new data.

### **c. Define and Explain Data Augmentation.**
*   **Definition:** A technique used to artificially increase the size and diversity of a training dataset by creating modified versions of existing data.
*   **Why it is needed:** Deep learning models require massive amounts of data to generalize. If data is scarce, the model will overfit.
*   **Techniques (Computer Vision):**
    1.  **Flipping:** Mirroring images horizontally or vertically.
    2.  **Rotation:** Rotating images by small degrees (e.g., 10°).
    3.  **Scaling/Zooming:** Zooming in on specific parts of the image.
    4.  **Translation:** Shifting the object within the frame.
    5.  **Color Jitter:** Changing brightness, contrast, or saturation.
*   **Benefit:** It forces the model to learn features that are "invariant" to these changes (e.g., a cat is still a cat even if it's upside down).

### **d. Explain Momentum in details.**
*   **Concept:** Momentum is an extension to the Gradient Descent optimizer that helps accelerate training and dampens oscillations.
*   **Mathematical Form:**
    1.  $v_t = \gamma v_{t-1} + \eta \nabla_\theta J(\theta)$
    2.  $\theta = \theta - v_t$
    *(Where $v$ is velocity, $\gamma$ is momentum coefficient (usually 0.9), and $\eta$ is learning rate).*
*   **Analogy:** Imagine a ball rolling down a hill. It gains speed (momentum) as it goes down. Even if there is a small bump (local minimum/saddle point), its "velocity" helps it roll over it.
*   **Benefits:**
    1.  **Faster Convergence:** Accelerates updates in directions with consistent gradients.
    2.  **Oscillation Damping:** Cancels out noisy updates in directions where gradients change frequently.
    3.  **Saddle Point Escape:** Helps the optimizer push through flat regions where gradients are near zero.

---

## 🏛️ Section III: CNN, RNN & Sequence Modeling

### **a. What is a convolution neural network (CNN)?**
*   **Definition:** A specialized type of neural network designed to process grid-like topology data, primarily images.
*   **Three Main Layers:**
    1.  **Convolutional Layer:** Uses learnable filters (kernels) to extract spatial features. Crucial concepts: **Stride** (step size), **Padding** (bordering), and **Receptive Field**.
    2.  **Pooling Layer:** Downsamples the feature maps (e.g., Max Pooling) to reduce dimensionality and provide translation invariance.
    3.  **Fully Connected (FC) Layer:** Standard architecture at the end to perform the final classification.
*   **Weight Sharing:** The same filter is used across the entire image, which massively reduces the number of parameters compared to a standard dense network.

### **b. Write a note on Auto-completion.**
*   **Concept:** A feature that predicts the next word or character in a sequence based on the context provided by previous entries.
*   **Mechanism:** Uses **Language Models** trained on massive text corpora. It calculates $P(w_n | w_1, w_2, ... w_{n-1})$.
*   **Deep Learning Models:**
    1.  **RNNs/LSTMs:** Good for short context but struggle with long sequences.
    2.  **Transformers (GPT):** The modern standard. Uses "Attention" mechanisms to look at all previous words simultaneously to provide highly accurate suggestions.
*   **Usage:** Search engines (Google), Messaging apps (WhatsApp/Gmail), and Code editors (GitHub Copilot).

### **c. Explain different types of RNN.**
RNNs are classified based on the mapping of Inputs to Outputs:
1.  **One-to-One:** Standard vanilla neural network (e.g., image classification).
2.  **One-to-Many:** One input, many outputs (e.g., **Image Captioning**—input one image, output a sentence).
3.  **Many-to-One:** Sequence input, single output (e.g., **Sentiment Analysis**—input a sentence, output a 'positive/negative' label).
4.  **Many-to-Many (Synchronous):** Equal number of inputs and outputs (e.g., **POS Tagging**—labelling every word in a sentence as a noun/verb).
5.  **Many-to-Many (Asynchronous):** Different counts (e.g., **Machine Translation**—an encoder-decoder architecture where a sentence of 5 words is translated into 7 words).

### **d. Write a note on computer vision.**
*   **Definition:** A field of AI that enables computers and systems to derive meaningful information from digital images, videos, and other visual inputs.
*   **Core Tasks:**
    1.  **Image Classification:** What is in the image?
    2.  **Object Detection:** Where is the object? (Boundary boxes).
    3.  **Image Segmentation:** Assigning a class to every single pixel (useful in autonomous driving).
*   **The DL Impact:** Before DL, vision relied on "Hand-crafted" features (like SIFT/HOG). Now, **CNNs** automatically learn features, leading to super-human performance in tasks like face recognition and medical diagnosis.

---

## 🏛️ Section IV: Autoencoders & Latent Models

### **a. Explain the concept of Probabilistic PCA (PPCA).**
*   **PCA Definition:** A linear method for dimensionality reduction that finds orthogonal directions of maximum variance.
*   **The Probabilistic Twist:** PPCA defines PCA as a **Latent Variable Model**.
*   **Mechanism:** It assumes that the observed data $x$ is generated from a lower-dimensional latent variable $z$ through a linear transformation plus some Gaussian noise: $x = Wz + \mu + \epsilon$.
*   **Significance:** Unlike standard PCA, PPCA allows for handling missing data, provides a likelihood for the data, and can be used in a Bayesian framework.

### **b. Write a short note on Slow Feature Analysis (SFA).**
*   **Principle:** In rapid data streams (like video), meaningful structural features (e.g., a car) change very slowly, while raw pixel values change extremely fast.
*   **The Goal:** To extract features that vary as slowly as possible over time while still carrying information about the scene.
*   **Mathematical Insight:** It minimizes the temporal derivative (rate of change) of the learned features.
*   **Usage:** Used in Unsupervised learning to learn invariant representations (e.g., learning that a face is the same even if the person moves or lighting changes).

### **c. What is the significance of Denoising Autoencoders (DAE)?**
*   **The Motivation:** A standard autoencoder might just learn the "Identity Function" (copying input to output) without learning any useful patterns.
*   **Working Principle:** DAE is trained to reconstruct clean data from a "corrupted" version (input with added noise).
*   **Significance:** 
    1.  **Feature Robustness:** It forces the model to learn the stable structure of the data that persists despite noise.
    2.  **Non-Linear Manifold Learning:** It discovers the low-dimensional manifold where the data lives.
    3.  **Pre-training:** Often used to initialize weights of deep networks to provide better starting points.

### **d. List Applications of Autoencoders.**
1.  **Dimensionality Reduction:** Compressing high-dimensional data into a small code (similar to PCA but non-linear).
2.  **Image Denoising:** Removing salt-and-pepper noise or blur from photos.
3.  **Anomaly Detection:** If a model is trained on normal data, it will have a very high reconstruction error when given "abnormal" data (e.g., fraud tokens, engine failure logs).
4.  **Generative Modelling:** VAEs (Variational Autoencoders) can generate new data samples.
5.  **Feature Learning:** Pre-training for deeper classification models.

---

## 🏛️ Section V: Generative & Bayesian Modeling

### **a. Define posterior probability in Bayesian statistics.**
*   **Definition:** The probability of parameters (or hypothesis) $\theta$ **after** observing the data $D$.
*   **Bayes' Rule:** $P(\theta | D) = \frac{P(D | \theta) \cdot P(\theta)}{P(D)}$.
*   **Terms:**
    1.  **$P(\theta | D)$ (Posterior):** What we want to find.
    2.  **$P(D | \theta)$ (Likelihood):** Probability of data given parameters.
    3.  **$P(\theta)$ (Prior):** Our initial belief before seeing data.
    4.  **$P(D)$ (Evidence):** Normalizing constant.
*   **Importance:** Deep learning uses this for **MAP (Maximum A Posteriori)** estimation and in Variational Inference.

### **b. Compare Discriminative vs. Generative Modeling.**
| Feature | Discriminative Model | Generative Model |
| :--- | :--- | :--- |
| **Goal** | Learns the boundary between classes. | Learns the distribution of each class. |
| **Probability** | $P(y|x)$ (Conditional) | $P(x,y)$ or $P(x)$ (Joint) |
| **Core Task** | Classification / Regression. | Generating new data / Density estimation. |
| **Complexity** | Usually simpler and more accurate for labels. | Complex; requires understanding data origin. |
| **Examples** | Logistic Regression, SVM, Random Forest. | Naive Bayes, GANs, VAEs, HMMs. |

### **c. Write a short note on Boltzmann Machines.**
*   **Definition:** An undirected, energy-based stochastic neural network with symmetric connections.
*   **Nodes:** Consists of **Visible nodes** (observed data) and **Hidden nodes** (latent features).
*   **Energy Minimization:** The network energy $E$ is minimized to reach thermal equilibrium. The probability of a state is given by the Boltzmann distribution: $P(s) \propto e^{-E(s)}$.
*   **Limitation:** General Boltzmann machines are "Intractable" (impossible to train) because every node is connected to every other node.
*   **Solution:** **Restricted Boltzmann Machines (RBMs)** remove intra-layer connections, making them a bipartite graph and trainable via Contrastive Divergence.

### **d. Explain Deep Belief Networks (DBN) with its Algorithm.**
*   **Definition:** A deep generative model composed of multiple layers of latent variables, typically built by stacking **RBMs**.
*   **Architecture:** The top two layers are undirected (RBM), while lower layers have directed downward connections.
*   **The Algorithm (Greedy Layer-Wise Pre-training):**
    1.  **Step 1:** Train the first layer as an RBM on raw inputs.
    2.  **Step 2:** Use the hidden activations of the 1st layer as inputs to the 2nd layer.
    3.  **Step 3:** Train the 2nd layer as an RBM.
    4.  **Step 4:** Repeat for all layers.
    5.  **Fine-Tuning:** Use backpropagation (supervised) on the whole stack to refine weights for a specific task (like classification).
*   **Significance:** DBNs were one of the first successful "Deep" architectures, overcoming the vanishing gradient problem by using unsupervised pre-training.
