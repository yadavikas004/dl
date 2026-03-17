# Deep Learning Exam Revision Guide

This guide is curated based on the current syllabus and recurring patterns in past question papers. Focus on these core areas to maximize your score.

---

## 1. Unit-wise Pointers (Key Concepts to Master)

### Unit I: Linear Algebra & Numerical Optimization
- **Core Matrix Operations:** Focus on Matrix-vector multiplication, Identity matrices, and Inverses.
- **Norms:** L1, L2, and $L^\infty$ norms. Understand why L1 leads to sparsity.
- **Eigen Decompositions:** Definition and its role in decomposing matrices into eigenvalues and eigenvectors.
- **Numerical Computation:** The concepts of **Overflow** (numbers too large) and **Underflow** (numbers too close to zero).
- **Optimization:** Gradient-based learning, Saddle points, and the dangers of "Poor Conditioning."

### Unit II: Deep Networks & Training
- **XOR Problem:** Why a single-layer perceptron fails and how hidden layers solve it.
- **Regularization:** L1 vs L2 regularization; **Bagging** (Bootstrap Aggregating); **Dropout** (randomly disabling neurons).
- **Backpropagation:** The chain rule for calculating gradients.
- **Challenges:** Identifying plateaus and local minima in the optimization landscape.

### Unit III: Convolution & Sequences
- **CNN Mechanism:** Understanding Kernels, Stride, Padding, and the role of **Pooling** (Max vs Average).
- **Recurrent Neural Networks (RNNs):** Unfolding of computational graphs; handling sequence data like natural language.
- **Data Augmentation:** Techniques to artificially increase dataset size (rotations, flips, etc).

### Unit IV: Research & Advanced Models
- **Autoencoders:** Denoising Autoencoders vs Undercomplete Autoencoders.
- **Representational Learning:** How models learn features.
- **Probabilistic PCA:** Comparing PCA with Factor Analysis.
- **Generative Models:** Generative Adversarial Networks (GANs) and Boltzmann Machines (RBMs).

---

## 2. Definition Questions & Answers

**Q1: Define a Saddle Point.**
**A:** A point where the gradient is zero, but it is a local maximum in one direction and a local minimum in another. Deep learning models often get stuck in saddle points rather than local minima.

**Q2: What is the "Identity Matrix"?**
**A:** A square matrix (usually denoted as $I$) with ones on the main diagonal and zeros elsewhere. Multiplying any matrix by the identity matrix leaves the original matrix unchanged ($A \cdot I = A$).

**Q3: Explain "Poor Conditioning."**
**A:** It refers to functions that change significantly with tiny changes in input. In matrices, a high **Condition Number** (ratio of largest to smallest eigenvalue) implies poor conditioning, making matrix inversion numerically unstable.

**Q4: What is the "Bagging" technique?**
**A:** Bagging (Bootstrap Aggregating) involves training multiple models on different subsets of the training data and averaging their predictions. It reduces variance and prevents overfitting.

**Q5: Briefly explain "RNN Unfolding."**
**A:** Unfolding is the process of mapping a recurrent computational graph (a loop) into a directed acyclic graph (a sequence of layers) where each layer represents a time step.

---

## 3. Mathematical Revision & Formulas

### Norms (Crucial for Regularization)
- **L1 Norm:** $\|x\|_1 = \sum_{i} |x_i|$ (Sum of absolute values; promotes sparsity)
- **L2 Norm:** $\|x\|_2 = \sqrt{\sum_{i} x_i^2}$ (Euclidean distance; commonly used for weight decay)

### Convolution calculation
**Formula:** $(f * g)(i) = \sum f(j)g(i-j)$
*Exam Shortcut:* Slide the kernel over the input and sum the products.
- **Example:** Input $[2, 3, 5]$, Kernel $[1, 2]$
- Step 1: $(2\times 1) + (3\times 2) = 8$
- Step 2: $(3\times 1) + (5\times 2) = 13$
- Result: $[8, 13]$

### Backpropagation Chain Rule
For $y = f(u)$ and $u = g(x)$, the derivative $\frac{dy}{dx}$ is:
$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$

### Eigenvalues/Eigenvectors
$A \mathbf{v} = \lambda \mathbf{v}$
Where $A$ is a square matrix, $\mathbf{v}$ is the eigenvector, and $\lambda$ is the eigenvalue.

---

## 4. Revision Tips for the Exam
1. **Draw Diagrams:** Always draw a 3x3 matrix for examples or a simple 4-neuron network for XOR.
2. **Comparison Tables:** Expect questions like "L1 vs L2" or "PCA vs Factor Analysis." Use tables for these.
3. **Keywords:** Use terms like "Numerical Stability," "Sparsity," "Vanishing Gradient," and "Loss Surface."
