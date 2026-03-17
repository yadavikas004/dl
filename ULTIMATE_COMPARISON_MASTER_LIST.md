# 🏁 The "Perfect Score" Comparison Master-List

University examiners love comparison questions because they are easy to grade. **If you see a "Compare vs" question, copy these tables exactly to get 100% marks in that section.**

---

### **1. Optimization: L1 Regularization vs. L2 Regularization**
| Feature | L1 (Lasso) | L2 (Ridge) |
| :--- | :--- | :--- |
| **Penalty Term** | $\lambda \sum |w_i|$ (Absolute) | $\lambda \sum w_i^2$ (Squared) |
| **Geometry** | Diamond / Rhombus | Circular / Spherical |
| **Resulting Weights** | Exact zeros (**Sparsity**) | Small but non-zero values |
| **Use Case** | Feature selection; interpretable models | Preventing large weights; general stability |
| **Effect on Model** | Compact, light model | Distributed, robust model |

---

### **2. Neural Architectures: CNN vs. RNN**
| Feature | Convolutional Neural Network (CNN) | Recurrent Neural Network (RNN) |
| :--- | :--- | :--- |
| **Input Type** | Spatial / Grid (Images, Video) | Sequential / Temporal (Text, Audio) |
| **Core Operation** | Sliding Filters (Convolution) | Hidden State Feedback (Loops) |
| **Parameter Sharing** | Sharing across space (image locations) | Sharing across time (sequence steps) |
| **Vanishing Gradient** | Not common (helped by ReLU) | Severe (fixed by LSTM/GRU/Gates) |
| **Focus** | Local features (edges, textures) | Historical context (previous words) |

---

### **3. Latent Models: Boltzmann (BM) vs. Restricted Boltzmann (RBM)**
| Feature | General Boltzmann Machine | Restricted Boltzmann Machine |
| :--- | :--- | :--- |
| **Connections** | Any node to any other node (Full) | Bipartite (Visible-to-Hidden only) |
| **Intra-layer** | Connections allowed within layers | **No connections** within a layer |
| **Training Speed** | Extremely slow (intractable) | Fast (Contrastive Divergence) |
| **Topology** | Complete graph | Bipartite graph |
| **Applications** | Theoretical research | Feature learning, Recommendations |

---

### **4. Dimensionality Reduction: PCA vs. ICA**
| Feature | Principal Component Analysis (PCA) | Independent Component Analysis (ICA) |
| :--- | :--- | :--- |
| **Information Type** | Variance / Uncorrelatedness | Statistical Independence |
| **Distribution** | Assumes Gaussian data | Works with **Non-Gaussian** data |
| **Orthogonality** | Components must be orthogonal | Components don't have to be orthogonal |
| **Example** | Image compression, noise reduction | **Cocktail Party Problem** (Source separation)|

---

### **5. Generative Strategy: Discriminative vs. Generative**
| Feature | Discriminative Models | Generative Models |
| :--- | :--- | :--- |
| **Probability** | $P(y|x)$ (Conditional) | $P(x,y)$ or $P(x)$ (Joint) |
| **Goal** | Find the decision boundary | Understand how data was created |
| **Examples** | SVM, Logistic Regression, MLP | GANs, VAEs, Naive Bayes |
| **New Data** | Cannot generate new samples | **Can synthesize** new data |

---

### **6. RNN Variants: LSTM vs. GRU**
| Feature | LSTM (Long Short-Term Memory) | GRU (Gated Recurrent Unit) |
| :--- | :--- | :--- |
| **Structure** | 3 Gates (Input, Forget, Output) | 2 Gates (Reset, Update) |
| **Cell State** | Keeps a long-term "Cell State" ($C_t$) | Merges state with hidden state ($h_t$) |
| **Complexity** | More complex; more parameters | Simpler; faster to train |
| **Performance** | Better for very long sequences | Better for smaller/medium datasets |

---

### **7. Numerical Stability: Overflow vs. Underflow**
| Feature | Underflow | Overflow |
| :--- | :--- | :--- |
| **Definition** | Numbers too small (close to 0) | Numbers too large (exceeding limit) |
| **Result** | Evaluations become strictly zero | Evaluations become `inf` or `NaN` |
| **Dataloss** | Gradient signal vanishes completely | Gradient signal oscillates wildly |
| **Fix** | Using `log` transformations | Gradient Clipping / Batch Norm |
