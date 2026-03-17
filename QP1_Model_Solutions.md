# 📄 Advanced Deep Learning (PPSCS402) - 6-Mark Comprehensive Solutions: QP1

This document provides high-scoring, detailed model answers for **EVERY** question in QP1. Each answer is expanded to meet the standard for a **6-mark university question**, including definitions, detailed mechanisms, diagrams (described), and importance.

---

## 🏛️ Section I

### **a. What are neurons? Explain the various activation functions used in neural networks.**
*   **Neuron (Perceptron):**
    *   The basic computational unit of a neural network modeled after biological neurons.
    *   **Mechanism:** It receives multiple inputs $x_1, x_2, ... x_n$. Each input is multiplied by a relative weight $w_i$. These weighted inputs are summed together along with a bias term $b$. The resulting "net input" (pre-activation $z$) is passed through an Activation Function $\phi(z)$ to produce the final output $y$.
    *   **Mathematical Form:** $y = \phi(\sum_{i=1}^{n} w_i x_i + b)$ or $y = \phi(W^T X + b)$.
*   **Activation Functions (Detailed):**
    1.  **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$. Squashes any real value into a range between 0 and 1. Primarily used for binary classification. *Critical weakness:* Causes vanishing gradients as the derivative becomes zero for very high or low inputs.
    2.  **Tanh (Hyperbolic Tangent):** Squashes values to $[-1, 1]$. It is zero-centered, meaning negative inputs are mapped strongly negative. It usually performs better than sigmoid but still suffers from saturation (vanishing gradients).
    3.  **ReLU (Rectified Linear Unit):** $f(x) = \max(0, x)$. Standard for building deep networks. It is computationally efficient and doesn't saturate in the positive domain.
    4.  **Leaky ReLU:** $f(x) = \max(0.01x, x)$. Used to fix the "Dying ReLU" problem where neurons stop learning because they only receive negative inputs.
    5.  **Softmax:** $\frac{e^{z_i}}{\sum e^{z_j}}$. Used specifically in the **output layer** for multi-class classification to provide a probability distribution.

### **b. Discuss the concept of linear regression in brief.**
*   **Core Concept:** Linear regression is a supervised learning algorithm used to predict a continuous numerical output based on one or more input features.
*   **The Equation:**
    *   Simple Linear Regression: $y = \beta_0 + \beta_1x + \epsilon$
    *   Multiple Linear Regression: $y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \epsilon$
*   **Key Components:**
    1.  **Dependent Variable ($y$):** The value we want to predict (Target).
    2.  **Independent Variable ($x$):** The features used for prediction.
    3.  **Slope ($\beta_1$):** Represents how much $y$ changes for a unit change in $x$.
    4.  **Intercept ($\beta_0$):** The value of $y$ when $x=0$.
*   **Objective (Cost Function):** The goal is to minimize the **Residual Sum of Squares (RSS)** or **Mean Squared Error (MSE)**. We often use **Gradient Descent** to iteratively update the weights ($\beta$) until the error is at its minimum.
*   **Assumptions:** Linearity, Independence of errors, Homoscedasticity (constant variance of errors), and Normal distribution of errors.

### **c. State the various applications of Deep Learning.**
Deep Learning has revolutionized many industries by automating complex feature extraction:
1.  **Computer Vision:** Face recognition (smartphones), Medical imaging (detecting tumors in X-rays/MRIs), and Object detection for self-driving cars.
2.  **Natural Language Processing (NLP):** Neural Machine Translation (Google Translate), Sentiment Analysis (Social media monitoring), and Large Language Models (ChatGPT, Claude).
3.  **Speech Recognition:** Virtual Assistants (Alexa, Siri), Speech-to-Text conversion for automated subtitling.
4.  **Healthcare:** Drug discovery (predicting molecular interaction) and personalized medicine based on genomic data.
5.  **Finance:** Algorithmic trading, fraud detection in real-time transactions, and credit scoring.
6.  **Recommendation Systems:** Movie recommendations (Netflix), Product recommendations (Amazon), and Content feeds (YouTube/TikTok).

### **d. Explain the concept of Overfitting in neural network models.**
*   **Definition:** Overfitting occurs when a neural network learns the training data "by heart," including its noise, outliers, and random fluctuations, instead of the general underlying pattern.
*   **The Problem:** The model achieves near 100% accuracy on training data but fails to generalize, leading to high error rates on the validation and test sets.
*   **Reasons for Overfitting:**
    1.  **High Capacity:** The network has too many layers/neurons relative to the amount of data.
    2.  **Insufficient Data:** The model doesn't have enough examples to see the true general pattern.
    3.  **Noisy Data:** The model tries to fit the random errors in the labels.
*   **Identification:** In a training plot, if the **Training Loss** continues to go down while the **Validation Loss** starts going up, the model is overfitting.
*   **Remedies (6-mark highlight):** Use Regularization (L1/L2), Dropout, Early Stopping, or add more training data.

---

## 🏛️ Section II

### **a. What do you understand by the term “Hyperparameter tuning”? Explain.**
*   **Hyperparameters vs. Parameters:** Unlike weights ($W$) and biases ($b$) which are learned by the model, hyperparameters are settings that must be defined **before** training starts.
*   **Examples:** Learning Rate, Number of Epochs, Batch Size, Number of Hidden Layers, and Dropout rate.
*   **Why Tuning is Required:** The performance of a model is highly sensitive to these values. For instance, a learning rate that is too high causes the model to diverge, while one that is too low makes training painfully slow.
*   **Tuning Strategies:**
    1.  **Grid Search:** Testing every possible combination from a set of values. (Exhaustive but slow).
    2.  **Random Search:** Testing random combinations. (Statistically more efficient than grid search).
    3.  **Bayesian Optimization:** Uses previous results to predict which hyperparameters will likely yield better performance.
    4.  **Hyperband:** An early-stopping-based approach that allocates more resources to promising configurations.

### **b. Write a short note on Inception networks.**
*   **The Motivation:** To create deep networks without a massive increase in computational cost while solving the problem of "what filter size is best?" (3x3? 5x5?).
*   **The Inception Module:** Instead of choosing one filter size, the network uses multiple filters (1x1, 3x3, and 5x5) and a Max-Pooling operation **in parallel** on the same level. Their outputs are then concatenated together.
*   **1x1 Convolutions (Bottleneck Layers):** Inception uses 1x1 convolutions to reduce the number of input channels before passing them to the expensive 3x3 and 5x5 convolutions. This "compresses" the data and saves millions of parameters.
*   **Global Average Pooling:** Instead of using massive Fully Connected layers at the end, Inception uses Global Average Pooling to reduce dimensions, preventing overfitting.
*   **Key Achievement:** GoogLeNet (Inception v1) won ILSVRC 2014, showing that depth and width can be increased efficiently.

### **c. What are backpropagation networks?**
*   **Definition:** Networks that use the Backpropagation algorithm to update weights based on the error.
*   **The Four Steps of Backprop:**
    1.  **Forward Pass:** Data flows through the network to generate a prediction $\hat{y}$.
    2.  **Loss Calculation:** The difference between $\hat{y}$ and true label $y$ is calculated using a loss function (e.g., MSE).
    3.  **Backward Pass (Chain Rule):** Starting from the output layer, the network calculates the gradient of the loss with respect to every weight. It uses the **Chain Rule** of calculus to pass the error signal backward through the layers.
    4.  **Weight Update:** An optimizer (like SGD) uses these gradients to adjust the weights: $W = W - \eta \cdot \frac{\partial Loss}{\partial W}$.
*   **Importance:** It is the backbone of all modern deep learning, allowing machines to "learn" from their mistakes by iteratively reducing error.

### **d. Explain the issue of vanishing and exploding gradients.**
*   **The Problem:** During backpropagation, gradients are multiplied by the weights of each layer.
*   **Vanishing Gradients:**
    *   **Cause:** Frequent with activation functions like Sigmoid where the derivative is very small ($<0.25$).
    *   **Effect:** Multiplying many small numbers ($0.1 \times 0.1 \times 0.1...$) makes the gradient effectively zero for early layers. These layers "vanish" because they stop receiving updates.
*   **Exploding Gradients:**
    *   **Cause:** Frequent in RNNs where the same weights are reused many times. If weights are large ($>1.0$), the gradient grows exponentially.
    *   **Effect:** Weight updates become massive, causing the model to crash or produce `NaN` values.
*   **Solutions:**
    1.  Use **ReLU** (derivative is 1, so it doesn't shrink gradients).
    2.  **Batch Normalization** (stabilizes weights).
    3.  **Gradient Clipping** (setting a max threshold for gradients to prevent explosion).
    4.  **Residual Connections (ResNets)** (provide a shortcut for gradients).

---

## 🏛️ Section III

### **a. What are autoencoders? How they are used in regularization?**
*   **Definition:** An unsupervised neural network that takes an input $X$, passes it through an **Encoder** to create a compressed latent vector $Z$ (the bottleneck), and then uses a **Decoder** to reconstruct the original input $\hat{X}$.
*   **Architecture:** Input Layer $\rightarrow$ Hidden Layer (smaller) $\rightarrow$ Output Layer (same size as input).
*   **Regularization Techniques:**
    1.  **Undercomplete Autoencoder:** By making the hidden layer much smaller than the input, the network is *forced* to learn only the most important features. This act of compression is itself a form of regularization.
    2.  **Sparse Autoencoder:** A penalty term is added to the loss function to ensure that only a few neurons in the hidden layer are active (non-zero) at a time.
    3.  **Denoising Autoencoder (DAE):** The network is given a "noisy" version of the image but told to reconstruct the "clean" version. This forces the model to learn the stable, high-level structural features of the data rather than the noise.

### **b. Explain the architecture of Hopfield networks.**
*   **Definition:** A recurrent, associative memory network that can store and retrieve patterns.
*   **Architecture Details:**
    1.  **Single Layer:** It is a single-layer network where every neuron is an input and an output at the same time.
    2.  **Full Connectivity:** Every neuron is connected to every other neuron in the network.
    3.  **Symmetric Weights:** The weight between neuron $i$ and $j$ is the same as $j$ to $i$ ($w_{ij} = w_{ji}$).
    4.  **No Self-Connections:** A neuron is never connected to itself ($w_{ii} = 0$).
*   **Energy Minimization:** The network works on the "Energy Surface" principle. Each stored pattern is a stable "energy well" (local minimum). If you give it a partial or noisy pattern, the network will "roll down" the energy surface until it reaches the closest stored stable pattern.
*   **Usage:** Pattern completion and solving optimization problems like the Traveling Salesman Problem.

### **c. State the applications of Generative Adversarial Networks (GANs).**
GANs are powerful for generating new, synthetic data that looks real:
1.  **Image Synthesis:** Generating photorealistic faces of people who don't exist.
2.  **Super-Resolution:** Converting low-resolution, blurry images into high-definition photos (SRGAN).
3.  **Style Transfer:** Applying the artistic style of one image (e.g., Starry Night) to another photograph.
4.  **Image-to-Image Translation:** Converting satellite maps to Google Maps view, or day photos to night photos (Pix2Pix).
5.  **Data Augmentation:** Creating synthetic medical data (X-rays, MRIs) where real data is scarce or sensitive.
6.  **Text-to-Image:** Systems like DALL-E or Midjourney that generate images based on text descriptions.
7.  **Video Generation:** Creating realistic video clips from static images or short prompts.

### **d. Discuss the training algorithms in Generative Adversarial Networks.**
*   **The Minimax Concept:** GAN training is a "Zero-Sum Game" between two networks: the **Generator (G)** and the **Discriminator (D)**.
*   **The Algorithm Steps:**
    1.  **Sample Noise:** The Generator takes a random noise vector $z$ as input.
    2.  **Generate Fake Data:** $G$ produces a fake sample $G(z)$.
    3.  **Train Discriminator ($D$):** Pass both **Real Data ($x$)** and **Fake Data ($G(z)$)** to the Discriminator. $D$ is updated to maximize its accuracy: label real as 1 and fake as 0.
    4.  **Train Generator ($G$):** The Generator is updated by calculating the gradient through the Discriminator. Its goal is to maximize the probability of $D$ labeling its fake data as "1" (Real).
*   **The Loss Function:** $\min_G \max_D \mathbb{E}_{x}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))]$.
*   **Equilibrium:** Training ends at a **Nash Equilibrium** where the Discriminator is completely confused and is only correct 50% of the time.

---

## 🏛️ Section IV

### **a. State the use of deep learning for language modelling.**
*   **Language Modelling Definition:** The task of predicting the next word in a sentence based on the context of previous words.
*   **Deep Learning Evolution:**
    1.  **Word Embeddings (Word2Vec/GloVe):** Represents words as dense vectors where similar words have similar vectors.
    2.  **RNNs & LSTMs:** The first major deep learning success in NLP. They process text sequentially and maintain a hidden state (memory) of previous words.
    3.  **Transformers (The Modern Standard):** Unlike RNNs, Transformers use **Self-Attention** to look at all words in a sentence simultaneously. This allows them to capture long-distance relationships (e.g., a pronoun at the end of a paragraph referring to a name at the start).
*   **Applications:** Autocomplete systems, Machine Translation, Voice Assistants, and Generative AI (GPT-4).

### **b. Why deep networks are considered a Q-function?**
*   **Reinforcement Learning Context:** In RL, we want to know the "value" of taking a certain action $a$ in a certain state $s$. This is called the Q-value, $Q(s, a)$.
*   **The Approximation Requirement:** In raw RL (Q-Learning), we use a table. But if our state is an image (millions of pixels), the table would be too massive to ever build.
*   **Deep Q-Networks (DQN):** A Deep Neural Network is used to **approximate** the Q-function. The network takes the state (e.g., game screen) as input and outputs the predicted Q-values for all possible actions.
*   **Conclusion:** The network "is" the Q-function because it performs the mapping $f(State, Action) \rightarrow Future\_Reward$. It allows RL to scale to complex problems like playing Atari games or controlling robots.

### **c. Write a short note on Reinforcement learning (RL).**
*   **Definition:** A branch of Machine Learning where an **Agent** learns to make decisions by performing **Actions** in an **Environment** to maximize a cumulative **Reward**.
*   **The RL Loop:**
    1.  Agent observes the current **State** ($s$).
    2.  Agent takes an **Action** ($a$) based on its **Policy** ($\pi$).
    3.  The Environment provides a **Reward** ($r$) and moves to a **New State** ($s'$).
*   **Key Differences:** Unlike supervised learning, there are no "labels" telling the agent the correct move. It only gets a feedback signal (Reward) which might be delayed (e.g., winning a game of chess after 50 moves).
*   **Components:** Policy (Agent's strategy), Reward Signal (Immediate feedback), Value Function (Long-term reward), and Model of environment (Optional).

### **d. Explain the sliding window approach for object localization.**
*   **Goal:** Localization not only detects *what* is in an image but *where* it is using a bounding box.
*   **Process:**
    1.  **Training:** A CNN is trained to classify small, cropped images of the target object (e.g., a car).
    2.  **Testing (The Slide):** A rectangular window of a specific size (e.g., 64x64) slides across the test image.
    3.  **Classification:** At every single step (pixel-by-pixel shift), the window content is cropped and fed into the CNN.
    4.  **Multi-Scale:** The process is repeated with different window sizes (large for close objects, small for far ones).
*   **Weakness:** It is extremely slow because it requires thousands of CNN passes for a single image.
*   **The Modern Answer:** This was replaced by "Region Proposal" (R-CNN) and eventually "Single-Shot" detectors like **YOLO** which divide the image into a grid and predict objects in one go.

---

## 🏛️ Section V

### **a. Explain multiclass classification in feedforward neural networks with an example.**
*   **Structure:** A multiclass network has the same number of output neurons as there are classes. (e.g., for 10 classes, there are 10 output nodes).
*   **The Softmax Layer:** The raw scores (logits) from the final layer are passed through a Softmax function. This ensures that:
    1.  All output values are between 0 and 1.
    2.  The sum of all outputs is exactly 1.0 (treating them as probabilities).
*   **The Loss Function:** We use **Categorical Cross-Entropy**. It measures how far our predicted probability distribution is from the true label (where the correct class is 1 and others are 0).
*   **Example (MNIST):**
    *   **Input:** An image of the number '7'.
    *   **Network:** Multiple hidden layers with ReLU.
    *   **Output Layer:** 10 nodes (0 to 9).
    *   **Result:** After training, the node at index 7 produces a value like 0.98, while others are near 0.

### **b. How to deal with Overfitting in neural network models?**
To secure 6 marks, list these standard "Best Practices":
1.  **L1/L2 Regularization:** Adding a penalty to the loss function based on the size of the weights (forces weights to stay small).
2.  **Dropout:** Randomly "turning off" a percentage of neurons during each training step. This prevents neurons from becoming too dependent on each other (**co-adaptation**).
3.  **Early Stopping:** Monitoring the validation error. As soon as it stops decreasing (even if training error is still going down), stop the training.
4.  **Data Augmentation:** Artificially increasing the data size by flipping, rotating, or cropping existing images.
5.  **Batch Normalization:** Normalizing the inputs to each layer so that training remains stable and faster, which implicitly acts as a minor regularizer.
6.  **Simplifying Architecture:** Reducing the number of layers or neurons if the model is too complex for the task.

### **c. Write a short note on Boltzmann machines (BM).**
*   **Core Logic:** An undirected, energy-based stochastic neural network. It consists of a visible layer (data) and a hidden layer (features).
*   **Stochastic Nature:** Unlike standard networks that produce a fixed number, BM units are probabilistic (they have a probability of being "on" or "off").
*   **Energy Principles:** Inspired by thermodynamics. The network tries to reach a state of minimum "Global Energy." Storing a piece of data corresponds to creating an "Energy Well" on the error surface.
*   **Restricted Boltzmann Machine (RBM):** The most practical version. It is "Restricted" because:
    *   Visible nodes ONLY connect to Hidden nodes.
    *   There are NO connections between visible-visible or hidden-hidden nodes.
*   **Usage:** RBMs were famous for the "Netflix Prize" (Collaborative filtering) and for pre-training deep belief networks.

### **d. How to balance exploration with exploitation?**
*   **Exploitation:** Using the move/strategy that the agent *already knows* works best to get immediate points.
*   **Exploration:** Trying a new or random move to see if it might be even better than the current best move.
*   **Balancing Strategies:**
    1.  **Epsilon-Greedy ($\epsilon$):** The agent chooses the best move $(1 - \epsilon)$ of the time. But with probability $\epsilon$, it chooses a completely random move. We usually start with $\epsilon=1$ (full exploration) and decrease it to $0.01$ (mostly exploitation) as the agent gets smarter.
    2.  **Softmax Strategy:** Actions with higher estimated values are chosen more often, but all actions have a non-zero probability of being picked.
    3.  **Upper Confidence Bound (UCB):** Adds a "bonus" to moves that the agent hasn't tried many times yet. This forces the agent to explore "unknown" territory.
*   **Significance:** Without this balance, an agent might get stuck doing a "good" move forever and never find the "perfect" move.
