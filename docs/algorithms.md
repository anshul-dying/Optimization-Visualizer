# Algorithms

This section provides a technical deep-dive into the machine learning models implemented in Optimizer-Lens. Every model is built from scratch using NumPy, focusing on mathematical clarity and vectorized performance.

## Linear regression

Linear Regression aims to find the best-fitting straight line through a set of points by minimizing the sum of squared differences (Mean Squared Error).

- **Objective Function:** $J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$
- **Optimization:** We use Gradient Descent to iteratively update weights $w$ and bias $b$.
- **Implementation Note:** The `core/linear_regression.py` module uses vectorized matrix subtraction and multiplication, significantly outperforming iterative loops.

## Logistic regression

For classification tasks, Logistic Regression applies a Sigmoid activation function to the linear output, mapping predictions to a probability range between 0 and 1.

- **Activation Function:** $\sigma(z) = \frac{1}{1 + e^{-z}}$
- **Loss Function:** Binary Cross-Entropy (Log-Loss).
- **Behavior:** In the playground, you'll see the decision boundary "rotate" to find the optimal separation between Class 0 (Blue) and Class 1 (Red).

## Support Vector Machines (SVM)

SVMs are powerful classifiers that search for the "maximum-margin hyperplane." The project implements the Sequential Minimal Optimization (SMO) algorithm for the quadratic optimization problem.

### Kernels

Kernels allow the SVM to perform non-linear classification by mapping data to a higher-dimensional space where it *is* linearly separable.

1.  **Linear Kernel:** $K(x, y) = x \cdot y$
2.  **Polynomial Kernel:** $K(x, y) = (x \cdot y + c)^d$
3.  **RBF (Gaussian) Kernel:** $K(x, y) = \exp(-\gamma ||x - y||^2)$
4.  **Sigmoid Kernel:** $K(x, y) = \tanh(\gamma x \cdot y + c)$

<!-- prettier-ignore -->
> [!TIP]
> The RBF kernel is exceptionally effective for the "Concentric Circles" and "Half Moons" datasets. Use the **Gamma** hyperparameter to control the "tightness" of the decision regions.

## Neural networks (Multi-Layer Perceptron)

The neural network implementation is a fully connected MLP with configurable depth and width. It demonstrates the power of backpropagation and non-linear activation.

- **Backpropagation:** The model calculates the gradient of the loss with respect to every weight using the chain rule, propagating errors from the output layer back to the input.
- **Activation Functions:**
    - **ReLU:** $f(x) = \max(0, x)$ — prevents vanishing gradients.
    - **Sigmoid/Tanh:** Traditional smooth activations for probability mapping.
    - **Leaky ReLU:** $f(x) = x$ if $x > 0$ else $0.01x$.
- **He Initialization:** We use He weight initialization to ensure stable training in deeper networks.

## Tree-based and ensemble methods

### Decision tree (CART)

Our decision tree implementation uses the Gini Impurity metric to find the optimal feature and threshold for every split.

- **Complexity:** $O(n \cdot d \cdot \log n)$ where $n$ is the number of samples and $d$ is the number of features.
- **Pruning:** Controlled through `max_depth` and `min_samples_split`.

### Random forest (Bagging)

A collection of decision trees, each trained on a "bootstrapped" (randomly sampled with replacement) subset of the data.

- **Diversity:** By only considering a random subset of features at each split, the forest ensures that trees are uncorrelated, leading to better generalization.

### Gradient boosting

A powerful ensemble technique that builds trees sequentially. Each new tree fits the *residuals* (errors) of the current ensemble.

- **Residual Fitting:** If the current model $F(x)$ predicts $\hat{y}$, the next tree $h(x)$ is trained to predict $y - \hat{y}$.
- **Learning Rate:** A lower learning rate (shrinkage) often leads to better performance but requires more trees.

<!-- prettier-ignore -->
> [!NOTE]
> All algorithms in the `core/` package include internal timers and memory tracking to provide the complexity metrics you see in the frontend "Analysis" tab.
