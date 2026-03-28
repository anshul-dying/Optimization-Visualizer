# Introduction

Optimizer-Lens is an advanced, full-stack interactive playground designed to demystify the "black box" of machine learning optimization. By providing real-time, epoch-by-epoch visualizations of model training, it offers a unique educational and debugging perspective that standard ML libraries often abstract away.

This project bridges the gap between high-level web interfaces and low-level algorithmic implementation. It is specifically designed for students and engineers who want to see the "why" behind model convergence, decision boundaries, and hyperparameter sensitivity.

## Core philosophy

The project is built on three foundational pillars:

1.  **Transparency:** Every algorithm is implemented from scratch using NumPy. No hidden abstractions—just linear algebra and calculus.
2.  **Immediacy:** Changes to hyperparameters like learning rates or kernel types result in instant visual feedback on the training canvas.
3.  **Experimental Rigor:** The playground supports "extreme values" and various noise levels, allowing you to stress-test algorithms against edge cases that often break production models.

## Why Optimizer-Lens?

In a typical ML workflow, you call `.fit()` and wait for a result. You get a final accuracy score, but you don't see the *path* the model took to get there. Optimizer-Lens changes this by:

- **Visualizing the Gradient:** Watch how a linear regression line "vibrates" and eventually settles as it finds the global minimum.
- **Observing the Kernel Trick:** See how an RBF kernel transforms a simple 2D space to capture complex, non-linear circular patterns.
- **Debugging Overfitting:** Increase the depth of a decision tree or the epochs of a neural network and watch the decision boundary grow increasingly erratic as it starts memorizing noise.

## Technical stack

Optimizer-Lens leverages modern technologies to deliver a seamless experience:

- **Frontend Engine:** Next.js 14 utilizing React Server Components for performance and HTML5 Canvas for high-frequency visualization updates.
- **Mathematical Core:** Python 3.8+ with NumPy for vectorized matrix operations, ensuring that scratch-built algorithms perform efficiently.
- **API Layer:** FastAPI provides a high-throughput, asynchronous bridge between the browser and the Python runtime.
- **Analytics:** Recharts-based telemetry for tracking loss and accuracy curves in real-time.

<!-- prettier-ignore -->
> [!IMPORTANT]
> Optimizer-Lens is an educational tool. While the algorithms are mathematically accurate, they are optimized for visual clarity and educational demonstration rather than large-scale production inference.

## Next steps

To begin your journey with Optimizer-Lens:

1.  **[Installation Guide](/docs/getting-started):** Set up your local environment in under 5 minutes.
2.  **[Architecture Deep-Dive](/docs/architecture):** Understand how the frontend and backend communicate.
3.  **[Algorithmic Catalog](/docs/algorithms):** Explore the mathematics behind our implementations.
