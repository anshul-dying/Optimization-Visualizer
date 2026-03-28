# Frontend

The Optimizer-Lens frontend is a high-performance Next.js 14 application that provides the interactive playground. It manages the training lifecycle, visualizes model convergence, and provides detailed performance analysis tools.

The application uses a reactive state management approach, primarily relying on React hooks (`useState`, `useEffect`, `useRef`). This ensures that changes to parameters like learning rates or noise levels are immediately reflected in the visualizations.

## Core architecture

The frontend is organized around several key layers:

- **State Management:** The `MLPlayground` component maintains the configuration of the machine learning model, the current dataset, and the training metrics.
- **Rendering Engine:** A specialized HTML5 Canvas layer handles the drawing of data points and the real-time update of decision boundaries.
- **API Communication:** A lightweight bridge that communicates with the FastAPI backend to send training requests and receive model weights and histories.
- **Analysis Layer:** Built with **Recharts**, this layer provides dynamic charts for tracking loss and accuracy curves during the training process.

## Visualization techniques

### Real-time canvas rendering

The heart of the playground is the canvas-based visualization. It allows you to see the model converge on the data in real-time.

- **Data points:** These are rendered as individual circles on the canvas. The color indicates the class label (Red for Class 1, Blue for Class 0).
- **Decision boundary:** For linear and logistic regression, the frontend calculates the hyperplane based on the current weights and bias returned by the model.
- **Non-linear boundaries:** For algorithms like SVM (with kernels) and Neural Networks, the frontend uses a grid-based prediction approach. It queries the model for predictions across a 2D grid and renders a heatmap to represent the decision regions.

### Interactive playback

When training with the "Real Algorithms" mode, the frontend does more than just show the final result. It performs an "animation playback" of the training history.

1.  The backend returns a list of metrics for every epoch.
2.  The frontend iterates through this history, updating the state at each step.
3.  The canvas and charts re-render to reflect the model's state at that specific point in time.

## Synthetic data generation

Optimizer-Lens includes several generators for common machine learning datasets. These are used to test how different algorithms handle various data distributions.

- **Linear Separable:** Data that can be perfectly split by a straight line.
- **Non-linear:** Data requiring non-linear boundaries, such as circular or moon-shaped distributions.
- **Extreme Outliers:** Tests how the model handles noise and points that are far from the main distribution.
- **Concentric Circles:** A classic test case for kernel-based SVMs and deep neural networks.

## UI components

The interface is built using **shadcn/ui** components, which provide a clean and consistent aesthetic. Key UI elements include:

- **Configuration Panel:** A sidebar for adjusting hyperparameters, dataset types, and training modes.
- **Execution & Analysis Panel:** The main area containing the decision boundary canvas and algorithmic breakdown.
- **Performance Metrics:** A dedicated section for tracking loss, accuracy, and confusion matrices.

<!-- prettier-ignore -->
> [!TIP]
> Use the **Extreme Values** switch to see how your chosen algorithm handles datasets with very large coordinate values. This often reveals stability issues in the mathematical implementation.

## Next steps

For more details on how the backend handles the computations, refer to the **Backend documentation**.
