# Optimizer-Lens

An interactive machine learning playground to visualize and test optimization algorithms.

## Features

- **8 Real ML Algorithms**: All algorithms fully implemented from scratch in Python
- **Interactive Playground**: Visualize decision boundaries and training metrics in real-time
- **Multiple Datasets**: Test with linear, non-linear, circles, moons, and more
- **Performance Metrics**: Track loss, accuracy, confusion matrix during training
- **Real-time Training**: Watch algorithms learn step-by-step
- **Dual Mode**: Switch between simulation (fast) and real implementations
- **Kernel Support**: Multiple kernel types for SVM (Linear, RBF, Polynomial, Sigmoid)
- **Customizable Networks**: Configure neural network architecture and activation functions

## Setup

### Frontend (Next.js)

```bash
# Install dependencies
pnpm install

# Run development server
pnpm dev
```

Visit `http://localhost:3000` to see the application.

### Backend (Python API - Required for Real Algorithms)

The playground can work in two modes:
1. **Simulation Mode** (default): Uses mathematical simulations to show algorithm behavior
2. **Real Mode**: Uses actual Python implementations of Linear and Logistic Regression

To enable **Real Mode**:

```bash
# Install Python dependencies
pip install -r requirements.txt

# Start the API server
python core/api.py
```

The API will run on `http://localhost:8000`

## Project Structure

```
├── app/
│   ├── page.tsx              # Home page
│   └── playground/
│       └── page.tsx          # Playground page
├── components/
│   ├── ml-playground.tsx     # Main playground component
│   └── ui/                   # UI components (shadcn/ui)
├── core/
│   ├── api.py               # FastAPI backend
│   ├── linear_regression.py # Linear Regression implementation
│   ├── logistic.py          # Logistic Regression implementation
│   └── optimizers/
│       └── linear.cpp       # C++ optimizers (future use)
└── requirements.txt         # Python dependencies
```

## Algorithms

### Implemented & Working ✅

All algorithms are now fully implemented and functional!

**1. Linear Regression** (`core/linear_regression.py`)
- Gradient Descent
- Stochastic Gradient Descent  
- Mean Squared Error loss
- R² score for accuracy

**2. Logistic Regression** (`core/logistic.py`)
- Sigmoid activation
- Binary Cross-Entropy loss
- Gradient Descent optimization

**3. Support Vector Machine** (`core/svm.py`)
- Linear, Polynomial, RBF, and Sigmoid kernels
- SMO (Sequential Minimal Optimization) algorithm
- Soft margin classification with C parameter
- Configurable gamma for RBF kernel

**4. Neural Network** (`core/neural_network.py`)
- Multi-layer perceptron
- Activation functions: ReLU, Sigmoid, Tanh, Leaky ReLU
- Configurable hidden layers
- Mini-batch gradient descent
- He weight initialization

**5. Decision Tree** (`core/decision_tree.py`)
- CART algorithm
- Gini impurity for splitting
- Configurable max depth and min samples
- Binary classification

**6. Random Forest** (`core/random_forest.py`)
- Ensemble of decision trees
- Bootstrap aggregating (bagging)
- Majority voting
- Configurable number of estimators

**7. Gradient Boosting** (`core/gradient_boosting.py`)
- Sequential ensemble learning
- Boosting with residual fitting
- Configurable learning rate
- Early stopping capability

**8. Kernel SVM** (Same as SVM with different kernel parameter)
- All kernel types: Linear, Polynomial, RBF, Sigmoid
- Kernel trick for non-linear decision boundaries

## How to Use

1. **Start the Frontend**: Run `pnpm dev` and open `http://localhost:3000/playground`

2. **Choose a Dataset**: Select from various dataset types (linear, circles, moons, etc.)

3. **Configure Algorithm**:
   - Select algorithm (Linear or Logistic for real implementations)
   - Adjust learning rate, epochs, and other hyperparameters

4. **(Optional) Enable Real Algorithms**:
   - Start the Python API: `python core/api.py`
   - Toggle "Use Real Algorithms" in the playground
   - Train button will now use actual implementations

5. **Train**: Click the Train button to see the algorithm learn

6. **Analyze**: View loss curves, accuracy metrics, and decision boundaries

## Bug Fixes Applied

### Linear Regression (`core/linear_regression.py`)
- ✅ Fixed parameter order in `_compute_gradient`
- ✅ Fixed loop iteration in `fit_sgd` (was `for i in range(indices)`, now `for i in indices`)

### Logistic Regression (`core/logistic.py`)
- ✅ Added `self` parameter to `_sigmoid` method
- ✅ Fixed matrix multiplication order in `_compute_prediction`
- ✅ Added missing `fit` method
- ✅ Added `get_params` method

### Playground (`components/ml-playground.tsx`)
- ✅ Added API integration for real algorithm training
- ✅ Added toggle to switch between simulation and real modes
- ✅ Added API status checking

## API Endpoints

### `GET /`
Health check

### `POST /train`
Train a model with given data and configuration

Request:
```json
{
  "algorithm": "linear" | "logistic",
  "X": [[x1, y1], [x2, y2], ...],
  "y": [label1, label2, ...],
  "learning_rate": 0.01,
  "epochs": 100
}
```

Response:
```json
{
  "metrics": [{"epoch": 1, "loss": 0.5, "accuracy": 0.8}, ...],
  "weights": [w1, w2],
  "bias": 0.5,
  "final_loss": 0.1,
  "final_accuracy": 0.95
}
```

### `POST /predict`
Make predictions with a trained model

## Technologies

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: Python, FastAPI, NumPy
- **Visualization**: Recharts, HTML Canvas
- **Future**: C++ optimizers with pybind11

## License

MIT
