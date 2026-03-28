# Backend

The Optimizer-Lens backend is a high-performance computation engine built with FastAPI and NumPy. It provides a robust API for training machine learning models and generating predictions for visualization.

The core of the backend is the `core/` package, which contains individual implementations of various machine learning algorithms. These are designed for educational clarity and high performance, with each algorithm implemented as a standalone class.

## API reference

The backend exposes several endpoints to support the frontend playground.

### `GET /`

Returns a simple status message confirming that the API is running. Use this to verify the connection between the frontend and backend.

### `POST /train`

The primary endpoint for training machine learning models. You must provide the algorithm name, dataset coordinates (`X`), labels (`y`), and any relevant hyperparameters.

**Request body:**

```json
{
  "algorithm": "svm",
  "X": [[1.2, 2.3], [0.5, -1.0], ...],
  "y": [1, 0, ...],
  "learning_rate": 0.01,
  "epochs": 100,
  "kernel": "rbf",
  "C": 1.0
}
```

**Response:**

Returns the training metrics for each epoch, including loss and accuracy. For linear models, it also returns final weights and bias.

### `POST /predict`

Generates predictions for a given set of input features using the latest trained model stored in memory.

**Request body:**

```json
{
  "X": [[1.5, 2.0], [0.0, -0.5], ...]
}
```

**Response:**

Returns a list of predicted class labels for each input point.

## Algorithmic structure

All algorithms in the `core/` directory follow a consistent pattern to ensure they can be easily used by the API layer.

1.  **Class-based implementation:** Each algorithm is implemented as a Python class (e.g., `SVM`, `NeuralNetwork`).
2.  **`fit(X, y)` method:** This method performs the core training logic and returns a list of dictionaries containing epoch-by-epoch metrics.
3.  **`predict(X)` method:** This method takes a batch of input points and returns their predicted class labels.
4.  **NumPy-based computations:** Most mathematical operations are vectorized using NumPy to ensure high performance.

## Implementation details

The backend uses several specific techniques to improve performance and accuracy:

- **Vectorized operations:** Rather than using Python loops, most calculations (like dot products and matrix operations) are performed using NumPy's vectorized functions.
- **Stateful model storage:** The `trained_model` global variable in `api.py` stores the latest trained model, allowing the `/predict` endpoint to generate results without re-training.
- **Error handling:** The API includes robust error handling to catch issues with invalid hyperparameter values or data distributions.

<!-- prettier-ignore -->
> [!NOTE]
> The backend does not persist models to disk. Every time you restart the FastAPI server, any previously trained models are cleared from memory.

## Next steps

If you are interested in contributing new algorithms or features to Optimizer-Lens, refer to the **Contributing documentation**.
