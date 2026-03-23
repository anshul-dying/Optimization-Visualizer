from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import numpy as np
from core.linear_regression import LinearRegression
from core.logistic import LogisticRegression
from core.svm import SVM
from core.neural_network import NeuralNetwork
from core.decision_tree import DecisionTree
from core.random_forest import RandomForest
from core.gradient_boosting import GradientBoosting

app = FastAPI()

# Global variable to store the latest trained model
trained_model = None

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TrainRequest(BaseModel):
    algorithm: str
    X: List[List[float]]
    y: List[float]
    learning_rate: Optional[float] = 0.01
    epochs: Optional[int] = 100
    batch_size: Optional[int] = 32
    C: Optional[float] = 1.0
    kernel: Optional[str] = 'linear'
    gamma: Optional[Any] = 'scale'
    degree: Optional[int] = 3
    hidden_layers: Optional[List[int]] = [8, 1]
    activation: Optional[str] = 'relu'
    n_estimators: Optional[int] = 10
    max_depth: Optional[int] = 10
    min_samples_split: Optional[int] = 2

class PredictRequest(BaseModel):
    X: List[List[float]]

@app.get("/")
async def root():
    return {"status": "ok", "message": "Optimizer Lens API is running"}

@app.post("/train")
async def train(request: TrainRequest):
    global trained_model
    X = np.array(request.X)
    y = np.array(request.y)
    lr = request.learning_rate
    epochs = request.epochs
    
    if request.algorithm == "linear":
        model = LinearRegression(learning_rate=lr, epochs=epochs)
        history = model.fit(X, y)
        trained_model = model
        preds = model.predict(X)
        return {
            "metrics": history,
            "weights": model.weights.tolist(),
            "bias": float(model.bias),
            "final_loss": history[-1]["loss"],
            "final_accuracy": history[-1]["accuracy"],
            "predictions": preds
        }
    
    elif request.algorithm == "logistic":
        model = LogisticRegression(learning_rate=lr, epochs=epochs)
        history = model.fit(X, y)
        trained_model = model
        preds = model.predict(X)
        return {
            "metrics": history,
            "weights": model.weights.tolist(),
            "bias": float(model.bias),
            "final_loss": history[-1]["loss"],
            "final_accuracy": history[-1]["accuracy"],
            "predictions": preds
        }

    elif request.algorithm == "svm" or request.algorithm == "kernel_svm":
        model = SVM(C=request.C, kernel=request.kernel, degree=request.degree, gamma=request.gamma, max_iter=epochs)
        history = model.fit(X, y)
        trained_model = model
        preds = model.predict(X)
        response = {
            "metrics": history,
            "final_loss": history[-1]["loss"],
            "final_accuracy": history[-1]["accuracy"],
            "predictions": preds
        }
        if request.kernel == 'linear':
            w_data = model.get_weights_linear()
            if w_data:
                response["weights"], response["bias"] = w_data
        return response

    elif request.algorithm == "neural_network":
        layers = [X.shape[1]] + request.hidden_layers
        if layers[-1] != 1: layers.append(1)
        model = NeuralNetwork(layers=layers, learning_rate=lr, epochs=epochs, activation=request.activation)
        history = model.fit(X, y)
        trained_model = model
        preds = model.predict(X)
        return {
            "metrics": history,
            "final_loss": history[-1]["loss"],
            "final_accuracy": history[-1]["accuracy"],
            "predictions": preds
        }

    elif request.algorithm == "decision_tree":
        model = DecisionTree(max_depth=request.max_depth, min_samples_split=request.min_samples_split)
        history = model.fit(X, y)
        trained_model = model
        preds = model.predict(X)
        return {
            "metrics": history,
            "final_loss": history[-1]["loss"],
            "final_accuracy": history[-1]["accuracy"],
            "predictions": preds
        }

    elif request.algorithm == "random_forest":
        model = RandomForest(n_estimators=request.n_estimators, max_depth=request.max_depth, min_samples_split=request.min_samples_split)
        history = model.fit(X, y)
        trained_model = model
        preds = model.predict(X)
        return {
            "metrics": history,
            "final_loss": history[-1]["loss"],
            "final_accuracy": history[-1]["accuracy"],
            "predictions": preds
        }

    elif request.algorithm == "gradient_boosting":
        model = GradientBoosting(n_estimators=request.n_estimators, learning_rate=lr, max_depth=request.max_depth)
        history = model.fit(X, y)
        trained_model = model
        preds = model.predict(X)
        return {
            "metrics": history,
            "final_loss": history[-1]["loss"],
            "final_accuracy": history[-1]["accuracy"],
            "predictions": preds
        }
    
    raise HTTPException(status_code=400, detail=f"Algorithm {request.algorithm} not implemented")

@app.post("/predict")
async def predict(request: PredictRequest):
    global trained_model
    if trained_model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet")
    
    X = np.array(request.X)
    preds = trained_model.predict(X)
    return {"predictions": preds}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
