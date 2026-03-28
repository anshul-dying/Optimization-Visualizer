# Architecture

This section provides a comprehensive deep-dive into the High-Level and Low-Level design of Optimizer-Lens. The system is architected as a decoupled, asynchronous machine learning environment where the frontend handles the visual "playback" of training history while the backend serves as a high-performance mathematical engine.

## High-Level Design (HLD)

The HLD illustrates the "Bridge Pattern" between the Next.js runtime and the Python mathematical core.

### System topology & data flow

This diagram tracks a single training request from the user's interaction in the UI through the computation layer and back to the canvas renderer.

```mermaid
sequenceDiagram
    participant U as User
    participant FE as Frontend (React/Next.js)
    participant API as FastAPI Backend
    participant CORE as NumPy Engine

    U->>FE: Set Hyperparameters & Click "Train"
    Note over FE: Generate Synthetic Dataset (X, y)
    FE->>API: POST /train (JSON Payload)
    Note right of API: Pydantic Validation (TrainRequest)
    API->>CORE: Initialize Model (e.g., SVM)
    CORE->>CORE: fit(X, y) - Epoch-by-Epoch Iteration
    CORE-->>API: Training History (Loss, Accuracy, Weights)
    API-->>FE: HTTP Response (JSON History Buffer)
    loop Animation Playback
        FE->>FE: Update React State (currentEpoch)
        FE->>FE: Re-render HTML5 Canvas (Decision Boundary)
        FE->>FE: Update Recharts (Loss/Acc Curves)
    end
    FE-->>U: Visual Convergence Complete
```

### Component interaction layer

```mermaid
graph LR
    subgraph Browser [Browser Layer]
        UI[shadcn/ui Configuration]
        State[React Hooks / State]
        Renderer[Canvas Rendering Engine]
    end

    subgraph Transport [Transport Layer]
        Fetch[Async Fetch / JSON]
    end

    subgraph Python [Computation Layer]
        F_API[FastAPI Endpoints]
        M_REG[Model Registry]
        NUMPY[Vectorized NumPy Core]
    end

    UI --> State
    State --> Fetch
    Fetch --> F_API
    F_API --> M_REG
    M_REG --> NUMPY
    NUMPY -- History --> F_API
    F_API -- JSON --> Fetch
    Fetch -- State Update --> State
    State --> Renderer
```

## Low-Level Design (LLD)

The LLD focuses on the internal mechanics of the NumPy engine and how the frontend renders non-linear boundaries.

### Core class hierarchy

Our Python backend uses a consistent interface for all machine learning models, allowing the API to handle diverse algorithms (Linear, SVM, Trees, NN) through a unified call pattern.

```mermaid
classDiagram
    class BaseModel {
        <<interface>>
        +fit(X, y) history
        +predict(X) labels
    }

    class LinearModels {
        +weights array
        +bias float
        +learning_rate float
        -_compute_gradient()
    }

    class KernelSVM {
        +C float
        +kernel_type string
        +alpha array
        +support_vectors array
        -_kernel(x1, x2)
        -smo_optimize()
    }

    class NeuralNetwork {
        +layers Layer[]
        +activation string
        +optimizer string
        -_forward_pass()
        -_backpropagate()
    }

    BaseModel <|-- LinearModels
    BaseModel <|-- KernelSVM
    BaseModel <|-- NeuralNetwork
```

### Boundary rendering strategy

One of the most complex parts of the frontend is rendering non-linear decision boundaries for algorithms like SVM and Neural Networks.

```mermaid
flowchart TD
    Start[Request Decision Regions] --> Grid[Create 50x50 Coordinate Grid]
    Grid --> Batch[Batch Points into X_grid]
    Batch --> API[POST /predict]
    API --> Res[Receive 2500 Predictions]
    Res --> Heatmap[Generate Heatmap Matrix]
    Heatmap --> Draw[Render Transparent Rects on Canvas]
    Draw --> Points[Overlay Class Data Points]
```

## Implementation principles

- **Vectorization over Iteration:** All mathematical operations in `core/` use NumPy's vectorized broadcasting rather than Python loops to minimize overhead during training.
- **Stateful API:** The backend maintains the `trained_model` in global memory, ensuring that the `/predict` endpoint is always instantaneous for a given dataset.
- **Canvas Buffering:** The frontend uses an off-screen canvas reference to calculate boundary positions before flushing them to the main UI thread, preventing flicker during high-speed training playback.

<!-- prettier-ignore -->
> [!TIP]
> To see these principles in action, open the browser's **Network Tab** while clicking "Train". You can inspect the massive JSON history buffer that fuels the real-time animations.
