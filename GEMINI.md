# Optimizer-Lens: Interactive ML Playground

Optimizer-Lens is a comprehensive, full-stack interactive playground designed to visualize and test machine learning algorithms. It bridges a modern Next.js frontend with a custom Python-based backend to provide real-time, epoch-by-epoch visualizations of model training.

## Project Overview

*   **Frontend**: Built with **Next.js 14**, **TypeScript**, **Tailwind CSS**, and **shadcn/ui**. It provides a rich, interactive UI for dataset generation, parameter configuration, and real-time canvas-based visualizations.
*   **Backend**: A **FastAPI** application (`core/api.py`) that implements various ML algorithms from scratch using **NumPy**. 
*   **Algorithms**: Includes Linear Regression, Logistic Regression, SVM (with Linear/RBF/Poly kernels), Neural Networks, Decision Trees, Random Forests, and Gradient Boosting.
*   **Key Feature**: Real-time "playback" of training history, allowing users to see decision boundaries converge epoch-by-epoch.

## Building and Running

### Prerequisites
*   Node.js (v18+) and **pnpm** (recommended) or npm.
*   Python (3.8+) and pip.

### Backend Setup
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Start the API server:
    ```bash
    python -m core.api
    ```
    The server runs on `http://localhost:8000` by default.

### Frontend Setup
1.  Install dependencies:
    ```bash
    pnpm install
    # OR
    npm install
    ```
2.  Configure environment variables (optional, defaults to localhost:8000):
    Create a `.env.local` file:
    ```text
    NEXT_PUBLIC_SERVER_URL=http://localhost:8000
    ```
3.  Run the development server:
    ```bash
    pnpm dev
    # OR
    npm run dev
    ```
    Access the playground at `http://localhost:3000/playground`.

## Development Conventions

*   **Frontend Architecture**: Follows the Next.js `app/` directory structure. UI components are located in `components/` and use the shadcn/ui pattern.
*   **State Management**: Uses React hooks (`useState`, `useEffect`, `useRef`) for local state and real-time animation control.
*   **Styling**: Uses **Tailwind CSS** for styling, with `cn` utility for class merging.
*   **Backend Structure**: All ML logic resides in the `core/` package. Each algorithm is implemented as a standalone class with a `.fit()` method that returns a training history.
*   **API Communication**: The frontend communicates with the backend via the `/train` POST endpoint, sending dataset coordinates and hyperparameters.
*   **Visualizations**: Real-time rendering is handled via HTML5 Canvas in the `MLPlayground` component.

## Key Files
*   `core/api.py`: FastAPI entry point and endpoint definitions.
*   `components/ml-playground.tsx`: The heart of the frontend, managing dataset generation, training state, and canvas rendering.
*   `core/`: Contains the NumPy-based algorithm implementations.
*   `app/playground/page.tsx`: The main route for the interactive playground.
