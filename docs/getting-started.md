# Getting started

Setting up Optimizer-Lens involves configuring both the Python-based backend and the Next.js frontend to work together. This guide walks you through the installation and execution steps for both environments.

Optimizer-Lens requires a local Python environment for the mathematical computations and a Node.js environment for the interactive UI. You must have both running to use the "Real Algorithms" mode in the playground.

## Prerequisites

Before starting the installation, ensure you have the following software installed on your machine:

- **Node.js:** version 18.0 or later.
- **pnpm:** (recommended) or **npm**.
- **Python:** version 3.8 or later.
- **pip:** Python package installer.

## Backend setup

The backend handles the core machine learning logic using FastAPI and NumPy. Follow these steps to set it up:

1.  Navigate to the project root directory.
2.  Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Start the FastAPI server:

    ```bash
    python -m core.api
    ```

    The server runs on `http://localhost:8000` by default. You should see a status message confirming that the API is running.

## Frontend setup

The frontend provides the interactive user interface for dataset manipulation and visualization. Follow these steps to set it up:

1.  Open a new terminal window and navigate to the project root directory.
2.  Install the Node.js dependencies:

    ```bash
    pnpm install
    # OR
    npm install
    ```

3.  Configure the environment variables (optional):

    Create a `.env.local` file in the root directory if you want to override the default backend URL:

    ```text
    NEXT_PUBLIC_SERVER_URL=http://localhost:8000
    ```

4.  Run the development server:

    ```bash
    pnpm dev
    # OR
    npm run dev
    ```

    Access the application by navigating to `http://localhost:3000/playground` in your web browser.

## Verifying the connection

Once both the backend and frontend are running, you can verify the connection in the playground:

1.  Look for the **Configuration** panel on the left side of the playground.
2.  Find the **Use Real Algorithms** toggle.
3.  Check the status message below the toggle. It should show a green checkmark indicating "API available."

<!-- prettier-ignore -->
> [!TIP]
> If the API is unavailable, ensure the FastAPI server is running on the correct port and that your browser is not blocking cross-origin requests.

## Next steps

Once you have the project running, explore the **Algorithms** documentation to understand the mathematical models behind the visualizations.
