# Contributing

We welcome contributions to Optimizer-Lens! Whether you are interested in implementing new machine learning algorithms, improving the frontend visualization, or fixing bugs, your help is appreciated.

To maintain the quality and consistency of the project, please follow these guidelines when contributing.

## Development workflow

The best way to contribute is to follow a systematic approach to development and testing.

1.  **Fork and clone:** Fork the repository on GitHub and clone it to your local machine.
2.  **Environment setup:** Follow the steps in the **Getting started** guide to set up your backend and frontend environments.
3.  **Create a branch:** Work on a new branch for your specific feature or fix.
4.  **Implement your changes:** Write clear, concise, and documented code.
5.  **Test your changes:** Ensure your changes work as expected in the playground and don't introduce regressions.
6.  **Submit a pull request:** Provide a detailed description of your changes and why they are necessary.

## Algorithmic contributions

When adding a new machine learning algorithm to the `core/` package:

- **Inherit from standard patterns:** Use a class-based structure with `fit(X, y)` and `predict(X)` methods.
- **Pure NumPy:** Prioritize the use of NumPy for all mathematical operations. Avoid introducing heavy external dependencies like scikit-learn or TensorFlow.
- **Return metrics:** Ensure the `fit` method returns a history of epoch-by-epoch metrics to support the playground's visualization.
- **Update the API:** Register your new algorithm in `core/api.py` and ensure the `TrainRequest` model includes any new hyperparameters.

## Frontend contributions

When contributing to the Next.js frontend:

- **Follow the component pattern:** Use shadcn/ui components where possible to maintain aesthetic consistency.
- **State management:** Use React hooks responsibly and avoid unnecessary re-renders.
- **Visualization:** If you are modifying the canvas rendering logic, ensure it remains performant even with large datasets.
- **Type safety:** Always use TypeScript for new components and utility functions.

## Code standards

To ensure the codebase remains maintainable:

- **Style:** Use consistent indentation and naming conventions.
- **Comments:** Provide comments for complex mathematical logic or architectural decisions.
- **Documentation:** If you add a new feature or hyperparameter, update the corresponding documentation files in the `docs/` folder.

<!-- prettier-ignore -->
> [!NOTE]
> All pull requests will be reviewed for clarity, performance, and adherence to the project's educational goals.

## Reporting issues

If you find a bug or have a feature request, please open an issue on the GitHub repository. Provide as much detail as possible, including steps to reproduce the issue and your environment details.
