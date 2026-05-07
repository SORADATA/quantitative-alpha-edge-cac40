# 🚀 Contribution Guide - AlphaEdge

Thank you for contributing to **AlphaEdge**! To maintain a robust pipeline and stay true to industry standards (inspired by **Qlib**), please follow these guidelines.

## 🌿 Branch Management

The project uses a two-layer branch model to ensure pipeline stability:
*   **`dev`**: The primary development branch. All new features (Alpha158, new models) and most bug fixes are merged here.
*   **`main`**: The stable branch used for production deployments and Hugging Face dataset synchronization.

## 🛠️ Development Workflow

To contribute code:
1.  **Fork**: Create a copy of the project on your GitHub account.
2.  **Feature Branch**: Implement your changes in a specific branch within your fork.
3.  **Respect Modularity**:
    *   **Signals**: New signal types (Models, Cache) must be implemented in `src/strategy/signals.py`.
    *   **Backtest**: Any changes regarding market friction, slippage, or execution logic belong in `src/pipeline/backtest.py`.
    *   **Features**: Integration of the 158 features (Alpha158) must be done in a vectorized manner to remain compatible with the `AlphaSignal` cache.

## 🧪 Testing & Validation

All changes must be validated before merging:
*   **Unit Tests**: Run `python -m unittest discover -s tests`.
*   **Realistic Backtest**: Ensure your changes do not introduce survivorship bias or ignore transaction costs (**Turnover Friction**).
*   **Signal Cache**: Verify that the `AlphaSignal` object is correctly instantiated via its class method to prevent attribute errors.

## 💡 Git Best Practices

We aim to keep the commit history compact and readable:
*   **Commit Messages**: Use a short summary followed by a detailed description if necessary (`git commit -m "feat: add alpha158" -m "detailed window description"`).
*   **Squash**: Combine multiple small commits into a single clean commit before submitting a Pull Request.
*   **Rebase**: Prefer `git rebase` to keep your branches up to date with the `dev` branch.

## 📖 Documentation

The project documentation is automatically generated from the code. If you modify a function, update its docstrings and check the documentation locally.

---

**Questions?** Check the open issues on the repository or refer to the internal architecture guides in the `src/` directory.