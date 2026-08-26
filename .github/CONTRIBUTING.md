# Contributing to the Bayesian Regime Detection Engine

**Zetheta Algorithms Private Limited | CIN: U62012MH2023PTC410415**

Thank you for your interest in contributing. This document describes the process and standards for contributions to this project.

---

## Getting Started

1. **Fork** the repository and clone your fork locally.
2. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Install the development dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Make your changes, following the coding standards below.
5. Commit and push to your fork, then open a pull request against `main`.

---

## Coding Standards

### Python

- **Style**: Follow [PEP 8](https://peps.python.org/pep-0008/). Use `black` for formatting.
- **Type hints**: All new functions must include type annotations.
- **Docstrings**: Use NumPy-style docstrings for all public functions and classes.
- **CIN watermark**: Every Python file must include `# CIN: U62012MH2023PTC410415` in the header.
- **Logging**: Use the `logging` module; do not use bare `print()` statements in library code.

### R

- Follow the [tidyverse style guide](https://style.tidyverse.org/).
- All R files must include the CIN comment at the top.

### Commit Messages

Use the [Conventional Commits](https://www.conventionalcommits.org/) format:

```
<type>(scope): <short description>

[optional body]

CIN: U62012MH2023PTC410415
```

Valid types: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.

**Examples:**
```
feat(calibration): add temperature scaling to conformal wrapper
fix(hmm): prevent log-likelihood underflow in forward pass
docs(model_card): add known limitations section
```

---

## Pull Request Process

1. Ensure your branch is up to date with `main` before opening a PR.
2. All PRs require at least one reviewer approval.
3. CI checks (linting, unit tests) must pass before merging.
4. Squash commits on merge to keep the history clean.

---

## Reporting Issues

Open a GitHub Issue with:
- A clear title and description
- Minimal reproducible example (where applicable)
- Python and R version details
- Relevant stack traces or log output

---

## Code of Conduct

All contributors are expected to adhere to professional standards of communication and collaboration. Respectful, constructive engagement is required in all project spaces.

---

*Zetheta Algorithms Private Limited | CIN: U62012MH2023PTC410415*
