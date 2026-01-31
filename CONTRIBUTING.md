# Contributing to NFMR

Thank you for your interest in contributing to NFMR! This document provides guidelines for contributing to the project.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
   ```bash
   git clone https://github.com/yourusername/nfmr.git
   cd nfmr
   ```

3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install development dependencies**:
   ```bash
   pip install -e ".[all]"
   ```

5. **Set up pre-commit hooks** (optional but recommended):
   ```bash
   pip install pre-commit
   pre-commit install
   ```

## Development Workflow

### Branching Strategy

- `main` - Stable release branch
- `develop` - Development branch for integration
- `feature/*` - Feature branches
- `bugfix/*` - Bug fix branches
- `docs/*` - Documentation updates

### Creating a Feature Branch

```bash
git checkout develop
git pull origin develop
git checkout -b feature/your-feature-name
```

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **isort** for import sorting
- **flake8** for linting
- **mypy** for type checking

Run all checks:
```bash
# Format code
black src tests
isort src tests

# Lint
flake8 src tests

# Type check
mypy src
```

### Writing Tests

- All new features should include tests
- Tests should be placed in the `tests/` directory
- Use pytest for testing
- Aim for high test coverage

Run tests:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_ontology.py -v
```

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Examples:
```
feat(core): add support for ALCH expressivity
fix(normalizer): handle edge case in definer creation
docs(readme): update installation instructions
```

## Pull Request Process

1. **Update your branch** with the latest changes from `develop`:
   ```bash
   git fetch origin
   git rebase origin/develop
   ```

2. **Push your changes**:
   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request** on GitHub:
   - Use a clear, descriptive title
   - Reference any related issues
   - Provide a detailed description of changes
   - Include screenshots/examples if applicable

4. **Address review feedback**:
   - Make requested changes
   - Push additional commits
   - Re-request review when ready

5. **Merge requirements**:
   - All CI checks must pass
   - At least one approval from maintainers
   - No unresolved conversations

## Code Review Guidelines

When reviewing PRs, consider:

- **Correctness**: Does the code work as intended?
- **Design**: Is the code well-structured and maintainable?
- **Tests**: Are there adequate tests?
- **Documentation**: Is the code well-documented?
- **Performance**: Are there any performance concerns?

## Reporting Issues

### Bug Reports

When reporting bugs, include:

1. **Description**: Clear description of the bug
2. **Steps to Reproduce**: Minimal steps to reproduce
3. **Expected Behavior**: What you expected to happen
4. **Actual Behavior**: What actually happened
5. **Environment**: Python version, OS, package versions
6. **Error Messages**: Full error traceback if applicable

### Feature Requests

When requesting features, include:

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: Your suggested approach
3. **Alternatives**: Other approaches you've considered
4. **Use Cases**: Example use cases

## Project Structure

```
nfmr/
├── src/
│   ├── core/           # Core forgetting algorithm
│   ├── rag/            # RAG pipeline components
│   ├── evaluation/     # Evaluation framework
│   └── utils/          # Utility functions
├── tests/              # Test files
├── docs/               # Documentation
├── experiments/        # Experiment scripts
├── data/               # Sample data
└── configs/            # Configuration files
```

## Documentation

- Use docstrings for all public functions and classes
- Follow Google-style docstrings
- Update README.md for user-facing changes
- Add type hints to all function signatures

Example docstring:
```python
def compute_module(
    kb: OntologyKB,
    target_vocab: Set[str],
    timeout: int = 300
) -> RetrievalResult:
    """
    Compute a zero-noise module for the given target vocabulary.
    
    This method eliminates all symbols not in the target vocabulary
    while preserving semantic equivalence over the target signature.
    
    Args:
        kb: The input knowledge base.
        target_vocab: Set of concept and role names to keep.
        timeout: Maximum time in seconds for the operation.
    
    Returns:
        RetrievalResult containing the computed module and metadata.
    
    Raises:
        TimeoutError: If the operation exceeds the timeout.
        MemoryError: If memory limit is exceeded.
    
    Example:
        >>> kb = OntologyKB.from_owl("ontology.owl")
        >>> result = compute_module(kb, {"Disease", "Symptom"})
        >>> print(result.module)
    """
```

## Questions?

If you have questions about contributing, feel free to:

- Open an issue with the `question` label
- Reach out to the maintainers

Thank you for contributing to NFMR! 🎉
