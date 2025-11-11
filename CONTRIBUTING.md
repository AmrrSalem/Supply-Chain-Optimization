# Contributing to Supply Chain Optimization

Thank you for your interest in contributing to the Supply Chain Inventory Optimization System! We welcome contributions from the community.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [How to Contribute](#how-to-contribute)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## 📜 Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please be respectful and constructive in all interactions.

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. We pledge to:

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Basic understanding of supply chain optimization concepts
- Familiarity with pandas, NumPy, and Streamlit

### First Time Contributors

If you're new to open source, here are some good first issues:

- Documentation improvements
- Adding tests
- Fixing typos
- Improving error messages
- Adding examples

Look for issues labeled `good first issue` or `help wanted`.

## 💻 Development Setup

1. **Fork the repository** on GitHub

2. **Clone your fork:**
   ```bash
   git clone https://github.com/YOUR_USERNAME/Supply-Chain-Optimization.git
   cd Supply-Chain-Optimization
   ```

3. **Add upstream remote:**
   ```bash
   git remote add upstream https://github.com/AmrrSalem/Supply-Chain-Optimization.git
   ```

4. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

5. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

6. **Install pre-commit hooks:**
   ```bash
   pre-commit install
   ```

7. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

## 🤝 How to Contribute

### Types of Contributions

**🐛 Bug Reports**
- Use the bug report template
- Include reproduction steps
- Provide system information
- Add error messages/logs

**✨ Feature Requests**
- Use the feature request template
- Explain the use case
- Describe expected behavior
- Consider implementation approach

**📝 Documentation**
- Fix typos and grammar
- Add examples
- Improve clarity
- Add missing documentation

**🧪 Tests**
- Add tests for uncovered code
- Improve test coverage
- Add integration tests
- Fix flaky tests

**🔧 Code Contributions**
- Bug fixes
- New features
- Performance improvements
- Refactoring

## 📏 Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with these specific guidelines:

**General:**
- Maximum line length: 100 characters
- Use 4 spaces for indentation (no tabs)
- Use descriptive variable names
- Add docstrings to all functions and classes

**Formatting:**
We use `black` for automatic code formatting:
```bash
black .
```

**Linting:**
We use `flake8` for linting:
```bash
flake8 .
```

**Type Hints:**
All functions should have type hints:
```python
def calculate_eoq(
    demand: float,
    order_cost: float,
    holding_cost: float
) -> float:
    """Calculate Economic Order Quantity."""
    return np.sqrt(2 * demand * order_cost / holding_cost)
```

**Imports:**
Organize imports in this order:
1. Standard library
2. Third-party packages
3. Local modules

```python
# Standard library
import os
from typing import Dict, List

# Third-party
import pandas as pd
import numpy as np

# Local
from config import settings
from utils.validators import validate_input
```

### Documentation Standards

**Docstrings:**
Use NumPy style docstrings:

```python
def optimize_inventory(
    products: pd.DataFrame,
    service_level: float = 0.95
) -> pd.DataFrame:
    """
    Optimize inventory levels for multiple products.

    Parameters
    ----------
    products : pd.DataFrame
        DataFrame containing product information with columns:
        product_id, demand, unit_cost, order_cost, lead_time
    service_level : float, default=0.95
        Target service level (0-1)

    Returns
    -------
    pd.DataFrame
        DataFrame with optimized order quantities and costs

    Raises
    ------
    ValueError
        If service_level is not between 0 and 1
    ValidationError
        If products DataFrame is missing required columns

    Examples
    --------
    >>> products = pd.DataFrame({
    ...     'product_id': ['A', 'B'],
    ...     'demand': [100, 200],
    ...     'unit_cost': [10, 20],
    ...     'order_cost': [50, 75],
    ...     'lead_time': [7, 14]
    ... })
    >>> results = optimize_inventory(products, service_level=0.95)
    >>> print(results['optimized_quantity'].sum())

    Notes
    -----
    This function uses the EOQ formula combined with safety stock
    calculations based on the specified service level.

    References
    ----------
    .. [1] Harris, F. W. (1913). "How many parts to make at once"
    """
    pass
```

### Code Organization

**File Structure:**
```
module_name.py
├── Imports
├── Constants
├── Type definitions
├── Helper functions
├── Main classes
└── Main execution (if __name__ == "__main__")
```

**Class Structure:**
```python
class ClassName:
    """Class docstring."""

    # Class variables
    class_var: ClassVar[int] = 10

    def __init__(self, ...):
        """Initialize."""
        # Instance variables
        self.instance_var = ...

    # Public methods
    def public_method(self):
        """Public method."""
        pass

    # Private methods
    def _private_method(self):
        """Private helper method."""
        pass

    # Magic methods last
    def __str__(self):
        """String representation."""
        return f"ClassName(...)"
```

## 🧪 Testing Guidelines

### Writing Tests

**Test Organization:**
```
tests/
├── unit/              # Unit tests
│   ├── test_optimization.py
│   └── test_calculations.py
├── integration/       # Integration tests
│   └── test_workflow.py
├── fixtures/          # Test data
│   └── sample_data.csv
└── conftest.py       # Shared fixtures
```

**Test Naming:**
```python
def test_should_calculate_eoq_correctly():
    """Test EOQ calculation with known inputs."""
    # Given
    demand = 1000
    order_cost = 50
    holding_cost = 5

    # When
    eoq = calculate_eoq(demand, order_cost, holding_cost)

    # Then
    expected = 141.42
    assert abs(eoq - expected) < 0.01
```

**Test Coverage:**
- Aim for 80%+ code coverage
- Test edge cases and error conditions
- Use parametrized tests for multiple scenarios

```python
import pytest

@pytest.mark.parametrize("demand,order_cost,holding_cost,expected", [
    (1000, 50, 5, 141.42),
    (2000, 100, 10, 200.00),
    (500, 25, 2.5, 100.00),
])
def test_eoq_calculation(demand, order_cost, holding_cost, expected):
    """Test EOQ with multiple parameter sets."""
    result = calculate_eoq(demand, order_cost, holding_cost)
    assert abs(result - expected) < 0.01
```

**Running Tests:**
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_optimization.py

# Run specific test
pytest tests/unit/test_optimization.py::test_eoq_calculation

# Run tests matching pattern
pytest -k "eoq"

# Run with verbose output
pytest -v
```

### Test Requirements

- All new features must include tests
- Bug fixes must include regression tests
- Maintain or improve code coverage
- All tests must pass before PR merge

## 🔄 Pull Request Process

### Before Submitting

1. **Update your branch:**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks:**
   ```bash
   # Format code
   black .

   # Check linting
   flake8 .

   # Type checking
   mypy .

   # Run tests
   pytest --cov

   # Security check
   bandit -r .
   ```

3. **Update documentation:**
   - Update README if needed
   - Add docstrings to new functions
   - Update CHANGELOG.md

4. **Commit your changes:**
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

   Use conventional commit format:
   - `feat:` New feature
   - `fix:` Bug fix
   - `docs:` Documentation changes
   - `test:` Adding tests
   - `refactor:` Code refactoring
   - `perf:` Performance improvement
   - `chore:` Maintenance tasks

### Submitting the PR

1. **Push your branch:**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request on GitHub:**
   - Use the PR template
   - Reference related issues
   - Provide clear description
   - Add screenshots if UI changes
   - Request review from maintainers

3. **PR Checklist:**
   - [ ] Code follows style guidelines
   - [ ] Tests added/updated
   - [ ] All tests pass
   - [ ] Documentation updated
   - [ ] No merge conflicts
   - [ ] Commits are clean and descriptive

### Review Process

- Maintainers will review your PR
- Address feedback promptly
- Keep discussion constructive
- Update PR based on feedback
- Once approved, PR will be merged

## 📝 Issue Guidelines

### Creating an Issue

**Bug Reports should include:**
- Clear, descriptive title
- Steps to reproduce
- Expected behavior
- Actual behavior
- System information (OS, Python version)
- Error messages/stack traces
- Screenshots if applicable

**Feature Requests should include:**
- Clear use case
- Proposed solution
- Alternative approaches considered
- Impact on existing functionality

### Issue Labels

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention needed
- `priority:high`: High priority
- `wontfix`: This will not be worked on

## 🌳 Branching Strategy

- `main`: Production-ready code
- `develop`: Integration branch
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `hotfix/*`: Critical production fixes
- `docs/*`: Documentation changes

## 📦 Release Process

1. Update version in `__init__.py`
2. Update CHANGELOG.md
3. Create release branch
4. Tag release: `git tag v1.0.0`
5. Push tag: `git push origin v1.0.0`
6. Create GitHub release with notes

## 🎯 Development Priorities

### Current Focus Areas

1. **Phase 1:** Testing infrastructure and documentation
2. **Phase 2:** CI/CD and security
3. **Phase 3:** Performance optimization
4. **Phase 4:** Advanced features

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed roadmap.

## 💬 Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and ideas
- **Pull Requests**: Code contributions

## 🙏 Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Acknowledged in documentation

## 📚 Resources

- [Python Style Guide (PEP 8)](https://www.python.org/dev/peps/pep-0008/)
- [NumPy Docstring Guide](https://numpydoc.readthedocs.io/en/latest/format.html)
- [pytest Documentation](https://docs.pytest.org/)
- [Git Best Practices](https://git-scm.com/book/en/v2)

## ❓ Questions?

If you have questions:
1. Check the documentation
2. Search existing issues
3. Create a new issue with the `question` label
4. Join our discussions

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to Supply Chain Optimization! 🎉**

*Together, we're making inventory optimization accessible to everyone.*
