# Pytest Test Suite for Quantum Diffusion Project

This guide assumes you're using [uv](https://github.com/astral-sh/uv) for Python package management.

## Installation

### Option 1: Install with all dependencies (Recommended)
```bash
cd /home/palash/diffusion
uv sync  # Install all dependencies from pyproject.toml
```

**If you get `uv: command not found` error**, install uv first:
```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
# Then sync dependencies
uv sync
```

### Option 2: Install pytest only (if not using full sync)
```bash
uv add --dev pytest>=7.0.0 pytest-cov>=4.0.0
```

## Running Tests

### Run all tests
```bash
pytest
pytest -v  # Verbose output
pytest -vv  # Extra verbose
```

### Run tests without slow tests (recommended for quick feedback)
```bash
pytest -m "not slow"
```

### Run specific test file
```bash
pytest tests/test_models.py
pytest tests/test_loss.py
pytest tests/test_functions.py
pytest tests/test_utils.py
pytest tests/test_integration.py
```

### Run specific test class
```bash
pytest tests/test_models.py::TestGeneralDynamicUNet
pytest tests/test_loss.py::TestPurity
```

### Run specific test function
```bash
pytest tests/test_models.py::TestGeneralDynamicUNet::test_unet_instantiation
pytest tests/test_loss.py::TestDensityMatrixCreation::test_density_matrix_hermitian
```

### Run with coverage report
```bash
pytest --cov=src --cov-report=html
# Open htmlcov/index.html in a browser
```

### Run with minimal output
```bash
pytest -q
pytest -q -m "not slow"
```

### Run tests in parallel (requires pytest-xdist)
```bash
uv add --dev pytest-xdist
pytest -n auto  # Use all CPU cores
```

## Test Structure

### conftest.py
Contains shared pytest fixtures used across all tests:
- `device`: Returns GPU if available, else CPU
- `random_seed`: Sets reproducible random seeds
- `vec_len`: Vector length for quantum states (D^N)
- `batch_size`: Default batch size for testing
- `random_state`: Random quantum state fixture
- `random_complex_state`: Random complex quantum state
- `pure_state`: Simple normalized pure state
- `time_batch`: Batch of time values for diffusion model

### test_models.py (17 tests)
Tests for neural network architectures:
- **TestGeneralDynamicUNet** (7 tests)
  - Model instantiation and configuration
  - Forward pass shape correctness
  - Gradient flow through model
  - Batch processing with various sizes
  - Time normalization handling
  - Evaluation mode consistency
  - Output dtype preservation

- **TestQuantumDiffusionMLP** (10 tests)
  - Model instantiation
  - Forward pass shapes
  - Gradient flow
  - Time input handling (scalar, 1D, batch)
  - Various batch sizes
  - Custom hidden dimensions
  - Output dtype preservation

### test_loss.py (13 tests)
Tests for loss functions and quantum metrics:
- **TestDensityMatrixCreation** (5 tests)
  - Correct tensor shapes
  - Hermiticity verification
  - Trace normalization to 1
  - Positive semidefiniteness
  - Single state handling

- **TestPurity** (6 tests)
  - Output shape correctness
  - Pure state purity = 1
  - Maximally mixed state purity
  - Valid purity range
  - Real-valued output
  - Range validation

- **TestIsPure** (2 tests)
  - Pure state detection
  - Tolerance parameter handling

- **TestPartialTraceLoss** (7 tests)
  - Scalar output verification
  - Non-negative loss values
  - Pure state behavior
  - Batch processing
  - Gradient flow
  - Deterministic computation

### test_functions.py (12 tests) 
Tests for core generation and training functions:
- **TestIndexingAndShapes** (3 tests)
  - Input shape handling (1D, 2D)
  - Output type validation
  - Value range checking

- **TestFineTuningWithLBFGS** (4 tests)
  - Output shape preservation
  - Loss reduction verification
  - State normalization preservation
  - Max iterations parameter

- **TestTrainingDataLoading** (3 tests)
  - Nonexistent data handling
  - Valid dataset structure
  - Tensor shape validation

- **TestGenerateAMEState** (5 tests) 
  - **SLOW** Basic state generation
  - **SLOW** Generated state normalization
  - **SLOW** Generation with L-BFGS fine-tuning
  - **SLOW** Different guidance scales
  - **SLOW** AME properties verification

### test_utils.py (9 tests)
Tests for utility functions:
- **TestDataGeneration** (5 tests)
  - **SLOW** Dataset structure validation
  - **SLOW** Tensor shape checking
  - **SLOW** File saving verification
  - **SLOW** Config dimension respect
  - Nonexistent data loading
  - **SLOW** Saved dataset loading

- **TestBenchmarking** (4 tests)
  - **SLOW** Basic benchmark execution
  - **SLOW** Result structure validation
  - **SLOW** Multiple configurations
  - **SLOW** Reasonable loss value checking
  - Results display formatting

### test_integration.py (15 tests)
End-to-end workflow tests:
- **TestModelPipeline** (3 tests)
  - UNet with loss computation
  - MLP with loss computation
  - Model state save/load

- **TestQuantumMetricsPipeline** (3 tests)
  - State → Density Matrix → Purity
  - State → Loss computation
  - Property verification pipeline

- **TestGenerationOptimizationPipeline** (3 tests)
  - **SLOW** L-BFGS optimization
  - **SLOW** Full generation pipeline
  - **SLOW** Generation with L-BFGS

- **TestBatchProcessing** (3 tests)
  - Batch loss computation
  - Batch density matrix
  - Batch purity computation

- **TestDeviceHandling** (2 tests)
  - Model device consistency
  - Loss computation device consistency

## Test Markers

Tests are marked for selective execution:
- `@pytest.mark.slow`: Long-running tests (generation, benchmarking)
- `@pytest.mark.integration`: End-to-end tests

### Exclude slow tests
```bash
pytest -m "not slow"
```

### Run only slow tests
```bash
pytest -m slow
```

### Run only integration tests
```bash
pytest -m integration
```

## Test Statistics

Total tests: **66**
- Unit tests: ~45
- Integration tests: ~15
- Slow tests (excluded by default): ~12

## Expected Test Execution Times

- **Quick run** (without slow tests): ~10-30 seconds
- **Full test suite** (including slow tests): ~2-5 minutes

## Debugging Tests

### Verbose output for failed tests
```bash
pytest -vv --tb=long
```

### Stop on first failure
```bash
pytest -x
```

### Show print statements
```bash
pytest -s
```

### Show local variables on failure
```bash
pytest -l
```

### Run with pdb debugger on failure
```bash
pytest --pdb
```

### Run specific test with debugging
```bash
pytest -s -vv tests/test_models.py::TestGeneralDynamicUNet::test_unet_instantiation
```

## Common Issues

### Issue: `ModuleNotFoundError: No module named 'pytest'`
**Solution**: Install pytest using uv
```bash
uv sync  # Install all dependencies including pytest
# Or add specifically if needed
uv add --dev pytest pytest-cov
```

### Issue: Tests fail with import errors
**Solution**: Install package in development mode using uv
```bash
cd /home/palash/diffusion
uv sync
```

### Issue: GPU tests fail
**Solution**: Tests automatically fall back to CPU if GPU unavailable
```bash
# Force CPU testing
CUDA_VISIBLE_DEVICES="" pytest tests/
```

### Issue: Slow tests timeout
**Solution**: Use longer timeout or skip slow tests
```bash
pytest -m "not slow"
pytest --timeout=600  # 10 minute timeout (requires pytest-timeout)
```

## Best Practices

1. **Run quick tests during development**
   ```bash
   pytest -m "not slow"
   ```

2. **Run full suite before commit**
   ```bash
   pytest
   ```

3. **Check coverage regularly**
   ```bash
   pytest --cov=src --cov-report=html
   ```

4. **Run specific tests when debugging**
   ```bash
   pytest -s -vv tests/test_file.py::TestClass::test_func
   ```

5. **Use fixtures for common setup**
   - Already defined in `conftest.py`
   - Add new fixtures there for reusability

## Adding New Tests

1. Create file `tests/test_*.py`
2. Import pytest and required modules
3. Create test functions starting with `test_`
4. Use fixtures from `conftest.py`
5. Remember to use `@pytest.mark.slow` for long tests
6. Run `pytest tests/test_*.py` to test

Example:
```python
import pytest
from src.core.functions import generate_ame_state

def test_my_feature(device, vec_len):
    """Test description."""
    # Arrange
    state = torch.randn(1, 2, vec_len, device=device)
    
    # Act
    result = some_function(state)
    
    # Assert
    assert result is not None
```

## References

- [pytest Documentation](https://docs.pytest.org/)
- [pytest Fixtures](https://docs.pytest.org/en/stable/how-to/fixtures.html)
- [pytest Markers](https://docs.pytest.org/en/stable/how-to/mark.html)
