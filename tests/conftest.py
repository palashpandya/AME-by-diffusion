"""
Shared pytest fixtures and configuration.
"""

import pytest
import torch
import numpy as np
from src import config


@pytest.fixture(scope="session")
def device():
    """Return the device to use for testing."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def random_seed():
    """Set random seeds for reproducibility."""
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    return seed


@pytest.fixture
def vec_len():
    """Vector length for quantum states."""
    return config.D ** config.N


@pytest.fixture
def batch_size():
    """Batch size for testing."""
    return 4


@pytest.fixture
def random_state(device, vec_len, batch_size):
    """Generate a random quantum state (real-imag format)."""
    state = torch.randn(batch_size, 2, vec_len, device=device)
    # Normalize
    state = state / torch.norm(state, dim=(1, 2), keepdim=True)
    return state


@pytest.fixture
def random_complex_state(device, vec_len):
    """Generate a random complex quantum state."""
    state = torch.randn(vec_len, dtype=torch.complex64, device=device)
    state = state / torch.norm(state)
    return state


@pytest.fixture
def pure_state(device, vec_len):
    """Generate a simple pure state (normalized random vector)."""
    state = torch.randn(vec_len, dtype=torch.complex64, device=device)
    state = state / torch.norm(state)
    return state


@pytest.fixture
def time_batch(device, batch_size):
    """Generate batch of time values for diffusion model."""
    return torch.rand(batch_size, 1, device=device)
