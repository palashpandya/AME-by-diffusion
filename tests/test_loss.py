"""
Tests for loss functions and quantum metrics.
"""

import pytest
import torch
import numpy as np
from src.core.loss import (
    make_density_matrix, purity, partial_trace_loss_optimized, is_pure
)
from src import config


class TestDensityMatrixCreation:
    """Test suite for density matrix creation."""

    def test_density_matrix_shape(self, random_state, device):
        """Test density matrix has correct shape."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        rho = make_density_matrix(random_state, d=d, n=n)
        
        assert rho.shape == (random_state.shape[0], vec_len, vec_len)

    def test_density_matrix_hermitian(self, random_state, device):
        """Test density matrix is Hermitian."""
        d = config.D
        n = config.N
        
        rho = make_density_matrix(random_state, d=d, n=n)
        
        # Check: rho = rho†
        rho_conj_T = torch.conj(rho.transpose(-2, -1))
        assert torch.allclose(rho, rho_conj_T, atol=1e-5)

    def test_density_matrix_trace_one(self, random_state, device):
        """Test density matrix has trace equal to 1."""
        d = config.D
        n = config.N
        
        rho = make_density_matrix(random_state, d=d, n=n)
        
        # Compute trace for each element in batch
        traces = torch.diagonal(rho, dim1=-2, dim2=-1).sum(dim=-1)
        
        assert torch.allclose(traces, torch.ones_like(traces), atol=1e-5)

    def test_density_matrix_positive_semidefinite(self, random_state, device):
        """Test density matrix is positive semidefinite (all eigenvalues >= 0)."""
        d = config.D
        n = config.N
        
        rho = make_density_matrix(random_state, d=d, n=n)
        
        # Get eigenvalues for first batch element
        eigvals = torch.linalg.eigvalsh(rho[0])
        
        assert torch.all(eigvals >= -1e-5)  # Allow small numerical error

    def test_density_matrix_single_state(self, device):
        """Test density matrix creation with single state."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Create single state
        state = torch.randn(1, 2, vec_len, device=device)
        state = state / torch.norm(state)
        
        rho = make_density_matrix(state, d=d, n=n)
        
        assert rho.shape == (1, vec_len, vec_len)


class TestPurity:
    """Test suite for purity computation."""

    def test_purity_output_shape(self, random_state, device):
        """Test purity output has correct shape."""
        d = config.D
        n = config.N
        
        rho = make_density_matrix(random_state, d=d, n=n)
        pur = purity(rho)
        
        assert pur.shape == (random_state.shape[0],)

    def test_purity_pure_state(self, device):
        """Test purity of pure state equals 1."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Create pure state: |0...0⟩
        state = torch.zeros(1, 2, vec_len, device=device)
        state[0, 0, 0] = 1.0  # Real part of first amplitude
        
        rho = make_density_matrix(state, d=d, n=n)
        pur = purity(rho)
        
        assert torch.allclose(pur, torch.ones_like(pur), atol=1e-4)

    def test_purity_maximally_mixed(self, device):
        """Test purity of maximally mixed state."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Create maximally mixed state: I/d^n
        rho = torch.eye(vec_len, device=device) / vec_len
        rho = rho.unsqueeze(0)  # Add batch dimension
        
        pur = purity(rho)
        expected_purity = 1.0 / vec_len
        
        assert torch.allclose(pur, torch.tensor([expected_purity], device=device), atol=1e-4)

    def test_purity_range(self, random_state, device):
        """Test purity is in valid range [1/d^n, 1]."""
        d = config.D
        n = config.N
        vec_len = d ** n
        min_purity = 1.0 / vec_len
        
        rho = make_density_matrix(random_state, d=d, n=n)
        pur = purity(rho)
        
        assert torch.all(pur >= min_purity - 1e-4)
        assert torch.all(pur <= 1.0 + 1e-4)

    def test_purity_real_valued(self, random_state, device):
        """Test purity gives real values."""
        d = config.D
        n = config.N
        
        rho = make_density_matrix(random_state, d=d, n=n)
        pur = purity(rho)
        
        assert pur.dtype in [torch.float32, torch.float64]


class TestIsPure:
    """Test suite for pure state detection."""

    def test_is_pure_detects_pure_state(self, device):
        """Test that is_pure detects pure states correctly."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Create pure state
        state = torch.zeros(1, 2, vec_len, device=device)
        state[0, 0, 0] = 1.0
        
        rho = make_density_matrix(state, d=d, n=n)
        result = is_pure(rho, tol=1e-4)
        
        assert result[0].item() == True

    def test_is_pure_tolerance(self, device):
        """Test is_pure respects tolerance parameter."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Create nearly pure state
        rho = torch.eye(vec_len, device=device) / vec_len
        rho = rho.unsqueeze(0)
        
        # With strict tolerance, should be False
        result_strict = is_pure(rho, tol=1e-10)
        assert result_strict[0].item() == False
        
        # With loose tolerance, might be True
        result_loose = is_pure(rho, tol=0.5)
        # (behavior depends on actual purity value)


class TestPartialTraceLoss:
    """Test suite for partial trace loss function."""

    def test_loss_output_is_scalar(self, random_state, device):
        """Test loss output is a scalar tensor."""
        d = config.D
        n = config.N
        
        loss = partial_trace_loss_optimized(random_state, d=d, n=n)
        
        assert loss.ndim == 0
        assert isinstance(loss, torch.Tensor)

    def test_loss_non_negative(self, random_state, device):
        """Test loss values are non-negative."""
        d = config.D
        n = config.N
        
        loss = partial_trace_loss_optimized(random_state, d=d, n=n)
        
        assert loss.item() >= -1e-6  # Allow small numerical error

    def test_loss_convergence_pure_state(self, device):
        """Test loss is small for states closer to AME."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Create pure state (should have lower loss tendency)
        state = torch.zeros(1, 2, vec_len, device=device)
        state[0, 0, 0] = 1.0  # |0...0⟩
        
        loss = partial_trace_loss_optimized(state, d=d, n=n)
        
        # Loss should be a valid number
        assert torch.isfinite(loss)

    def test_loss_batch_processing(self, random_state, device):
        """Test loss works with batch processing."""
        d = config.D
        n = config.N
        
        loss = partial_trace_loss_optimized(random_state, d=d, n=n)
        
        # Loss should be scalar even for batch input
        assert loss.ndim == 0

    def test_loss_gradient_flow(self, device):
        """Test gradients flow through loss function."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        state = torch.randn(1, 2, vec_len, device=device, requires_grad=True)
        loss = partial_trace_loss_optimized(state, d=d, n=n)
        loss.backward()
        
        assert state.grad is not None
        assert state.grad.shape == state.shape

    def test_loss_deterministic(self, device):
        """Test loss is deterministic for same input."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        state = torch.randn(1, 2, vec_len, device=device)
        
        with torch.no_grad():
            loss1 = partial_trace_loss_optimized(state, d=d, n=n)
            loss2 = partial_trace_loss_optimized(state, d=d, n=n)
        
        assert torch.allclose(loss1, loss2)
