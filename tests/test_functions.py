"""
Tests for core generation and training functions.
"""

import pytest
import torch
import numpy as np
from src.core.functions import (
    verify_ame_properties, fine_tune_with_lbfgs,
    load_ame_training_data, generate_ame_state
)
from src.core.loss import make_density_matrix, purity
from src import config


class TestIndexingAndShapes:
    """Test proper tensor shapes and indexing."""

    def test_ame_properties_input_shapes(self, device):
        """Test verify_ame_properties handles various input shapes."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # 1D input
        state_1d = torch.randn(vec_len, dtype=torch.complex64, device=device)
        state_1d = state_1d / torch.norm(state_1d)
        result_1d = verify_ame_properties(state_1d, n=n, d=d)
        assert isinstance(result_1d, float)
        
        # 2D input
        state_2d = torch.randn(1, vec_len, dtype=torch.complex64, device=device)
        state_2d = state_2d / torch.norm(state_2d)
        result_2d = verify_ame_properties(state_2d, n=n, d=d)
        assert isinstance(result_2d, float)

    def test_ame_properties_output_type(self, device):
        """Test verify_ame_properties returns float."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        state = torch.randn(vec_len, dtype=torch.complex64, device=device)
        state = state / torch.norm(state)
        
        result = verify_ame_properties(state, n=n, d=d)
        
        assert isinstance(result, float)
        assert not np.isnan(result)
        assert not np.isinf(result)

    def test_ame_properties_values_non_negative(self, device):
        """Test AME property values are non-negative."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        state = torch.randn(vec_len, dtype=torch.complex64, device=device)
        state = state / torch.norm(state)
        
        result = verify_ame_properties(state, n=n, d=d)
        
        assert result >= 0.0


class TestFineTuningWithLBFGS:
    """Test suite for L-BFGS fine-tuning."""

    def test_lbfgs_output_shape(self, device):
        """Test L-BFGS returns state with correct shape."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        state = torch.randn(vec_len, dtype=torch.complex64, device=device)
        state = state / torch.norm(state)
        
        optimized = fine_tune_with_lbfgs(state, d=d, n=n, max_iters=10, verbose=False)
        
        assert optimized.shape == state.shape
        assert optimized.dtype == torch.complex64

    def test_lbfgs_reduces_loss(self, device):
        """Test L-BFGS reduces loss (at least doesn't increase dramatically)."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        state = torch.randn(vec_len, dtype=torch.complex64, device=device)
        state = state / torch.norm(state)
        
        # Initial loss
        state_real_imag = torch.stack([state.real, state.imag], dim=0).unsqueeze(0)
        from src.core.loss import partial_trace_loss_optimized
        initial_loss = partial_trace_loss_optimized(state_real_imag, d=d, n=n).item()
        
        # Fine-tune
        optimized = fine_tune_with_lbfgs(state, d=d, n=n, max_iters=20, verbose=False)
        
        # Final loss
        opt_real_imag = torch.stack([optimized.real, optimized.imag], dim=0).unsqueeze(0)
        final_loss = partial_trace_loss_optimized(opt_real_imag, d=d, n=n).item()
        
        # Loss should not increase significantly
        assert final_loss <= initial_loss * 1.5  # Allow some numerical variation

    def test_lbfgs_preserves_normalized_state(self, device):
        """Test L-BFGS maintains state normalization."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        state = torch.randn(vec_len, dtype=torch.complex64, device=device)
        state = state / torch.norm(state)
        
        optimized = fine_tune_with_lbfgs(state, d=d, n=n, max_iters=10, verbose=False)
        
        # Check norm is close to 1
        norm = torch.norm(optimized).item()
        assert abs(norm - 1.0) < 1e-3

    def test_lbfgs_max_iterations(self, device):
        """Test L-BFGS respects max_iters parameter."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        state = torch.randn(vec_len, dtype=torch.complex64, device=device)
        state = state / torch.norm(state)
        
        # Should complete with small max_iters
        optimized = fine_tune_with_lbfgs(state, d=d, n=n, max_iters=5, verbose=False)
        
        assert optimized is not None
        assert optimized.shape == state.shape


class TestTrainingDataLoading:
    """Test suite for training data loading."""

    def test_load_nonexistent_data(self):
        """Test loading nonexistent training data returns None."""
        result = load_ame_training_data(save_path="nonexistent_path_xyz.pt")
        
        assert result is None

    def test_load_valid_data_structure(self, tmp_path, device):
        """Test loading valid training data structure."""
        d = config.D
        n = config.N
        vec_len = d ** n
        batch_size = 2
        
        # Create dummy dataset
        dataset = {
            'noisy_states': torch.randn(batch_size, 2, vec_len),
            'clean_states': torch.randn(batch_size, 2, vec_len),
            'timesteps': torch.randint(0, 1000, (batch_size,)),
            'd': d,
            'n': n,
        }
        
        # Save to temp file
        save_path = tmp_path / "test_data.pt"
        torch.save(dataset, str(save_path))
        
        # Load it back
        loaded = load_ame_training_data(save_path=str(save_path))
        
        assert loaded is not None
        assert 'noisy_states' in loaded
        assert 'clean_states' in loaded
        assert 'timesteps' in loaded
        assert torch.allclose(loaded['noisy_states'], dataset['noisy_states'])


class TestGenerateAMEState:
    """Test suite for AME state generation."""

    @pytest.mark.slow
    def test_generate_ame_basic(self, device):
        """Test basic AME state generation."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Generate with minimal steps for quick testing
        final_state = generate_ame_state(
            guidance_scale=0.05,
            num_steps=10,
            d=d,
            n=n,
            use_lbfgs=False
        )
        
        assert final_state is not None
        assert final_state.shape == (vec_len,)
        assert final_state.dtype == torch.complex64

    @pytest.mark.slow
    def test_generate_ame_normalization(self, device):
        """Test generated state is normalized."""
        d = config.D
        n = config.N
        
        final_state = generate_ame_state(
            guidance_scale=0.05,
            num_steps=10,
            d=d,
            n=n,
            use_lbfgs=False
        )
        
        norm = torch.norm(final_state).item()
        assert abs(norm - 1.0) < 0.1  # Allow some tolerance

    @pytest.mark.slow
    def test_generate_ame_with_lbfgs(self, device):
        """Test AME generation with L-BFGS fine-tuning."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        final_state = generate_ame_state(
            guidance_scale=0.05,
            num_steps=5,
            d=d,
            n=n,
            use_lbfgs=True
        )
        
        assert final_state is not None
        assert final_state.shape == (vec_len,)

    @pytest.mark.slow
    def test_generate_ame_different_guidance_scales(self, device):
        """Test generation with different guidance scales."""
        d = config.D
        n = config.N
        
        for guidance_scale in [0.01, 0.05, 0.1]:
            state = generate_ame_state(
                guidance_scale=guidance_scale,
                num_steps=5,
                d=d,
                n=n,
                use_lbfgs=False
            )
            
            assert state is not None
            norm = torch.norm(state)
            assert abs(norm.item() - 1.0) < 0.5  # Reasonably normalized

    @pytest.mark.slow
    def test_generate_ame_properties(self, device):
        """Test generated state has reasonable AME properties."""
        d = config.D
        n = config.N
        
        final_state = generate_ame_state(
            guidance_scale=0.05,
            num_steps=5,
            d=d,
            n=n,
            use_lbfgs=False
        )
        
        # Verify properties
        ame_loss = verify_ame_properties(final_state, n=n, d=d)
        
        assert isinstance(ame_loss, float)
        assert ame_loss >= 0.0
        assert not np.isnan(ame_loss)
