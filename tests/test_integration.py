"""
Integration tests for end-to-end workflows.
"""

import pytest
import torch
from src.models.unet import GeneralDynamicUNet
from src.models.mlp import QuantumDiffusionMLP
from src.core.functions import (
    verify_ame_properties, fine_tune_with_lbfgs, generate_ame_state
)
from src.core.loss import make_density_matrix, purity, partial_trace_loss_optimized
from src import config


class TestModelPipeline:
    """Test complete model pipeline workflows."""

    def test_unet_with_loss_pipeline(self, device):
        """Test UNet integration with loss computation."""
        model = GeneralDynamicUNet().to(device)
        vec_len = config.D ** config.N
        
        # Random input
        x = torch.randn(1, 2, vec_len, device=device, requires_grad=True)
        t = torch.randn(1, 1, device=device)
        
        # Forward pass
        output = model(x, t)
        
        # Compute loss
        loss = partial_trace_loss_optimized(output, d=config.D, n=config.N)
        
        # Backprop
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(loss)

    def test_mlp_with_loss_pipeline(self, device):
        """Test MLP integration with loss computation."""
        vec_len = config.D ** config.N
        model = QuantumDiffusionMLP(vec_len=vec_len).to(device)
        
        # Random input
        x = torch.randn(1, 2, vec_len, device=device, requires_grad=True)
        t = torch.randn(1, device=device)
        
        # Forward pass
        output = model(x, t)
        
        # Compute loss
        loss = partial_trace_loss_optimized(output, d=config.D, n=config.N)
        
        # Backprop
        loss.backward()
        
        assert x.grad is not None
        assert not torch.isnan(loss)

    def test_model_state_dict_save_load(self, device, tmp_path):
        """Test model state can be saved and loaded."""
        model1 = GeneralDynamicUNet().to(device)
        vec_len = config.D ** config.N
        
        # Forward pass to generate state
        x = torch.randn(1, 2, vec_len, device=device)
        t = torch.randn(1, 1, device=device)
        out1 = model1(x, t)
        
        # Save state
        save_path = tmp_path / "model.pt"
        torch.save(model1.state_dict(), str(save_path))
        
        # Load into new model
        model2 = GeneralDynamicUNet().to(device)
        model2.load_state_dict(torch.load(str(save_path)))
        model2.eval()
        model1.eval()
        
        # Compare outputs
        with torch.no_grad():
            out2 = model2(x, t)
        
        assert torch.allclose(out1, out2)


class TestQuantumMetricsPipeline:
    """Test quantum metrics workflow."""

    def test_state_to_density_matrix_to_purity(self, device):
        """Test full pipeline from state to purity."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Create random state
        state = torch.randn(1, 2, vec_len, device=device)
        state = state / torch.norm(state)
        
        # Convert to density matrix
        rho = make_density_matrix(state, d=d, n=n)
        
        # Compute purity
        pur = purity(rho)
        
        # Verify ranges
        assert 1.0 / vec_len <= pur.item() <= 1.0

    def test_state_to_loss(self, device):
        """Test full pipeline from state to loss."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Create random state
        state = torch.randn(1, 2, vec_len, device=device)
        
        # Compute loss
        loss = partial_trace_loss_optimized(state, d=d, n=n)
        
        # Verify it's a valid scalar
        assert loss.ndim == 0
        assert loss.item() >= 0.0
        assert not torch.isnan(loss)

    def test_verification_pipeline(self, device):
        """Test AME property verification pipeline."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Create state
        state = torch.randn(vec_len, dtype=torch.complex64, device=device)
        state = state / torch.norm(state)
        
        # Verify properties
        ame_loss = verify_ame_properties(state, n=n, d=d)
        
        assert isinstance(ame_loss, float)
        assert ame_loss >= 0.0


class TestGenerationOptimizationPipeline:
    """Test complete generation with optimization."""

    @pytest.mark.slow
    def test_lbfgs_optimization_pipeline(self, device):
        """Test L-BFGS optimization on generated state."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        # Start with random state
        state = torch.randn(vec_len, dtype=torch.complex64, device=device)
        state = state / torch.norm(state)
        
        # Get initial loss
        state_real_imag = torch.stack([state.real, state.imag], dim=0).unsqueeze(0)
        initial_loss = partial_trace_loss_optimized(state_real_imag, d=d, n=n).item()
        
        # Optimize with L-BFGS
        optimized = fine_tune_with_lbfgs(state, d=d, n=n, max_iters=20, verbose=False)
        
        # Get final loss
        opt_real_imag = torch.stack([optimized.real, optimized.imag], dim=0).unsqueeze(0)
        final_loss = partial_trace_loss_optimized(opt_real_imag, d=d, n=n).item()
        
        # Verify optimization worked
        assert torch.isfinite(torch.tensor(initial_loss))
        assert torch.isfinite(torch.tensor(final_loss))
        assert abs(torch.norm(optimized).item() - 1.0) < 1e-3

    @pytest.mark.slow
    def test_full_generation_pipeline(self, device):
        """Test complete generation pipeline."""
        d = config.D
        n = config.N
        
        # Generate state with short steps for testing
        state = generate_ame_state(
            guidance_scale=0.05,
            num_steps=10,
            d=d,
            n=n,
            use_lbfgs=False
        )
        
        # Verify properties
        assert state is not None
        assert not torch.isnan(state).any()
        assert abs(torch.norm(state).item() - 1.0) < 0.2
        
        # Compute AME loss
        ame_loss = verify_ame_properties(state, n=n, d=d)
        assert isinstance(ame_loss, float)
        assert ame_loss >= 0.0

    @pytest.mark.slow
    def test_generation_with_lbfgs_pipeline(self, device):
        """Test full generation with L-BFGS refinement."""
        d = config.D
        n = config.N
        
        # Generate with L-BFGS
        state = generate_ame_state(
            guidance_scale=0.05,
            num_steps=5,
            d=d,
            n=n,
            use_lbfgs=True
        )
        
        # Verify result
        assert state is not None
        assert abs(torch.norm(state).item() - 1.0) < 1e-3
        
        # Get properties
        ame_loss = verify_ame_properties(state, n=n, d=d)
        assert isinstance(ame_loss, float)


class TestBatchProcessing:
    """Test batch processing in pipelines."""

    def test_batch_loss_computation(self, device):
        """Test loss computation on batches."""
        d = config.D
        n = config.N
        vec_len = d ** n
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            states = torch.randn(batch_size, 2, vec_len, device=device)
            loss = partial_trace_loss_optimized(states, d=d, n=n)
            
            # Loss should be scalar regardless of batch size
            assert loss.ndim == 0
            assert not torch.isnan(loss)

    def test_batch_density_matrix(self, device):
        """Test density matrix computation on batches."""
        d = config.D
        n = config.N
        vec_len = d ** n
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            states = torch.randn(batch_size, 2, vec_len, device=device)
            states = states / torch.norm(states, dim=(1, 2), keepdim=True)
            
            rho = make_density_matrix(states, d=d, n=n)
            
            assert rho.shape == (batch_size, vec_len, vec_len)

    def test_batch_purity(self, device):
        """Test purity computation on batches."""
        d = config.D
        n = config.N
        vec_len = d ** n
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            states = torch.randn(batch_size, 2, vec_len, device=device)
            states = states / torch.norm(states, dim=(1, 2), keepdim=True)
            
            rho = make_density_matrix(states, d=d, n=n)
            pur = purity(rho)
            
            assert pur.shape == (batch_size,)
            assert torch.all(pur >= 0.0)
            assert torch.all(pur <= 1.0 + 1e-4)


class TestDeviceHandling:
    """Test device handling across pipelines."""

    def test_model_device_consistency(self, device):
        """Test model maintains device consistency."""
        model = GeneralDynamicUNet().to(device)
        vec_len = config.D ** config.N
        
        x = torch.randn(1, 2, vec_len, device=device)
        t = torch.randn(1, 1, device=device)
        
        output = model(x, t)
        
        assert output.device == x.device
        assert all(p.device == x.device for p in model.parameters())

    def test_loss_device_consistency(self, device):
        """Test loss computation maintains device consistency."""
        d = config.D
        n = config.N
        vec_len = d ** n
        
        state = torch.randn(1, 2, vec_len, device=device)
        loss = partial_trace_loss_optimized(state, d=d, n=n)
        
        # Loss should be on same device
        assert loss.device == state.device
