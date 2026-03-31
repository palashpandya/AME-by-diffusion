"""
Tests for model architectures (UNet and MLP).
"""

import pytest
import torch
import torch.nn as nn
from src.models.unet import GeneralDynamicUNet
from src.models.mlp import QuantumDiffusionMLP
from src import config


class TestGeneralDynamicUNet:
    """Test suite for the UNet architecture."""

    def test_unet_instantiation(self):
        """Test that UNet can be instantiated with correct configuration."""
        model = GeneralDynamicUNet()
        assert model is not None
        assert model.d == config.D
        assert model.n == config.N

    def test_unet_forward_pass_shape(self, device):
        """Test that forward pass returns correct output shape."""
        model = GeneralDynamicUNet().to(device)
        vec_len = config.D ** config.N
        batch_size = 2
        
        x = torch.randn(batch_size, 2, vec_len, device=device)
        t = torch.randn(batch_size, 1, device=device)
        
        output = model(x, t)
        
        assert output.shape == (batch_size, 2, vec_len)
        assert output.dtype == x.dtype

    def test_unet_forward_pass_gradient_flow(self, device):
        """Test that gradients flow through the model."""
        model = GeneralDynamicUNet().to(device)
        vec_len = config.D ** config.N
        
        x = torch.randn(1, 2, vec_len, device=device, requires_grad=True)
        t = torch.randn(1, 1, device=device)
        
        output = model(x, t)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_unet_different_batch_sizes(self, device):
        """Test model with various batch sizes."""
        model = GeneralDynamicUNet().to(device)
        vec_len = config.D ** config.N
        
        for batch_size in [1, 2, 4, 8]:
            x = torch.randn(batch_size, 2, vec_len, device=device)
            t = torch.randn(batch_size, 1, device=device)
            output = model(x, t)
            
            assert output.shape[0] == batch_size
            assert output.shape == (batch_size, 2, vec_len)

    def test_unet_time_normalization(self, device):
        """Test model handles time values in [0, 1] range."""
        model = GeneralDynamicUNet().to(device)
        vec_len = config.D ** config.N
        
        x = torch.randn(1, 2, vec_len, device=device)
        
        # Test with normalized time
        t_low = torch.tensor([[0.0]], device=device)
        t_high = torch.tensor([[1.0]], device=device)
        t_mid = torch.tensor([[0.5]], device=device)
        
        out_low = model(x, t_low)
        out_high = model(x, t_high)
        out_mid = model(x, t_mid)
        
        assert out_low.shape == out_high.shape == out_mid.shape

    def test_unet_eval_mode(self, device):
        """Test model in evaluation mode (no dropout/norm behavior)."""
        model = GeneralDynamicUNet().to(device)
        model.eval()
        
        vec_len = config.D ** config.N
        x = torch.randn(1, 2, vec_len, device=device)
        t = torch.randn(1, 1, device=device)
        
        # Two forward passes in eval mode should give identical results
        with torch.no_grad():
            out1 = model(x, t)
            out2 = model(x, t)
        
        assert torch.allclose(out1, out2)

    def test_unet_output_dtype(self, device):
        """Test output maintains input dtype."""
        model = GeneralDynamicUNet().to(device)
        vec_len = config.D ** config.N
        
        # Test with float32
        x_f32 = torch.randn(1, 2, vec_len, dtype=torch.float32, device=device)
        t_f32 = torch.randn(1, 1, dtype=torch.float32, device=device)
        output_f32 = model(x_f32, t_f32)
        assert output_f32.dtype == torch.float32


class TestQuantumDiffusionMLP:
    """Test suite for the MLP model."""

    def test_mlp_instantiation(self, vec_len):
        """Test that MLP can be instantiated with correct dimensions."""
        model = QuantumDiffusionMLP(vec_len=vec_len)
        assert model is not None
        assert model.input_dim == (vec_len * 2) + 1

    def test_mlp_forward_pass_shape(self, device, vec_len):
        """Test that MLP forward pass returns correct shape."""
        model = QuantumDiffusionMLP(vec_len=vec_len).to(device)
        batch_size = 4
        
        x = torch.randn(batch_size, 2, vec_len, device=device)
        t = torch.randn(batch_size, device=device)
        
        output = model(x, t)
        
        assert output.shape == (batch_size, 2, vec_len)

    def test_mlp_forward_pass_gradient_flow(self, device, vec_len):
        """Test that gradients flow through the MLP."""
        model = QuantumDiffusionMLP(vec_len=vec_len).to(device)
        
        x = torch.randn(1, 2, vec_len, device=device, requires_grad=True)
        t = torch.randn(1, device=device)
        
        output = model(x, t)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist for model parameters
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_mlp_scalar_time(self, device, vec_len):
        """Test MLP handles scalar time input."""
        model = QuantumDiffusionMLP(vec_len=vec_len).to(device)
        batch_size = 2
        
        x = torch.randn(batch_size, 2, vec_len, device=device)
        t = torch.tensor(0.5, device=device)  # scalar
        
        output = model(x, t)
        assert output.shape[0] == batch_size

    def test_mlp_1d_time(self, device, vec_len):
        """Test MLP handles 1D time tensor."""
        model = QuantumDiffusionMLP(vec_len=vec_len).to(device)
        batch_size = 2
        
        x = torch.randn(batch_size, 2, vec_len, device=device)
        t = torch.randn(batch_size, device=device)  # 1D
        
        output = model(x, t)
        assert output.shape[0] == batch_size

    def test_mlp_different_batch_sizes(self, device, vec_len):
        """Test MLP with various batch sizes."""
        model = QuantumDiffusionMLP(vec_len=vec_len).to(device)
        
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 2, vec_len, device=device)
            t = torch.randn(batch_size, device=device)
            output = model(x, t)
            
            assert output.shape == (batch_size, 2, vec_len)

    def test_mlp_custom_hidden_dim(self, device, vec_len):
        """Test MLP with custom hidden dimensions."""
        model = QuantumDiffusionMLP(
            vec_len=vec_len,
            hidden_dim=512,
            num_layers=2
        ).to(device)
        
        x = torch.randn(1, 2, vec_len, device=device)
        t = torch.randn(1, device=device)
        
        output = model(x, t)
        assert output.shape == (1, 2, vec_len)

    def test_mlp_output_dtype(self, device, vec_len):
        """Test MLP output maintains input dtype."""
        model = QuantumDiffusionMLP(vec_len=vec_len).to(device)
        
        x = torch.randn(1, 2, vec_len, dtype=torch.float32, device=device)
        t = torch.randn(1, dtype=torch.float32, device=device)
        output = model(x, t)
        
        assert output.dtype == torch.float32
