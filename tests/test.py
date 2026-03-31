"""
Unit tests for the quantum diffusion project using unittest framework.

Demonstrates:
- Test class structure and inheritance from unittest.TestCase
- setUp and tearDown methods
- Various assertion methods
- Testing torch tensors and numpy operations
"""

import unittest
import torch
import numpy as np
from src.models.mlp import QuantumDiffusionMLP
from src.core.loss import partial_trace_loss_optimized
from src.core.functions import verify_ame_properties
from src import config


class TestQuantumDiffusionMLP(unittest.TestCase):
    """Test suite for the QuantumDiffusionMLP model."""
    
    def setUp(self):
        """Initialize test fixtures before each test."""
        # Use project-wide AME dimensions from config
        self.vec_len = config.D ** config.N
        self.model = QuantumDiffusionMLP(vec_len=self.vec_len)
        self.batch_size = 4
    
    def tearDown(self):
        """Clean up after each test."""
        # PyTorch automatically manages memory, but you could clear cache here
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    def test_model_initialization(self):
        """Test that the model initializes with correct dimensions."""
        self.assertIsNotNone(self.model)
        self.assertEqual(self.model.input_dim, (self.vec_len * 2) + 1)
    
    def test_forward_pass_output_shape(self):
        """Test that forward pass returns correct output shape."""
        x = torch.randn(self.batch_size, 2, self.vec_len)
        t = torch.randn(self.batch_size)
        
        output = self.model(x, t)
        
        expected_shape = (self.batch_size, 2, self.vec_len)
        self.assertEqual(output.shape, expected_shape)
    
    def test_forward_pass_scalar_time(self):
        """Test forward pass with scalar time input."""
        x = torch.randn(self.batch_size, 2, self.vec_len)
        t = torch.tensor(0.5)  # Scalar time
        
        output = self.model(x, t)
        self.assertEqual(output.shape[0], self.batch_size)
    
    def test_forward_pass_different_batch_sizes(self):
        """Test that model handles different batch sizes correctly."""
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 2, self.vec_len)
            t = torch.randn(batch_size)
            output = self.model(x, t)
            self.assertEqual(output.shape[0], batch_size)


class TestPartialTraceLoss(unittest.TestCase):
    """Test suite for the partial_trace_loss function."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.d = 3  # Qudit dimension
        self.vec_len = self.d ** config.N  # For `N` qudits/qutrits
    
    def test_loss_output_is_scalar(self):
        """Test that loss output is a scalar tensor."""
        state = torch.randn(1, 2, self.vec_len)
        loss = partial_trace_loss_optimized(state, d=self.d, n=config.N) # Added n=config.N
        
        self.assertEqual(loss.ndim, 0)  # Scalar tensor
        self.assertIsInstance(loss, torch.Tensor)
    
    def test_loss_is_non_negative(self):
        """Test that loss values are non-negative."""
        state = torch.randn(1, 2, self.vec_len)
        loss = partial_trace_loss_optimized(state, d=self.d, n=config.N) # Added n=config.N
        
        self.assertGreaterEqual(loss.item(), 0.0)
    
    def test_loss_with_batch(self):
        """Test loss computation with multiple states in batch."""
        batch_size = 4
        state = torch.randn(batch_size, 2, self.vec_len)
        loss = partial_trace_loss_optimized(state, d=self.d, n=config.N) # Added n=config.N
        
        self.assertEqual(loss.ndim, 0)  # Still returns single scalar


class TestVerifyAMEProperties(unittest.TestCase):
    """Test suite for the verify_ame_properties function."""
    
    def setUp(self):
        """Initialize test fixtures."""
        self.d = 3
        self.vec_len = self.d ** config.N
    
    def test_verify_returns_float(self):
        """Test that verify_ame_properties returns a float."""
        state = torch.randn(self.vec_len, dtype=torch.complex64)
        result = verify_ame_properties(state, d=self.d)
        
        self.assertIsInstance(result, float)
    
    def test_verify_with_2d_input(self):
        """Test verify_ame_properties with 2D input."""
        state = torch.randn(1, self.vec_len, dtype=torch.complex64)
        result = verify_ame_properties(state, d=self.d)
        
        self.assertIsInstance(result, float)
        self.assertGreaterEqual(result, 0.0)
    
    def test_verify_handles_normalized_state(self):
        """Test that verify works with normalized quantum states."""
        state = torch.randn(self.vec_len, dtype=torch.complex64)
        state = state / torch.norm(state)  # Normalize
        
        result = verify_ame_properties(state, d=self.d)
        self.assertLess(result, 100)  # Should be reasonable value


class TestDataTypes(unittest.TestCase):
    """Test data type handling and conversions."""
    
    def test_tensor_to_numpy_conversion(self):
        """Test converting torch tensor to numpy array."""
        tensor = torch.randn(3, 4)
        array = tensor.numpy()
        
        self.assertEqual(type(array), np.ndarray)
        self.assertEqual(array.shape, (3, 4))
    
    def test_numpy_to_tensor_conversion(self):
        """Test converting numpy array to torch tensor."""
        array = np.random.randn(3, 4)
        tensor = torch.from_numpy(array)
        
        self.assertEqual(type(tensor), torch.Tensor)
        self.assertEqual(tensor.shape, torch.Size([3, 4]))


# Test Suite Grouping (optional: organize related tests)
def suite():
    """Create a test suite combining all tests."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestQuantumDiffusionMLP))
    suite.addTest(unittest.makeSuite(TestPartialTraceLoss))
    suite.addTest(unittest.makeSuite(TestVerifyAMEProperties))
    suite.addTest(unittest.makeSuite(TestDataTypes))
    return suite


if __name__ == '__main__':
    # Run tests with minimal output
    unittest.main(verbosity=1)
