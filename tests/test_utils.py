"""
Tests for utility functions (data generation and benchmarking).
"""

import pytest
import torch
import os
from src.utils.data import generate_training_dataset, load_training_dataset
from src.utils.benchmark import benchmark_ame_generation, display_benchmark_results
from src import config


class TestDataGeneration:
    """Test suite for training data generation."""

    @pytest.mark.slow
    def test_generate_training_dataset_structure(self, tmp_path):
        """Test generated dataset has correct structure."""
        save_path = str(tmp_path / "test_data.pt")
        
        dataset = generate_training_dataset(
            num_samples=2,  # Small for testing
            save_path=save_path
        )
        
        assert dataset is not None
        assert 'noisy_states' in dataset
        assert 'clean_states' in dataset
        assert 'timesteps' in dataset
        assert 'd' in dataset
        assert 'n' in dataset

    @pytest.mark.slow
    def test_generate_training_dataset_shapes(self, tmp_path):
        """Test generated dataset has correct tensor shapes."""
        save_path = str(tmp_path / "test_data.pt")
        num_samples = 2
        d = config.D
        n = config.N
        vec_len = d ** n
        
        dataset = generate_training_dataset(
            num_samples=num_samples,
            save_path=save_path
        )
        
        # Check shapes
        assert dataset['noisy_states'].shape[1] == 2  # Real and imag channels
        assert dataset['clean_states'].shape[1] == 2
        assert dataset['timesteps'].dtype in [torch.int32, torch.int64]

    @pytest.mark.slow
    def test_generate_training_dataset_saved_file(self, tmp_path):
        """Test dataset is saved to disk correctly."""
        save_path = str(tmp_path / "test_data.pt")
        
        dataset = generate_training_dataset(
            num_samples=2,
            save_path=save_path
        )
        
        # Check file exists
        assert os.path.exists(save_path)
        
        # Try to load it back
        loaded = torch.load(save_path)
        assert loaded is not None
        assert 'noisy_states' in loaded

    @pytest.mark.slow
    def test_generate_training_dataset_dimensions(self, tmp_path):
        """Test dataset respects config dimensions."""
        save_path = str(tmp_path / "test_data.pt")
        
        dataset = generate_training_dataset(
            num_samples=2,
            save_path=save_path
        )
        
        assert dataset['d'] == config.D
        assert dataset['n'] == config.N

    def test_load_training_dataset_nonexistent(self):
        """Test loading nonexistent dataset returns None."""
        result = load_training_dataset(save_path="nonexistent_xyz.pt")
        
        assert result is None

    @pytest.mark.slow
    def test_load_training_dataset_saved_file(self, tmp_path):
        """Test loading previously generated dataset."""
        save_path = str(tmp_path / "test_data.pt")
        
        # Generate dataset
        generated = generate_training_dataset(
            num_samples=2,
            save_path=save_path
        )
        
        # Load it
        loaded = load_training_dataset(save_path=save_path)
        
        assert loaded is not None
        assert torch.allclose(loaded['noisy_states'], generated['noisy_states'])
        assert torch.allclose(loaded['clean_states'], generated['clean_states'])


class TestBenchmarking:
    """Test suite for benchmarking utilities."""

    @pytest.mark.slow
    def test_benchmark_basic(self):
        """Test basic benchmarking execution."""
        results = benchmark_ame_generation(
            guidance_scales=[0.05],
            num_steps_list=[5],
            pretrain=False
        )
        
        assert results is not None
        assert len(results) == 1
        assert isinstance(results, list)

    @pytest.mark.slow
    def test_benchmark_result_structure(self):
        """Test benchmark results have correct structure."""
        results = benchmark_ame_generation(
            guidance_scales=[0.05],
            num_steps_list=[5],
            pretrain=False
        )
        
        result = results[0]
        assert 'guidance_scale' in result
        assert 'num_steps' in result
        assert 'ame_loss' in result
        assert 'loss_history' in result

    @pytest.mark.slow
    def test_benchmark_multiple_configs(self):
        """Test benchmark with multiple configurations."""
        results = benchmark_ame_generation(
            guidance_scales=[0.03, 0.05],
            num_steps_list=[5],
            pretrain=False
        )
        
        assert len(results) == 2
        assert results[0]['guidance_scale'] == 0.03
        assert results[1]['guidance_scale'] == 0.05

    @pytest.mark.slow
    def test_benchmark_loss_values_reasonable(self):
        """Test benchmark produces reasonable loss values."""
        results = benchmark_ame_generation(
            guidance_scales=[0.05],
            num_steps_list=[5],
            pretrain=False
        )
        
        for result in results:
            ame_loss = result['ame_loss']
            assert isinstance(ame_loss, float)
            assert ame_loss >= 0.0
            assert not float('inf') == ame_loss

    def test_display_benchmark_results_basic(self, capsys):
        """Test benchmark results display."""
        results = [
            {
                'guidance_scale': 0.1,
                'num_steps': 100,
                'ame_loss': 0.5,
                'loss_history': []
            },
            {
                'guidance_scale': 0.2,
                'num_steps': 200,
                'ame_loss': 0.3,
                'loss_history': []
            }
        ]
        
        display_benchmark_results(results)
        
        captured = capsys.readouterr()
        assert '0.1' in captured.out or 'Guidance Scale' in captured.out
        assert '0.5' in captured.out or 'AME Loss' in captured.out
