"""
Project-wide configuration for AME generation.

Define `D` (qudit dimension) and `N` (number of parties/qudits) here
so the rest of the codebase uses a single source of truth.
"""

# Default AME parameters
D = 6
N = 4

# Benchmarking configurations (pre-train epochs, guidance scales and step counts)
BENCHMARK_EPOCHS = 250
BENCHMARK_GUIDANCE_SCALES = [0.2, 0.5, 0.7]
BENCHMARK_NUM_STEPS = [1000]

# Training data generation config
TRAINING_DATA_NUM_SAMPLES = 100  # Number of AME states to generate
TRAINING_DATA_SAVE_PATH = 'training_data_ame_UNET.pt'  # Path to save dataset
TRAINING_DATA_GENERATION_STEPS = 500  # Steps per AME state generation
TRAINING_DATA_NOISE_LEVELS = 3  # Number of noise level variants per state