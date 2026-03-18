"""
Generate training data for diffusion model pre-training using AME states.

This script generates AME states with varying parameters and creates
noisy training samples for the diffusion model pre-training stage.
"""

import torch
import numpy as np
import config
from main import generate_ame_state, partial_trace_loss
import os

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def generate_training_dataset(num_samples=10, save_path='training_data_ame.pt'):
    """
    Generate AME states and create noisy training samples.
    
    Args:
        num_samples: Number of AME states to generate
        save_path: Path to save the training dataset
    
    Returns:
        Dict with noisy states, clean states, and timesteps
    """
    d = config.D
    n = config.N
    vec_len = d ** n
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Generating {num_samples} training AME({n},{d}) states...")
    
    clean_states = []
    
    # Generate AME states with varying guidance scales for diversity
    guidance_scales = np.linspace(0.05, 0.9, num_samples)
    
    for idx, gs in enumerate(guidance_scales):
        try:
            final_state, _ = generate_ame_state(
                guidance_scale=gs,
                num_steps=config.TRAINING_DATA_GENERATION_STEPS,
                verbose=False,
                d=d,
                n=n,
                pretrain=False,  # Don't pre-train during data generation
                use_lbfgs=False
            )
            
            # Verify it's a valid state
            final_state = final_state.to(device)
            state_real_imag = torch.stack([final_state.real, final_state.imag], dim=0).unsqueeze(0)
            loss = partial_trace_loss(state_real_imag, d=d, n=n)
            
            clean_states.append(final_state)
            
        except Exception as e:
            continue
    
    if not clean_states:
        return None
    
    # Stack clean states
    clean_states_tensor = torch.stack(clean_states).to(device)
    batch_size = clean_states_tensor.shape[0]
    
    # Create multiple noisy versions of each state (at different noise levels)
    num_noise_levels = config.TRAINING_DATA_NOISE_LEVELS
    noisy_states_list = []
    timesteps_list = []
    clean_states_list = []
    
    timestep_values = [
        int(1000 * 0.2),  # Low noise
        int(1000 * 0.4),  # Medium noise
        int(1000 * 0.6),  # 
        int(1000 * 0.8),  # 
        int(1000 * 0.95), # Heavy noise
    ]
    
    for noise_idx, t_value in enumerate(timestep_values):
        t = torch.full((batch_size,), t_value, device=device)
        noise = torch.randn(batch_size, vec_len, dtype=clean_states_tensor.dtype, device=device)
        
        # Diffusion schedule: α(t) from 1 (clean) to 0 (total noise)
        alpha = torch.cos(t.float() * np.pi / 2000).view(-1, 1) ** 2
        x_noisy = torch.sqrt(alpha) * clean_states_tensor + torch.sqrt(1 - alpha) * noise
        
        noisy_states_list.append(x_noisy)
        timesteps_list.append(t)
        clean_states_list.append(clean_states_tensor)
    
    # Concatenate all noise levels
    all_noisy = torch.cat(noisy_states_list, dim=0)
    all_timesteps = torch.cat(timesteps_list, dim=0)
    all_clean = torch.cat(clean_states_list, dim=0)
    
    # Convert complex to real-imag representation for model input
    noisy_real_imag = torch.stack([all_noisy.real, all_noisy.imag], dim=1)
    clean_real_imag = torch.stack([all_clean.real, all_clean.imag], dim=1)
    
    dataset = {
        'noisy_states': noisy_real_imag.cpu(),
        'clean_states': clean_real_imag.cpu(),
        'timesteps': all_timesteps.cpu(),
        'd': d,
        'n': n,
        'num_base_samples': batch_size,
        'num_noise_levels': num_noise_levels,
    }
    
    # Save to disk
    torch.save(dataset, save_path)
    print(f"Training dataset saved ({len(all_noisy)} samples)")
    
    return dataset


def load_training_dataset(save_path='training_data_ame.pt'):
    """Load a previously saved training dataset."""
    if not os.path.exists(save_path):
        return None
    
    dataset = torch.load(save_path)
    return dataset


if __name__ == "__main__":
    # Use configuration from config.py
    dataset = generate_training_dataset(
        num_samples=config.TRAINING_DATA_NUM_SAMPLES,
        save_path=config.TRAINING_DATA_SAVE_PATH
    )
