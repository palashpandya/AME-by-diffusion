import torch
import torch.nn as nn
from diffusers import DDPMScheduler as SCHEDULER
from ..models.unet import GeneralDynamicUNet
import numpy as np
from .. import config
import os
from .loss import make_density_matrix, purity, partial_trace_loss_optimized
import matplotlib.pyplot as plt

# Global configurations for functions in this module
SEED = 42 # Fixed seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- VERIFICATION FUNCTION ---
def verify_ame_properties(final_state, n=config.N, d=config.D):
    """
    Computes the AME quality metric (partial trace loss) on the final state.
    Lower values indicate better AME properties.
    
    Args:
        final_state: Complex tensor of shape (1, vec_len) or (vec_len,)
        d: Qudit dimension
    
    Returns:
        Float scalar representing the AME loss
    """
    # Ensure correct shape and convert to real-imag format for loss computation
    if final_state.ndim == 1:
        final_state = final_state.unsqueeze(0)  # Add batch dimension if needed
    
    # Stack real and imaginary parts
    state_real_imag = torch.stack([final_state.real, final_state.imag], dim=1)
    loss = partial_trace_loss_optimized(state_real_imag, n=n, d=d) #+ torch.abs(1/purity(make_density_matrix(state_real_imag, d=d, n=config.N)) - 1.0)  # Add purity penalty
    return loss.item()


# --- TRAINING DATA LOADING ---
def load_ame_training_data(save_path=config.TRAINING_DATA_SAVE_PATH):
    """Load pre-generated AME training dataset.

    Args:
        save_path: Path to the saved training dataset

    Returns:
        Dictionary with training data, or None if not found
    """
    if not os.path.exists(save_path):
        return None

    try:
        dataset = torch.load(save_path)
        return dataset
    except Exception as e:
        print(f"Error loading training data from {save_path}: {e}")
        return None


# --- MODEL PRE-TRAINING ---
def train_diffusion_model(model, num_epochs=100, batch_size=16, d=config.D, n=config.N, dataset=None):
    """Pre-train the diffusion model on AME training data.

    Args:
        model: The diffusion model to train
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        d, n: Qudit dimension and number of qudits
        dataset: Pre-generated training dataset. If None, attempts to load from disk.

    Raises:
        FileNotFoundError: If training dataset cannot be found or loaded
    """
    device = next(model.parameters()).device
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    # Try to load dataset if not provided
    if dataset is None:
        dataset = load_ame_training_data()

    # Require dataset to exist
    if dataset is None or 'noisy_states' not in dataset:
        raise FileNotFoundError(
            f"Training data required but not found at {config.TRAINING_DATA_SAVE_PATH}\n"
            f"Generate it with: python generate_training_data.py"
        )

    noisy_data = dataset['noisy_states'].to(device)
    clean_data = dataset['clean_states'].to(device)
    timesteps_data = dataset['timesteps'].to(device)
    num_samples = len(noisy_data)

    print(f"Training diffusion model on {num_samples} samples...")

    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0

        # Shuffle indices
        indices = torch.randperm(num_samples, device=device)

        # Process batches
        for batch_start in range(0, num_samples, batch_size):
            batch_end = min(batch_start + batch_size, num_samples)
            batch_indices = indices[batch_start:batch_end]

            x_noisy = noisy_data[batch_indices]
            x_clean = clean_data[batch_indices]
            t_batch = timesteps_data[batch_indices]

            # Model predicts noise residual
            t_vec = t_batch.view(-1, 1).float()
            pred_noise = model(x_noisy, t_vec)

            # Compute noise residual (difference from clean state)
            noise_residual = x_noisy - x_clean

            # MSE loss and AME penalty 
            loss = nn.functional.mse_loss(pred_noise, noise_residual) + partial_trace_loss_optimized(x_noisy, d=d, n=n) * 0.5 

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.6f}")

    model.eval()

# --- L-BFGS FINE-TUNING ---
def fine_tune_with_lbfgs(state_complex, d, n, max_iters=500, verbose=False):
    """Fine-tune state using L-BFGS for very tight convergence.

    Args:
        state_complex: Complex-valued state tensor of shape (vec_len,)
        d, n: Qudit dimension and number of qudits
        max_iters: Maximum iterations for L-BFGS
    """
    # Convert to real representation for optimization
    state_real = state_complex.real.to(torch.float64).clone().detach().requires_grad_(True)
    state_imag = state_complex.imag.to(torch.float64).clone().detach().requires_grad_(True)
    
    params = [state_real, state_imag]
    optimizer = torch.optim.LBFGS( params, lr=1, max_iter=1000, tolerance_change=1e-16,tolerance_grad=1e-16, line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        
        state_real_imag = torch.stack([state_real, state_imag], dim=0).unsqueeze(0)

        loss = partial_trace_loss_optimized(state_real_imag, d=d, n=n)
        loss.backward()
        return loss

    for it in range(max_iters // 20): # L-BFGS max_iter is per step, not per closure call
        optimizer.step(closure)

    with torch.no_grad():
        state_final = torch.complex(state_real.detach().to(torch.float32), state_imag.detach().to(torch.float32))

    return state_final

# --- PRETRAIN WRAPPER ---
def pretrain_diffusion_model(save_path='pretrained_model.pt', num_epochs=None, batch_size=16, d=None, n=None):
    """Pre-train diffusion model once and save to disk.
    
    Args:
        save_path: Path to save trained model weights
        num_epochs: Number of training epochs (uses config.BENCHMARK_EPOCHS if None)
        batch_size: Batch size for training
        d, n: Qudit dimension and number of qudits (uses config if None)
    
    Returns:
        Trained model
    """
    if num_epochs is None:
        num_epochs = config.BENCHMARK_EPOCHS
    if d is None:
        d = config.D
    if n is None:
        n = config.N
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Pre-training diffusion model for AME({n},{d}) on device: {device}") 
    
    # Load dataset
    dataset = load_ame_training_data()
    if dataset is None or 'noisy_states' not in dataset:
        raise FileNotFoundError(
            f"Training data required but not found at {config.TRAINING_DATA_SAVE_PATH}\n"
            f"Generate it with: python generate_training_data.py"
        )
    
    # Use dimensions from dataset if available
    if 'd' in dataset and 'n' in dataset:
        d = dataset['d']
        n = dataset['n']
    
    vec_len = d**n
    model = GeneralDynamicUNet().to(device)
    # model = QuantumDiffusionMLP(vec_len, hidden_dim=2048, num_layers=10).to(device) # Commented out in original
    
    # Train
    train_diffusion_model(model, num_epochs=num_epochs, batch_size=batch_size, d=d, n=n, dataset=dataset)
    
    # Save
    torch.save(model.state_dict(), save_path)
    print(f"Pre-trained model saved to {save_path}")
    
    return model

# --- MAIN GENERATION FUNCTION (generate_ame_state2) ---
def generate_ame_state2(guidance_scale=0.2, num_steps=2000, verbose=False, d=config.D, n=config.N, use_lbfgs=True, pretrain=True, model_path='pretrained_model.pt'):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # If pretraining, load dataset to ensure model matches data dimensions
    dataset = None
    if pretrain:
        dataset = load_ame_training_data()
        if dataset is not None and 'd' in dataset and 'n' in dataset:
            # Use dimensions from the training data
            d = dataset['d']
            n = dataset['n']
            if d!=config.D or n!=config.N:
                print(f"Warning: Training data dimensions (d={d}, n={n}) differ from config (D={config.D}, N={config.N}). Using training data dimensions for generation.")


    vec_len = d**n
    # model = QuantumDiffusionMLP(vec_len, hidden_dim=2048, num_layers=10).to(device) # Commented out in original
    model = GeneralDynamicUNet().to(device)

    # Load pre-trained weights if available
    if pretrain and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        # Warn if pre-trained model not found
        if pretrain:
            print(f"Warning: Pre-trained model not found at {model_path}")
            print("Run pretrain_diffusion_model() first for better results")

    model.eval()

    print(f"Generating AME({n},{d}) with guidance_scale={guidance_scale}, num_steps={num_steps} on device: {device}")

    scheduler = SCHEDULER(num_train_timesteps=num_steps, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.set_timesteps(num_steps)

    x = torch.randn((1, 2, vec_len), device=device)

    loss_history = []  # Track loss trajectory

    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            t_vec = torch.full((1, 1), t.item()/scheduler.config.num_train_timesteps, device=device)
            noise_pred = model(x, t_vec)

        x0 = scheduler.step(noise_pred, t, x).pred_original_sample
        # Adaptive guidance: increase as we progress
        progress = i / len(scheduler.timesteps)
        adaptive_scale = guidance_scale * (1.0 + 5.0 * progress)  # Scales from 1x to 5x

        # Gradient-based guidance
        x = x.requires_grad_(True)
        loss = partial_trace_loss_optimized(x0, d=d, n=n)
        grads = torch.autograd.grad(loss, x)[0]

        with torch.no_grad():
            x = x - adaptive_scale * grads
            # Normalize complex vector
            mag = torch.sqrt(torch.sum(x**2, dim=(1,2), keepdim=True))
            x = x / mag
            x = scheduler.step(noise_pred, t, x).prev_sample

        # Loss trajectory logging
        if i % 50 == 0:
            current_loss = partial_trace_loss_optimized(x, d=d, n=n).item()
            current_purity = purity(make_density_matrix(x, d=d, n=n)).item()
            loss_history.append((i, current_loss))
            if verbose:
                print(f"Step {i}/{len(scheduler.timesteps)} - Loss: {current_loss:.6f}, Purity: {current_purity:.6f}")

    # Final fine-tuning with L-BFGS
    if use_lbfgs:
        x_final = torch.complex(x[:,0,:], x[:,1,:]).squeeze()
        x_optimized = fine_tune_with_lbfgs(x_final, d=d, n=n, verbose=verbose)
        print("Purity after lbfgs fine-tuning:", purity(make_density_matrix(torch.stack([x_optimized.real, x_optimized.imag], dim=0).unsqueeze(0), d=d, n=n)).item())
        return x_optimized.cpu(), loss_history
    else:
        result = torch.complex(x[:,0,:], x[:,1,:]).squeeze(0).detach().cpu()
        return result, loss_history
    
def generate_ame_state(guidance_scale=0.1, num_steps=1000, d=config.D, n=config.N, use_lbfgs=True):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec_len = d**n
    
    # Initialize Model
    model = GeneralDynamicUNet().to(device)
    model.eval() # Ensure dropout/norm layers are in eval mode

    # Setup Scheduler
    scheduler = SCHEDULER(num_train_timesteps=num_steps, beta_schedule="squaredcos_cap_v2", clip_sample=False)
    scheduler.set_timesteps(num_steps)

    # Initial Random Noise (Batch, Channels, Length)
    initial_state_path = f"data/AME_{config.N}_{config.D}_state_vector.npy"
    try:
        # Load the complex vector and convert to the model's expected real-imag format
        loaded_vector = torch.from_numpy(np.load(initial_state_path)).to(device)
        x = torch.stack([loaded_vector.real, loaded_vector.imag], dim=0).unsqueeze(0)
        print(f"Successfully loaded initial state from {initial_state_path}")
    except FileNotFoundError:
        print(f"Initial state file not found at {initial_state_path}. Starting from random noise.")
        x = torch.randn((1, 2, vec_len), device=device)

    x = x / torch.norm(x) # Normalize the initial state

    for i, t in enumerate(scheduler.timesteps):
        # 1. Predict Noise Residual
        with torch.no_grad():
            # Standardize t for the UNet (0 to 1 range)
            t_input = torch.full((1, 1), t.item() / scheduler.config.num_train_timesteps, device=device)
            noise_pred = model(x, t_input)

        # 2. ALGEBRAIC RECONSTRUCTION OF x0 (The "Peek")
        # Extract alpha_bar from scheduler for this specific timestep
        alpha_bar_t = scheduler.alphas_cumprod[t]
        
        # Enable gradients for guidance nudge
        x = x.detach().requires_grad_(True)
        
        # Direct formula for x0: (x_t - sqrt(1-alpha)*noise) / sqrt(alpha)
        x0_recon = (x - torch.sqrt(1 - alpha_bar_t) * noise_pred) / torch.sqrt(alpha_bar_t)
        
        # 3. SPECTRAL GUIDANCE
        # Calculate AME loss on the clean reconstruction
        loss = partial_trace_loss_optimized(x0_recon, d=d, n=n)
        
        # Compute gradient of loss with respect to noisy input x
        grads = torch.autograd.grad(loss, x)[0]
        
        # 4. NORMALIZED NUDGE
        # We normalize the gradient and apply adaptive scaling
        progress = i / len(scheduler.timesteps)
        adaptive_scale = guidance_scale * (1.0 + 2.0 * progress) # Stronger guidance as noise decreases
        
        with torch.no_grad():
            # Directional nudge
            norm_grads = grads / (torch.norm(grads) + 1e-9)
            x = x - adaptive_scale * norm_grads
            
            # 5. STANDARD SCHEDULER STEP (t -> t-1)
            x = scheduler.step(noise_pred, t, x).prev_sample
            
            # 6. GLOBAL PHASE PINNING
            # Convert to complex to find phase of the first element
            psi_c = torch.complex(x[0, 0, :], x[0, 1, :])
            phase_shift = torch.angle(psi_c[0])
            
            # Rotate x to keep first element real
            cos_p = torch.cos(-phase_shift)
            sin_p = torch.sin(-phase_shift)
            
            real_new = x[:, 0, :] * cos_p - x[:, 1, :] * sin_p
            imag_new = x[:, 0, :] * sin_p + x[:, 1, :] * cos_p
            x = torch.stack([real_new, imag_new], dim=1)
            
            # Normalize to maintain pure state manifold
            x = x / torch.norm(x)

        if i % 100 == 0:
            print(f"Step {i}/{num_steps} | Recon AME Loss: {loss.item():.6f}")

    # Final result
    final_complex = torch.complex(x[0, 0, :], x[0, 1, :])
    
    if use_lbfgs:
        print("Starting High-Precision L-BFGS Fine-tuning...")
        final_complex = fine_tune_with_lbfgs(final_complex, d, n)

    return final_complex
