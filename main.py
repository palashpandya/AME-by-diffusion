import torch
import torch.nn as nn
from diffusers import DDIMScheduler
import numpy as np

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# --- 1. MODEL DEFINITION ---
class QuantumDiffusionMLP(nn.Module):
    def __init__(self, vec_len):
        super().__init__()
        self.input_dim = (vec_len * 2) + 1
        
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 512),
            # nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, vec_len * 2) 
        )

    def forward(self, x, t):
        batch_size = x.shape[0]
        x_flat = x.view(batch_size, -1)
        
        if t.ndim == 1:
            t = t.unsqueeze(-1)
        elif t.ndim == 0:
            t = t.expand(batch_size, 1)
        
        combined = torch.cat([x_flat, t.float()], dim=-1)
        out = self.net(combined)
        return out.view(batch_size, 2, -1)

# --- 2. MULTI-PARTITION LOSS ---
def partial_trace_loss(state_real_imag, d=3):
    psi = torch.complex(state_real_imag[:, 0, :], state_real_imag[:, 1, :])
    psi = psi / torch.norm(psi, dim=-1, keepdim=True)
    psi_tensor = psi.view(-1, d, d, d, d)
    
    target = torch.eye(d**2, device=psi.device).unsqueeze(0) / (d**2)
    total_loss = 0
    
    # Check three primary balanced bipartitions for 4 qubits/qutrits
    # (0,1 vs 2,3), (0,2 vs 1,3), (0,3 vs 1,2)
    partitions = [
        ('B r s k l, B t u k l -> B r s t u', "01|23"),
        ('B r k s l, B t k u l -> B r s t u', "02|13"),
        ('B r k l s, B t k l u -> B r s t u', "03|12")
    ]
    
    for einsum_str, label in partitions:
        rho_A = torch.einsum(einsum_str, psi_tensor, psi_tensor.conj())
        rho_A = rho_A.reshape(-1, d**2, d**2)
        total_loss += torch.linalg.matrix_norm(rho_A - target, ord='fro').mean()
        
    return total_loss / len(partitions)

# --- 3. MAIN GENERATION FUNCTION ---
def generate_ame_state(guidance_scale=0.2, num_steps=2000, verbose=False):
    d, n = 3, 4
    vec_len = d**n
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = QuantumDiffusionMLP(vec_len).to(device)
    model.eval() 
    
    scheduler = DDIMScheduler(num_train_timesteps=num_steps, beta_schedule="squaredcos_cap_v2",clip_sample=False)
    scheduler.set_timesteps(num_steps)

    x = torch.randn((1, 2, vec_len), device=device)
    
    loss_history = []  # Track loss trajectory

    if verbose:
        print(f"Generating AME({n},{d}) state with guidance_scale={guidance_scale}, num_steps={num_steps}...")

    for i, t in enumerate(scheduler.timesteps):
        with torch.no_grad():
            t_vec = torch.full((1, 1), t, device=device)
            noise_pred = model(x, t_vec)
        
        x = scheduler.step(noise_pred, t, x).prev_sample

        # Guidance
        x = x.detach().requires_grad_(True)
        loss = partial_trace_loss(x, d=d)
        grads = torch.autograd.grad(loss, x)[0]
        
        with torch.no_grad():
            scale_factor = 1.#torch.clamp(torch.sin((t / num_steps) * 3.14159 / 2), min=0.1)
            current_guidance = guidance_scale * scale_factor
            x = x - current_guidance * grads
            # Normalize complex vector
            mag = torch.sqrt(torch.sum(x**2, dim=(1,2), keepdim=True))
            x = x / mag

        if verbose and i % max(1, len(scheduler.timesteps) // 5) == 0:
            print(f"  Step {i}/{len(scheduler.timesteps)}: Loss = {loss.item():.6f}")
        
        # Loss trajectory logging
        if i % 50 == 0:
            current_loss = partial_trace_loss(x, d=d).item()
            loss_history.append((i, current_loss))
            print(f"    guidance_scale={guidance_scale}: Step {i} loss = {current_loss:.6f}")

    return torch.complex(x[:,0,:], x[:,1,:]).detach().cpu(), loss_history


if __name__ == "__main__":
    from benchmark import benchmark_ame_generation, display_benchmark_results, plot_loss_trajectories
    
    # Run benchmarking with specified parameter ranges
    results = benchmark_ame_generation(
        guidance_scales=[0.05, 0.1, 0.2],
        num_steps_list=[ 1000, 2000, 4000]
    )
    
    # Display formatted results
    display_benchmark_results(results)
    
    # Plot loss trajectories
    plot_loss_trajectories(results)
    
    print("Benchmarking Complete!")
    