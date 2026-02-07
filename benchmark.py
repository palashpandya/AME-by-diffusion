"""
Benchmarking module for AME (Absolutely Maximally Entangled) state generation.

This module contains functions for:
- Verifying AME properties of generated quantum states
- Running benchmarks across different parameter configurations
- Visualizing loss trajectories and results
"""

import torch
import numpy as np
import itertools
from tabulate import tabulate
import matplotlib.pyplot as plt

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

from main import generate_ame_state, partial_trace_loss


# --- VERIFICATION FUNCTION ---
def verify_ame_properties(final_state, d=3):
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
    loss = partial_trace_loss(state_real_imag, d=d)
    return loss.item()


# --- BENCHMARKING FUNCTION ---
def benchmark_ame_generation(guidance_scales=[0.1, 0.2], num_steps_list=[2000, 4000]):
    """
    Benchmark the AME generation with different guidance scales and number of steps.
    
    Args:
        guidance_scales: List of guidance scale values to test
        num_steps_list: List of number of steps to test
    
    Returns:
        results: List of dictionaries with benchmark results
    """
    results = []
    
    print("Starting Benchmarking...")
    print(f"Testing {len(guidance_scales)} guidance scales × {len(num_steps_list)} step counts = {len(guidance_scales) * len(num_steps_list)} configurations\n")
    
    total = len(guidance_scales) * len(num_steps_list)
    current = 0
    
    for guidance_scale, num_steps in itertools.product(guidance_scales, num_steps_list):
        current += 1
        print(f"[{current}/{total}] Running with guidance_scale={guidance_scale}, num_steps={num_steps}...")
        
        final_state, loss_history = generate_ame_state(guidance_scale=guidance_scale, num_steps=num_steps, verbose=False)
        ame_loss = verify_ame_properties(final_state, d=3)
        
        results.append({
            'guidance_scale': guidance_scale,
            'num_steps': num_steps,
            'ame_loss': ame_loss,
            'loss_history': loss_history
        })
        
        print(f"  → AME Loss: {ame_loss:.6f}\n")
    
    return results


# --- DISPLAY RESULTS ---
def display_benchmark_results(results):
    """Display benchmark results in a formatted table."""
    table_data = []
    for result in results:
        table_data.append([
            result['guidance_scale'],
            result['num_steps'],
            f"{result['ame_loss']:.6f}"
        ])
    
    
    print("\nBENCHMARK RESULTS\n")
    
    print(tabulate(table_data, 
                   headers=['Guidance Scale', 'Num Steps', 'AME Loss'],
                   tablefmt='grid',
                   floatfmt='.6f'))
    


# --- PLOTTING FUNCTION ---
def plot_loss_trajectories(results):
    """
    Plot loss trajectories for all configurations.
    Creates three visualization figures:
    1. Loss vs Step for each guidance scale (grouped by num_steps)
    2. Loss vs Step for each num_steps (grouped by guidance_scale)
    3. Heatmap of final losses across all configurations
    """
    guidance_scales = sorted(set(r['guidance_scale'] for r in results))
    num_steps_list = sorted(set(r['num_steps'] for r in results))
    
    # Figure 1: Grouped by guidance_scale
    n_scales = len(guidance_scales)
    rows = (n_scales + 1) // 2
    cols = 2 if n_scales > 1 else 1
    
    fig1, axes1 = plt.subplots(rows, cols, figsize=(14, 5*rows))
    if rows == 1 and cols == 1:
        axes1 = np.array([axes1])
    else:
        axes1 = axes1.flatten()
    
    fig1.suptitle('Loss Trajectory by Guidance Scale', fontsize=16, fontweight='bold')
    
    for idx, gs in enumerate(guidance_scales):
        ax = axes1[idx]
        for num_steps in num_steps_list:
            result = next((r for r in results if r['guidance_scale'] == gs and r['num_steps'] == num_steps), None)
            if result and result['loss_history']:
                steps, losses = zip(*result['loss_history'])
                ax.plot(steps, losses, marker='o', label=f'Steps: {num_steps}', linewidth=2)
        
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(f'Guidance Scale = {gs}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if needed
    for idx in range(len(guidance_scales), len(axes1)):
        axes1[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('loss_trajectories_by_guidance2.png', dpi=300, bbox_inches='tight')
    print("Saved: loss_trajectories_by_guidance2.png")
    
    # Figure 2: Grouped by num_steps
    n_steps = len(num_steps_list)
    rows2 = (n_steps + 1) // 2
    cols2 = 2 if n_steps > 1 else 1
    
    fig2, axes2 = plt.subplots(rows2, cols2, figsize=(14, 5*rows2))
    if rows2 == 1 and cols2 == 1:
        axes2 = np.array([axes2])
    else:
        axes2 = axes2.flatten()
    
    fig2.suptitle('Loss Trajectory by Number of Steps', fontsize=16, fontweight='bold')
    
    for idx, ns in enumerate(num_steps_list):
        ax = axes2[idx]
        for gs in guidance_scales:
            result = next((r for r in results if r['guidance_scale'] == gs and r['num_steps'] == ns), None)
            if result and result['loss_history']:
                steps, losses = zip(*result['loss_history'])
                ax.plot(steps, losses, marker='s', label=f'Guidance: {gs}', linewidth=2)
        
        ax.set_xlabel('Step', fontsize=11)
        ax.set_ylabel('Loss', fontsize=11)
        ax.set_title(f'Num Steps = {ns}', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide extra subplots if needed
    for idx in range(len(num_steps_list), len(axes2)):
        axes2[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('loss_trajectories_by_steps2.png', dpi=300, bbox_inches='tight')
    print("Saved: loss_trajectories_by_steps2.png")
    
    # Figure 3: Final loss heatmap
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    
    # Create matrix for heatmap
    loss_matrix = np.zeros((len(guidance_scales), len(num_steps_list)))
    for i, gs in enumerate(guidance_scales):
        for j, ns in enumerate(num_steps_list):
            result = next((r for r in results if r['guidance_scale'] == gs and r['num_steps'] == ns), None)
            if result:
                loss_matrix[i, j] = result['ame_loss']
    
    im = ax3.imshow(loss_matrix, cmap='RdYlGn_r', aspect='auto')
    ax3.set_xticks(range(len(num_steps_list)))
    ax3.set_yticks(range(len(guidance_scales)))
    ax3.set_xticklabels(num_steps_list)
    ax3.set_yticklabels(guidance_scales)
    ax3.set_xlabel('Number of Steps', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Guidance Scale', fontsize=12, fontweight='bold')
    ax3.set_title('Final AME Loss Heatmap (Lower is Better)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(guidance_scales)):
        for j in range(len(num_steps_list)):
            text = ax3.text(j, i, f'{loss_matrix[i, j]:.4f}',
                          ha="center", va="center", color="black", fontsize=10, fontweight='bold')
    
    plt.colorbar(im, ax=ax3, label='AME Loss')
    plt.tight_layout()
    plt.savefig('loss_heatmap2.png', dpi=300, bbox_inches='tight')
    print("Saved: loss_heatmap2.png")


# --- VERIFICATION REPORTING ---
def print_ame_verification(state_vector, d=3):
    """
    Checks the entanglement properties of the state across all 2-party cuts.
    
    Args:
        state_vector: Complex tensor of shape (vec_len,)
        d: Qudit dimension
    """
    print("\n--- AME Verification Report ---")
    
    # Ensure state is a tensor and reshaped correctly
    psi = state_vector.view(d, d, d, d)
    
    # Define the balanced bipartitions (cuts)
    # Each tuple contains: (einsum_string, label)
    partitions = [
        ('abij, abkl -> ijkl', "Qubits 0,1 | 2,3"),
        ('aibj, akbl -> ijkl', "Qubits 0,2 | 1,3"),
        ('aijb, aklb -> ijkl', "Qubits 0,3 | 1,2")
    ]
    
    target_purity = 1.0 / (d**2)
    max_entropy = np.log(d**2)
    
    for einsum_str, label in partitions:
        # 1. Compute Reduced Density Matrix (RDM)
        # We contract 2 indices to leave a (d^2 x d^2) matrix
        rho = torch.einsum(einsum_str, psi, psi.conj())
        rho = rho.reshape(d**2, d**2)
        
        # 2. Calculate Purity: Tr(rho^2)
        purity = torch.trace(torch.matmul(rho, rho)).real.item()
        
        # 3. Calculate Von Neumann Entropy: -Tr(rho log rho)
        # We use eigvals because rho is Hermitian
        eigvals = torch.linalg.eigvalsh(rho)
        # Filter out near-zero eigenvalues for numerical stability
        eigvals = eigvals[eigvals > 1e-10]
        entropy = -torch.sum(eigvals * torch.log(eigvals)).item()
        
        print(f"Partition {label}:")
        print(f"  > Purity:  {purity:.6f} (Target: {target_purity:.6f})")
        print(f"  > Entropy: {entropy:.6f} (Target: {max_entropy:.6f})")
