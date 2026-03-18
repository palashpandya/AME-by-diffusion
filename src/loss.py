import torch
import torch.nn as nn
from diffusers import DDPMScheduler as SCHEDULER
from unet.unet import GeneralDynamicUNet
import numpy as np
import config
from itertools import combinations
import string

DIM_LIST = [config.D] * config.N  # For partial trace calculations
SEED = 42 # Fixed seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)



# --- Make Density Matrix ---
def make_density_matrix(state_real_imag, d, n):
    """Convert real-imag tensor to complex state and form density matrix."""
    psi = torch.complex(state_real_imag[:, 0, :], state_real_imag[:, 1, :])
    rho = torch.matmul( psi.conj().transpose(-2, -1), psi)  # from vector to density matrix (outer product)
    # rho = psi + psi.conj().transpose(-2, -1)  # Density matrix (unnormalized)
    trace = torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1, keepdim=True)
    rho = rho / (trace.unsqueeze(-1))  # Normalize to trace 1
    return rho


# --- Density Matrix Purity Utilities ---
def purity(rho: torch.Tensor) -> torch.Tensor:
    """Return the purity of a density matrix ``rho``.
    The input may contain a batch dimension; 
    the returned tensor has the same leading dimensions as ``rho``
    excluding the final two.

    Args:
        rho: complex-valued density matrix of shape ``(..., d, d)``.

    Returns:
        A real tensor of purities with shape ``(...)``.
    """
    # matrix multiplication handles batch dimensions automatically
    squared = torch.matmul(rho, rho)
    # trace over last two dims
    tr = torch.diagonal(squared, dim1=-2, dim2=-1).sum(-1)
    return tr.real


def is_pure(rho: torch.Tensor, tol: float = 1e-6) -> torch.Tensor:
    """Check whether ``rho`` is (approximately) a pure state.

    This simply computes the purity and compares it to 1 within ``tol``.
    Works with batched inputs; returns a boolean tensor of the same leading
    shape as ``rho`` except for the last two matrix dimensions.

    Args:
        rho: density matrix with shape ``(..., d, d)``.
        tol: numerical tolerance for purity comparison (default ``1e-6``).

    Returns:
        A boolean tensor where ``True`` indicates a (nearly) pure density
        matrix.
    """
    pur = purity(rho)
    return torch.abs(pur - 1.0) <= tol

# --- Partial Trace on the given subsystems ---
def partial_trace(rho, tout, dims):
    """
    Takes the partial trace of rho over subsystems in tout.
    rho: The density matrix (Batch, d^n, d^n) or (d^n, d^n)
    tout: List/tuple of indices to trace out
    dims: List of local dimensions [d, d, ..., d]
    """
    n = config.N

    num_indices = 2 * n
    indices = list(string.ascii_lowercase)[:num_indices]
    
    # Logic to trace out: identify the row/column indices for the same subsystem
    # and give them the same letter so einsum sums over them.
    indices_list = list(indices)
    for sys in tout:
        indices_list[sys + n] = indices_list[sys]
    
    # Reshape rho into its multi-partite form (d, d, ..., d, d, d, ..., d)
    # rho1 shape becomes (d_1, d_2... d_n, d_1, d_2... d_n)
    rho1 = rho.reshape(*(dims + dims))
    
    # Contract (trace)
    einsum_str = ''.join(indices_list)
    rho_traced = torch.einsum(einsum_str, rho1)
    
    # Calculate new dimensions for the resulting matrix
    newdims = [dims[i] for i in range(len(dims)) if i not in tout]
    new_size = int(torch.prod(torch.tensor(newdims)))
    
    return rho_traced.reshape(new_size, new_size)


# --- MULTI-PARTITION LOSS ---
# def partial_trace_loss(state_real_imag, d=3, n=4, purity_weight=1.0):
#     """Compute AME loss for n qudits of dimension d.
    
#     Args:
#         state_real_imag: Input state in real-imag format
#         d: Local dimension of each qudit
#         n: Number of qudits
#         purity_weight: Weight for the global purity constraint (default 1.0)
    
#     Returns:
#         Total loss (partial trace loss + purity constraint)
#     """
    
#     rho = make_density_matrix(state_real_imag, d, n)  # Ensure we have a proper density matrix
    
#     # Constraint 1: Global state must be pure (purity ≈ 1)
#     pur = purity(rho)
#     purity_loss = torch.mean(torch.abs(pur - 1.0))  # Penalize deviation from purity=1
    
#     # Constraint 2: All partial traces must be maximally mixed
#     #initialize loss
#     partial_trace_loss_val = 0.
#     total_combinations = 0

#     for num_trout in range(1, n//2 + 1):
#         # Iterate through all combinations of subsystems to trace out
#         comb_sys = list(combinations(range(n), num_trout))

#         # Target maximally mixed state (Identity)
#         remaining_dim = d**(n - num_trout)
#         target = torch.eye(remaining_dim, device=rho.device).unsqueeze(0) / remaining_dim

#         for sys in comb_sys:
#             # Calculate partial trace for this combination
#             reduced_rho = partial_trace(rho, tout=sys, dims=DIM_LIST)
#             # Add Frobenius norm difference to loss
#             partial_trace_loss_val += torch.linalg.matrix_norm(reduced_rho - target, ord='fro')
#         total_combinations += len(comb_sys)
    
#     # Normalize partial trace loss by number of combinations and state dimension scale
#     partial_trace_loss_val = partial_trace_loss_val / total_combinations
    
#     print(f"Total combinations: {total_combinations}, Purity: {pur.item():.6f}, PT Loss: {partial_trace_loss_val.item():.6f}")
    
#     # Combined loss with purity weight
#     total_loss = partial_trace_loss_val + purity_weight * purity_loss
#     return total_loss


#     total_loss = partial_trace_loss_val + purity_weight * purity_loss
#     return total_loss

# --- MULTI-PARTITION LOSS ---
# def partial_trace_loss(state_real_imag, d=config.D, n=config.N, purity_weight=1.0):
#     """Compute AME loss for n qudits of dimension d.
    
#     Args:
#         state_real_imag: Input state in real-imag format
#         d: Local dimension of each qudit
#         n: Number of qudits
#         purity_weight: Weight for the global purity constraint (default 1.0)

#     Returns:
#         Total loss (partial trace loss + purity constraint)
#     """
#     rho = make_density_matrix(state_real_imag, d, n)  # Ensure we have a proper density matrix

#     # Constraint 1: Global state must be pure (purity ≈ 1)
#     pur = purity(rho)
#     purity_loss = torch.mean(torch.abs(pur - 1.0))  # Penalize deviation from purity=1

#     # Constraint 2: All k-qudit reduced states (k <= n/2) must be maximally mixed
#     partial_trace_loss_val = 0.
#     total_combinations = 0
#     all_indices = set(range(n))

#     # k is the size of the subsystem we are checking
#     for k in range(1, n // 2 + 1):
#         # Iterate through all subsystems of size k
#         subsystems_to_check = list(combinations(range(n), k))

#         # Target is a maximally mixed state of dimension d**k
#         target_dim = d**k
#         target = torch.eye(target_dim, device=rho.device, dtype=rho.dtype).unsqueeze(0) / target_dim

#         for sys_to_keep in subsystems_to_check:
#             # To get the reduced state of 'sys_to_keep', we trace out its complement
#             sys_to_trace_out = tuple(all_indices - set(sys_to_keep))

#             # Calculate partial trace over the other n-k systems
#             reduced_rho = partial_trace(rho, tout=sys_to_trace_out, dims=DIM_LIST)

#             # Add Frobenius norm difference to loss
#             partial_trace_loss_val += torch.linalg.matrix_norm(reduced_rho - target, ord='fro')
#         total_combinations += len(subsystems_to_check)

#     # Normalize partial trace loss by number of combinations
#     if total_combinations > 0:
#         partial_trace_loss_val = partial_trace_loss_val / total_combinations

#     # Combined loss with purity weight
#     total_loss = partial_trace_loss_val + purity_weight * purity_loss
#     return total_loss

def partial_trace_loss_optimized(state_real_imag, d, n):
    """
    AME loss using Singular Values (Spectral Loss) and Global Phase Fixing.
    """
    # 1. Convert to complex and Normalize
    psi = torch.complex(state_real_imag[:, 0, :], state_real_imag[:, 1, :])
    psi = psi / (torch.norm(psi, dim=-1, keepdim=True) + 1e-12)
    
    # 2. FIX GLOBAL PHASE [New]
    # Rotate the state so the first component is real/positive.
    # This removes the U(1) symmetry which can cause optimization drift.
    phase = torch.angle(psi[:, 0:1])
    psi = psi * torch.exp(-1j * phase)
    
    batch_size = psi.shape[0]
    psi_tensor = psi.view(batch_size, *([d] * n))
    
    pt_loss = 0.
    k = n // 2 
    subsystems = list(combinations(range(n), k))
    
    # Target singular value for a maximally mixed state: 1/sqrt(d^k)
    target_s = 1.0 / np.sqrt(d**k)
    
    for sys_a in subsystems:
        sys_b = [i for i in range(n) if i not in sys_a]
        # Move subsystem A indices to front
        permuted = psi_tensor.permute(0, *[i+1 for i in sys_a], *[i+1 for i in sys_b])
        # Flatten into Matrix M (d^k x d^(n-k))
        matrix_m = permuted.reshape(batch_size, d**k, -1)
        
        # 3. SPECTRAL LOSS [New]
        # Instead of purity (sum of s^4), we penalize s directly.
        # This provides a constant gradient force even as we get very close to zero.
        s = torch.linalg.svdvals(matrix_m)
        pt_loss += torch.mean((s - target_s)**2)

    return pt_loss / len(subsystems)

def partial_trace_loss_optimized_old(state_real_imag, d=config.D, n=config.N, purity_weight=1.0):
    """
    Optimized AME loss using Schmidt coefficients/Purity proxy.
    Directly operates on the state vector to avoid O(d^2n) density matrices.
    """
    # 1. Normalize and convert to complex vector
    psi = torch.complex(state_real_imag[:, 0, :], state_real_imag[:, 1, :])
    psi = psi / torch.norm(psi, dim=-1, keepdim=True) 
    
    batch_size = psi.shape[0]
    # Reshape to multi-partite tensor (Batch, d, d, ..., d)
    psi_tensor = psi.view(batch_size, *([d] * n))
    
    pt_loss = 0.
    # Only need to check k = n // 2; smaller k are implied
    k = n // 2 
    subsystems = list(combinations(range(n), k))
    
    for sys_a in subsystems:
        # Move indices of subsystem A to the front
        sys_b = [i for i in range(n) if i not in sys_a]
        permuted = psi_tensor.permute(0, *[i+1 for i in sys_a], *[i+1 for i in sys_b])
        
        # Reshape into a matrix M of size (d^k, d^(n-k))
        matrix_m = permuted.reshape(batch_size, d**k, -1)
        
        # Reduced density matrix rho_A = M @ M^dagger
        # We want rho_A to be I / d^k. This is equivalent to M having 
        # singular values all equal to 1/sqrt(d^k).
        rho_a = torch.matmul(matrix_m, matrix_m.conj().transpose(-2, -1))
        
        # Purity Trace(rho_A^2) should be 1/d^k
        purity_val = torch.real(torch.linalg.matrix_norm(rho_a, ord='fro')**2)
        target_purity = 1.0 / (d**k)
        
        pt_loss += torch.mean(torch.abs(purity_val - target_purity))

    return (pt_loss / len(subsystems))