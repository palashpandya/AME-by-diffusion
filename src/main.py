import torch
import torch.nn as nn
from diffusers import DDPMScheduler as SCHEDULER
from .models.unet import GeneralDynamicUNet
import numpy as np
from . import config
import os
from .core.loss import make_density_matrix, purity, partial_trace_loss_optimized
from .core.functions import (
    verify_ame_properties, fine_tune_with_lbfgs, generate_ame_state,
    load_ame_training_data, train_diffusion_model, pretrain_diffusion_model
)
import matplotlib.pyplot as plt

# All function definitions are now in core/functions.py and imported above




if __name__ == "__main__":

    # final_state, loss_history = generate_ame_state(guidance_scale=0.05, num_steps=2000, verbose=True, d=config.D, n=config.N, pretrain=False, use_lbfgs=True)
    final_state = generate_ame_state(guidance_scale=0.05, num_steps=50,  d=config.D, n=config.N, use_lbfgs=True)
    ame_loss = verify_ame_properties(final_state, d=config.D, n=config.N)
    # print(f"Final AME Loss: {ame_loss:.6f}")
    # result = make_density_matrix(torch.stack([final_state.real, final_state.imag], dim=0).unsqueeze(0), d=config.D, n=config.N)
    # io.mmwrite(f"data/AME_{config.N}_{config.D}_diffusion.mtx", result.squeeze().cpu().numpy())
    # print("now with pre-training") # This line is commented out, but its presence suggests it might be used later.

    # Pre-train the diffusion model once
    # pretrain_diffusion_model(num_epochs=config.BENCHMARK_EPOCHS)
    # final_state, loss_history = generate_ame_state(guidance_scale=0.05, num_steps=500, verbose=True, d=config.D, n=config.N, pretrain=True, use_lbfgs=True)
    # ame_loss = verify_ame_properties(final_state, d=config.D)
    # print(f"Final AME Loss with pre-training: {ame_loss:.6f}")

    # check bfgs:
    vec_len = config.D ** config.N
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x_final = final_state.to(device)
    x_optimized = fine_tune_with_lbfgs(x_final, d=config.D, n=config.N, verbose=True)
    rho = make_density_matrix(torch.stack([x_optimized.real, x_optimized.imag], dim=0).unsqueeze(0), d=config.D, n=config.N)
    partial_trace_loss_value = partial_trace_loss_optimized(torch.stack([x_optimized.real, x_optimized.imag], dim=0).unsqueeze(0), d=config.D, n=config.N).item()
    print(f"Partial Trace Loss after L-BFGS fine-tuning: {partial_trace_loss_value:.6f}")
    print("Purity after lbfgs fine-tuning:", purity(rho).item())
    
    x_optimized = fine_tune_with_lbfgs(x_optimized, d=config.D, n=config.N, verbose=True)
    rho = make_density_matrix(torch.stack([x_optimized.real, x_optimized.imag], dim=0).unsqueeze(0), d=config.D, n=config.N)
    partial_trace_loss_value = partial_trace_loss_optimized(torch.stack([x_optimized.real, x_optimized.imag], dim=0).unsqueeze(0), d=config.D, n=config.N).item()
    
    print(f"Partial Trace Loss after L-BFGS fine-tuning: {partial_trace_loss_value:.6f}")
    print("Purity after lbfgs fine-tuning:", purity(rho).item())
    plt.matshow(rho.squeeze().cpu().numpy().real, cmap='viridis')
    plt.colorbar()
    plt.title("Real part of Density Matrix after L-BFGS Fine-tuning")
    plt.savefig('density_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()  # Clean up memory

    # Create data directory if it doesn't exist
    if not os.path.exists('data'):
        os.makedirs('data')

    # Save the final state vector instead of the full density matrix
    state_vector_path = f"data/AME_{config.N}_{config.D}_state_vector.npy"
    np.save(state_vector_path, x_optimized.cpu().numpy())
    print(f"Saved final state vector to {state_vector_path}")


    # Run benchmarking with specified parameter ranges
    # from benchmark import benchmark_ame_generation, display_benchmark_results, plot_loss_trajectories
    # results = benchmark_ame_generation(
    #     guidance_scales=config.BENCHMARK_GUIDANCE_SCALES,
    #     num_steps_list=config.BENCHMARK_NUM_STEPS,
    #     pretrain=False  # Do not use pre-trained model
    # )

    # Display formatted results
    # display_benchmark_results(results)

    # Plot loss trajectories
    # plot_loss_trajectories(results)

    print("\nBenchmarking Complete!")

    
