# baselines/random_edit.py
import torch

def random_edit_baseline(embeddings, alpha=1.0):
    """
    Applies a random perturbation to the input embeddings as a baseline for comparison.
    The perturbation is generated from a standard normal distribution and scaled by alpha.
    
    Args:
        embeddings (torch.Tensor): Original image embeddings.
        alpha (float): Scaling factor for the random noise.
        
    Returns:
        torch.Tensor: Embeddings with random noise added.
    """
    print(f"Running Random Edit Baseline with alpha={alpha}...")
    
    noise = torch.randn_like(embeddings) * alpha
    edited_embeddings = embeddings + noise
    
    print("Random Edit Baseline complete.")
    return edited_embeddings

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    # Create dummy embeddings
    embed_dim = 512
    num_samples = 100
    dummy_embeddings = torch.randn(num_samples, embed_dim)
    
    print("Testing random_edit_baseline...")
    edited_embeddings = random_edit_baseline(dummy_embeddings, alpha=0.5)
    
    print(f"Original embeddings shape: {dummy_embeddings.shape}")
    print(f"Edited embeddings shape: {edited_embeddings.shape}")
    
    # Check if embeddings have actually changed
    assert not torch.allclose(dummy_embeddings, edited_embeddings, atol=1e-6), "Embeddings did not change!"
    
    # Check the magnitude of change (should be roughly proportional to alpha)
    change_magnitude = torch.norm(edited_embeddings - dummy_embeddings, dim=-1).mean().item()
    print(f"Average change magnitude per embedding: {change_magnitude:.4f}")
    print("Random Edit Baseline testing complete.")