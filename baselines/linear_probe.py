# baselines/linear_probe.py

# Re-using the LatentProbe class from surgery for consistency in this baseline
# The distinction is more in how and when it's used within main.py
from surgery.latent_probe import LatentProbe

def run_linear_probe_baseline(embeddings: torch.Tensor, labels: torch.Tensor,
                              model_name="Original CLIP Embeddings") -> dict:
    """
    Runs a linear probe baseline on the given embeddings.

    Args:
        embeddings (torch.Tensor): Image embeddings (features).
        labels (torch.Tensor): Corresponding attribute labels (e.g., 0 or 1).
        model_name (str): A descriptive name for the embeddings being probed.

    Returns:
        dict: Dictionary containing evaluation metrics for the linear probe.
    """
    print(f"\n--- Running Linear Probe Baseline on {model_name} ---")
    probe = LatentProbe()
    probe.train(embeddings, labels)
    results = probe.evaluate(embeddings, labels)
    return results

# Example usage (within main.py context):
# from baselines.linear_probe import run_linear_probe_baseline
# from models.clip_wrapper import CLIPModelWrapper
# from data.celeba_loader import CelebADataset
#
# # Assuming you have original_embeddings and binary_labels
# linear_probe_baseline_results = run_linear_probe_baseline(original_embeddings, binary_labels)
# print(f"Linear Probe Baseline Results: {linear_probe_baseline_results}")