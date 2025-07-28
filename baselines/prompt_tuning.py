# baselines/prompt_tuning.py
import torch
import torch.nn.functional as F
from tqdm import tqdm

def prompt_tuning_baseline(clip_wrapper, embeddings, labels, concept_positive, concept_negative):
    """
    Implements a prompt tuning baseline for classification.
    It creates text embeddings for positive and negative concepts and classifies
    image embeddings based on cosine similarity to these text embeddings.
    
    Args:
        clip_wrapper (CLIPModelWrapper): An instance of the CLIPModelWrapper.
        embeddings (torch.Tensor): Image embeddings to classify.
        labels (torch.Tensor): True labels corresponding to the embeddings (0 for negative, 1 for positive).
        concept_positive (str): The string representing the positive concept (e.g., "Smiling").
        concept_negative (str): The string representing the negative concept (e.g., "Not_Smiling").
        
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Predicted labels (0 or 1).
            - torch.Tensor: Confidence scores (cosine similarity to positive concept).
    """
    print(f"Running Prompt Tuning Baseline for '{concept_positive}' vs '{concept_negative}'...")

    # Create text prompts
    positive_prompt = f"A photo of a {concept_positive.lower()}"
    negative_prompt = f"A photo of a {concept_negative.lower()}"

    # Embed the prompts
    with torch.no_grad():
        positive_text_emb = clip_wrapper.embed_text([positive_prompt]).squeeze(0).to(embeddings.device)
        negative_text_emb = clip_wrapper.embed_text([negative_prompt]).squeeze(0).to(embeddings.device)

    # Normalize embeddings for cosine similarity
    embeddings_norm = F.normalize(embeddings, p=2, dim=-1)
    positive_text_emb_norm = F.normalize(positive_text_emb, p=2, dim=-1)
    negative_text_emb_norm = F.normalize(negative_text_emb, p=2, dim=-1)

    # Calculate cosine similarity
    sim_positive = torch.matmul(embeddings_norm, positive_text_emb_norm)
    sim_negative = torch.matmul(embeddings_norm, negative_text_emb_norm)

    # Predict based on which similarity is higher
    predictions = (sim_positive > sim_negative).long()
    confidences = sim_positive # Use similarity to positive as confidence

    print("Prompt Tuning Baseline complete.")
    return predictions.cpu(), confidences.cpu()

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    from models.clip_wrapper import CLIPModelWrapper
    import torch

    # Mock CLIPWrapper for testing
    class MockCLIPWrapper:
        def __init__(self, embed_dim=512):
            self.embed_dim = embed_dim
            self.device = 'cpu'

        def embed_text(self, prompts):
            # Simulate text embeddings (e.g., "smiling" is high in dim 0, "not smiling" is low)
            if "smiling" in prompts[0].lower():
                return torch.tensor([[1.0, 0.2, 0.3] + [0.0]*(self.embed_dim-3)]) # Positive concept
            elif "not smiling" in prompts[0].lower():
                return torch.tensor([[-1.0, 0.1, 0.2] + [0.0]*(self.embed_dim-3)]) # Negative concept
            return torch.randn(len(prompts), self.embed_dim)

    # Create dummy embeddings and labels
    embed_dim = 512
    num_samples = 100
    dummy_embeddings = torch.randn(num_samples, embed_dim)
    dummy_labels = torch.randint(0, 2, (num_samples,)).long() # 0 or 1

    # Adjust some embeddings to be more 'positive' or 'negative' for clear test
    dummy_embeddings[dummy_labels == 1, 0] += 2.0 # Positive examples
    dummy_embeddings[dummy_labels == 0, 0] -= 2.0 # Negative examples

    mock_clip = MockCLIPWrapper(embed_dim=embed_dim)

    print("Testing prompt_tuning_baseline...")
    predictions, confidences = prompt_tuning_baseline(
        mock_clip, dummy_embeddings, dummy_labels, "Smiling", "Not_Smiling"
    )

    print(f"Predicted labels shape: {predictions.shape}")
    print(f"Confidence scores shape: {confidences.shape}")
    print(f"Sample true labels: {dummy_labels[:10].tolist()}")
    print(f"Sample predicted labels: {predictions[:10].tolist()}")

    # Calculate accuracy for testing
    accuracy = (predictions == dummy_labels).float().mean().item()
    print(f"Dummy accuracy: {accuracy:.4f}")
    print("Prompt Tuning Baseline testing complete.")