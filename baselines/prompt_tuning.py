# baselines/prompt_tuning.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
# Import evaluation metrics to calculate accuracy and mean confidences
from evaluation.metrics import calculate_accuracy, calculate_mean_confidence # Added imports

def prompt_tuning_baseline(clip_wrapper, embeddings, labels, concept_dict, device): # Modified signature
    """
    Implements a prompt tuning baseline for classification.
    It creates text embeddings for positive and negative concepts and classifies
    image embeddings based on cosine similarity to these text embeddings.
    
    Args:
        clip_wrapper (CLIPModelWrapper): An instance of the CLIPModelWrapper.
        embeddings (torch.Tensor): Image embeddings to classify.
        labels (torch.Tensor): True labels corresponding to the embeddings (0 for negative, 1 for positive).
        concept_dict (dict): A dictionary containing 'positive' and 'negative' concept strings.
                             e.g., {'positive': "Smiling", 'negative': "Not_Smiling"}. # Modified arg description
        device (str): The device ('cuda' or 'cpu') to perform computations on. # Added device argument
        
    Returns:
        tuple: A tuple containing:
            - float: Classification accuracy.
            - float: Mean confidence score for positive true labels.
            - float: Mean confidence score for negative true labels.
    """
    concept_positive_str = concept_dict['positive'] # Extract positive concept string
    concept_negative_str = concept_dict['negative'] # Extract negative concept string

    print(f"Running Prompt Tuning Baseline for '{concept_positive_str}' vs '{concept_negative_str}' on device: {device}...")

    # Create text prompts
    positive_prompt = f"A photo of a {concept_positive_str.lower()}" # Corrected: Use extracted string
    negative_prompt = f"A photo of a {concept_negative_str.lower()}" # Corrected: Use extracted string

    # Embed the prompts and move to device
    with torch.no_grad():
        positive_text_emb = clip_wrapper.embed_text([positive_prompt]).squeeze(0).to(device) # Ensure on device
        negative_text_emb = clip_wrapper.embed_text([negative_prompt]).squeeze(0).to(device) # Ensure on device

    # Move image embeddings to device
    embeddings_on_device = embeddings.to(device)

    # Normalize embeddings for cosine similarity
    embeddings_norm = F.normalize(embeddings_on_device, p=2, dim=-1)
    positive_text_emb_norm = F.normalize(positive_text_emb, p=2, dim=-1)
    negative_text_emb_norm = F.normalize(negative_text_emb, p=2, dim=-1)

    # Calculate cosine similarity
    sim_positive = torch.matmul(embeddings_norm, positive_text_emb_norm)
    sim_negative = torch.matmul(embeddings_norm, negative_text_emb_norm)

    # Predict based on which similarity is higher
    predictions = (sim_positive > sim_negative).long()
    confidences = sim_positive # Use similarity to positive as confidence

    # Calculate evaluation metrics
    accuracy = calculate_accuracy(predictions.cpu(), labels.cpu()) # Ensure labels are on CPU for metrics
    mean_pos_conf, mean_neg_conf = calculate_mean_confidence(confidences.cpu(), labels.cpu()) # Ensure labels are on CPU for metrics

    print("Prompt Tuning Baseline complete.")
    # Return 3 values: accuracy, mean_pos_conf, mean_neg_conf
    return accuracy, mean_pos_conf, mean_neg_conf

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
                return torch.tensor([[1.0, 0.2, 0.3] + [0.0]*(self.embed_dim-3)], device=self.device) # Positive concept
            elif "not smiling" in prompts[0].lower():
                return torch.tensor([[-1.0, 0.1, 0.2] + [0.0]*(self.embed_dim-3)], device=self.device) # Negative concept
            return torch.randn(len(prompts), self.embed_dim, device=self.device)

    # Create dummy embeddings and labels
    embed_dim = 512
    num_samples = 100
    dummy_embeddings = torch.randn(num_samples, embed_dim)
    dummy_labels = torch.randint(0, 2, (num_samples,)).long() # 0 or 1

    # Adjust some embeddings to be more 'positive' or 'negative' for clear test
    dummy_embeddings[dummy_labels == 1, 0] += 2.0 # Positive examples
    dummy_embeddings[dummy_labels == 0, 0] -= 2.0 # Negative examples

    mock_clip = MockCLIPWrapper(embed_dim=embed_dim)
    mock_concept_dict = {'positive': 'Smiling', 'negative': 'Not_Smiling'}

    print("Testing prompt_tuning_baseline...")
    # Now expecting 3 return values in the test
    accuracy, mean_pos_conf, mean_neg_conf = prompt_tuning_baseline(
        mock_clip, dummy_embeddings, dummy_labels, mock_concept_dict, 'cpu' # Pass dictionary and device
    )

    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Mean Positive Confidence: {mean_pos_conf:.4f}")
    print(f"Test Mean Negative Confidence: {mean_neg_conf:.4f}")
    print("Prompt Tuning Baseline testing complete.")