# evaluation/metrics.py
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def calculate_accuracy(predictions, true_labels):
    """
    Calculates classification accuracy.
    
    Args:
        predictions (torch.Tensor): Predicted labels (0 or 1).
        true_labels (torch.Tensor): True labels (0 or 1).
        
    Returns:
        float: Accuracy score.
    """
    predictions_np = predictions.cpu().numpy()
    true_labels_np = true_labels.cpu().numpy()
    acc = accuracy_score(true_labels_np, predictions_np)
    print(f"Accuracy: {acc:.4f}")
    return acc

def calculate_cosine_similarity(embedding1, embedding2):
    """
    Calculates the cosine similarity between two sets of embeddings.
    
    Args:
        embedding1 (torch.Tensor): First set of embeddings.
        embedding2 (torch.Tensor): Second set of embeddings (must have same shape as embedding1).
        
    Returns:
        torch.Tensor: A tensor of cosine similarities, shape (N,).
    """
    if embedding1.shape != embedding2.shape:
        raise ValueError("Embeddings must have the same shape for cosine similarity calculation.")
    
    embedding1_norm = F.normalize(embedding1.float(), p=2, dim=-1)
    embedding2_norm = F.normalize(embedding2.float(), p=2, dim=-1)
    
    # Cosine similarity between corresponding vectors in the two sets
    similarity = F.cosine_similarity(embedding1_norm, embedding2_norm, dim=-1)
    
    print(f"Calculated average cosine similarity: {similarity.mean().item():.4f}")
    return similarity

def calculate_mean_confidence(confidences, labels):
    """
    Calculates the mean confidence for positive and negative classes.
    
    Args:
        confidences (torch.Tensor): Confidence scores (e.g., similarity to positive concept).
        labels (torch.Tensor): True labels (0 for negative, 1 for positive).
        
    Returns:
        tuple: (mean_conf_positive, mean_conf_negative)
    """
    positive_confidences = confidences[labels == 1]
    negative_confidences = confidences[labels == 0]
    
    mean_conf_pos = positive_confidences.mean().item() if len(positive_confidences) > 0 else 0.0
    mean_conf_neg = negative_confidences.mean().item() if len(negative_confidences) > 0 else 0.0
    
    print(f"Mean confidence for positive samples: {mean_conf_pos:.4f}")
    print(f"Mean confidence for negative samples: {mean_conf_neg:.4f}")
    return mean_conf_pos, mean_conf_neg

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    # Dummy data for testing
    num_samples = 100
    dummy_predictions = torch.randint(0, 2, (num_samples,)).long()
    dummy_true_labels = torch.randint(0, 2, (num_samples,)).long()

    acc = calculate_accuracy(dummy_predictions, dummy_true_labels)
    print(f"Test Accuracy: {acc}")

    embed_dim = 512
    dummy_emb1 = torch.randn(num_samples, embed_dim)
    dummy_emb2 = dummy_emb1 + 0.1 * torch.randn(num_samples, embed_dim) # Slightly perturbed
    
    sim = calculate_cosine_similarity(dummy_emb1, dummy_emb2)
    print(f"Test Average Cosine Similarity: {sim.mean().item()}")

    dummy_confidences = torch.randn(num_samples) # Some random scores
    dummy_conf_labels = torch.randint(0, 2, (num_samples,)).long()
    
    mean_pos, mean_neg = calculate_mean_confidence(dummy_confidences, dummy_conf_labels)
    print(f"Test Mean Positive Confidence: {mean_pos}")
    print(f"Test Mean Negative Confidence: {mean_neg}")