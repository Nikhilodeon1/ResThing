# evaluation/retrieval.py

import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

def evaluate_retrieval(query_embeddings: torch.Tensor,
                       gallery_embeddings: torch.Tensor,
                       query_labels: torch.Tensor,
                       gallery_labels: torch.Tensor,
                       top_k: int = 5) -> dict:
    """
    Evaluates retrieval performance based on cosine similarity.

    Args:
        query_embeddings (torch.Tensor): Embeddings of query images.
        gallery_embeddings (torch.Tensor): Embeddings of gallery images (database).
        query_labels (torch.Tensor): Labels corresponding to query embeddings.
        gallery_labels (torch.Tensor): Labels corresponding to gallery embeddings.
        top_k (int): Number of top most similar items to consider for accuracy.

    Returns:
        dict: Retrieval metrics, e.g., accuracy@k.
    """
    print(f"\n--- Evaluating Retrieval (Top-K = {top_k}) ---")

    if query_embeddings.is_cuda:
        query_embeddings_np = query_embeddings.cpu().numpy()
    else:
        query_embeddings_np = query_embeddings.numpy()

    if gallery_embeddings.is_cuda:
        gallery_embeddings_np = gallery_embeddings.cpu().numpy()
    else:
        gallery_embeddings_np = gallery_embeddings.numpy()

    if query_labels.is_cuda:
        query_labels_np = query_labels.cpu().numpy()
    else:
        query_labels_np = query_labels.numpy()

    if gallery_labels.is_cuda:
        gallery_labels_np = gallery_labels.cpu().numpy()
    else:
        gallery_labels_np = gallery_labels.numpy()

    correct_retrievals = 0
    total_queries = query_embeddings_np.shape[0]

    for i in tqdm(range(total_queries), desc="Retrieving matches"):
        query_emb = query_embeddings_np[i].reshape(1, -1)
        query_label = query_labels_np[i]

        # Compute cosine similarity with all gallery embeddings
        similarities = cosine_similarity(query_emb, gallery_embeddings_np)[0]

        # Get indices of top_k most similar items
        top_k_indices = np.argsort(similarities)[::-1][:top_k]

        # Check if any of the top_k retrieved items have the same label as the query
        if query_label in gallery_labels_np[top_k_indices]:
            correct_retrievals += 1

    accuracy_at_k = correct_retrievals / total_queries
    print(f"Retrieval Accuracy@{top_k}: {accuracy_at_k:.4f}")

    return {f"retrieval_accuracy_at_{top_k}": accuracy_at_k}

# Example usage (within main.py context):
# from evaluation.retrieval import evaluate_retrieval
# from models.clip_wrapper import CLIPModelWrapper
# from data.celeba_loader import CelebADataset
#
# # Assuming you have original_embeddings, original_labels, and a gallery_embeddings, gallery_labels
# # (e.g., query set could be a subset of the gallery, or two distinct sets)
# # For simplicity, let's use a split of the original dataset for query and gallery
#
# # Assuming original_embeddings and binary_labels are available from your main script
# # For demonstration, let's split them into query and gallery (e.g., first 100 for query, rest for gallery)
# num_queries = min(100, original_embeddings.shape[0] // 2)
# query_embs = original_embeddings[:num_queries]
# query_lbls = binary_labels[:num_queries]
# gallery_embs = original_embeddings[num_queries:]
# gallery_lbls = binary_labels[num_queries:]
#
# retrieval_results = evaluate_retrieval(query_embs, gallery_embs, query_lbls, gallery_lbls, top_k=5)
# print(f"Retrieval results: {retrieval_results}")