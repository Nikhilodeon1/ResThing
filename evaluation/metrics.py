# evaluation/metrics.py

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, average_precision_score
from sklearn.linear_model import LogisticRegression # More suitable for binary classification
from sklearn.model_selection import train_test_split
import numpy as np
import json
import os
from scipy import stats # For statistical tests

def calculate_accuracy(predictions, labels):
    """
    Calculates classification accuracy.
    Args:
        predictions (torch.Tensor): Predicted labels (e.g., 0 or 1).
        labels (torch.Tensor): True labels.
    Returns:
        float: Accuracy score.
    """
    if predictions.dim() > 1:
        predictions = predictions.squeeze()
    if labels.dim() > 1:
        labels = labels.squeeze()
        
    return accuracy_score(labels.cpu().numpy(), predictions.cpu().numpy())

def calculate_cosine_similarity(embeddings1, embeddings2):
    """
    Calculates cosine similarity between two sets of embeddings.
    Args:
        embeddings1 (torch.Tensor): First set of embeddings.
        embeddings2 (torch.Tensor): Second set of embeddings.
    Returns:
        torch.Tensor: Cosine similarities for each corresponding pair.
    """
    # Normalize embeddings to unit vectors
    embeddings1_norm = F.normalize(embeddings1, p=2, dim=1)
    embeddings2_norm = F.normalize(embeddings2, p=2, dim=1)
    
    # Calculate dot product, which is cosine similarity for normalized vectors
    # Ensure inputs are float32 for torch.matmul
    similarities = torch.sum(embeddings1_norm.float() * embeddings2_norm.float(), dim=1)
    return similarities

def calculate_mean_confidence(confidences, labels):
    """
    Calculates mean confidence for positive and negative samples separately.
    Args:
        confidences (torch.Tensor): Model output probabilities/confidences for the positive class.
        labels (torch.Tensor): True labels (1 for positive, -1 for negative, as per CelebA).
    Returns:
        tuple: (mean_pos_conf, mean_neg_conf)
    """
    if confidences.dim() > 1:
        confidences = confidences.squeeze()
    if labels.dim() > 1:
        labels = labels.squeeze()

    pos_confidences = confidences[labels == 1]
    neg_confidences = confidences[labels == -1]

    mean_pos_conf = pos_confidences.mean().item() if len(pos_confidences) > 0 else 0.0
    mean_neg_conf = neg_confidences.mean().item() if len(neg_confidences) > 0 else 0.0

    return mean_pos_conf, mean_neg_conf


# --- NEWLY IMPLEMENTED FUNCTIONS ---

def evaluate_probe_accuracy(embeddings, labels, save_path=None):
    """
    Trains a simple linear probe (Logistic Regression) to classify embeddings
    and evaluates its accuracy and F1-score.
    
    Args:
        embeddings (torch.Tensor): Embeddings to be classified.
        labels (torch.Tensor): True labels (should be 0/1, convert if -1/1).
        save_path (str, optional): Path to save results JSON. Defaults to None.
    
    Returns:
        tuple: (accuracy, f1)
    """
    print(f"Evaluating probe accuracy for embeddings of shape {embeddings.shape}...")

    # Convert labels from -1/1 to 0/1 if necessary
    labels_np = labels.cpu().numpy()
    if all(l in [-1, 1] for l in labels_np):
        labels_np = np.where(labels_np == 1, 1, 0) # Convert -1 to 0

    X = embeddings.cpu().numpy()
    y = labels_np

    if len(np.unique(y)) < 2:
        print("Warning: Only one class present in labels for probe. Cannot train classifier.")
        results = {"accuracy": 0.0, "f1_score": 0.0, "message": "Only one class present in labels."}
        if save_path:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=4)
        return 0.0, 0.0

    # Use a fixed random state for reproducibility within this function
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Use LogisticRegression as a linear probe.
    # Set max_iter for convergence, solver for robustness.
    model = LogisticRegression(random_state=42, max_iter=1000, solver='liblinear') 
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred) # F1-score is good for binary classification

    results = {
        "accuracy": float(accuracy),
        "f1_score": float(f1),
        "num_samples": len(y),
        "num_train_samples": len(y_train),
        "num_test_samples": len(y_test)
    }

    print(f"Probe Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Probe results saved to {save_path}")

    return accuracy, f1

def evaluate_retrieval(query_embeddings, query_labels, query_image_ids, gallery_embeddings, gallery_labels, gallery_image_ids, top_k=5, save_path=None):
    """
    Evaluates retrieval performance based on cosine similarity.
    For each query, it retrieves top_k most similar items from the gallery
    and checks if relevant items (same label) are retrieved.
    Calculates mean Average Precision (mAP) and Top-K accuracy.

    Args:
        query_embeddings (torch.Tensor): Embeddings of query images.
        query_labels (torch.Tensor): Labels of query images.
        query_image_ids (list): Image IDs of query images.
        gallery_embeddings (torch.Tensor): Embeddings of gallery images.
        gallery_labels (torch.Tensor): Labels of gallery images.
        gallery_image_ids (list): Image IDs of gallery images.
        top_k (int): Number of top similar items to retrieve.
        save_path (str, optional): Path to save results JSON. Defaults to None.

    Returns:
        tuple: (mAP, top_k_accuracy)
    """
    print(f"Evaluating retrieval (Top-K={top_k}) for {len(query_embeddings)} queries...")

    # Ensure tensors are on CPU for numpy conversion and float32 for calculations
    query_embeddings_np = query_embeddings.cpu().numpy().astype(np.float32)
    query_labels_np = query_labels.cpu().numpy()
    gallery_embeddings_np = gallery_embeddings.cpu().numpy().astype(np.float32)
    gallery_labels_np = gallery_labels.cpu().numpy()

    # Convert labels from -1/1 to 0/1 for consistency if needed (especially for AP)
    if all(l in [-1, 1] for l in query_labels_np):
        query_labels_np = np.where(query_labels_np == 1, 1, 0)
    if all(l in [-1, 1] for l in gallery_labels_np):
        gallery_labels_np = np.where(gallery_labels_np == 1, 1, 0)

    # Normalize embeddings for cosine similarity
    query_embeddings_norm = query_embeddings_np / np.linalg.norm(query_embeddings_np, axis=1, keepdims=True)
    gallery_embeddings_norm = gallery_embeddings_np / np.linalg.norm(gallery_embeddings_np, axis=1, keepdims=True)

    average_precisions = []
    top_k_correct_count = 0

    for i in range(len(query_embeddings_norm)):
        query_embedding = query_embeddings_norm[i]
        query_label = query_labels_np[i]
        query_id = query_image_ids[i]

        # Calculate cosine similarities with all gallery items
        similarities = np.dot(gallery_embeddings_norm, query_embedding)

        # Create a mask to exclude the query itself from the gallery
        # This assumes unique image IDs. If not, needs more robust handling.
        mask = np.ones(len(gallery_image_ids), dtype=bool)
        if query_id in gallery_image_ids: # Check if query image ID is in gallery
            try:
                query_in_gallery_idx = gallery_image_ids.index(query_id)
                mask[query_in_gallery_idx] = False # Exclude the query itself
            except ValueError:
                # If query_id not found for some reason (e.g., in different split/gallery)
                pass

        similarities_filtered = similarities[mask]
        gallery_labels_filtered = gallery_labels_np[mask]
        
        # Sort by similarity in descending order
        sorted_indices = np.argsort(similarities_filtered)[::-1]

        # Get top-k retrieved labels and their relevance
        top_k_retrieved_labels_for_query = gallery_labels_filtered[sorted_indices[:top_k]]
        relevance_scores_for_ap = similarities_filtered[sorted_indices] # Use scores for AP calculation
        relevance_binary_for_ap = (gallery_labels_filtered[sorted_indices] == query_label).astype(int) # Binary relevance

        # Calculate Average Precision for this query
        if len(relevance_binary_for_ap) > 0 and np.sum(relevance_binary_for_ap) > 0: # Ensure there are relevant items to retrieve
             ap = average_precision_score(relevance_binary_for_ap, relevance_scores_for_ap)
             average_precisions.append(ap)
        elif len(relevance_binary_for_ap) > 0: # No relevant items, AP is 0
             average_precisions.append(0.0)


        # Check for Top-K accuracy: Is the query label present in top-k retrieved?
        if query_label in top_k_retrieved_labels_for_query:
            top_k_correct_count += 1
            
    mAP = np.mean(average_precisions) if len(average_precisions) > 0 else 0.0
    top_k_accuracy = top_k_correct_count / len(query_embeddings_norm) if len(query_embeddings_norm) > 0 else 0.0

    results = {
        "mean_average_precision": float(mAP),
        f"top_{top_k}_accuracy": float(top_k_accuracy),
        "num_queries": len(query_embeddings),
        "num_gallery_items": len(gallery_embeddings),
        "top_k": top_k
    }

    print(f"Retrieval mAP (Top-{top_k}): {mAP:.4f}, Top-{top_k} Accuracy: {top_k_accuracy:.4f}")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Retrieval results saved to {save_path}")

    return mAP, top_k_accuracy

def perform_statistical_test(data1, data2, test_name="Comparison"):
    """
    Performs an independent Student's t-test to compare two sets of data and prints the result.
    Assumes independent samples and similar variances. Use `perform_paired_statistical_test`
    for paired data.
    
    Args:
        data1 (list/np.array): First set of data (e.g., accuracies from method A).
        data2 (list/np.array): Second set of data (e.g., accuracies from method B).
        test_name (str): Name of the comparison for printout.
    """
    if not data1 or not data2:
        print(f"Skipping statistical test for '{test_name}': Insufficient data.")
        return
    
    if len(data1) < 2 or len(data2) < 2:
        print(f"Skipping statistical test for '{test_name}': Need at least 2 samples per group for meaningful t-test.")
        return

    # Perform independent t-test (assuming equal variance for simplicity, can use ttest_ind(equal_var=False) for Welch's)
    t_stat, p_value = stats.ttest_ind(data1, data2)

    print(f"\n--- Independent T-Test Results for '{test_name}' ---")
    print(f"    Mean 1: {np.mean(data1):.4f} (Std: {np.std(data1):.4f})")
    print(f"    Mean 2: {np.mean(data2):.4f} (Std: {np.std(data2):.4f})")
    print(f"    T-statistic: {t_stat:.4f}")
    print(f"    P-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"    Result: Statistically significant difference (p < {alpha}).")
    else:
        print(f"    Result: No statistically significant difference (p >= {alpha}).")
    print("-------------------------------------------------")


def perform_paired_statistical_test(data1, data2, test_name="Paired Comparison"):
    """
    Performs a paired Student's t-test to compare two sets of data from paired observations.
    Suitable for comparing performance metrics (e.g., accuracy) from the same runs.

    Args:
        data1 (list/np.array): First set of paired data.
        data2 (list/np.array): Second set of paired data.
        test_name (str): Name of the comparison for printout.
    """
    if not data1 or not data2:
        print(f"Skipping paired statistical test for '{test_name}': Insufficient data.")
        return
    
    if len(data1) != len(data2):
        print(f"Skipping paired statistical test for '{test_name}': Data sets must have the same length for paired test.")
        return
    
    if len(data1) < 2:
        print(f"Skipping paired statistical test for '{test_name}': Need at least 2 pairs of samples for meaningful paired t-test.")
        return

    # Perform paired t-test
    t_stat, p_value = stats.ttest_rel(data1, data2)

    print(f"\n--- Paired T-Test Results for '{test_name}' ---")
    print(f"    Mean 1: {np.mean(data1):.4f} (Std: {np.std(data1):.4f})")
    print(f"    Mean 2: {np.mean(data2):.4f} (Std: {np.std(data2):.4f})")
    print(f"    Difference (Mean1 - Mean2): {np.mean(np.array(data1) - np.array(data2)):.4f}")
    print(f"    T-statistic: {t_stat:.4f}")
    print(f"    P-value: {p_value:.4f}")

    alpha = 0.05
    if p_value < alpha:
        print(f"    Result: Statistically significant difference (p < {alpha}).")
    else:
        print(f"    Result: No statistically significant difference (p >= {alpha}).")
    print("-------------------------------------------------")

def get_model_predictions(encoder, embeddings, concept, device):
    """
    Gets the model's (prompt tuning) predictions and confidences for given embeddings.
    This essentially re-runs the classification part of prompt_tuning_baseline
    to get per-sample results.

    Args:
        encoder (EncoderWrapper): The encoder model wrapper.
        embeddings (torch.Tensor): Embeddings to classify.
        concept (dict): Dictionary with 'positive' and 'negative' concept descriptions.
        device (str): Device to use ('cuda' or 'cpu').

    Returns:
        tuple: (predictions (torch.Tensor), confidences (torch.Tensor))
    """
    # Generate text embeddings for positive and negative prompts
    pos_text_embedding = encoder.embed_text(concept['positive']).to(device)
    neg_text_embedding = encoder.embed_text(concept['negative']).to(device)

    # Normalize image and text embeddings for cosine similarity
    image_embeddings_norm = F.normalize(embeddings.to(device), p=2, dim=1)
    pos_text_embedding_norm = F.normalize(pos_text_embedding, p=2, dim=0) # dim=0 because it's a single vector
    neg_text_embedding_norm = F.normalize(neg_text_embedding, p=2, dim=0)

    # Calculate cosine similarity with positive and negative concept embeddings
    pos_similarities = torch.matmul(image_embeddings_norm, pos_text_embedding_norm.T)
    neg_similarities = torch.matmul(image_embeddings_norm, neg_text_embedding_norm.T)

    # Get predictions based on which concept is more similar
    # Convert similarities to a 2D tensor for softmax (batch_size, 2)
    combined_similarities = torch.stack([neg_similarities, pos_similarities], dim=1)
    
    # Apply softmax to get confidences for 0 (negative) and 1 (positive)
    confidences_softmax = torch.softmax(combined_similarities, dim=1)
    
    # The confidence for the positive class (label 1)
    positive_class_confidences = confidences_softmax[:, 1] 

    # Predictions: 1 if pos_similarity > neg_similarity, else 0 (mapping -1 to 0 if labels are -1/1)
    predictions = (pos_similarities > neg_similarities).int() # 0 or 1

    return predictions.cpu(), positive_class_confidences.cpu()