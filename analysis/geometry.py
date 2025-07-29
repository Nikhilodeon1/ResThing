# analysis/geometry.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_vector_properties(vector, vector_name="Vector"):
    """
    Calculates and prints basic properties of a given vector.
    Args:
        vector (torch.Tensor): The vector to analyze.
        vector_name (str): Name of the vector for printout.
    Returns:
        dict: A dictionary of calculated properties.
    """
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32)

    norm = torch.norm(vector).item()
    mean_abs_component = torch.mean(torch.abs(vector)).item()
    std_component = torch.std(vector).item()
    min_component = torch.min(vector).item()
    max_component = torch.max(vector).item()
    
    # Sparsity: percentage of components close to zero (e.g., within 1e-6)
    sparsity = (torch.sum(torch.abs(vector) < 1e-6).item() / vector.numel()) * 100

    properties = {
        f'{vector_name}_Norm': norm,
        f'{vector_name}_Mean_Abs_Component': mean_abs_component,
        f'{vector_name}_Std_Component': std_component,
        f'{vector_name}_Min_Component': min_component,
        f'{vector_name}_Max_Component': max_component,
        f'{vector_name}_Sparsity_Percent': sparsity
    }

    print(f"\n--- Geometric Properties for '{vector_name}' ---")
    for k, v in properties.items():
        print(f"  {k.replace(f'{vector_name}_', '').replace('_', ' ')}: {v:.4f}")
    print("-------------------------------------------------")
    
    return properties


def compare_vectors_geometry(vector1, vector2, name1="Vector 1", name2="Vector 2"):
    """
    Compares two vectors geometrically by calculating their cosine similarity and angle.
    Args:
        vector1 (torch.Tensor): First vector.
        vector2 (torch.Tensor): Second vector.
        name1 (str): Name of the first vector.
        name2 (str): Name of the second vector.
    Returns:
        dict: A dictionary of comparison metrics.
    """
    if not isinstance(vector1, torch.Tensor):
        vector1 = torch.tensor(vector1, dtype=torch.float32)
    if not isinstance(vector2, torch.Tensor):
        vector2 = torch.tensor(vector2, dtype=torch.float32)

    # Ensure vectors are 1D (flatten if needed for dot product)
    vector1_flat = vector1.flatten()
    vector2_flat = vector2.flatten()

    # Normalize to unit vectors
    vector1_norm = F.normalize(vector1_flat, p=2, dim=0)
    vector2_norm = F.normalize(vector2_flat, p=2, dim=0)

    # Cosine similarity
    cosine_sim = torch.dot(vector1_norm, vector2_norm).item()
    
    # Angle in degrees
    # Clamp to avoid NaN from floating point errors outside [-1, 1] range for arccos
    angle_rad = torch.acos(torch.clamp(torch.tensor(cosine_sim), -1.0 + 1e-7, 1.0 - 1e-7)).item()
    angle_deg = np.degrees(angle_rad)

    comparison_metrics = {
        f'Cosine_Similarity_{name1}_vs_{name2}': cosine_sim,
        f'Angle_Degrees_{name1}_vs_{name2}': angle_deg
    }

    print(f"\n--- Geometric Comparison: '{name1}' vs '{name2}' ---")
    print(f"  Cosine Similarity: {cosine_sim:.4f}")
    print(f"  Angle (Degrees): {angle_deg:.4f}")
    print("-------------------------------------------------")

    return comparison_metrics


def plot_vector_component_distribution(vector, vector_name, save_path):
    """
    Plots the distribution of components of a vector.
    Args:
        vector (torch.Tensor): The vector whose components to plot.
        vector_name (str): Name of the vector for plot title.
        save_path (str): Path to save the plot.
    """
    if not isinstance(vector, torch.Tensor):
        vector = torch.tensor(vector, dtype=torch.float32)
    
    plt.figure(figsize=(8, 6))
    sns.histplot(vector.cpu().numpy(), kde=True, bins=50)
    plt.title(f'Distribution of Components for {vector_name}')
    plt.xlabel('Component Value')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Component distribution plot for '{vector_name}' saved to {save_path}")

def plot_pairwise_similarity_heatmap(similarity_matrix, labels, title, save_path):
    """
    Plots a heatmap of pairwise similarities between vectors.
    Args:
        similarity_matrix (np.array): 2D array of pairwise similarities.
        labels (list): List of names corresponding to the vectors.
        title (str): Title for the heatmap.
        save_path (str): Path to save the plot.
    """
    plt.figure(figsize=(len(labels) + 2, len(labels) + 2))
    sns.heatmap(similarity_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                xticklabels=labels, yticklabels=labels, linewidths=.5)
    plt.title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Pairwise similarity heatmap saved to {save_path}")