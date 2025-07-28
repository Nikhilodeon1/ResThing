# evaluation/visualize.py
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import torch

def create_tsne_plot(embeddings, labels, title="t-SNE Plot of Embeddings", save_path=None):
    """
    Generates and optionally saves a t-SNE plot of embeddings.
    
    Args:
        embeddings (torch.Tensor): Embeddings to visualize (N, D).
        labels (torch.Tensor): Corresponding labels (N,).
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    print(f"Generating t-SNE plot for {len(embeddings)} embeddings...")
    
    if len(embeddings) > 5000:
        print("Warning: t-SNE can be slow for large datasets. Consider sampling a subset.")
        # For demonstration, let's sample if too large
        sample_indices = np.random.choice(len(embeddings), 5000, replace=False)
        embeddings_np = embeddings[sample_indices].cpu().numpy()
        labels_np = labels[sample_indices].cpu().numpy()
    else:
        embeddings_np = embeddings.numpy()
        labels_np = labels.cpu().numpy()

    # Create TSNE model
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    # n_jobs=-1 for parallel processing if supported by your sklearn version and numpy setup
    
    # Fit and transform
    try:
        tsne_results = tsne.fit_transform(embeddings_np)
    except Exception as e:
        print(f"Error during t-SNE computation: {e}")
        print("Falling back to PCA for dimensionality reduction if t-SNE fails often.")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        tsne_results = pca.fit_transform(embeddings_np)
        title = title + " (PCA Fallback)"


    # Plotting
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_results[:, 0], tsne_results[:, 1],
        c=labels_np,
        cmap='viridis', # Or 'RdBu' for binary labels
        alpha=0.7,
        s=10
    )
    plt.colorbar(scatter, ticks=np.unique(labels_np), label='Label')
    plt.title(title)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"t-SNE plot saved to {save_path}")
    else:
        plt.show()
    plt.close() # Close the plot to free memory


def plot_cosine_similarity_hist(similarities, title="Cosine Similarities", save_path=None):
    """
    Plots a histogram of cosine similarities.
    
    Args:
        similarities (torch.Tensor): Tensor of cosine similarities.
        title (str): Title for the plot.
        save_path (str, optional): Path to save the plot. If None, displays the plot.
    """
    print(f"Generating histogram for {len(similarities)} cosine similarities...")
    
    plt.figure(figsize=(8, 6))
    plt.hist(similarities.cpu().numpy(), bins=50, edgecolor='black', alpha=0.7)
    plt.title(title)
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.grid(axis='y', alpha=0.75)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Cosine similarity histogram saved to {save_path}")
    else:
        plt.show()
    plt.close()

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    # Dummy data for testing
    num_samples = 1000
    embed_dim = 512
    dummy_embeddings = torch.randn(num_samples, embed_dim)
    dummy_labels = torch.randint(0, 2, (num_samples,)).long()
    
    # Create some separation for better visualization
    dummy_embeddings[dummy_labels == 0] -= 1.0
    dummy_embeddings[dummy_labels == 1] += 1.0

    print("Testing create_tsne_plot...")
    create_tsne_plot(dummy_embeddings, dummy_labels, title="Test t-SNE Plot", save_path="./temp_outputs/tsne_test.png")

    # Dummy similarities
    dummy_similarities = torch.rand(num_samples) * 2 - 1 # range from -1 to 1
    print("\nTesting plot_cosine_similarity_hist...")
    plot_cosine_similarity_hist(dummy_similarities, title="Test Cosine Similarity Histogram", save_path="./temp_outputs/cosine_hist_test.png")
    
    print("\nVisualization testing complete. Check './temp_outputs/' for generated plots.")