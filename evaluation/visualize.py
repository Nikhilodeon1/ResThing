# evaluation/visualize.py

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from sklearn.manifold import TSNE
import torch
from PIL import Image
import shutil # For copying images
from datetime import datetime # Added for report timestamp
# NEW: Import io_utils for image loading if needed, though PIL Image.open is direct
# from utils.io_utils import load_torch_tensor # Not directly needed for images here


def create_tsne_plot(embeddings, labels, title, save_path):
    """
    Generates and saves a t-SNE plot of embeddings.
    """
    if embeddings.shape[0] == 0:
        print(f"Skipping t-SNE plot for '{title}': No embeddings to plot.")
        return

    # Ensure embeddings are on CPU and convert to numpy
    embeddings_np = embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()

    # Handle labels that might be -1/1, convert to 0/1 for plotting colors
    unique_labels = np.unique(labels_np)
    if -1 in unique_labels and 1 in unique_labels:
        labels_for_plot = np.where(labels_np == 1, 1, 0) # Convert -1 to 0
    else:
        labels_for_plot = labels_np

    n_components = 2 # Always 2 for 2D plot
    perplexity = min(30, max(1, embeddings_np.shape[0] - 1)) # Perplexity should be less than n_samples
    if embeddings_np.shape[0] < 5: # t-SNE requires at least 5 samples
        print(f"Warning: Not enough samples ({embeddings_np.shape[0]}) for t-SNE plot. Skipping.")
        return

    tsne = TSNE(n_components=n_components, random_state=42, perplexity=perplexity, n_iter=1000)
    try:
        tsne_results = tsne.fit_transform(embeddings_np)
    except Exception as e:
        print(f"Error during t-SNE fitting for '{title}': {e}. Skipping plot.")
        return

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=labels_for_plot, cmap='viridis', alpha=0.6)
    plt.title(title)
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.colorbar(scatter, ticks=unique_labels, label='Label') # Show original labels in colorbar
    plt.grid(True, linestyle='--', alpha=0.6)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"t-SNE plot saved to {save_path}")

def plot_cosine_similarity_hist(similarities, title, save_path):
    """
    Plots a histogram of cosine similarities.
    """
    if similarities.numel() == 0:
        print(f"Skipping cosine similarity histogram for '{title}': No similarities to plot.")
        return

    plt.figure(figsize=(8, 6))
    sns.histplot(similarities.cpu().numpy(), bins=50, kde=True)
    plt.title(title)
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Frequency')
    plt.grid(True, linestyle='--', alpha=0.6)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Cosine similarity histogram saved to {save_path}")

def plot_vector_trajectories(original_embeddings, edited_embeddings, labels, direction, save_path, num_samples=50):
    """
    Plots trajectories of embeddings before and after editing, showing the direction vector.
    Project embeddings onto the direction vector for 1D visualization.
    """
    if original_embeddings.shape[0] == 0:
        print(f"Skipping vector trajectory plot: No embeddings to plot.")
        return

    # Ensure tensors are on CPU and convert to numpy
    original_embeddings_np = original_embeddings.cpu().numpy()
    edited_embeddings_np = edited_embeddings.cpu().numpy()
    labels_np = labels.cpu().numpy()
    direction_np = direction.cpu().numpy()

    # Normalize direction for projection
    direction_norm = direction_np / np.linalg.norm(direction_np)

    # Project embeddings onto the direction vector
    proj_orig = np.dot(original_embeddings_np, direction_norm)
    proj_edited = np.dot(edited_embeddings_np, direction_norm)

    # Select a subset of samples for clarity
    if num_samples > original_embeddings_np.shape[0]:
        num_samples = original_embeddings_np.shape[0]
    
    # Ensure representative samples, perhaps by picking some positive and some negative
    pos_indices = np.where(labels_np == 1)[0]
    neg_indices = np.where(labels_np == -1)[0]

    selected_indices = []
    # Try to get roughly half positive and half negative
    if len(pos_indices) > 0:
        selected_indices.extend(np.random.choice(pos_indices, min(len(pos_indices), num_samples // 2), replace=False))
    if len(neg_indices) > 0:
        selected_indices.extend(np.random.choice(neg_indices, min(len(neg_indices), num_samples - len(selected_indices)), replace=False))
    
    # If not enough positive/negative, just fill with random from all
    if len(selected_indices) < num_samples:
        all_indices = np.arange(original_embeddings_np.shape[0])
        remaining_indices = np.setdiff1d(all_indices, selected_indices)
        selected_indices.extend(np.random.choice(remaining_indices, num_samples - len(selected_indices), replace=False))
    
    selected_indices = np.array(selected_indices)


    plt.figure(figsize=(10, 8))
    
    # Plot trajectories
    for i in selected_indices:
        color = 'blue' if labels_np[i] == -1 else 'red' # Blue for negative, Red for positive
        plt.plot([proj_orig[i], proj_edited[i]], [0, 1], color=color, alpha=0.6, marker='o', linestyle='-')
    
    # Add start and end points
    plt.scatter(proj_orig[selected_indices], np.zeros_like(proj_orig[selected_indices]), 
                c=[('red' if l == 1 else 'blue') for l in labels_np[selected_indices]], 
                label='Original Embeddings', s=100, zorder=5, edgecolor='k')
    plt.scatter(proj_edited[selected_indices], np.ones_like(proj_edited[selected_indices]), 
                c=[('red' if l == 1 else 'blue') for l in labels_np[selected_indices]], 
                label='Edited Embeddings', s=100, zorder=5, edgecolor='k', marker='X')

    plt.yticks([0, 1], ['Original', 'Edited'])
    plt.xlabel('Projection onto Concept Direction')
    plt.title('Embedding Trajectories along Concept Direction')
    
    # Create custom legend handles to avoid duplicate labels
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.grid(True, linestyle='--', alpha=0.6)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Vector trajectory plot saved to {save_path}")


def visualize_failure_modes(encoder, original_embeddings, edited_embeddings, original_labels, 
                            original_predictions, edited_predictions, image_ids, output_dir, 
                            cfg, concept, num_samples_to_save=10):
    """
    Identifies and visualizes images where latent surgery causes a failure (e.g., accuracy decrease).
    Generates a markdown report.

    Args:
        encoder (EncoderWrapper): The encoder model wrapper.
        original_embeddings (torch.Tensor): Embeddings before surgery.
        edited_embeddings (torch.Tensor): Embeddings after surgery.
        original_labels (torch.Tensor): True labels (-1/1).
        original_predictions (torch.Tensor): Predictions from original embeddings (0/1).
        edited_predictions (torch.Tensor): Predictions from edited embeddings (0/1).
        image_ids (list): List of image IDs (e.g., filenames) corresponding to embeddings.
        output_dir (str): Base output directory.
        cfg (dict): The full configuration dictionary, used to find data_root.
        concept (dict): Dictionary with 'positive' and 'negative' concept descriptions.
        num_samples_to_save (int): Number of failure examples to save for each category.
    """
    print("\n--- Identifying and Visualizing Failure Modes ---")
    
    failure_viz_dir = os.path.join(output_dir, 'failure_modes')
    os.makedirs(failure_viz_dir, exist_ok=True)

    # Convert labels from -1/1 to 0/1 for comparison with predictions
    labels_01 = (original_labels == 1).int()

    # Define failure categories
    failures = {
        "Original_Correct_Surgery_Incorrect": [],
    }

    # Iterate through each sample to find failures
    for i in range(len(original_embeddings)):
        orig_correct = (original_predictions[i] == labels_01[i]).item()
        edited_correct = (edited_predictions[i] == labels_01[i]).item()

        # Failure type 1: Original correct, but surgery made it incorrect
        if orig_correct and not edited_correct:
            failures["Original_Correct_Surgery_Incorrect"].append(i)
        
        # NOTE: For "Original_Incorrect_Surgery_Still_Incorrect_But_Worse", it's harder to define "worse" objectively
        # without a clear metric (e.g., confidence drop, further from decision boundary).
        # We will focus on "Original Correct, Surgery Incorrect" as it's a clear regression.

    markdown_report_path = os.path.join(output_dir, 'failure_modes_report.md')
    with open(markdown_report_path, 'w') as f:
        f.write("# Latent Surgery Failure Modes Report\n\n")
        f.write(f"Report generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"This report highlights examples where latent surgery led to undesired outcomes, primarily focusing on cases where initially correct classifications became incorrect after surgery.\n\n")

        for failure_type, indices in failures.items():
            f.write(f"## {failure_type.replace('_', ' ')}\n\n")
            f.write(f"Total instances found: {len(indices)}\n\n")

            samples_to_show = min(num_samples_to_save, len(indices))
            if samples_to_show == 0:
                f.write("No examples found for this category.\n\n")
                continue

            f.write(f"Showing {samples_to_show} examples:\n\n")
            
            selected_indices = np.random.choice(indices, samples_to_show, replace=False)
            
            for k, idx in enumerate(selected_indices):
                image_filename = image_ids[idx] # This is typically just the filename, e.g., '000001.jpg'
                
                # Reconstruct full image path based on dataset type and config
                dataset_name = cfg['dataset'].lower()
                data_root = cfg['data_root']
                image_full_path = None

                if dataset_name == 'celeba':
                    image_full_path = os.path.join(data_root, 'img_align_celeba', image_filename)
                elif dataset_name == 'cub-200':
                    # CUB's image_ids are usually full paths relative to images_dir (e.g., '001.Black_footed_Albatross/Black_Footed_Albatross_0001_796.jpg')
                    image_full_path = os.path.join(data_root, 'CUB_200_2011', 'images', image_filename) # Adjusted path for CUB
                else:
                    print(f"Warning: Unknown dataset '{dataset_name}'. Cannot determine image path for ID: {image_filename}.")


                saved_image_filename = f"{failure_type}_{image_filename.replace('.jpg', '')}_{k}.jpg" 
                # Ensure saved_image_filename is safe for paths (replace '/' with '_')
                saved_image_filename = saved_image_filename.replace(os.path.sep, '_')
                
                saved_image_path = os.path.join(failure_viz_dir, saved_image_filename)
                relative_image_path = os.path.join('failure_modes', saved_image_filename) # Path relative to output_dir for markdown

                if image_full_path and os.path.exists(image_full_path):
                    shutil.copy(image_full_path, saved_image_path)
                    f.write(f"### Sample {k+1}: Image ID {image_filename}\n\n")
                    f.write(f"True Label: {original_labels[idx].item()} ({concept['positive'] if original_labels[idx].item() == 1 else concept['negative']})\n")
                    f.write(f"Original Prediction: {original_predictions[idx].item()} ({'Correct' if original_predictions[idx].item() == labels_01[idx].item() else 'Incorrect'})\n")
                    f.write(f"Edited Prediction: {edited_predictions[idx].item()} ({'Correct' if edited_predictions[idx].item() == labels_01[idx].item() else 'Incorrect'})\n")
                    f.write(f"![{image_filename}]({relative_image_path})\n\n")
                else:
                    f.write(f"### Sample {k+1}: Image ID {image_filename} (Image not found or not saved)\n\n")
                    f.write(f"True Label: {original_labels[idx].item()}\n")
                    f.write(f"Original Prediction: {original_predictions[idx].item()}\n")
                    f.write(f"Edited Prediction: {edited_predictions[idx].item()}\n\n")

            f.write("---\n\n") # Separator
    
    print(f"Failure modes report saved to {markdown_report_path}")

# NEW: Function to visualize retrieval examples
def visualize_retrieval_examples(
    query_img_ids: list,
    query_original_embeddings: torch.Tensor,
    query_edited_embeddings: torch.Tensor,
    gallery_img_ids: list,
    gallery_embeddings: torch.Tensor,
    original_nn_indices: torch.Tensor, # Indices into gallery_img_ids/embeddings
    edited_nn_indices: torch.Tensor,   # Indices into gallery_img_ids/embeddings
    cfg: dict,
    num_queries_to_visualize: int = 5,
    num_nn_to_show: int = 5,
    output_dir: str = None
):
    """
    Visualizes retrieval examples (query image + its nearest neighbors)
    before and after latent surgery.
    
    Args:
        query_img_ids (list): Image IDs of the query images.
        query_original_embeddings (torch.Tensor): Original embeddings of query images.
        query_edited_embeddings (torch.Tensor): Edited embeddings of query images.
        gallery_img_ids (list): Image IDs of all gallery images.
        gallery_embeddings (torch.Tensor): Embeddings of all gallery images.
        original_nn_indices (torch.Tensor): Indices of k-NN in gallery for original queries.
                                             Shape: (num_queries, num_nn_to_show)
        edited_nn_indices (torch.Tensor): Indices of k-NN in gallery for edited queries.
                                           Shape: (num_queries, num_nn_to_show)
        cfg (dict): Configuration dictionary, used to get data_root.
        num_queries_to_visualize (int): Number of query examples to visualize.
        num_nn_to_show (int): Number of nearest neighbors to display for each query.
        output_dir (str, optional): Directory to save the visualization grids.
    """
    print("\n--- Visualizing Retrieval Examples ---")
    if output_dir is None:
        output_dir = os.path.join(cfg['output_dir'], 'retrieval_examples')
    os.makedirs(output_dir, exist_ok=True)

    # Ensure tensors are on CPU for numpy operations
    gallery_embeddings_np = gallery_embeddings.cpu().numpy()

    # Select random queries to visualize
    num_queries_to_visualize = min(num_queries_to_visualize, len(query_img_ids))
    selected_query_indices = np.random.choice(len(query_img_ids), num_queries_to_visualize, replace=False)

    dataset_name = cfg['dataset'].lower()
    data_root = cfg['data_root']

    for i, query_idx in enumerate(selected_query_indices):
        query_id = query_img_ids[query_idx]
        
        # Get paths for query image
        query_image_path = _get_image_full_path(query_id, dataset_name, data_root)
        if not query_image_path or not os.path.exists(query_image_path):
            print(f"Warning: Query image not found for ID {query_id}. Skipping visualization.")
            continue

        fig, axes = plt.subplots(2, num_nn_to_show + 1, figsize=(2 * (num_nn_to_show + 1), 4.5)) # +1 for query image

        # Load query image
        try:
            query_img = Image.open(query_image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading query image {query_image_path}: {e}. Skipping.")
            plt.close(fig)
            continue

        # Plot query image
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title(f'Query: {query_id}\n(Original)', fontsize=8)
        axes[0, 0].axis('off')
        axes[1, 0].imshow(query_img)
        axes[1, 0].set_title(f'Query: {query_id}\n(Edited)', fontsize=8)
        axes[1, 0].axis('off')

        # Plot original nearest neighbors
        original_nn_ids = [gallery_img_ids[idx] for idx in original_nn_indices[query_idx, :num_nn_to_show].tolist()]
        for j, nn_id in enumerate(original_nn_ids):
            nn_path = _get_image_full_path(nn_id, dataset_name, data_root)
            if nn_path and os.path.exists(nn_path):
                try:
                    nn_img = Image.open(nn_path).convert("RGB")
                    axes[0, j + 1].imshow(nn_img)
                    axes[0, j + 1].set_title(f'NN {j+1}: {nn_id}', fontsize=8)
                except Exception as e:
                    print(f"Error loading original NN image {nn_path}: {e}")
            axes[0, j + 1].axis('off')

        # Plot edited nearest neighbors
        edited_nn_ids = [gallery_img_ids[idx] for idx in edited_nn_indices[query_idx, :num_nn_to_show].tolist()]
        for j, nn_id in enumerate(edited_nn_ids):
            nn_path = _get_image_full_path(nn_id, dataset_name, data_root)
            if nn_path and os.path.exists(nn_path):
                try:
                    nn_img = Image.open(nn_path).convert("RGB")
                    axes[1, j + 1].imshow(nn_img)
                    axes[1, j + 1].set_title(f'NN {j+1}: {nn_id}', fontsize=8)
                except Exception as e:
                    print(f"Error loading edited NN image {nn_path}: {e}")
            axes[1, j + 1].axis('off')
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"retrieval_grid_{query_id.replace(os.path.sep, '_')}.png")
        plt.savefig(plot_path)
        plt.close(fig)
        print(f"Saved retrieval grid for query {query_id} to {plot_path}")


# Helper function to get full image path
def _get_image_full_path(image_id, dataset_name, data_root):
    if dataset_name == 'celeba':
        return os.path.join(data_root, 'img_align_celeba', image_id)
    elif dataset_name == 'cub-200':
        return os.path.join(data_root, 'CUB_200_2011', 'images', image_id)
    else:
        print(f"Warning: Unknown dataset '{dataset_name}'. Cannot determine image path for ID: {image_id}.")
        return None

# NEW: Function to plot heatmap of cosine similarity between direction vectors
def plot_direction_heatmap(
    direction_vectors: dict, # e.g., {'concept_A': tensor_A, 'concept_B': tensor_B}
    save_path: str = None
):
    """
    Generates a heatmap of pairwise cosine similarities between multiple direction vectors.

    Args:
        direction_vectors (dict): A dictionary where keys are concept names (str)
                                  and values are their corresponding direction vectors (torch.Tensor).
        save_path (str, optional): Directory and filename to save the heatmap.
                                   If None, plots are shown.
    """
    print("\n--- Generating Direction Vector Cosine Similarity Heatmap ---")
    if not direction_vectors:
        print("No direction vectors provided for heatmap. Skipping.")
        return

    concept_names = list(direction_vectors.keys())
    vectors = [vec.cpu().numpy() for vec in direction_vectors.values()]

    if len(vectors) < 2:
        print("At least two direction vectors are needed to create a similarity heatmap. Skipping.")
        return

    # Calculate pairwise cosine similarities
    num_vectors = len(vectors)
    similarity_matrix = np.eye(num_vectors) # Initialize with 1s on diagonal (self-similarity)

    for i in range(num_vectors):
        for j in range(i + 1, num_vectors):
            vec1 = vectors[i]
            vec2 = vectors[j]
            # Ensure vectors are normalized
            vec1_norm = vec1 / np.linalg.norm(vec1)
            vec2_norm = vec2 / np.linalg.norm(vec2)
            similarity = np.dot(vec1_norm, vec2_norm)
            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity # Matrix is symmetric

    plt.figure(figsize=(8, 7))
    sns.heatmap(
        similarity_matrix,
        annot=True,
        cmap='coolwarm',
        fmt=".2f",
        linewidths=.5,
        xticklabels=concept_names,
        yticklabels=concept_names
    )
    plt.title('Cosine Similarity between Concept Direction Vectors')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
        print(f"Direction vector heatmap saved to {save_path}")
    else:
        plt.show()