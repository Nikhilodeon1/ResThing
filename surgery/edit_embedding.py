# surgery/edit_embedding.py
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt # NEW: Import matplotlib for plotting
import seaborn as sns # NEW: Import seaborn for enhanced plots
import os # NEW: Import os for path operations
from utils.io_utils import save_numpy # NEW: Import save_numpy for saving data for plots if needed

def apply_surgery(clip_wrapper, dataloader, direction_vector, alpha):
    """
    Applies latent space surgery to image embeddings.
    The transformation is defined as z' = z + alpha * v, where z is the original embedding,
    v is the direction vector, and alpha is the scaling factor.
    
    Args:
        clip_wrapper (CLIPModelWrapper): An instance of the CLIPModelWrapper.
                                         Used to get original embeddings via get_latents.
        dataloader (torch.utils.data.DataLoader): DataLoader providing images for editing.
        direction_vector (torch.Tensor): The pre-computed unit direction vector.
        alpha (float): The scaling factor for the direction vector.
    
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The edited image embeddings.
            - torch.Tensor: The original image embeddings.
            - torch.Tensor: The corresponding labels.
            - list: The corresponding image IDs/names.
    """
    original_embeddings, labels, img_ids = clip_wrapper.get_latents(dataloader)
    
    print(f"Applying latent surgery with alpha={alpha} to {len(original_embeddings)} embeddings...")
    
    # Ensure direction_vector is on the same device as embeddings if computation happens on device
    direction_vector = direction_vector.to(original_embeddings.device)
    
    # Expand direction_vector to match the batch dimension for element-wise addition
    edited_embeddings = original_embeddings + alpha * direction_vector
    
    print("Latent surgery applied.")
    return edited_embeddings, original_embeddings, labels, img_ids


# NEW: Function to plot concept weight changes
def plot_concept_weight_change(
    original_embeddings: torch.Tensor, 
    edited_embeddings: torch.Tensor, 
    direction_vector: torch.Tensor, 
    img_ids: list, 
    cfg: dict,
    num_samples_to_plot: int = 5,
    output_dir: str = None
):
    """
    Generates bar charts showing the change in concept weight for example images
    before and after latent surgery.
    Concept weight is calculated as the projection of the embedding onto the direction vector.

    Args:
        original_embeddings (torch.Tensor): Original image embeddings.
        edited_embeddings (torch.Tensor): Edited image embeddings.
        direction_vector (torch.Tensor): The concept direction vector.
        img_ids (list): List of image IDs/names corresponding to the embeddings.
        cfg (dict): Configuration dictionary, used for concept names.
        num_samples_to_plot (int): Number of example images to plot.
        output_dir (str, optional): Directory to save the plots. If None, plots are shown.
    """
    if output_dir is None:
        output_dir = os.path.join(cfg['output_dir'], 'concept_weight_charts')
    os.makedirs(output_dir, exist_ok=True)

    # Move direction_vector to CPU for numpy operations if it's on GPU
    direction_vector_np = direction_vector.cpu().numpy()

    # Calculate concept weights (projection onto the direction vector)
    # Using numpy for dot product as embeddings might be on CPU already from apply_surgery
    original_weights = torch.matmul(original_embeddings.cpu(), direction_vector.cpu()).numpy()
    edited_weights = torch.matmul(edited_embeddings.cpu(), direction_vector.cpu()).numpy()

    # Select random samples to plot
    # Ensure we don't pick more samples than available
    num_samples_to_plot = min(num_samples_to_plot, len(img_ids))
    
    # For more consistent visualization of typical changes,
    # it might be beneficial to pick samples where the change is notable,
    # or a mix of positive/negative concepts. For now, random selection.
    
    indices = torch.randperm(len(img_ids))[:num_samples_to_plot].tolist()

    concept_name = cfg['concept']['positive'] # Assuming positive concept defines the direction

    print(f"Generating concept weight change charts for {num_samples_to_plot} samples...")

    for i, idx in enumerate(indices):
        img_id = img_ids[idx]
        orig_w = original_weights[idx]
        edited_w = edited_weights[idx]

        fig, ax = plt.subplots(figsize=(6, 4))
        
        # Prepare data for plotting
        labels = ['Original', 'Edited']
        weights = [orig_w, edited_w]
        colors = ['skyblue', 'lightcoral']

        sns.barplot(x=labels, y=weights, palette=colors, ax=ax)

        ax.set_ylabel(f'Concept Weight ({concept_name})')
        ax.set_title(f'Concept Weight Change for {img_id} (α={cfg["surgery_alpha"]})')
        ax.axhline(0, color='grey', linewidth=0.8) # Add a line at 0 for reference
        
        # Add numerical values on top of bars
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='center', 
                        xytext=(0, 5), 
                        textcoords='offset points')

        plt.tight_layout()
        plot_path = os.path.join(output_dir, f"concept_weight_change_{img_id}.png")
        plt.savefig(plot_path)
        plt.close(fig) # Close the plot to free memory
        print(f"Saved concept weight chart for {img_id} to {plot_path}")

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    from models.clip_wrapper import CLIPModelWrapper
    from surgery.direction_finder import compute_direction
    from torch.utils.data import Dataset, DataLoader
    import torch
    import sys

    # Mock DataLoader for testing apply_surgery (same as direction_finder's mock)
    class MockCelebADataset(Dataset):
        def __init__(self, num_samples=100, embed_dim=512):
            self.num_samples = num_samples
            self.embed_dim = embed_dim
            self.embeddings = torch.randn(num_samples, embed_dim)
            self.embeddings[num_samples//2:] += 2 * torch.randn(num_samples//2, embed_dim)
            self.embeddings[num_samples//2:, 0] += 5 # Stronger shift in first dimension for "positive"
            
            self.labels = torch.cat([torch.zeros(num_samples//2), torch.ones(num_samples//2)]).long()
            self.img_ids = [f"mock_img_{i:03d}" for i in range(num_samples)]
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Return a dummy image tensor, label, and img_id
            return torch.randn(3, 224, 224), self.labels[idx], self.img_ids[idx]

    # Mock CLIPWrapper (same as direction_finder's mock)
    class MockCLIPWrapper:
        def __init__(self, embed_dim=512):
            self.embed_dim = embed_dim
            self.device = 'cpu'

        def get_latents(self, dataloader):
            mock_embeddings = []
            mock_labels = []
            mock_img_ids = []
            # In a real scenario, this would extract embeddings from images via CLIP
            # For mock, we retrieve pre-defined embeddings
            for img_tensor, label, img_id in dataloader:
                # Find the true embedding for this img_id from the mock dataset
                # Assuming img_id is a list/tuple of one ID for batch_size=1
                idx = dataloader.dataset.img_ids.index(img_id[0]) 
                mock_embeddings.append(dataloader.dataset.embeddings[idx].unsqueeze(0))
                mock_labels.append(label.item())
                mock_img_ids.append(img_id[0])
            return torch.cat(mock_embeddings, dim=0), torch.tensor(mock_labels), mock_img_ids

    mock_cfg = {
        'concept': {'positive': 'Smiling', 'negative': 'Not_Smiling'},
        'batch_size': 16, # Increased batch size for more realistic mock
        'surgery_alpha': 1.0, # Test alpha
        'output_dir': './outputs_test_edit_embedding', # Test output directory
        'save_tsne': True # Placeholder, not used directly here
    }
    
    mock_dataset = MockCelebADataset(num_samples=50) # Reduced samples for quick test
    mock_dataloader = DataLoader(mock_dataset, batch_size=mock_cfg['batch_size'], shuffle=False)
    mock_clip = MockCLIPWrapper(embed_dim=mock_dataset.embed_dim)

    # First, compute the direction vector
    direction_vector = compute_direction(mock_clip, mock_dataloader, mock_cfg)
    print(f"\nDirection vector computed for apply_surgery test: {direction_vector[:5]}")

    # Now, apply surgery
    print("\nTesting apply_surgery...")
    edited_embs, original_embs, labels, img_ids = apply_surgery(
        mock_clip, mock_dataloader, direction_vector, mock_cfg['surgery_alpha']
    )

    print(f"Original embeddings shape: {original_embs.shape}")
    print(f"Edited embeddings shape: {edited_embs.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of image IDs: {len(img_ids)}")

    # Verify the transformation for a few samples
    sample_idx = 0
    print(f"\nOriginal embedding (sample {sample_idx}, first 5 dims): {original_embs[sample_idx, :5]}")
    print(f"Edited embedding (sample {sample_idx}, first 5 dims): {edited_embs[sample_idx, :5]}")
    expected_edited_embs = original_embs[sample_idx] + mock_cfg['surgery_alpha'] * direction_vector
    print(f"Expected edited embedding (sample {sample_idx}, first 5 dims): {expected_edited_embs[:5]}")
    
    # Check if the transformation was applied correctly
    assert torch.allclose(edited_embs[sample_idx], expected_edited_embs, atol=1e-6), "Surgery not applied correctly!"
    print("Surgery application verified for sample.")

    # NEW: Test plot_concept_weight_change
    print("\nTesting plot_concept_weight_change...")
    try:
        plot_concept_weight_change(
            original_embs,
            edited_embs,
            direction_vector,
            img_ids,
            mock_cfg,
            num_samples_to_plot=3,
            output_dir=os.path.join(mock_cfg['output_dir'], 'concept_weight_charts_test')
        )
        print("plot_concept_weight_change test finished. Check the output directory.")
    except Exception as e:
        print(f"Error during plot_concept_weight_change test: {e}")
        sys.exit(1)