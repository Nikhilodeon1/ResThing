# surgery/edit_embedding.py
import torch
from tqdm import tqdm

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
    # Although get_latents returns CPU embeddings, for large scale ops, it might be faster on GPU
    # Let's ensure consistency for safety
    direction_vector = direction_vector.to(original_embeddings.device)
    
    # Expand direction_vector to match the batch dimension for element-wise addition
    # No need to expand explicitly if broadcasting handles it, but good to be aware.
    # original_embeddings.shape is (N, D), direction_vector.shape is (D,)
    # Addition will broadcast correctly.
    edited_embeddings = original_embeddings + alpha * direction_vector
    
    print("Latent surgery applied.")
    return edited_embeddings, original_embeddings, labels, img_ids

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    from models.clip_wrapper import CLIPModelWrapper
    from surgery.direction_finder import compute_direction
    from torch.utils.data import Dataset, DataLoader
    import torch

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
            for img_tensor, label, img_id in dataloader:
                # Find the true embedding for this img_id from the mock dataset
                idx = dataloader.dataset.img_ids.index(img_id[0])
                mock_embeddings.append(dataloader.dataset.embeddings[idx].unsqueeze(0))
                mock_labels.append(label.item())
                mock_img_ids.append(img_id[0])
            return torch.cat(mock_embeddings, dim=0), torch.tensor(mock_labels), mock_img_ids

    mock_cfg = {
        'concept': {'positive': 'Smiling', 'negative': 'Not_Smiling'},
        'batch_size': 32,
        'surgery_alpha': 1.0 # Test alpha
    }
    
    mock_dataset = MockCelebADataset()
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