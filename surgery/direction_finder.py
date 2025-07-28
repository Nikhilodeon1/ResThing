# surgery/direction_finder.py
import torch

def compute_direction(clip_wrapper, dataloader, cfg):
    """
    Computes the semantic direction vector in CLIP's latent space.
    The direction is computed as the normalized vector from the mean embedding
    of 'negative' concept images to the mean embedding of 'positive' concept images.
    
    Args:
        clip_wrapper (CLIPModelWrapper): An instance of the CLIPModelWrapper.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset (e.g., CelebA).
                                                Assumes it provides (images, labels, image_ids),
                                                where labels are 0 for negative, 1 for positive.
        cfg (dict): Configuration dictionary containing 'concept' (positive/negative strings).
    Returns:
        torch.Tensor: The normalized direction vector.
    """
    positive_embeddings = []
    negative_embeddings = []

    print(f"Computing direction vector for concept: {cfg['concept']['positive']} vs {cfg['concept']['negative']}")

    # Ensure get_latents is called from clip_wrapper to get all embeddings
    all_embeddings, all_labels, _ = clip_wrapper.get_latents(dataloader)

    # Filter embeddings based on labels
    for i, label in enumerate(all_labels):
        if label == 1: # Assuming 1 for positive concept
            positive_embeddings.append(all_embeddings[i])
        elif label == 0: # Assuming 0 for negative concept
            negative_embeddings.append(all_embeddings[i])

    if not positive_embeddings:
        raise ValueError(f"No positive concept embeddings found for '{cfg['concept']['positive']}'. "
                         "Check dataset labels or filtering logic.")
    if not negative_embeddings:
        raise ValueError(f"No negative concept embeddings found for '{cfg['concept']['negative']}'. "
                         "Check dataset labels or filtering logic.")

    mean_positive_embedding = torch.stack(positive_embeddings).mean(dim=0)
    mean_negative_embedding = torch.stack(negative_embeddings).mean(dim=0)

    # Compute the raw direction vector
    raw_direction = mean_positive_embedding - mean_negative_embedding

    # Normalize the direction vector to get a unit vector
    direction_unit_vector = raw_direction / torch.norm(raw_direction)
    
    print(f"Direction vector computed. Shape: {direction_unit_vector.shape}")
    return direction_unit_vector.cpu()

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    from models.clip_wrapper import CLIPModelWrapper
    from torch.utils.data import Dataset, DataLoader
    import torch

    # Mock DataLoader for testing direction_finder
    class MockCelebADataset(Dataset):
        def __init__(self, num_samples=100, embed_dim=512):
            self.num_samples = num_samples
            self.embed_dim = embed_dim
            # Create synthetic embeddings and labels
            # 50 positive (label 1), 50 negative (label 0)
            self.embeddings = torch.randn(num_samples, embed_dim)
            # Make positive embeddings slightly shifted in one dimension for a clear direction
            self.embeddings[num_samples//2:] += 2 * torch.randn(num_samples//2, embed_dim) # Add some noise
            self.embeddings[num_samples//2:, 0] += 5 # Stronger shift in first dimension for "positive"
            
            self.labels = torch.cat([torch.zeros(num_samples//2), torch.ones(num_samples//2)]).long()
            self.img_ids = [f"mock_img_{i:03d}" for i in range(num_samples)]
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # This mock dataset provides dummy image tensors, labels, and image IDs.
            # get_latents in CLIPModelWrapper will process these.
            return torch.randn(3, 224, 224), self.labels[idx], self.img_ids[idx]

    # Mock CLIPWrapper (doesn't actually run CLIP, just returns pre-made embeddings for get_latents)
    class MockCLIPWrapper:
        def __init__(self, embed_dim=512):
            self.embed_dim = embed_dim
            self.device = 'cpu' # For mock

        def get_latents(self, dataloader):
            # Simulate get_latents by returning pre-generated embeddings and labels
            mock_embeddings = []
            mock_labels = []
            mock_img_ids = []
            for img_tensor, label, img_id in dataloader:
                # For this mock, we assume the dataloader gives us the 'true' embeddings
                # In real scenario, get_latents internally calls embed_image
                mock_embeddings.append(dataloader.dataset.embeddings[dataloader.dataset.img_ids.index(img_id[0])].unsqueeze(0))
                mock_labels.append(label.item())
                mock_img_ids.append(img_id[0])
            return torch.cat(mock_embeddings, dim=0), torch.tensor(mock_labels), mock_img_ids

    mock_cfg = {
        'concept': {'positive': 'Smiling', 'negative': 'Not_Smiling'},
        'batch_size': 32
    }
    
    # Initialize mock data and loader
    mock_dataset = MockCelebADataset()
    mock_dataloader = DataLoader(mock_dataset, batch_size=mock_cfg['batch_size'], shuffle=False)

    # Initialize mock CLIP wrapper
    mock_clip = MockCLIPWrapper(embed_dim=mock_dataset.embed_dim)

    print("Testing compute_direction with mock data...")
    direction_vector = compute_direction(mock_clip, mock_dataloader, mock_cfg)
    print(f"Computed direction vector (first 5 dims): {direction_vector[:5]}")
    print(f"Norm of direction vector: {torch.norm(direction_vector)}") # Should be close to 1.0