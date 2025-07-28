# surgery/direction_finder.py
import torch

# surgery/direction_finder.py

import torch
import numpy as np # Added this import if not already present

def compute_direction(clip_wrapper, dataloader, cfg):
    print("\n--- Phase 2.1: Computing Semantic Direction ---")

    # Get all embeddings and labels from the dataloader
    # The dataloader now provides images preprocessed by CLIP's preprocess_image
    # And labels which are 1 or -1 for the 'Smiling' attribute.
    all_embeddings, all_labels, _ = clip_wrapper.get_latents(dataloader)

    # Ensure labels are PyTorch tensors if they aren't already
    if not isinstance(all_labels, torch.Tensor):
        all_labels = torch.tensor(all_labels, dtype=torch.long) # Use long for labels

    # Get the target attribute name from config (e.g., 'Smiling')
    target_attribute_name = cfg['concept']['positive'] # This will be 'Smiling'

    # Filter embeddings based on the actual labels (1 for present, -1 for absent)
    # The 'positive' concept corresponds to label 1
    positive_concept_embeddings = all_embeddings[all_labels == 1]
    # The 'negative' concept corresponds to label -1
    negative_concept_embeddings = all_embeddings[all_labels == -1]


    if len(positive_concept_embeddings) == 0:
        raise ValueError(f"No positive concept embeddings found for '{target_attribute_name}'. "
                         "Check dataset labels or filtering logic.")
    if len(negative_concept_embeddings) == 0:
        # This is the specific error you were getting, now handled by looking for -1
        raise ValueError(f"No negative concept embeddings found (i.e., no -1 labels for '{target_attribute_name}'). "
                         "Check dataset labels or filtering logic.")

    print(f"Found {len(positive_concept_embeddings)} positive embeddings for '{target_attribute_name}'.")
    print(f"Found {len(negative_concept_embeddings)} negative embeddings for '{target_attribute_name}'.")


    # Calculate the mean embedding for each concept
    mean_positive_embedding = positive_concept_embeddings.mean(dim=0)
    mean_negative_embedding = negative_concept_embeddings.mean(dim=0)

    # Compute the direction vector
    direction = mean_positive_embedding - mean_negative_embedding

    # Normalize the direction vector
    direction = direction / direction.norm()

    print(f"Semantic direction for '{target_attribute_name}' computed.")
    return direction

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