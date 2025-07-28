# data/celeba_loader.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data.transforms import get_image_transforms # Assuming transforms.py is in the same 'data' directory

class CelebADataset(Dataset):
    def __init__(self, root_dir, transform=None, concept_positive=None, concept_negative=None):
        """
        Args:
            root_dir (string): Directory with all the images and annotation file.
                               Expected structure: root_dir/img_align_celeba/ and root_dir/list_attr_celeba.csv
            transform (callable, optional): Optional transform to be applied on an image.
            concept_positive (string): The attribute name for positive samples (e.g., "Smiling").
            concept_negative (string): The attribute name for negative samples (e.g., "Not_Smiling").
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        self.attr_path = os.path.join(root_dir, 'list_attr_celeba.csv')
        self.transform = transform

        if not os.path.exists(self.attr_path):
            raise FileNotFoundError(f"Annotation file not found at: {self.attr_path}. Please check the path and extraction.")
        
        self.attr_df = pd.read_csv(self.attr_path, delim_whitespace=True) # Use delim_whitespace as per provided format [cite: 242]

        self.data_filtered = []
        if concept_positive and concept_negative:
            # Filter for images that are clearly positive or negative for the concept
            # Assuming '1' for positive, '-1' for negative [cite: 243]
            positive_attr = self.attr_df[self.attr_df[concept_positive] == 1]
            negative_attr = self.attr_df[self.attr_df[concept_negative] == 1] # This might be incorrect based on sample data - let's refine.

            # Re-evaluating based on sample data: attributes are binary, 1 for present, -1 for absent.
            # So, for "Smiling" positive is 1, negative is -1.
            self.data_filtered = self.attr_df[
                (self.attr_df[concept_positive] == 1) | (self.attr_df[concept_positive] == -1)
            ].copy()
            self.data_filtered['label'] = (self.data_filtered[concept_positive] == 1).astype(int)
        else:
            print("Warning: No concepts provided for filtering. Loading all images.")
            self.data_filtered = self.attr_df.copy()
            self.data_filtered['label'] = 0 # Default label if not filtering

        if self.data_filtered.empty:
            raise ValueError("No data found after filtering. Check concept names or dataset integrity.")

    def __len__(self):
        return len(self.data_filtered)

    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.data_filtered.iloc[idx]['image_id'])
        image = Image.open(img_name).convert('RGB')
        label = self.data_filtered.iloc[idx]['label']

        if self.transform:
            image = self.transform(image)

        return image, label, self.data_filtered.iloc[idx]['image_id'] # Also return image_id for traceability


def get_celeba_dataloaders(cfg):
    """
    Returns train and test dataloaders for CelebA.
    """
    transform = get_image_transforms()
    
    # In a real scenario, you'd split into train/test. For simplicity, we'll load the full dataset for now.
    # A common approach is to use a subset for finding directions and another for evaluation.
    # For now, we'll assume the full filtered dataset for both if no explicit split is given.
    
    concept_positive = cfg['concept']['positive']
    concept_negative = cfg['concept']['negative']
    
    full_dataset = CelebADataset(
        root_dir=cfg['celeba_root_dir'],
        transform=transform,
        concept_positive=concept_positive,
        concept_negative=concept_negative
    )

    # For demonstration, we'll use a simple 80/20 split. In practice, CelebA has official splits.
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # This requires scikit-learn for train_test_split, or manual splitting
    from torch.utils.data import random_split
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=os.cpu_count() // 2 or 1 # Use half available CPU cores
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1
    )
    
    print(f"CelebA: Loaded {len(full_dataset)} images. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    return train_loader, test_loader

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    from utils.config_loader import load_config
    try:
        cfg = load_config("../config.yaml") # Adjust path as needed for local testing
        if cfg['dataset'] == 'CelebA':
            print(f"Attempting to load CelebA from: {cfg['celeba_root_dir']}")
            train_loader, test_loader = get_celeba_dataloaders(cfg)
            print(f"Train loader has {len(train_loader)} batches.")
            print(f"Test loader has {len(test_loader)} batches.")
            
            # Test iterating through a batch
            for i, (images, labels, img_ids) in enumerate(train_loader):
                print(f"Batch {i+1}: Images shape: {images.shape}, Labels shape: {labels.shape}")
                print(f"Sample labels: {labels[:5]}")
                print(f"Sample image IDs: {img_ids[:5]}")
                break
        else:
            print("CelebA loader test skipped as 'dataset' in config is not 'CelebA'.")
    except Exception as e:
        print(f"Error during CelebA loader test: {e}")