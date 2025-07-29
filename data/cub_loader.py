# data/cub_loader.py

import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
# Assuming transforms.py is no longer needed as preprocess_fn is passed
# from data.transforms import get_image_transforms # REMOVE THIS LINE if no longer used

class CUBDataset(Dataset):
    def __init__(self, root_dir, target_attribute, transform=None):
        """
        Args:
            root_dir (string): Directory with CUB-200-2011 dataset.
                               Expected structure: root_dir/images/, root_dir/attributes/,
                               root_dir/images.txt, root_dir/image_class_labels.txt,
                               root_dir/classes.txt, root_dir/train_test_split.txt
            target_attribute (str): The specific attribute to use for binary classification
                                    (e.g., "has_bill_shape_conical").
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.transform = transform
        self.target_attribute = target_attribute

        # Load necessary CUB metadata files
        self.images_df = pd.read_csv(os.path.join(root_dir, 'images.txt'), sep=' ', names=['image_id', 'image_name'])
        self.train_test_split_df = pd.read_csv(os.path.join(root_dir, 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_img'])
        
        # Load attribute names and image attribute labels
        self.attributes_df = pd.read_csv(os.path.join(root_dir, 'attributes', 'attributes.txt'), sep=' ', names=['attribute_id', 'attribute_name'])
        self.image_attributes_df = pd.read_csv(os.path.join(root_dir, 'attributes', 'image_attribute_labels.txt'), sep=' ', names=['image_id', 'attribute_id', 'attribute_value', 'confidence_level', 'time'])

        # Merge dataframes to get image names and attribute values
        self.data = pd.merge(self.images_df, self.image_attributes_df, on='image_id')
        self.data = pd.merge(self.data, self.attributes_df, on='attribute_id')
        
        # Filter for the target attribute
        self.data_filtered = self.data[self.data['attribute_name'] == self.target_attribute].copy()

        if self.data_filtered.empty:
            raise ValueError(f"Target attribute '{self.target_attribute}' not found or no images for it. "
                             "Please check attribute name in attributes.txt or data integrity.")

        # Map attribute_value (1 for present, 0 for absent/not mentioned) to -1/1 labels
        # The CUB attributes are typically 1 (present) or 0 (absent/not applicable).
        # We need to map 1 -> 1 (positive concept) and 0 -> -1 (negative concept)
        self.data_filtered['label'] = self.data_filtered['attribute_value'].apply(lambda x: 1 if x == 1 else -1)

        # Get unique image IDs and their corresponding labels for the target attribute
        # If an image has multiple entries for the same attribute (e.g., different confidences), take the first.
        self.data_filtered = self.data_filtered.drop_duplicates(subset=['image_id', 'attribute_name'])
        
        # Ensure image_id is the index for easy lookup
        self.data_filtered = self.data_filtered.set_index('image_id')

        self.image_ids = self.data_filtered.index.tolist()
        self.labels = self.data_filtered['label'].loc[self.image_ids].values

        # Debugging the labels to ensure they contain -1s and 1s
        unique_labels = pd.Series(self.labels).unique()
        print(f"DEBUG (CUBDataset): Unique labels for '{self.target_attribute}': {unique_labels}")
        if not (-1 in unique_labels and 1 in unique_labels):
            print(f"WARNING: Labels for '{self.target_attribute}' do not contain both 1 and -1. "
                  "This might affect direction finding for binary attributes.")
        
        if self.data_filtered.empty:
            raise ValueError("No data found after filtering for target attribute. Check attribute name or dataset integrity.")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        img_name_full_path = self.images_df[self.images_df['image_id'] == img_id]['image_name'].iloc[0]
        img_path = os.path.join(self.img_dir, img_name_full_path)
        
        image = Image.open(img_path).convert('RGB')
        label = int(self.labels[idx]) # Ensure label is an integer (1 or -1)

        if self.transform:
            image = self.transform(image)

        return image, label, img_name_full_path # Return full image name for consistency

def get_cub_dataloaders(cfg, preprocess_fn): # Added preprocess_fn argument
    """
    Returns train and test dataloaders for CUB-200.
    """
    # The 'concept' in config.yaml for CUB should now specify the target attribute.
    # Example: concept: { positive: "has_bill_shape_conical" }
    target_attribute = cfg['concept']['positive'] # Assuming positive concept is the target attribute name

    full_dataset = CUBDataset(
        root_dir=cfg['cub_root_dir'],
        target_attribute=target_attribute,
        transform=preprocess_fn # Pass the encoder's preprocess_fn
    )

    # Use a custom collate_fn to handle batching of preprocessed images
    def custom_collate_fn(batch):
        images = torch.cat([item[0] for item in batch], dim=0)
        labels = torch.tensor([item[1] for item in batch])
        img_ids = [item[2] for item in batch]
        return images, labels, img_ids

    # Use the train_test_split.txt file to create train and test splits
    split_df = pd.read_csv(os.path.join(cfg['cub_root_dir'], 'train_test_split.txt'), sep=' ', names=['image_id', 'is_training_img'])
    
    # Filter image IDs based on the target attribute and then apply train/test split
    # Ensure only images relevant to the target attribute are considered for splitting
    relevant_image_ids = full_dataset.data_filtered.index.tolist()
    split_df_relevant = split_df[split_df['image_id'].isin(relevant_image_ids)]

    train_image_ids = split_df_relevant[split_df_relevant['is_training_img'] == 1]['image_id'].tolist()
    test_image_ids = split_df_relevant[split_df_relevant['is_training_img'] == 0]['image_id'].tolist()

    # Create datasets for train and test splits using Subset
    from torch.utils.data import Subset
    train_indices = [full_dataset.image_ids.index(img_id) for img_id in train_image_ids if img_id in full_dataset.image_ids]
    test_indices = [full_dataset.image_ids.index(img_id) for img_id in test_image_ids if img_id in full_dataset.image_ids]

    train_dataset = Subset(full_dataset, train_indices)
    test_dataset = Subset(full_dataset, test_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=os.cpu_count() // 2 or 1,
        collate_fn=custom_collate_fn # Use custom collate function
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1,
        collate_fn=custom_collate_fn # Use custom collate function
    )
    
    print(f"CUB-200: Loaded {len(full_dataset)} images for attribute '{target_attribute}'.")
    print(f"Train: {len(train_dataset)} images, Test: {len(test_dataset)} images.")
    print(f"Train loader has {len(train_loader)} batches.")
    print(f"Test loader has {len(test_loader)} batches.")
    return train_loader, test_loader

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    from utils.config_loader import load_config
    from models.encoder_wrapper import EncoderWrapper # For mock preprocess_fn

    # Mock preprocess_fn for testing
    class MockPreprocessFn:
        def __call__(self, images):
            # Simulate CLIP preprocessing: adds batch dim, converts to float32
            if isinstance(images, list): # Handle batch of PIL images
                return torch.stack([torch.randn(3, 224, 224) for _ in images])
            return torch.randn(1, 3, 224, 224) # Single image

    try:
        cfg = load_config("../config.yaml") 
        cfg['dataset'] = 'CUB-200' # Temporarily set for testing
        
        # IMPORTANT: You must ensure this attribute exists in CUB's attributes.txt
        # and has both 0 and 1 values in image_attribute_labels.txt
        cfg['concept']['positive'] = "has_bill_shape_conical" # Example CUB attribute
        cfg['concept']['negative'] = "not_has_bill_shape_conical" # Placeholder for consistency, actual filtering uses -1

        # Make sure cub_root_dir points to your CUB dataset
        # This path must contain 'images', 'attributes', 'images.txt', 'train_test_split.txt', etc.
        cfg['cub_root_dir'] = "/content/data/cub" # Adjust to your actual path

        print(f"Attempting to load CUB-200 from: {cfg['cub_root_dir']}")
        train_loader, test_loader = get_cub_dataloaders(cfg, MockPreprocessFn())
        
        for i, (images, labels, img_names) in enumerate(train_loader):
            print(f"Batch {i+1}: Images shape: {images.shape}, Labels shape: {labels.shape}")
            print(f"Sample labels: {labels[:5]}")
            print(f"Sample image names: {img_names[:5]}")
            break
    except Exception as e:
        print(f"Error during CUB loader test: {e}")