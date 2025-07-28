# data/celeba_loader.py
import os
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from data.transforms import get_image_transforms # Assuming transforms.py is in the same 'data' directory

class CelebADataset(Dataset):
    def __init__(self, root_dir, target_attribute, transform=None):
        """
        Args:
            root_dir (str): Directory with all the CelebA image folders and attribute list.
            target_attribute (str): The attribute to target (e.g., 'Smiling').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.img_dir = os.path.join(root_dir, 'img_align_celeba') # Standard CelebA image folder name
        self.attr_file = os.path.join(root_dir, 'list_attr_celeba.csv')

        print(f"DEBUG (CelebADataset): Attempting to load attribute file from: {self.attr_file}")
        if not os.path.exists(self.attr_file):
            raise FileNotFoundError(f"DEBUG (CelebADataset): Attribute file NOT FOUND at: {self.attr_file}. Please check your path.")

        self.attr_df = pd.read_csv(self.attr_file)
        
        # CelebA attribute file has 'image_id' as the first column, which needs to be the index
        self.attr_df = self.attr_df.set_index('image_id')

        print(f"DEBUG (CelebADataset): Columns loaded into attr_df: {self.attr_df.columns.tolist()}")

        self.target_attribute = target_attribute

        # Ensure the target_attribute exists in the DataFrame's columns
        if self.target_attribute not in self.attr_df.columns:
            raise KeyError(f"DEBUG (CelebADataset): Configured target attribute '{self.target_attribute}' not found in loaded CSV columns. Available: {self.attr_df.columns.tolist()}")

        # The core of the fix:
        # Instead of filtering into separate positive/negative indices,
        # we load all image IDs and their corresponding labels for the target attribute.
        # The 'labels' will contain 1s and -1s for the target_attribute (e.g., 'Smiling').
        self.image_ids = self.attr_df.index.tolist()
        self.labels = self.attr_df[self.target_attribute].loc[self.image_ids].values

        # Debugging the labels to ensure they contain -1s and 1s
        unique_labels = pd.Series(self.labels).unique()
        print(f"DEBUG (CelebADataset): Unique labels for '{self.target_attribute}': {unique_labels}")
        if not (-1 in unique_labels and 1 in unique_labels):
            print(f"WARNING: Labels for '{self.target_attribute}' do not contain both 1 and -1. "
                  "This might affect direction finding for binary attributes.")
        
        # No need for self.positive_indices or self.negative_indices for the DataLoader's __getitem__
        # The direction_finder will handle separating them based on the labels.


    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            transformed_image = self.transform(image)
            # DEBUG
            # print(f"\nDEBUG (CelebADataset __getitem__ for {img_name}):")
            # print(f"  Type after transform: {type(transformed_image)}")
            # if isinstance(transformed_image, torch.Tensor):
            #     print(f"  Shape after transform: {transformed_image.shape}")
            #     print(f"  Number of dimensions: {transformed_image.dim()}")
            # else:
            #     print(f"  Transformed image is NOT a torch.Tensor, it's {type(transformed_image)}!")
            # END DEBUG
            return transformed_image, label, img_name
        else:
            raise ValueError("Transform must be provided for CelebADataset to preprocess images for CLIP.")

# Helper function to get dataloaders
# Modify this function to accept the CLIP preprocessor
def get_celeba_dataloaders(cfg, clip_preprocess): # Added clip_preprocess argument
    """
    Creates and returns CelebA train and test DataLoaders.
    """
    # Assuming cfg contains 'celeba_root_dir', 'batch_size', 'num_workers'
    # And that the dataset split logic (if any) is handled internally by CelebADataset
    # For simplicity, we'll create one dataset and then split (or assume dataset handles split)

    # Let's modify based on the full main.py's usage which calls get_celeba_dataloaders(cfg)
    # and expects it to return train_loader, test_loader
    # This means this function itself needs to instantiate CLIPModelWrapper or get its preprocess
    # A cleaner approach is for main.py to handle transform passing to the dataset directly
    # But since main.py calls get_celeba_dataloaders(cfg), we adapt here.

    # The current main.py structure has a slight coupling here.
    # The `main.py` I provided earlier had this:
    # `dataset = CelebADataset(root_dir=cfg['celeba_root_dir'], target_attribute=cfg['concept']['positive'], transform=clip_model.preprocess_image)`
    # This is the preferred way.

    # If you intend to keep `train_loader, test_loader = get_celeba_dataloaders(cfg)` in main.py:
    # Then get_celeba_dataloaders needs to know about `clip_preprocess`.

    # Let's adjust main.py to directly instantiate dataset and then dataloaders
    # This removes the need for `get_celeba_dataloaders` to deal with `clip_preprocess`
    # and aligns with the more direct style of the blueprint.
    # The current `get_celeba_dataloaders` in your provided code doesn't take `clip_preprocess`.

    # For now, let's assume get_celeba_dataloaders directly creates datasets and splits.
    # We will need the CLIP preprocessor, which usually comes from the model.

    # Option 1: Pass clip_model to get_celeba_dataloaders (cleaner for this structure)
    # main.py would call: train_loader, test_loader = get_celeba_dataloaders(cfg, clip)
    # And then this function would do: transform=clip.preprocess_image

    # Option 2: Define a default transform if not passed (less ideal as it's not CLIP-specific)
    # For a minimal fix to the KeyError, let's just make sure the path is correct.

    # The error is in CelebADataset.__init__ when it tries to access the column.
    # The `get_celeba_dataloaders` function's role is primarily to return the loaders.
    # It *must* receive the correct `clip_preprocess` to pass to the dataset.
    # Let's adjust main.py to pass it properly.

    # For a quick fix, let's define a dummy transform if not passed.
    # But the real fix is passing CLIP's transform.

    # Let's stick to the minimal change for the debug, which is inside CelebADataset.__init__
    # as that's where the KeyError happens.

    # For the data loaders, assuming a simple split for demonstration.
    # In a real scenario, you'd likely have a fixed train/test split.
    full_dataset = CelebADataset(
        root_dir=cfg['celeba_root_dir'],
        target_attribute=cfg['concept']['positive'],
        transform=clip_preprocess # This needs to be passed in from main.py
    )

    # Define a custom collate_fn for the DataLoader
    # This collate_fn handles the batching of data yielded by __getitem__
    def custom_collate_fn(batch):
        # batch is a list of tuples: (transformed_image, label, img_name)
        # where transformed_image is already a 4D tensor (1, C, H, W)
        
        images = torch.cat([item[0] for item in batch], dim=0) # Concatenate along the batch dimension
        labels = torch.tensor([item[1] for item in batch]) # Convert labels to a tensor
        img_ids = [item[2] for item in batch] # Keep image IDs as a list

        return images, labels, img_ids

    # Simple split (you might have a fixed split based on CelebA's image IDs)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg.get('num_workers', 2),
        collate_fn=custom_collate_fn # <<< ADD THIS
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg.get('num_workers', 2),
        collate_fn=custom_collate_fn # <<< ADD THIS
    )

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