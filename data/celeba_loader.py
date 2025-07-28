# data/celeba_loader.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
        self.img_dir = os.path.join(root_dir, 'images') # Assuming images are in a subfolder 'images'
        self.attr_file = os.path.join(root_dir, 'list_attr_celeba.csv')

        # --- DIAGNOSTIC ADDITION START ---
        print(f"DEBUG (CelebADataset): Attempting to load attribute file from: {self.attr_file}")
        if not os.path.exists(self.attr_file):
            raise FileNotFoundError(f"DEBUG (CelebADataset): Attribute file NOT FOUND at: {self.attr_file}. Please check your path.")
        # --- DIAGNOSTIC ADDITION END ---

        self.attr_df = pd.read_csv(self.attr_file)
        self.attr_df = self.attr_df.set_index('image_id') # Assuming 'image_id' is the column name for the index

        # --- DIAGNOSTIC ADDITION START ---
        print(f"DEBUG (CelebADataset): Columns loaded into attr_df: {self.attr_df.columns.tolist()}")
        # --- DIAGNOSTIC ADDITION END ---

        self.target_attribute = target_attribute

        # This is the line (or very near it) that caused the KeyError
        # Make sure the target_attribute is in the columns list *here*
        if self.target_attribute not in self.attr_df.columns:
            raise KeyError(f"DEBUG (CelebADataset): Configured target attribute '{self.target_attribute}' not found in loaded CSV columns. Available: {self.attr_df.columns.tolist()}")


        # Filter for positive and negative samples based on the target attribute
        # Assuming original labels are 1 and -1 as per CelebA standard
        self.image_ids = self.attr_df.index.tolist()
        self.labels = self.attr_df[self.target_attribute].loc[self.image_ids].values

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_name = self.image_ids[idx]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label, img_name

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

    # Simple split (you might have a fixed split based on CelebA's image IDs)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=cfg.get('num_workers', 2) # Use .get() for optional num_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=cfg.get('num_workers', 2)
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