# data/cub_loader.py
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data.transforms import get_image_transforms # Assuming transforms.py is in the same 'data' directory

class CUBDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_classes=None):
        """
        Args:
            root_dir (string): Directory with CUB-200-2011 dataset.
                               Expected structure: root_dir/images/, root_dir/image_class_labels.txt,
                               root_dir/classes.txt
            transform (callable, optional): Optional transform to be applied on an image.
            target_classes (list, optional): List of class names to filter the dataset (e.g., ['Indigo_Bunting', 'Black_Tern']).
                                             If None, loads all classes.
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.images_file = os.path.join(root_dir, 'images.txt')
        self.class_labels_file = os.path.join(root_dir, 'image_class_labels.txt')
        self.classes_file = os.path.join(root_dir, 'classes.txt')
        self.transform = transform

        if not all(os.path.exists(f) for f in [self.images_file, self.class_labels_file, self.classes_file]):
            raise FileNotFoundError(f"One or more required CUB files not found in {root_dir}. "
                                    "Please ensure 'images.txt', 'image_class_labels.txt', and 'classes.txt' exist.")

        self.images_df = pd.read_csv(self.images_file, sep=' ', names=['image_id', 'image_name'])
        self.labels_df = pd.read_csv(self.class_labels_file, sep=' ', names=['image_id', 'class_id'])
        self.classes_df = pd.read_csv(self.classes_file, sep=' ', names=['class_id', 'class_name'])

        # Merge dataframes to get image names and class names
        self.data = pd.merge(self.images_df, self.labels_df, on='image_id')
        self.data = pd.merge(self.data, self.classes_df, on='class_id')

        self.data_filtered = self.data.copy()
        if target_classes:
            self.data_filtered = self.data_filtered[self.data_filtered['class_name'].isin(target_classes)]
            if self.data_filtered.empty:
                raise ValueError(f"No images found for target classes: {target_classes}. Check class names.")
            # Map class names to integer labels for training
            unique_classes = sorted(self.data_filtered['class_name'].unique())
            self.class_to_idx = {name: idx for idx, name in enumerate(unique_classes)}
            self.data_filtered['label'] = self.data_filtered['class_name'].map(self.class_to_idx)
        else:
            print("Warning: No target classes provided for filtering CUB. Loading all classes with original class_id as label.")
            self.data_filtered['label'] = self.data_filtered['class_id'] - 1 # Adjust to 0-indexed if needed

        if self.data_filtered.empty:
            raise ValueError("No data found after filtering. Check target classes or dataset integrity.")

    def __len__(self):
        return len(self.data_filtered)

    def __getitem__(self, idx):
        img_info = self.data_filtered.iloc[idx]
        img_name = os.path.join(self.img_dir, img_info['image_name'])
        image = Image.open(img_name).convert('RGB')
        label = img_info['label']

        if self.transform:
            image = self.transform(image)

        return image, label, img_info['image_name']


def get_cub_dataloaders(cfg):
    """
    Returns train and test dataloaders for CUB-200.
    """
    transform = get_image_transforms()
    
    # CUB doesn't have explicit positive/negative concepts like CelebA attributes.
    # For latent surgery, we might define concepts based on specific bird species
    # or visual attributes (if available in a separate annotation).
    # For now, we'll load specific classes if 'concept' is a list in config.
    target_classes = cfg['concept'].get('cub_classes') if 'cub_classes' in cfg['concept'] else None

    full_dataset = CUBDataset(
        root_dir=cfg['cub_root_dir'],
        transform=transform,
        target_classes=target_classes
    )

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    from torch.utils.data import random_split
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size'],
        shuffle=True,
        num_workers=os.cpu_count() // 2 or 1
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg['batch_size'],
        shuffle=False,
        num_workers=os.cpu_count() // 2 or 1
    )
    
    print(f"CUB-200: Loaded {len(full_dataset)} images. Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    return train_loader, test_loader

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    from utils.config_loader import load_config
    try:
        # Example config for CUB:
        # dataset: CUB-200
        # concept:
        #   cub_classes: ["Indigo_Bunting", "Black_Tern"]
        # cub_root_dir: "/path/to/CUB_200_2011"
        cfg = load_config("../config.yaml") 
        cfg['dataset'] = 'CUB-200' # Temporarily set for testing
        # Ensure your config.yaml has a 'cub_classes' key under 'concept' for filtering or remove it to load all
        cfg['concept']['cub_classes'] = ["Indigo_Bunting", "Black_Tern", "Tree_Sparrow"] 
        # Make sure cub_root_dir points to your CUB dataset
        cfg['cub_root_dir'] = "/content/drive/MyDrive/Paper2/CUB_200_2011" # Dummy path, replace with your actual path

        if cfg['dataset'] == 'CUB-200':
            print(f"Attempting to load CUB-200 from: {cfg['cub_root_dir']}")
            train_loader, test_loader = get_cub_dataloaders(cfg)
            print(f"Train loader has {len(train_loader)} batches.")
            print(f"Test loader has {len(test_loader)} batches.")
            
            for i, (images, labels, img_names) in enumerate(train_loader):
                print(f"Batch {i+1}: Images shape: {images.shape}, Labels shape: {labels.shape}")
                print(f"Sample labels: {labels[:5]}")
                print(f"Sample image names: {img_names[:5]}")
                break
        else:
            print("CUB-200 loader test skipped as 'dataset' in config is not 'CUB-200'.")
    except Exception as e:
        print(f"Error during CUB loader test: {e}")