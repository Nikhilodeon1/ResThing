# data/imagenet_loader.py
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder # ImageFolder is good for ImageNet-like structures
from data.transforms import get_image_transforms # Assuming transforms.py is in the same 'data' directory

# This loader is designed for a subset of ImageNet or for specific classes.
# For full ImageNet, it's usually recommended to use torchvision.datasets.ImageNet directly
# if you have the full dataset organized in the standard way.

class ImageNetSubsetDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_classes=None):
        """
        Args:
            root_dir (string): Root directory of the ImageNet (or subset) dataset.
                               Assumes structure like root_dir/train/class_name/image.jpg
            transform (callable, optional): Optional transform to be applied on an image.
            target_classes (list, optional): List of class names (e.g., ['n02119789', 'n02089973'] or human-readable names).
                                            If None, loads all classes found.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # ImageFolder expects root/class_name/image.jpg structure
        self.image_folder_dataset = ImageFolder(root=self.root_dir, transform=self.transform)
        
        # Filter by target_classes if provided
        self.samples = []
        if target_classes:
            # Create a mapping from class name to its index in ImageFolder's classes
            class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.image_folder_dataset.classes)}
            
            # Identify indices of target classes
            target_indices = [class_to_idx[tc] for tc in target_classes if tc in class_to_idx]
            if not target_indices:
                raise ValueError(f"None of the specified target classes {target_classes} found in dataset.")

            # Filter samples
            for path, class_idx in self.image_folder_dataset.samples:
                if class_idx in target_indices:
                    self.samples.append((path, class_idx))
            
            if not self.samples:
                raise ValueError("No images found after filtering by target classes. Check class names or dataset path.")
        else:
            self.samples = self.image_folder_dataset.samples
            print("Warning: No target classes provided for ImageNet. Loading all classes found in the root directory.")
        
        self.classes = self.image_folder_dataset.classes # All classes found by ImageFolder
        self.class_to_idx = self.image_folder_dataset.class_to_idx


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Return image_name (basename of the path) for consistency with other loaders
        image_name = os.path.basename(path)
        return image, label, image_name


def get_imagenet_dataloaders(cfg):
    """
    Returns train and test dataloaders for ImageNet (or a subset).
    """
    transform = get_image_transforms()
    
    # ImageNet typically uses specific train/val subdirectories
    train_root = os.path.join(cfg['imagenet_root_dir'], 'train')
    val_root = os.path.join(cfg['imagenet_root_dir'], 'val')

    # 'concept' in config.yaml for ImageNet might specify a list of classes to load
    target_classes = cfg['concept'].get('imagenet_classes') if 'imagenet_classes' in cfg['concept'] else None

    # For ImageNet, it's common to use the 'val' split for evaluation.
    # We'll create a full dataset from 'train' for direction finding and a separate 'val' dataset for testing.
    # If target_classes are specified, they'll apply to both.

    train_dataset = ImageNetSubsetDataset(
        root_dir=train_root,
        transform=transform,
        target_classes=target_classes
    )
    test_dataset = ImageNetSubsetDataset( # Use a separate test split if available, otherwise split train
        root_dir=val_root, # Assuming 'val' is the test set
        transform=transform,
        target_classes=target_classes
    )

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
    
    print(f"ImageNet: Loaded Train: {len(train_dataset)}, Test: {len(test_dataset)}")
    return train_loader, test_loader

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    from utils.config_loader import load_config
    try:
        # Example config for ImageNet:
        # dataset: ImageNet
        # concept:
        #   imagenet_classes: ["n02120079", "n02120505"] # Example WordNet IDs for cat breeds
        # imagenet_root_dir: "/path/to/ImageNet"
        cfg = load_config("../config.yaml") 
        cfg['dataset'] = 'ImageNet' # Temporarily set for testing
        # Ensure your config.yaml has 'imagenet_classes' under 'concept' for filtering or remove to load all
        cfg['concept']['imagenet_classes'] = ["n02088364", "n02089025"] # Example: "beagle", "dalmatian"
        cfg['imagenet_root_dir'] = "/content/drive/MyDrive/Paper2/ImageNet" # Dummy path, replace with your actual path

        if cfg['dataset'] == 'ImageNet':
            print(f"Attempting to load ImageNet from: {cfg['imagenet_root_dir']}")
            train_loader, test_loader = get_imagenet_dataloaders(cfg)
            print(f"Train loader has {len(train_loader)} batches.")
            print(f"Test loader has {len(test_loader)} batches.")
            
            for i, (images, labels, img_names) in enumerate(train_loader):
                print(f"Batch {i+1}: Images shape: {images.shape}, Labels shape: {labels.shape}")
                print(f"Sample labels: {labels[:5]}")
                print(f"Sample image names: {img_names[:5]}")
                break
        else:
            print("ImageNet loader test skipped as 'dataset' in config is not 'ImageNet'.")
    except Exception as e:
        print(f"Error during ImageNet loader test: {e}")