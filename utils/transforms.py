# data/transforms.py
from torchvision import transforms

def get_image_transforms(target_size=224):
    """
    Defines image transformations for consistent preprocessing across datasets.
    CLIP models typically expect 224x224 input and normalized images.
    """
    return transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])