# models/clip_wrapper.py
import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

class CLIPModelWrapper:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None):
        """
        Initializes the CLIP model and processor.
        Args:
            model_name (str): The name of the CLIP model to load.
            device (str, optional): The device to run the model on (e.g., 'cuda', 'cpu').
                                    If None, automatically detects.
        """
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP model '{model_name}' on device: {self.device}")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval() # Set model to evaluation mode

    def embed_image(self, images):
        """
        Embeds a batch of images into CLIP's latent space.
        Args:
            images (torch.Tensor or PIL.Image.Image or list of PIL.Image.Image):
                Image(s) to embed. If a Tensor, assumed to be pre-processed.
        Returns:
            torch.Tensor: Image embeddings.
        """
        with torch.no_grad():
            if isinstance(images, torch.Tensor):
                # If images are already a tensor (e.g., from DataLoader), assume pre-processed
                inputs = {'pixel_values': images.to(self.device)}
            else:
                # Otherwise, process raw images (PIL.Image or list of PIL.Image)
                inputs = self.processor(images=images, return_tensors="pt").to(self.device)
            
            image_features = self.model.get_image_features(**inputs)
            return image_features.cpu() # Return to CPU for consistency

    def embed_text(self, prompts):
        """
        Embeds a batch of text prompts into CLIP's latent space.
        Args:
            prompts (str or list of str): Text prompt(s) to embed.
        Returns:
            torch.Tensor: Text embeddings.
        """
        with torch.no_grad():
            inputs = self.processor(text=prompts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            return text_features.cpu() # Return to CPU for consistency

    def get_latents(self, dataloader):
        """
        Iterates through a dataloader and computes CLIP embeddings for all images.
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing images.
        Returns:
            torch.Tensor: Concatenated image embeddings.
            list: Concatenated labels.
            list: Concatenated image IDs/names.
        """
        all_embeddings = []
        all_labels = []
        all_img_ids = []

        print(f"Generating embeddings for {len(dataloader.dataset)} images...")
        for batch_images, batch_labels, batch_img_ids in tqdm(dataloader, desc="Embedding images"):
            embeddings = self.embed_image(batch_images)
            all_embeddings.append(embeddings)
            all_labels.extend(batch_labels.tolist())
            all_img_ids.extend(batch_img_ids)
        
        return torch.cat(all_embeddings, dim=0), torch.tensor(all_labels), all_img_ids

# Example of how to use this outside the project main flow for testing
if __name__ == '__main__':
    from PIL import Image
    import requests
    import numpy as np

    # Dummy image for testing
    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    clip_wrapper = CLIPModelWrapper()

    # Test image embedding
    print("Testing image embedding...")
    image_embedding = clip_wrapper.embed_image(image)
    print(f"Image embedding shape: {image_embedding.shape}") # Expected: torch.Size([1, 512])

    # Test text embedding
    print("\nTesting text embedding...")
    text_embedding = clip_wrapper.embed_text(["a photo of a cat", "a photo of a dog"])
    print(f"Text embedding shape: {text_embedding.shape}") # Expected: torch.Size([2, 512])

    # Test get_latents with a dummy dataloader
    print("\nTesting get_latents with a dummy dataloader (requires a mock DataLoader)...")
    class MockDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples=10, img_size=224):
            self.num_samples = num_samples
            self.img_size = img_size
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create a dummy image (random tensor)
            dummy_image = torch.randn(3, self.img_size, self.img_size) 
            dummy_label = idx % 2 # Simple binary label
            dummy_id = f"img_{idx:04d}"
            return dummy_image, dummy_label, dummy_id

    mock_dataloader = DataLoader(MockDataset(num_samples=5), batch_size=2, shuffle=False)
    
    # Temporarily set model to CPU for this mock test if on GPU and image tensors are directly used
    original_device = clip_wrapper.device
    clip_wrapper.model.to('cpu') 

    all_embs, all_lbls, all_ids = clip_wrapper.get_latents(mock_dataloader)
    print(f"All embeddings shape: {all_embs.shape}")
    print(f"All labels shape: {all_lbls.shape}")
    print(f"Number of image IDs: {len(all_ids)}")
    print(f"Sample embeddings (first 2): \n{all_embs[:2]}")
    print(f"Sample labels: {all_lbls}")
    print(f"Sample IDs: {all_ids}")

    # Restore model device
    clip_wrapper.model.to(original_device)
    print(f"\nCLIPModelWrapper testing complete.")