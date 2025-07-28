# models/clip_wrapper.py

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image # Import PIL for type hinting/clarity

class CLIPModelWrapper:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        print(f"Loading CLIP model: {model_name} to {device}")
        
        # Load processor and model
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.model.eval() # Set model to evaluation mode

    def embed_image(self, images):
        """
        Embeds a batch of preprocessed image tensors.
        This method expects images that have already gone through preprocess_image.
        
        Args:
            images (torch.Tensor): A batch of preprocessed image tensors (pixel_values).
        Returns:
            torch.Tensor: Image embeddings (on CPU).
        """
        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            raise ValueError(
                "Input to embed_image must be a 4D preprocessed tensor (batch, channels, height, width). "
                "Did you pass images through clip.preprocess_image first?"
            )
        
        with torch.no_grad():
            # Pass the preprocessed pixel_values directly to the model
            image_features = self.model.get_image_features(pixel_values=images.to(self.device))
            return image_features.cpu() # Return to CPU as per main.py's expectation

    def embed_text(self, texts):
        """
        Embeds a list of text strings.
        Args:
            texts (list[str]): A list of text strings.
        Returns:
            torch.Tensor: Text embeddings (on CPU).
        """
        with torch.no_grad():
            # The processor handles tokenization and padding for text
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            text_features = self.model.get_text_features(**inputs)
            return text_features.cpu() # Return to CPU

    def get_latents(self, dataloader):
        """
        Collects all image embeddings (latents) and their corresponding labels
        from a DataLoader.
        
        Args:
            dataloader (torch.utils.data.DataLoader): DataLoader providing images and labels.
        
        Returns:
            tuple: (all_embeddings, all_labels, all_image_ids)
                   all_embeddings (torch.Tensor): Concatenated image embeddings.
                   all_labels (torch.Tensor): Concatenated labels.
                   all_image_ids (list): List of image IDs from the dataset.
        """
        all_embeddings = []
        all_labels = []
        all_image_ids = []
        
        # Ensure model is in evaluation mode (should already be from __init__, but good practice)
        self.model.eval() 
        
        with torch.no_grad():
            for batch_images, batch_labels, batch_img_ids in dataloader:
                # 1. Preprocess images using the dedicated preprocess_image method
                #    This returns pixel_values tensor on CPU
                preprocessed_images = self.preprocess_image(batch_images)
                
                # 2. Embed the preprocessed images
                #    embed_image expects pixel_values tensor and moves it to the correct device internally
                embeddings = self.embed_image(preprocessed_images) 
                
                # Collect embeddings and labels, ensuring they are on CPU
                all_embeddings.append(embeddings.cpu())
                all_labels.append(batch_labels.cpu())
                all_image_ids.extend(batch_img_ids) # Extend the list for image IDs

        # Concatenate all collected tensors
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return all_embeddings, all_labels, all_image_ids

    # --- ADD THIS NEW METHOD ---
    def preprocess_image(self, images):
        """
        Applies CLIP's required image preprocessing.
        
        Args:
            images: A single PIL Image, a list of PIL Images, or a batch of NumPy arrays.
                    (The CLIPProcessor is flexible here).
        Returns:
            torch.Tensor: Preprocessed image tensor(s) (pixel_values) suitable for CLIP.
                          This will be on CPU, as it's typically moved to device by the Dataloader later.
        """
        # The CLIPProcessor's __call__ method (or `preprocess` if called directly on feature_extractor)
        # handles resizing, normalization, etc. It returns a BatchEncoding, from which we get pixel_values.
        # It's important that this function returns just the pixel_values tensor.
        processed_input = self.processor(images=images, return_tensors="pt")
        return processed_input.pixel_values # This returns the tensor on CPU by default
    # --- END NEW METHOD ---