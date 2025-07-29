# models/encoder_wrapper.py

import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image # For type hinting/clarity

class EncoderWrapper:
    def __init__(self, model_name="openai/clip-vit-base-patch32", device="cpu"):
        self.device = device
        self.model_name = model_name
        print(f"Loading encoder: {model_name} to {device}")
        
        # Load processor and model using Auto classes for generality
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval() # Set model to evaluation mode

        # Determine model type for specific embedding extraction logic
        self.is_clip_model = "clip" in model_name.lower()
        print(f"Detected model type: {'CLIP' if self.is_clip_model else 'Generic/Vision-only'}")

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
                "Did you pass images through preprocess_image first?"
            )
        
        with torch.no_grad():
            inputs = {'pixel_values': images.to(self.device)}
            outputs = self.model(**inputs)

            if self.is_clip_model:
                # CLIP models typically have a get_image_features method or return image_embeds
                image_features = outputs.image_embeds if hasattr(outputs, 'image_embeds') else outputs.pooler_output
            elif hasattr(outputs, 'last_hidden_state'):
                # DINOv2 and similar vision transformers return last_hidden_state
                # Often, pooling is needed (e.g., mean pooling across tokens, or using the CLS token)
                # For simplicity, let's mean pool tokens (excluding CLS if present at index 0)
                # Assuming outputs.last_hidden_state is (batch_size, num_tokens, hidden_dim)
                # This might need refinement based on specific model architecture.
                image_features = outputs.last_hidden_state.mean(dim=1)
            else:
                raise NotImplementedError(f"Embedding extraction not implemented for model type: {self.model_name}. "
                                          "Please check model output structure.")
            
            return image_features.cpu() # Return to CPU

    def embed_text(self, texts):
        """
        Embeds a list of text strings. Only available for CLIP models.
        Args:
            texts (list[str]): A list of text strings.
        Returns:
            torch.Tensor: Text embeddings (on CPU).
        """
        if not self.is_clip_model:
            raise NotImplementedError(f"Text embedding is not available for model: {self.model_name}")
            
        with torch.no_grad():
            inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            outputs = self.model(**inputs)
            text_features = outputs.text_embeds if hasattr(outputs, 'text_embeds') else outputs.pooler_output
            return text_features.cpu() # Return to CPU

    def preprocess_image(self, images):
        """
        Applies the encoder's required image preprocessing.
        
        Args:
            images: A single PIL Image, a list of PIL Images, or a batch of NumPy arrays.
        Returns:
            torch.Tensor: Preprocessed image tensor(s) (pixel_values) suitable for the encoder.
                          This will be on CPU by default.
        """
        # AutoProcessor's __call__ method handles resizing, normalization, etc.
        # It returns a BatchEncoding, from which we get pixel_values.
        processed_input = self.processor(images=images, return_tensors="pt")
        return processed_input.pixel_values

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
        
        self.model.eval() 
        
        with torch.no_grad():
            for batch_images, batch_labels, batch_img_ids in dataloader:
                # batch_images from the DataLoader are ALREADY preprocessed tensors
                # (pixel_values) because preprocess_image was applied as the dataset's transform.
                embeddings = self.embed_image(batch_images) 
                
                all_embeddings.append(embeddings.cpu())
                all_labels.append(batch_labels.cpu())
                all_image_ids.extend(batch_img_ids)

        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return all_embeddings, all_labels, all_image_ids