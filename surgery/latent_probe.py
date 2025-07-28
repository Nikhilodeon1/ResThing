# surgery/latent_probe.py

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

class LatentProbe:
    def __init__(self, random_state=42):
        """
        Initializes the LatentProbe with a Logistic Regression model.
        """
        self.model = LogisticRegression(random_state=random_state, max_iter=1000)
        self.probe_name = "Logistic Regression Probe"

    def train(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Trains the probe on the provided embeddings and labels.

        Args:
            embeddings (torch.Tensor): Image embeddings (features).
            labels (torch.Tensor): Corresponding attribute labels (e.g., 0 or 1, or -1 and 1).
        """
        if embeddings.is_cuda:
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings.numpy()

        if labels.is_cuda:
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels.numpy()

        print(f"Training {self.probe_name}...")
        self.model.fit(embeddings_np, labels_np)
        print(f"{self.probe_name} training complete.")

    def evaluate(self, embeddings: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Evaluates the probe on the provided embeddings and labels.

        Args:
            embeddings (torch.Tensor): Image embeddings (features).
            labels (torch.Tensor): Corresponding attribute labels.

        Returns:
            dict: Dictionary containing evaluation metrics (accuracy, F1-score).
        """
        if embeddings.is_cuda:
            embeddings_np = embeddings.cpu().numpy()
        else:
            embeddings_np = embeddings.numpy()

        if labels.is_cuda:
            labels_np = labels.cpu().numpy()
        else:
            labels_np = labels.numpy()

        predictions = self.model.predict(embeddings_np)
        accuracy = accuracy_score(labels_np, predictions)
        f1 = f1_score(labels_np, predictions, average='binary') # Use 'binary' for 2-class problems

        print(f"Probe Evaluation - Accuracy: {accuracy:.4f}, F1-Score: {f1:.4f}")
        return {"probe_accuracy": accuracy, "probe_f1_score": f1}

# Example usage (within main.py context):
# from surgery.latent_probe import LatentProbe
# from models.clip_wrapper import CLIPModelWrapper
# from data.celeba_loader import CelebADataset
#
# # Assuming you have original_embeddings, edited_embeddings, and binary_labels
# probe = LatentProbe()
#
# # Train probe on original embeddings
# probe.train(original_embeddings, binary_labels)
# original_probe_results = probe.evaluate(original_embeddings, binary_labels)
#
# # Evaluate probe on edited embeddings
# edited_probe_results = probe.evaluate(edited_embeddings, binary_labels)
# print(f"Probe results on original embeddings: {original_probe_results}")
# print(f"Probe results on edited embeddings: {edited_probe_results}")