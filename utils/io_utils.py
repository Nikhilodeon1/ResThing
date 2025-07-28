# utils/io_utils.py

import os
import json
import pickle
import torch

def save_json(data: dict, file_path: str):
    """Saves a dictionary to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Data saved to {file_path}")

def load_json(file_path: str) -> dict:
    """Loads a dictionary from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    print(f"Data loaded from {file_path}")
    return data

def save_pickle(data, file_path: str):
    """Saves any Python object to a pickle file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {file_path}")

def load_pickle(file_path: str):
    """Loads a Python object from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    print(f"Data loaded from {file_path}")
    return data

def save_torch_tensor(tensor: torch.Tensor, file_path: str):
    """Saves a PyTorch tensor to a file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    torch.save(tensor, file_path)
    print(f"Tensor saved to {file_path}")

def load_torch_tensor(file_path: str) -> torch.Tensor:
    """Loads a PyTorch tensor from a file."""
    tensor = torch.load(file_path)
    print(f"Tensor loaded from {file_path}")
    return tensor