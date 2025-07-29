# baselines/splice_baseline.py

import torch
import splice
import numpy as np

def run_splice_baseline(clip_wrapper, original_embeddings, train_loader, cfg, device):
    """
    Implements the SpLiCE baseline for latent space editing,
    performing targeted semantic manipulation of w_sparse weights.

    Args:
        clip_wrapper: An instance of CLIPModelWrapper (not directly used for SpLiCE's
                      decomposition/recomposition but may be useful for mapping concepts).
        original_embeddings (torch.Tensor): The embeddings of the test set images
                                            that need to be edited.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training set (not directly used
                      by SpLiCE's manipulation API here, but could be for learning concept
                      strengths if needed in a more complex SpLiCE implementation).
        cfg (dict): Configuration dictionary, containing 'concept' (positive/negative)
                    and 'surgery_alpha' for the manipulation factor.
        device (str): The device ('cuda' or 'cpu') to perform computations on.

    Returns:
        torch.Tensor: The embeddings after applying the SpLiCE transformation
                      with targeted semantic edits.
    """
    print(f"--- Running SpLiCE Baseline for targeted semantic manipulation on device: {device} ---")

    # 1. Load SpLiCE model and vocabulary
    # Map your CLIP model name to a SpLiCE compatible backbone name for loading.
    # The new info suggests `splice.load(name="open_clip:ViT-B-32", vocabulary="laion", vocabulary_size=10000)`
    splice_model_load_name_map = {
        "openai/clip-vit-base-patch32": "open_clip:ViT-B-32",
        "openai/clip-vit-base-patch16": "open_clip:ViT-B-16",
        # Add other mappings if you use different CLIP models or DINOv2 (if SpLiCE supports it directly)
    }
    
    current_clip_model_name = cfg.get('alternative_encoder_name') or cfg['model_name']
    splice_load_name = splice_model_load_name_map.get(current_clip_model_name, None)

    if splice_load_name is None:
        print(f"WARNING: CLIP model '{current_clip_model_name}' not explicitly mapped to a known SpLiCE load name. "
              "Using 'open_clip:ViT-B-32' as default for SpLiCE initialization. Ensure compatibility for accurate results.")
        splice_load_name = "open_clip:ViT-B-32" # Fallback

    try:
        # Use vocabulary="laion" and vocabulary_size=10000 as per the new guide
        sp = splice.load(name=splice_load_name, vocabulary="laion", vocabulary_size=10000, device=device)
        sp.eval() # Set to evaluation mode
        print(f"SpLiCE model loaded: {splice_load_name} with 'laion' vocabulary (size 10000) on {device}")
    except Exception as e:
        print(f"Error loading SpLiCE model: {e}")
        print("Please ensure the 'splice' package is installed correctly (`pip install splice`) "
              "and the specified model/vocabulary combination is available within SpLiCE's capabilities.")
        print("Returning original embeddings as placeholder due to SpLiCE initialization failure.")
        return original_embeddings.to(device)

    # Get the vocabulary for concept indexing
    try:
        vocab = splice.get_vocabulary("laion")
        print(f"SpLiCE vocabulary loaded with {len(vocab)} concepts.")
    except Exception as e:
        print(f"Error loading SpLiCE vocabulary: {e}")
        print("Returning original embeddings as placeholder due to vocabulary failure.")
        return original_embeddings.to(device)

    # Identify target concept indices
    positive_concept = cfg['concept']['positive'] # e.g., "Smiling"
    negative_concept = cfg['concept']['negative'] # e.g., "Not_Smiling"

    pos_idx = -1
    neg_idx = -1

    if positive_concept in vocab:
        pos_idx = vocab.index(positive_concept)
        print(f"Found positive concept '{positive_concept}' at index {pos_idx}")
    else:
        print(f"WARNING: Positive concept '{positive_concept}' not found in SpLiCE vocabulary.")

    if negative_concept in vocab:
        neg_idx = vocab.index(negative_concept)
        print(f"Found negative concept '{negative_concept}' at index {neg_idx}")
    else:
        print(f"WARNING: Negative concept '{negative_concept}' not found in SpLiCE vocabulary.")


    # Move original_embeddings to the correct device for decomposition
    original_embeddings_on_device = original_embeddings.to(device)

    # 2. Decompose original embeddings to sparse weights
    print("Decomposing original embeddings into sparse concept weights...")
    try:
        # The previous documentation used sp.decompose(z) for a batch of embeddings
        w_sparse_orig, _, _ = sp.decompose(original_embeddings_on_device, return_l0=True, return_cos=True)
        print(f"Decomposition complete. w_sparse_orig shape: {w_sparse_orig.shape}")
    except Exception as e:
        print(f"Error during SpLiCE decomposition: {e}")
        print("Returning original embeddings as placeholder due to decomposition failure.")
        return original_embeddings.to(device)

    # 3. Manipulate target concept(s) in w_sparse
    print("Manipulating target concept weights in sparse space...")
    w_new = w_sparse_orig.clone()
    
    # Use surgery_alpha from cfg for amplification/suppression
    gamma = cfg.get('surgery_alpha', 1.0) # Default to 1.0 if not found

    if pos_idx != -1:
        # Amplify positive concept: w_new[:, target_idx] = γ * w_sparse[:, target_idx]
        w_new[:, pos_idx] = gamma * w_sparse_orig[:, pos_idx]
        print(f"Amplify '{positive_concept}' concept weight by factor {gamma}.")

    if neg_idx != -1:
        # Suppress negative concept: w_new[:, target_idx] = 0
        w_new[:, neg_idx] = 0
        print(f"Suppress '{negative_concept}' concept weight (set to 0).")
    
    # Ensure w_new remains non-negative as w_sparse should be
    w_new = torch.relu(w_new)


    # 4. Recompose to CLIP-like embedding
    print("Recomposing edited embeddings from manipulated sparse weights...")
    try:
        splice_edited_embeddings = sp.recompose(w_new)
        print("Recomposition complete.")
    except Exception as e:
        print(f"Error during SpLiCE recomposition: {e}")
        print("Returning original embeddings as placeholder due to recomposition failure.")
        return original_embeddings.to(device)

    # Ensure output embeddings are on the correct device
    print("SpLiCE baseline execution finished.")
    return splice_edited_embeddings.to(device)