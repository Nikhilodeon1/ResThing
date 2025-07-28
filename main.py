import torch
import os
import pandas as pd # For logging results
import numpy as np  # Added for numpy operations in evaluation

# Existing imports from your provided code
from utils.config_loader import load_config
from data.celeba_loader import get_celeba_dataloaders
from data.cub_loader import get_cub_dataloaders
from data.imagenet_loader import get_imagenet_dataloaders # Will load all images for ImageNet, no labels for concepts though
from models.clip_wrapper import CLIPModelWrapper
from surgery.direction_finder import compute_direction # Assuming this computes the direction vector
from surgery.edit_embedding import apply_surgery # Assuming this applies surgery and returns edited/original embeddings
from baselines.prompt_tuning import prompt_tuning_baseline
from baselines.random_edit import random_edit_baseline
from evaluation.metrics import calculate_accuracy, calculate_cosine_similarity, calculate_mean_confidence
from evaluation.visualize import create_tsne_plot, plot_cosine_similarity_hist

# NEW IMPORTS for optional files
from surgery.latent_probe import LatentProbe
from baselines.linear_probe import run_linear_probe_baseline
from evaluation.retrieval import evaluate_retrieval
from utils.io_utils import save_json, load_json, save_pickle, load_pickle, save_torch_tensor, load_torch_tensor


def main():
    # Load configuration
    cfg = load_config("config.yaml")
    print("Configuration loaded:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create output directory if it doesn't exist
    os.makedirs(cfg["output_dir"], exist_ok=True)
    
    # Phase 1: Data Preparation & Loading
    train_loader, test_loader = None, None
    if cfg['dataset'] == 'CelebA':
        train_loader, test_loader = get_celeba_dataloaders(cfg)
    elif cfg['dataset'] == 'CUB-200':
        train_loader, test_loader = get_cub_dataloaders(cfg)
    elif cfg['dataset'] == 'ImageNet':
        train_loader, test_loader = get_imagenet_dataloaders(cfg)
    else:
        raise ValueError(f"Unsupported dataset specified in config: {cfg['dataset']}")

    print(f"\nPhase 1: Data Preparation & Loading Complete.")
    print(f"Train DataLoader batches: {len(train_loader)}")
    print(f"Test DataLoader batches: {len(test_loader)}")

    # Phase 2: Model Setup & Latent Surgery Implementation
    print("\n--- Phase 2: Model Setup & Latent Surgery Implementation ---")

    clip = CLIPModelWrapper(cfg["model_name"], device=device) # Pass device to CLIPModelWrapper

    # Compute semantic direction vector using the training data
    direction = compute_direction(clip, train_loader, cfg) # Ensure compute_direction handles device correctly
    print(f"Computed semantic direction vector shape: {direction.shape}")

    # Apply latent surgery to test embeddings
    # We will get original and edited embeddings for the test set
    edited_embeddings_surgery, original_embeddings_test, test_labels, test_img_ids = apply_surgery(
        clip, test_loader, direction, cfg["surgery_alpha"]
    )
    print(f"Original test embeddings shape: {original_embeddings_test.shape}")
    print(f"Edited test embeddings (Latent Surgery) shape: {edited_embeddings_surgery.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    print(f"Number of test image IDs: {len(test_img_ids)}")

    print("\nPhase 2: Model Setup & Latent Surgery Implementation Complete.")

    # Phase 3: Baseline Comparison & Evaluation
    print("\n--- Phase 3: Baseline Comparison & Evaluation ---")

    results = [] # To store all evaluation results

    # --- Prepare labels for probes/retrieval ---
    # Convert labels to 0/1 for sklearn (assuming test_labels are -1/1 as commonly done for attributes)
    # If your test_labels are already 0/1, this conversion is safe but redundant.
    # Ensure test_labels is on CPU and then converted to numpy if needed
    labels_for_probe_retrieval = test_labels.cpu()
    if labels_for_probe_retrieval.min() == -1 and labels_for_probe_retrieval.max() == 1:
        labels_for_probe_retrieval = (labels_for_probe_retrieval + 1) // 2 # Maps -1 to 0, 1 to 1


    # --- Evaluation for Latent Surgery ---
    print("\nEvaluating Latent Surgery:")
    
    # Re-use prompt text embeddings as "classifiers"
    pos_text_prompt = f"A photo of a {cfg['concept']['positive'].lower()}"
    neg_text_prompt = f"A photo of a {cfg['concept']['negative'].lower()}"
    text_embeddings = clip.embed_text([pos_text_prompt, neg_text_prompt]).to(device) # Ensure text embeddings are on correct device
    
    # Evaluate original embeddings w.r.t. concept prompts
    original_preds, original_confidences = prompt_tuning_baseline(
        clip, original_embeddings_test, test_labels, cfg['concept']['positive'], cfg['concept']['negative']
    ) 
    original_accuracy = calculate_accuracy(original_preds, test_labels)
    original_mean_pos_conf, original_mean_neg_conf = calculate_mean_confidence(original_confidences, test_labels)
    results.append({
        'Method': 'Original Embeddings (Prompt Baseline)',
        'Accuracy': original_accuracy,
        'Mean_Positive_Confidence': original_mean_pos_conf,
        'Mean_Negative_Confidence': original_mean_neg_conf,
        'Cosine_Similarity_to_Original': 1.0 # By definition
    })

    # Evaluate edited embeddings w.r.t. concept prompts
    edited_preds_surgery, edited_confidences_surgery = prompt_tuning_baseline(
        clip, edited_embeddings_surgery, test_labels, cfg['concept']['positive'], cfg['concept']['negative']
    )
    surgery_accuracy = calculate_accuracy(edited_preds_surgery, test_labels)
    surgery_mean_pos_conf, surgery_mean_neg_conf = calculate_mean_confidence(edited_confidences_surgery, test_labels)
    
    # Cosine similarity between original and edited embeddings (for change magnitude)
    cos_sim_orig_to_surgery = calculate_cosine_similarity(original_embeddings_test, edited_embeddings_surgery).mean().item()

    results.append({
        'Method': 'Latent Surgery',
        'Accuracy': surgery_accuracy,
        'Mean_Positive_Confidence': surgery_mean_pos_conf,
        'Mean_Negative_Confidence': surgery_mean_neg_conf,
        'Cosine_Similarity_to_Original': cos_sim_orig_to_surgery
    })

    # --- Baselines ---
    # Random Edit Baseline
    if cfg.get('run_random_edit_baseline', True):
        print("\nEvaluating Random Edit Baseline:")
        edited_embeddings_random = random_edit_baseline(original_embeddings_test, alpha=cfg["surgery_alpha"]) # Use same alpha
        
        random_preds, random_confidences = prompt_tuning_baseline(
            clip, edited_embeddings_random, test_labels, cfg['concept']['positive'], cfg['concept']['negative']
        )
        random_accuracy = calculate_accuracy(random_preds, test_labels)
        random_mean_pos_conf, random_mean_neg_conf = calculate_mean_confidence(random_confidences, test_labels)
        cos_sim_orig_to_random = calculate_cosine_similarity(original_embeddings_test, edited_embeddings_random).mean().item()

        results.append({
            'Method': 'Random Edit Baseline',
            'Accuracy': random_accuracy,
            'Mean_Positive_Confidence': random_mean_pos_conf,
            'Mean_Negative_Confidence': random_mean_neg_conf,
            'Cosine_Similarity_to_Original': cos_sim_orig_to_random
        })

    # NEW: Linear Probe Baseline
    if cfg.get('run_linear_probe_baseline', False):
        print("\n--- Running Linear Probe Baseline ---")
        linear_probe_results = run_linear_probe_baseline(original_embeddings_test, labels_for_probe_retrieval)
        results.append({
            'Method': 'Linear Probe Baseline',
            'Probe_Accuracy': linear_probe_results.get('probe_accuracy'),
            'Probe_F1_Score': linear_probe_results.get('probe_f1_score'),
            'Notes': 'Probe on Original Embeddings'
        })
        save_json(linear_probe_results, os.path.join(cfg['output_dir'], "linear_probe_baseline_results.json"))


    # --- Optional Evaluations ---
    # NEW: Latent Probe on Latent Surgery Embeddings
    if cfg.get('run_latent_probe', False):
        print("\n--- Running Latent Probe on Edited Embeddings ---")
        latent_probe = LatentProbe()
        # Train probe on ORIGINAL embeddings, then evaluate on EDITED embeddings
        # This helps determine if the edit shifted the attribute classification.
        latent_probe.train(original_embeddings_test, labels_for_probe_retrieval)
        edited_probe_results = latent_probe.evaluate(edited_embeddings_surgery, labels_for_probe_retrieval)
        
        results.append({
            'Method': 'Latent Probe (Post-Surgery)',
            'Probe_Accuracy': edited_probe_results.get('probe_accuracy'),
            'Probe_F1_Score': edited_probe_results.get('probe_f1_score'),
            'Notes': 'Probe trained on original, evaluated on edited embeddings'
        })
        save_json(edited_probe_results, os.path.join(cfg['output_dir'], "latent_probe_edited_results.json"))


    # NEW: Retrieval Evaluation
    if cfg.get('run_retrieval_eval', False):
        print("\n--- Running Retrieval Evaluation ---")
        # For retrieval, we can use the test set as both query and gallery
        # A more sophisticated setup might involve separate query/gallery sets.
        # Ensure top_k is set in config.yaml
        top_k = cfg.get('retrieval_top_k', 5)

        print(f"Evaluating retrieval for Original Embeddings (Top-K={top_k}):")
        retrieval_results_original = evaluate_retrieval(
            query_embeddings=original_embeddings_test,
            gallery_embeddings=original_embeddings_test, # Using same set as gallery
            query_labels=labels_for_probe_retrieval,
            gallery_labels=labels_for_probe_retrieval,
            top_k=top_k
        )
        results.append({
            'Method': 'Retrieval (Original Embeddings)',
            f'Retrieval_Accuracy_@{top_k}': retrieval_results_original.get(f'retrieval_accuracy_at_{top_k}'),
            'Notes': 'Querying original embeddings against original embeddings'
        })
        save_json(retrieval_results_original, os.path.join(cfg['output_dir'], "retrieval_original_results.json"))

        print(f"\nEvaluating retrieval for Edited Embeddings (Top-K={top_k}):")
        retrieval_results_edited = evaluate_retrieval(
            query_embeddings=edited_embeddings_surgery, # Querying with edited embeddings
            gallery_embeddings=original_embeddings_test, # Querying against original gallery
            query_labels=labels_for_probe_retrieval,
            gallery_labels=labels_for_probe_retrieval,
            top_k=top_k
        )
        results.append({
            'Method': 'Retrieval (Edited Embeddings)',
            f'Retrieval_Accuracy_@{top_k}': retrieval_results_edited.get(f'retrieval_accuracy_at_{top_k}'),
            'Notes': 'Querying edited embeddings against original embeddings'
        })
        save_json(retrieval_results_edited, os.path.join(cfg['output_dir'], "retrieval_edited_results.json"))


    # --- Visualizations ---
    if cfg.get('save_tsne', True):
        # t-SNE plot for Original Embeddings
        create_tsne_plot(
            original_embeddings_test.cpu().numpy(), # Ensure numpy array on CPU
            test_labels.cpu().numpy(),             # Ensure numpy array on CPU
            title=f"t-SNE of Original Embeddings ({cfg['dataset']} - {cfg['concept']['positive']}/{cfg['concept']['negative']})",
            save_path=os.path.join(cfg['output_dir'], 'tsne_original.png')
        )

        # t-SNE plot for Latent Surgery Edited Embeddings
        create_tsne_plot(
            edited_embeddings_surgery.cpu().numpy(), # Ensure numpy array on CPU
            test_labels.cpu().numpy(),               # Ensure numpy array on CPU
            title=f"t-SNE of Latent Surgery Embeddings (Alpha={cfg['surgery_alpha']})",
            save_path=os.path.join(cfg['output_dir'], 'tsne_latent_surgery.png')
        )

        if cfg.get('run_random_edit_baseline', True):
            # t-SNE plot for Random Edit Baseline Embeddings
            create_tsne_plot(
                edited_embeddings_random.cpu().numpy(), # Ensure numpy array on CPU
                test_labels.cpu().numpy(),              # Ensure numpy array on CPU
                title=f"t-SNE of Random Edit Embeddings (Alpha={cfg['surgery_alpha']})",
                save_path=os.path.join(cfg['output_dir'], 'tsne_random_edit.png')
            )
            
        # Cosine similarity histogram of change from original to edited (Latent Surgery)
        cos_sim_dist_surgery = calculate_cosine_similarity(original_embeddings_test, edited_embeddings_surgery)
        plot_cosine_similarity_hist(
            cos_sim_dist_surgery,
            title=f"Cosine Similarity (Original vs Latent Surgery Edited) Distribution",
            save_path=os.path.join(cfg['output_dir'], 'cosine_similarity_surgery_hist.png')
        )

        if cfg.get('run_random_edit_baseline', True):
            # Cosine similarity histogram of change from original to edited (Random Edit)
            cos_sim_dist_random = calculate_cosine_similarity(original_embeddings_test, edited_embeddings_random)
            plot_cosine_similarity_hist(
                cos_sim_dist_random,
                title=f"Cosine Similarity (Original vs Random Edit Edited) Distribution",
                save_path=os.path.join(cfg['output_dir'], 'cosine_similarity_random_hist.png')
            )


    # Log results to a CSV file
    results_df = pd.DataFrame(results)
    results_csv_path = os.path.join(cfg['output_dir'], 'evaluation_results.csv')
    results_df.to_csv(results_csv_path, index=False)
    print(f"\nEvaluation results saved to {results_csv_path}")
    print("\nSummary of Results:")
    print(results_df)

    print("\nPhase 3: Baseline Comparison & Evaluation Complete.")

    # In Phase 4, we will focus on reporting, reproducibility, and documentation.

if __name__ == '__main__':
    # Ensure config.yaml and dummy data setup from Phase 2 is in place
    if not os.path.exists("config.yaml"):
        print("Please ensure config.yaml and dummy data setup from Phase 2 are in place or replace with real data paths.")
        print("Run the 'main()' function or directly create the necessary files/folders.")
    else:
        main()