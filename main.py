# main.ipynb (or main.py)
import torch
import os
import pandas as pd # For logging results

from utils.config_loader import load_config
from data.celeba_loader import get_celeba_dataloaders
from data.cub_loader import get_cub_dataloaders
from data.imagenet_loader import get_imagenet_dataloaders
from models.clip_wrapper import CLIPModelWrapper
from surgery.direction_finder import compute_direction
from surgery.edit_embedding import apply_surgery
from baselines.prompt_tuning import prompt_tuning_baseline
from baselines.random_edit import random_edit_baseline
from evaluation.metrics import calculate_accuracy, calculate_cosine_similarity, calculate_mean_confidence
from evaluation.visualize import create_tsne_plot, plot_cosine_similarity_hist

def main():
    # Load configuration
    cfg = load_config("config.yaml")
    print("Configuration loaded:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

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

    clip = CLIPModelWrapper(cfg["model_name"])

    # Compute semantic direction vector using the training data
    direction = compute_direction(clip, train_loader, cfg)
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

    # --- Evaluation for Latent Surgery ---
    print("\nEvaluating Latent Surgery:")
    # For classification with latent surgery, we can use the direction vector itself as a classifier
    # A simple classifier: dot product with direction vector, then threshold.
    # More robust: use the concept text embeddings (from prompt tuning) as targets for both original and edited
    
    # Re-use prompt text embeddings as "classifiers"
    pos_text_prompt = f"A photo of a {cfg['concept']['positive'].lower()}"
    neg_text_prompt = f"A photo of a {cfg['concept']['negative'].lower()}"
    text_embeddings = clip.embed_text([pos_text_prompt, neg_text_prompt]).to(original_embeddings_test.device)
    pos_concept_emb = text_embeddings[0]
    neg_concept_emb = text_embeddings[1]

    # Evaluate original embeddings w.r.t. concept prompts
    original_preds, original_confidences = prompt_tuning_baseline(
        clip, original_embeddings_test, test_labels, cfg['concept']['positive'], cfg['concept']['negative']
    ) # Re-using the prompt_tuning_baseline logic for evaluation
    original_accuracy = calculate_accuracy(original_preds, test_labels)
    original_mean_pos_conf, original_mean_neg_conf = calculate_mean_confidence(original_confidences, test_labels)
    results.append({
        'Method': 'Original Embeddings',
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

    # Prompt Tuning Baseline (already effectively run as the classification method)
    # The 'Original Embeddings' entry in `results` already covers the performance of prompt tuning
    # when applied directly to original embeddings. We don't need a separate "run" for this baseline.
    # The prompt_tuning_baseline function itself is used as the classification mechanism.
    
    # --- Visualizations ---
    if cfg.get('save_tsne', True):
        # t-SNE plot for Original Embeddings
        create_tsne_plot(
            original_embeddings_test,
            test_labels,
            title=f"t-SNE of Original Embeddings ({cfg['dataset']} - {cfg['concept']['positive']}/{cfg['concept']['negative']})",
            save_path=os.path.join(cfg['output_dir'], 'tsne_original.png')
        )

        # t-SNE plot for Latent Surgery Edited Embeddings
        create_tsne_plot(
            edited_embeddings_surgery,
            test_labels,
            title=f"t-SNE of Latent Surgery Embeddings (Alpha={cfg['surgery_alpha']})",
            save_path=os.path.join(cfg['output_dir'], 'tsne_latent_surgery.png')
        )

        if cfg.get('run_random_edit_baseline', True):
            # t-SNE plot for Random Edit Baseline Embeddings
            create_tsne_plot(
                edited_embeddings_random,
                test_labels,
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