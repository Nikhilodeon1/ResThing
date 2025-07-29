# main.py

import os
import torch
import yaml
import pandas as pd
from datetime import datetime

# Assuming existing imports like:
from models.clip_wrapper import CLIPModelWrapper
from data.celeba_loader import get_celeba_dataloaders # Ensure this is correct
from surgery.direction_finder import compute_direction
from surgery.edit_embedding import apply_latent_surgery
from evaluation.metrics import (
    evaluate_probe_accuracy, evaluate_retrieval,
    compute_cosine_similarity_change,
    # Potentially new geometric analysis functions will go here
    # Potentially new statistical significance functions will go here
)
from evaluation.visualize import (
    create_tsne_plot, plot_cosine_similarity_hist,
    # Potentially new failure mode visualization functions will go here
)
# Import baselines
from baselines.prompt_tuning import run_prompt_tuning_baseline
from baselines.random_edit import run_random_edit_baseline
from baselines.linear_probe import run_linear_probe_baseline
# Placeholder import for SpLiCE baseline
# from baselines.splice_baseline import run_splice_baseline # Will be added later

def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    print("Configuration loaded from config.yaml")
    print("Configuration loaded:")
    for key, value in cfg.items():
        print(f"  {key}: {value}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create output directory
    output_dir = cfg['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Prepare results storage
    all_results = []
    
    # --- NEW: Loop for Statistical Significance ---
    num_runs = cfg.get('num_runs_for_stats', 1)
    if num_runs > 1:
        print(f"\n--- Running experiment {num_runs} times for statistical significance ---")
    
    for run_idx in range(num_runs):
        print(f"\n--- Starting Run {run_idx + 1}/{num_runs} ---")
        # Ensure reproducibility for each run
        torch.manual_seed(run_idx) # Use run_idx as seed for different runs

        # --- Phase 1: Model Setup & Data Loading ---
        print("\n--- Phase 1: Model Setup & Data Loading ---")

        # --- NEW: Encoder Selection ---
        current_model_name = cfg['model_name']
        if cfg.get('alternative_encoder_name') and run_idx == 0: # Only log once per run
            current_model_name = cfg['alternative_encoder_name']
            print(f"Using alternative encoder: {current_model_name}")
        elif run_idx == 0:
            print(f"Using default encoder: {current_model_name}")

        clip = CLIPModelWrapper(current_model_name, device)
        
        # Data loaders (assuming CelebA for now)
        train_loader, test_loader = get_celeba_dataloaders(cfg, clip.preprocess_image)
        
        # Get embeddings and labels for the test set
        print("\nGetting test set embeddings...")
        test_embeddings_orig, test_labels, test_img_ids = clip.get_latents(test_loader)
        
        print(f"Original test embeddings shape: {test_embeddings_orig.shape}")
        print(f"Test labels shape: {test_labels.shape}")
        print(f"Number of test image IDs: {len(test_img_ids)}")

        # --- Phase 1 Complete ---

        # --- Phase 2: Latent Surgery Implementation ---
        print("\n--- Phase 2: Latent Surgery Implementation ---")

        # Compute semantic direction
        direction = compute_direction(clip, train_loader, cfg) # Ensure compute_direction handles device correctly
        
        # Apply latent surgery to test embeddings
        print(f"Applying latent surgery with alpha={cfg['surgery_alpha']} to {len(test_embeddings_orig)} embeddings...")
        test_embeddings_edited = apply_latent_surgery(test_embeddings_orig, direction, cfg['surgery_alpha'])
        
        print("Latent surgery applied.")
        print(f"Edited test embeddings (Latent Surgery) shape: {test_embeddings_edited.shape}")

        print("\n--- Phase 2: Model Setup & Latent Surgery Implementation Complete. ---")

        # --- Phase 3: Baseline Comparison & Evaluation ---
        print("\n--- Phase 3: Baseline Comparison & Evaluation ---")

        run_results = {
            'Run_Index': run_idx + 1,
            'Model_Name': current_model_name,
            'Concept_Positive': cfg['concept']['positive'],
            'Concept_Negative': cfg['concept']['negative'],
            'Surgery_Alpha': cfg['surgery_alpha']
        }

        # Evaluate Latent Surgery
        print("\nEvaluating Latent Surgery:")
        # Latent Surgery uses prompt baseline for evaluation of concept change
        ls_prompt_accuracy, ls_pos_conf, ls_neg_conf = run_prompt_tuning_baseline(
            clip, test_embeddings_edited, test_labels, cfg['concept'], device
        )
        run_results['Latent_Surgery_Prompt_Accuracy'] = ls_prompt_accuracy
        run_results['Latent_Surgery_Prompt_Pos_Conf'] = ls_pos_conf
        run_results['Latent_Surgery_Prompt_Neg_Conf'] = ls_neg_conf
        
        ls_cosine_sim_change = compute_cosine_similarity_change(test_embeddings_orig, test_embeddings_edited)
        run_results['Latent_Surgery_Cosine_Sim_Change'] = ls_cosine_sim_change
        print(f"Calculated average cosine similarity: {ls_cosine_sim_change:.4f}")

        # Run Prompt Tuning Baseline (Original Embeddings)
        if cfg.get('run_prompt_baseline', True):
            print("\nRunning Prompt Tuning Baseline for Original Embeddings...")
            pt_accuracy, pt_pos_conf, pt_neg_conf = run_prompt_tuning_baseline(
                clip, test_embeddings_orig, test_labels, cfg['concept'], device
            )
            run_results['Original_Prompt_Accuracy'] = pt_accuracy
            run_results['Original_Prompt_Pos_Conf'] = pt_pos_conf
            run_results['Original_Prompt_Neg_Conf'] = pt_neg_conf
            print("Prompt Tuning Baseline complete.")

        # Run Random Edit Baseline
        if cfg.get('run_random_edit_baseline', True):
            print("\nEvaluating Random Edit Baseline:")
            random_edited_embeddings = run_random_edit_baseline(test_embeddings_orig, cfg['surgery_alpha'])
            
            re_prompt_accuracy, re_pos_conf, re_neg_conf = run_prompt_tuning_baseline(
                clip, random_edited_embeddings, test_labels, cfg['concept'], device
            )
            run_results['Random_Edit_Prompt_Accuracy'] = re_prompt_accuracy
            run_results['Random_Edit_Prompt_Pos_Conf'] = re_pos_conf
            run_results['Random_Edit_Prompt_Neg_Conf'] = re_neg_conf

            re_cosine_sim_change = compute_cosine_similarity_change(test_embeddings_orig, random_edited_embeddings)
            run_results['Random_Edit_Cosine_Sim_Change'] = re_cosine_sim_change
            print(f"Calculated average cosine similarity: {re_cosine_sim_change:.4f}")
            print("Random Edit Baseline complete.")

        # --- NEW: Run SpLiCE Baseline ---
        if cfg.get('run_splice_baseline', False):
            print("\n--- Running SpLiCE Baseline ---")
            # This function needs to be implemented in baselines/splice_baseline.py
            # For now, it's a placeholder.
            # splice_edited_embeddings = run_splice_baseline(clip, test_embeddings_orig, train_loader, cfg, device)
            # print("SpLiCE Baseline complete.")
            # Evaluate SpLiCE similar to latent surgery
            # splice_prompt_accuracy, _, _ = run_prompt_tuning_baseline(
            #     clip, splice_edited_embeddings, test_labels, cfg['concept'], device
            # )
            # run_results['SpLiCE_Prompt_Accuracy'] = splice_prompt_accuracy
            print("SpLiCE Baseline placeholder active. Implement 'run_splice_baseline' function.")
        
        # Run Linear Probe Baseline (Original and Edited)
        if cfg.get('run_linear_probe_baseline', True):
            print("\n--- Running Linear Probe Baseline ---")
            print("\n--- Running Linear Probe Baseline on Original CLIP Embeddings ---")
            lp_orig_accuracy, lp_orig_f1 = evaluate_probe_accuracy(
                test_embeddings_orig, test_labels, 
                os.path.join(output_dir, 'linear_probe_baseline_results.json')
            )
            run_results['Linear_Probe_Original_Accuracy'] = lp_orig_accuracy
            run_results['Linear_Probe_Original_F1'] = lp_orig_f1

        if cfg.get('run_latent_probe', True):
            print("\n--- Running Latent Probe on Edited Embeddings ---")
            lp_edited_accuracy, lp_edited_f1 = evaluate_probe_accuracy(
                test_embeddings_edited, test_labels,
                os.path.join(output_dir, 'latent_probe_edited_results.json')
            )
            run_results['Latent_Probe_Edited_Accuracy'] = lp_edited_accuracy
            run_results['Latent_Probe_Edited_F1'] = lp_edited_f1

        # Run Retrieval Evaluation
        if cfg.get('run_retrieval_eval', True):
            print("\n--- Running Retrieval Evaluation ---")
            
            # For Original Embeddings
            print("Evaluating retrieval for Original Embeddings (Top-K=5):")
            retrieval_orig_accuracy = evaluate_retrieval(
                test_embeddings_orig, test_labels, test_img_ids, cfg['retrieval_top_k'],
                os.path.join(output_dir, 'retrieval_original_results.json')
            )
            run_results['Retrieval_Accuracy_Original'] = retrieval_orig_accuracy

            # For Edited Embeddings
            print("Evaluating retrieval for Edited Embeddings (Top-K=5):")
            retrieval_edited_accuracy = evaluate_retrieval(
                test_embeddings_edited, test_labels, test_img_ids, cfg['retrieval_top_k'],
                os.path.join(output_dir, 'retrieval_edited_results.json')
            )
            run_results['Retrieval_Accuracy_Edited'] = retrieval_edited_accuracy

        # --- NEW: Geometric / Theoretical Analysis ---
        if cfg.get('enable_geometric_analysis', False):
            print("\n--- Performing Geometric / Theoretical Analysis ---")
            # Example: Direction orthogonality or angle distributions
            # These functions need to be added to evaluation/metrics.py
            # ortho_score = compute_direction_orthogonality(direction, some_other_direction)
            # angle_dist = compute_embedding_angle_distribution(test_embeddings_orig)
            # run_results['Orthogonality_Score'] = ortho_score
            # run_results['Angle_Distribution_Metric'] = angle_dist
            print("Geometric analysis placeholder active. Implement specific analysis functions.")

        # --- NEW: Failure Mode Visualization ---
        if cfg.get('enable_failure_mode_viz', False):
            print("\n--- Preparing Failure Mode Visualizations ---")
            # This function needs to identify and visualize specific failures.
            # Requires more context on what constitutes a "failure" and how to visualize.
            # Example: visualize_failure_modes(original_images, edited_images, original_labels, edited_predictions, output_dir)
            print("Failure mode visualization placeholder active. Implement 'visualize_failure_modes'.")

        # --- Visualizations ---
        print(f"\nGenerating t-SNE plot for {len(test_embeddings_orig)} embeddings...")
        create_tsne_plot(test_embeddings_orig, test_labels, os.path.join(output_dir, 'tsne_original.png'), "t-SNE of Original Embeddings")

        print(f"Generating t-SNE plot for {len(test_embeddings_edited)} embeddings...")
        create_tsne_plot(test_embeddings_edited, test_labels, os.path.join(output_dir, 'tsne_latent_surgery.png'), "t-SNE of Latent Surgery Edited Embeddings")

        if cfg.get('run_random_edit_baseline', True):
            print(f"Generating t-SNE plot for {len(random_edited_embeddings)} embeddings...")
            create_tsne_plot(random_edited_embeddings, test_labels, os.path.join(output_dir, 'tsne_random_edit.png'), "t-SNE of Random Edit Embeddings")
        
        # Plot cosine similarity change histograms
        print(f"Generating histogram for {len(test_embeddings_orig)} cosine similarities...")
        plot_cosine_similarity_hist(ls_cosine_sim_change, os.path.join(output_dir, 'cosine_similarity_surgery_hist.png'), "Cosine Similarity Change (Latent Surgery)")

        if cfg.get('run_random_edit_baseline', True):
            print(f"Generating histogram for {len(random_edited_embeddings)} cosine similarities...")
            plot_cosine_similarity_hist(re_cosine_sim_change, os.path.join(output_dir, 'cosine_similarity_random_hist.png'), "Cosine Similarity Change (Random Edit)")
        
        all_results.append(run_results)
        print(f"\n--- Run {run_idx + 1}/{num_runs} Complete ---")


    # --- NEW: Statistical Significance Calculation (after all runs) ---
    if num_runs > 1:
        print("\n--- Computing Statistical Significance ---")
        results_df_multi_run = pd.DataFrame(all_results)
        
        # Example: Compute mean and std dev for a metric across runs
        # You'd need to select specific metrics here.
        # This part requires specific statistical test implementations in evaluation/metrics.py
        # For example:
        # mean_ls_acc = results_df_multi_run['Latent_Surgery_Prompt_Accuracy'].mean()
        # std_ls_acc = results_df_multi_run['Latent_Surgery_Prompt_Accuracy'].std()
        # print(f"Latent Surgery Prompt Accuracy (Mean ± Std): {mean_ls_acc:.4f} ± {std_ls_acc:.4f}")

        # Placeholder for actual statistical tests (e.g., t-tests between methods)
        # from evaluation.statistical_tests import run_statistical_tests # New file
        # run_statistical_tests(results_df_multi_run)
        print("Statistical significance calculation placeholder active. Implement specific tests.")


    # Save summary of results
    final_results_df = pd.DataFrame(all_results)
    output_csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    final_results_df.to_csv(output_csv_path, index=False)
    print(f"\nEvaluation results saved to {output_csv_path}")

    print("\nSummary of Results:")
    print(final_results_df)

    print("\nPhase 3: Baseline Comparison & Evaluation Complete.")

if __name__ == '__main__':
    main()