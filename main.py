# main.py

import os
import torch
import yaml
import pandas as pd
from datetime import datetime
from scipy import stats # For statistical tests
import random # For geometric analysis random vector
import shutil # For copying images for failure visualization

from models.encoder_wrapper import EncoderWrapper
# Import all potential dataset loaders. You must ensure these files exist and have get_dataloaders
from data.celeba_loader import get_celeba_dataloaders
from data.cub_loader import get_cub_dataloaders 
# from data.imagenet_loader import get_imagenet_dataloaders 

# NEW IMPORTS FOR NEW FEATURES
import analysis.geometry # For geometric insights
from evaluation.metrics import (
    calculate_accuracy,
    calculate_cosine_similarity,
    calculate_mean_confidence,
    evaluate_probe_accuracy,
    evaluate_retrieval,
    perform_statistical_test,
    perform_paired_statistical_test, # NEW: Paired t-test
    get_model_predictions, # NEW: Get per-sample predictions
)
from evaluation.visualize import (
    create_tsne_plot,
    plot_cosine_similarity_hist,
    plot_vector_trajectories,
    visualize_failure_modes, # NEW: Failure mode visualization
)
# Import baselines
from baselines.prompt_tuning import prompt_tuning_baseline
from baselines.random_edit import random_edit_baseline
from baselines.splice_baseline import run_splice_baseline


def get_dataset_dataloaders(cfg, preprocess_fn):
    """
    Dynamically loads the correct dataset dataloaders and datasets based on config.
    Returns: (train_loader, test_loader, train_dataset, test_dataset)
    """
    dataset_name = cfg['dataset'].lower()
    if dataset_name == 'celeba':
        # get_celeba_dataloaders should be updated to return train_dataset, test_dataset
        train_loader, test_loader, train_dataset, test_dataset = get_celeba_dataloaders(cfg, preprocess_fn)
        return train_loader, test_loader, train_dataset, test_dataset
    elif dataset_name == 'cub-200': 
        # get_cub_dataloaders should be updated to return train_dataset, test_dataset
        train_loader, test_loader, train_dataset, test_dataset = get_cub_dataloaders(cfg, preprocess_fn)
        return train_loader, test_loader, train_dataset, test_dataset
    # elif dataset_name == 'imagenet': 
    #     # If imagenet is implemented, it should also return the datasets
    #     return get_imagenet_dataloaders(cfg, preprocess_fn)
    else:
        raise ValueError(f"Unknown dataset: {cfg['dataset']}. Check config.yaml and data loaders.")


def main():
    # Load configuration
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)

    print("Configuration loaded from config.yaml")
    print("Configuration loaded:")
    for key, value in cfg.items():
        print(f"    {key}: {value}")

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create output directory
    output_dir = cfg['output_dir']
    print(f"DEBUG: Loaded output_dir from config: '{output_dir}'")
    os.makedirs(output_dir, exist_ok=True)

    # Prepare results storage
    all_results = []
    
    # --- Loop for Statistical Significance ---
    num_runs = cfg.get('num_runs_for_stats', 1)
    if num_runs > 1:
        print(f"\n--- Running experiment {num_runs} times for statistical significance ---")
    
    # Determine the model name to use across all runs for this experiment
    model_to_use_for_experiment = cfg['model_name'] # Default
    if cfg.get('alternative_encoder_name') and cfg['alternative_encoder_name'] != "":
        model_to_use_for_experiment = cfg['alternative_encoder_name'] # Override if alternative specified

    print(f"\nModel for this experiment: {model_to_use_for_experiment}")

    for run_idx in range(num_runs):
        print(f"\n--- Starting Run {run_idx + 1}/{num_runs} ---")
        # Ensure reproducibility for each run (optional, for different random seeds per run)
        # torch.manual_seed(run_idx) 
        # np.random.seed(run_idx) # If using numpy random functions

        # --- Phase 1: Model Setup & Data Loading ---
        print("\n--- Phase 1: Model Setup & Data Loading ---")

        # Current model name for this specific run (will be consistent if no changes within runs)
        current_model_name = model_to_use_for_experiment
        
        # Instantiate EncoderWrapper
        encoder = EncoderWrapper(current_model_name, device)
        
        # Data loaders (dynamically load based on config)
        # Ensure get_dataset_dataloaders returns train_dataset and test_dataset
        train_loader, test_loader, train_dataset, test_dataset = get_dataset_dataloaders(cfg, encoder.preprocess_image) 
        
        # Get embeddings and labels for the test set
        print("\nGetting test set embeddings...")
        test_embeddings_orig_for_ls, test_labels_for_ls, test_img_ids_for_ls = encoder.get_latents(test_loader)
        
        print(f"Original test embeddings shape: {test_embeddings_orig_for_ls.shape}")
        print(f"Test labels shape: {test_labels_for_ls.shape}")
        print(f"Number of test image IDs: {len(test_img_ids_for_ls)}")

        # --- Phase 1 Complete ---

        # --- Phase 2: Latent Surgery Implementation ---
        print("\n--- Phase 2: Latent Surgery Implementation ---")

        # Compute semantic direction
        direction = compute_direction(encoder, train_loader, cfg)
        
        # Apply latent surgery to test embeddings
        print(f"Applying latent surgery with alpha={cfg['surgery_alpha']}...")
        test_embeddings_edited_ls, _, _, _ = apply_surgery(
            encoder, test_loader, direction, cfg['surgery_alpha']
        )
        
        print("Latent surgery applied.")
        print(f"Edited test embeddings (Latent Surgery) shape: {test_embeddings_edited_ls.shape}")

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

        # Get per-sample predictions for failure analysis
        print("\nGetting per-sample predictions for original embeddings...")
        orig_predictions, orig_confidences = get_model_predictions(
            encoder, test_embeddings_orig_for_ls, cfg['concept'], device
        )
        print("Getting per-sample predictions for edited embeddings...")
        edited_predictions, edited_confidences = get_model_predictions(
            encoder, test_embeddings_edited_ls, cfg['concept'], device
        )


        # Evaluate Latent Surgery (using prompt tuning)
        print("\nEvaluating Latent Surgery (Prompt Tuning Classifier):")
        ls_prompt_accuracy, ls_pos_conf, ls_neg_conf = prompt_tuning_baseline(
            encoder, test_embeddings_edited_ls, test_labels_for_ls, cfg['concept'], device
        )
        run_results['Latent_Surgery_Prompt_Accuracy'] = ls_prompt_accuracy
        run_results['Latent_Surgery_Prompt_Pos_Conf'] = ls_pos_conf
        run_results['Latent_Surgery_Prompt_Neg_Conf'] = ls_neg_conf
        
        ls_cosine_sim_change = calculate_cosine_similarity(test_embeddings_orig_for_ls, test_embeddings_edited_ls).mean().item()
        run_results['Latent_Surgery_Cosine_Sim_Change'] = ls_cosine_sim_change
        print(f"Calculated average cosine similarity (Latent Surgery): {ls_cosine_sim_change:.4f}")

        # Run Prompt Tuning Baseline (Original Embeddings)
        if cfg.get('run_prompt_baseline', True):
            print("\nRunning Prompt Tuning Baseline for Original Embeddings...")
            pt_accuracy, pt_pos_conf, pt_neg_conf = prompt_tuning_baseline(
                encoder, test_embeddings_orig_for_ls, test_labels_for_ls, cfg['concept'], device
            )
            run_results['Original_Prompt_Accuracy'] = pt_accuracy
            run_results['Original_Prompt_Pos_Conf'] = pt_pos_conf
            run_results['Original_Prompt_Neg_Conf'] = pt_neg_conf
            print("Prompt Tuning Baseline complete.")

        # Run Random Edit Baseline
        if cfg.get('run_random_edit_baseline', True):
            print("\nEvaluating Random Edit Baseline:")
            random_edited_embeddings = random_edit_baseline(test_embeddings_orig_for_ls, cfg['surgery_alpha'])
            
            re_prompt_accuracy, re_pos_conf, re_neg_conf = prompt_tuning_baseline(
                encoder, random_edited_embeddings, test_labels_for_ls, cfg['concept'], device
            )
            run_results['Random_Edit_Prompt_Accuracy'] = re_prompt_accuracy
            run_results['Random_Edit_Prompt_Pos_Conf'] = re_pos_conf
            run_results['Random_Edit_Prompt_Neg_Conf'] = re_neg_conf

            re_cosine_sim_change = calculate_cosine_similarity(test_embeddings_orig_for_ls, random_edited_embeddings).mean().item()
            run_results['Random_Edit_Cosine_Sim_Change'] = re_cosine_sim_change
            print(f"Calculated average cosine similarity (Random Edit): {re_cosine_sim_change:.4f}")
            print("Random Edit Baseline complete.")

        # Run SpLiCE Baseline
        if cfg.get('run_splice_baseline', False):
            print("\n--- Running SpLiCE Baseline ---")
            splice_edited_embeddings = run_splice_baseline(
                encoder, test_embeddings_orig_for_ls, train_loader, cfg, device
            )
            print("SpLiCE Baseline complete.")
            
            splice_prompt_accuracy, splice_pos_conf, splice_neg_conf = prompt_tuning_baseline(
                encoder, splice_edited_embeddings, test_labels_for_ls, cfg['concept'], device
            )
            run_results['SpLiCE_Prompt_Accuracy'] = splice_prompt_accuracy
            run_results['SpLiCE_Prompt_Pos_Conf'] = splice_pos_conf
            run_results['SpLiCE_Prompt_Neg_Conf'] = splice_neg_conf

            splice_cosine_sim_change = calculate_cosine_similarity(test_embeddings_orig_for_ls, splice_edited_embeddings).mean().item()
            run_results['SpLiCE_Cosine_Sim_Change'] = splice_cosine_sim_change
            print(f"Calculated average cosine similarity (SpLiCE): {splice_cosine_sim_change:.4f}")

        # Run Linear Probe Baseline (Original Embeddings)
        if cfg.get('run_linear_probe_baseline', True):
            print("\n--- Running Linear Probe Baseline ---")
            lp_orig_accuracy, lp_orig_f1 = evaluate_probe_accuracy(
                test_embeddings_orig_for_ls, 
                test_labels_for_ls, 
                os.path.join(output_dir, f'linear_probe_baseline_results_run{run_idx}.json') # Unique name per run
            )
            run_results['Linear_Probe_Original_Accuracy'] = lp_orig_accuracy
            run_results['Linear_Probe_Original_F1'] = lp_orig_f1
            print("Linear Probe Baseline complete.")

        # Run Latent Probe (Edited Embeddings)
        if cfg.get('run_latent_probe', True):
            print("\n--- Running Latent Probe ---")
            lp_edited_accuracy, lp_edited_f1 = evaluate_probe_accuracy(
                test_embeddings_edited_ls, 
                test_labels_for_ls,
                os.path.join(output_dir, f'latent_probe_edited_results_run{run_idx}.json') # Unique name per run
            )
            run_results['Latent_Probe_Edited_Accuracy'] = lp_edited_accuracy
            run_results['Latent_Probe_Edited_F1'] = lp_edited_f1
            print("Latent Probe complete.")

        # Run Retrieval Evaluation
        if cfg.get('run_retrieval_eval', True):
            print("\n--- Running Retrieval Evaluation ---")
            # For Original Embeddings
            print("Evaluating retrieval for Original Embeddings:")
            retrieval_orig_mAP, retrieval_orig_topK_acc = evaluate_retrieval(
                test_embeddings_orig_for_ls, test_labels_for_ls, test_img_ids_for_ls, 
                test_embeddings_orig_for_ls, test_labels_for_ls, test_img_ids_for_ls, # Use test set as gallery
                cfg['retrieval_top_k'],
                os.path.join(output_dir, f'retrieval_original_results_run{run_idx}.json')
            )
            run_results['Retrieval_mAP_Original'] = retrieval_orig_mAP
            run_results[f'Retrieval_Top{cfg["retrieval_top_k"]}_Accuracy_Original'] = retrieval_orig_topK_acc

            # For Edited Embeddings
            print("Evaluating retrieval for Edited Embeddings:")
            retrieval_edited_mAP, retrieval_edited_topK_acc = evaluate_retrieval(
                test_embeddings_edited_ls, test_labels_for_ls, test_img_ids_for_ls,
                test_embeddings_edited_ls, test_labels_for_ls, test_img_ids_for_ls, # Use edited set as gallery
                cfg['retrieval_top_k'],
                os.path.join(output_dir, f'retrieval_edited_results_run{run_idx}.json')
            )
            run_results['Retrieval_mAP_Edited'] = retrieval_edited_mAP
            run_results[f'Retrieval_Top{cfg["retrieval_top_k"]}_Accuracy_Edited'] = retrieval_edited_topK_acc
            print("Retrieval Evaluation complete.")

        # --- Geometric / Theoretical Analysis ---
        if cfg.get('enable_geometric_analysis', False):
            print("\n--- Performing Geometric / Theoretical Analysis ---")
            # Analyze properties of the learned direction vector
            direction_properties = analysis.geometry.calculate_vector_properties(direction, "Concept_Direction")
            run_results.update(direction_properties)

            # Compare concept direction with a random vector
            random_direction = torch.randn_like(direction)
            random_direction_comparison = analysis.geometry.compare_vectors_geometry(
                direction, random_direction, "Concept_Direction", "Random_Direction"
            )
            run_results.update(random_direction_comparison)

            # Plot distribution of components of the direction vector
            analysis.geometry.plot_vector_component_distribution(
                direction, "Concept_Direction", 
                os.path.join(output_dir, f'geometric_analysis_direction_distribution_run{run_idx}.png')
            )

            # Placeholder for future multiple concept direction comparison:
            # If you have multiple concept directions (e.g., from different runs/definitions),
            # you can collect them and use plot_pairwise_similarity_heatmap.
            # E.g., concept_directions = [direction_concept1, direction_concept2]
            # similarity_matrix = ... # compute pairwise similarities
            # analysis.geometry.plot_pairwise_similarity_heatmap(similarity_matrix, ['Concept1', 'Concept2'], "Concept Direction Similarities", ...)
            
            print("Geometric Analysis complete.")

        # --- Failure Mode Visualization ---
        if cfg.get('enable_failure_mode_viz', False):
            print("\n--- Preparing Failure Mode Visualizations ---")
            visualize_failure_modes(
                encoder, 
                test_embeddings_orig_for_ls, 
                test_embeddings_edited_ls, 
                test_labels_for_ls, 
                orig_predictions, 
                edited_predictions, 
                test_img_ids_for_ls, 
                output_dir,
                cfg, # Pass full config for data_root access to reconstruct image paths
                cfg['concept'],
                num_samples_to_save=cfg.get('num_failure_viz_samples', 10)
            )
            print("Failure Mode Visualizations complete.")


        # --- Visualizations ---
        if cfg.get('save_tsne', True):
            print(f"\nGenerating t-SNE plot for {len(test_embeddings_orig_for_ls)} embeddings (Original)...")
            create_tsne_plot(test_embeddings_orig_for_ls, test_labels_for_ls, "t-SNE of Original Embeddings", os.path.join(output_dir, f'tsne_original_run{run_idx}.png'))

            print(f"Generating t-SNE plot for {len(test_embeddings_edited_ls)} embeddings (Latent Surgery)...")
            create_tsne_plot(test_embeddings_edited_ls, test_labels_for_ls, "t-SNE of Latent Surgery Edited Embeddings", os.path.join(output_dir, f'tsne_latent_surgery_run{run_idx}.png'))

            if cfg.get('run_random_edit_baseline', True):
                if 'random_edited_embeddings' in locals():
                    print(f"Generating t-SNE plot for {len(random_edited_embeddings)} embeddings (Random Edit)...")
                    create_tsne_plot(random_edited_embeddings, test_labels_for_ls, "t-SNE of Random Edit Embeddings", os.path.join(output_dir, f'tsne_random_edit_run{run_idx}.png'))
            
            if cfg.get('run_splice_baseline', False):
                if 'splice_edited_embeddings' in locals():
                    print(f"Generating t-SNE plot for {len(splice_edited_embeddings)} embeddings (SpLiCE)...")
                    create_tsne_plot(splice_edited_embeddings, test_labels_for_ls, "t-SNE of SpLiCE Edited Embeddings", os.path.join(output_dir, f'tsne_splice_run{run_idx}.png'))
            
            # Generate vector trajectory plot
            if cfg.get('save_vector_trajectory', True):
                print(f"Generating vector trajectory plot for {len(test_embeddings_orig_for_ls)} embeddings...")
                plot_vector_trajectories(
                    test_embeddings_orig_for_ls, 
                    test_embeddings_edited_ls, 
                    test_labels_for_ls, 
                    direction, # Pass the direction vector
                    os.path.join(output_dir, f'vector_trajectory_run{run_idx}.png'),
                    num_samples=cfg.get('num_trajectory_samples', 50) # Plot trajectories for a subset of samples for clarity
                )

        # Plot cosine similarity change histograms
        print(f"Generating histogram for cosine similarities (Latent Surgery)...")
        plot_cosine_similarity_hist(
            calculate_cosine_similarity(test_embeddings_orig_for_ls, test_embeddings_edited_ls), 
            "Cosine Similarity Change (Latent Surgery)", 
            os.path.join(output_dir, f'cosine_similarity_surgery_hist_run{run_idx}.png')
        )
        
        if cfg.get('run_random_edit_baseline', True):
            if 'random_edited_embeddings' in locals():
                print(f"Generating histogram for cosine similarities (Random Edit)...")
                plot_cosine_similarity_hist(
                    calculate_cosine_similarity(test_embeddings_orig_for_ls, random_edited_embeddings), 
                    "Cosine Similarity Change (Random Edit)", 
                    os.path.join(output_dir, f'cosine_similarity_random_hist_run{run_idx}.png')
                )
        
        if cfg.get('run_splice_baseline', False):
            if 'splice_edited_embeddings' in locals():
                print(f"Generating histogram for cosine similarities (SpLiCE)...")
                plot_cosine_similarity_hist(
                    calculate_cosine_similarity(test_embeddings_orig_for_ls, splice_edited_embeddings), 
                    "Cosine Similarity Change (SpLiCE)", 
                    os.path.join(output_dir, f'cosine_similarity_splice_hist_run{run_idx}.png')
                )

        all_results.append(run_results)
        print(f"\n--- Run {run_idx + 1}/{num_runs} Complete ---")


    # --- Statistical Significance Calculation (after all runs) ---
    if num_runs > 1:
        print("\n--- Computing Statistical Significance ---")
        results_df_multi_run = pd.DataFrame(all_results)
        
        # Example: Compare Latent Surgery vs. Original Prompt Accuracy (Paired)
        if 'Latent_Surgery_Prompt_Accuracy' in results_df_multi_run.columns and \
           'Original_Prompt_Accuracy' in results_df_multi_run.columns:
            
            ls_accs = results_df_multi_run['Latent_Surgery_Prompt_Accuracy'].tolist()
            orig_accs = results_df_multi_run['Original_Prompt_Accuracy'].tolist()
            
            print("\nComparing Latent Surgery Prompt Accuracy vs. Original Prompt Accuracy (Paired t-test):")
            perform_paired_statistical_test(ls_accs, orig_accs, "Latent Surgery vs Original Prompt Accuracy")
        
        # Example: Latent Surgery vs. Random Edit Prompt Accuracy (Paired)
        if 'Latent_Surgery_Prompt_Accuracy' in results_df_multi_run.columns and \
           'Random_Edit_Prompt_Accuracy' in results_df_multi_run.columns:
            ls_accs = results_df_multi_run['Latent_Surgery_Prompt_Accuracy'].tolist()
            re_accs = results_df_multi_run['Random_Edit_Prompt_Accuracy'].tolist()
            print("\nComparing Latent Surgery Prompt Accuracy vs. Random Edit Prompt Accuracy (Paired t-test):")
            perform_paired_statistical_test(ls_accs, re_accs, "Latent Surgery vs Random Edit Prompt Accuracy")

        # Example: Latent Surgery vs. SpLiCE Prompt Accuracy (Paired)
        if 'Latent_Surgery_Prompt_Accuracy' in results_df_multi_run.columns and \
           'SpLiCE_Prompt_Accuracy' in results_df_multi_run.columns:
            ls_accs = results_df_multi_run['Latent_Surgery_Prompt_Accuracy'].tolist()
            splice_accs = results_df_multi_run['SpLiCE_Prompt_Accuracy'].tolist()
            print("\nComparing Latent Surgery Prompt Accuracy vs. SpLiCE Prompt Accuracy (Paired t-test):")
            perform_paired_statistical_test(ls_accs, splice_accs, "Latent Surgery vs SpLiCE Prompt Accuracy")


        # Save the full multi-run results DataFrame
        multi_run_csv_path = os.path.join(output_dir, 'multi_run_evaluation_results.csv')
        results_df_multi_run.to_csv(multi_run_csv_path, index=False)
        print(f"\nFull multi-run results saved to {multi_run_csv_path}")

    # Save summary of results (if only one run, this is the same as all_results)
    final_results_df = pd.DataFrame(all_results)
    output_csv_path = os.path.join(output_dir, 'evaluation_results.csv')
    final_results_df.to_csv(output_csv_path, index=False)
    print(f"\nEvaluation results saved to {output_csv_path}")

    print("\nSummary of Results:")
    print(final_results_df)

    print("\nPhase 3: Baseline Comparison & Evaluation Complete.")

if __name__ == '__main__':
    main()