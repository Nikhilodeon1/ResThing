# evaluation/plot_results.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def plot_accuracy_comparison(df: pd.DataFrame, output_dir: str, filename_suffix: str = ""):
    """
    Plots a bar chart comparing the average accuracy of different methods.
    Assumes df contains 'Original_Prompt_Accuracy', 'Latent_Surgery_Prompt_Accuracy',
    'Random_Edit_Prompt_Accuracy', and optionally 'SpLiCE_Prompt_Accuracy'.
    """
    print("\n--- Generating Accuracy Comparison Chart ---")
    
    accuracy_cols = [
        'Original_Prompt_Accuracy',
        'Latent_Surgery_Prompt_Accuracy',
        'Random_Edit_Prompt_Accuracy'
    ]
    if 'SpLiCE_Prompt_Accuracy' in df.columns:
        accuracy_cols.append('SpLiCE_Prompt_Accuracy')

    # Filter for columns that actually exist in the DataFrame
    existing_accuracy_cols = [col for col in accuracy_cols if col in df.columns]

    if not existing_accuracy_cols:
        print("No relevant accuracy columns found for comparison chart. Skipping.")
        return

    # Calculate mean accuracy for each method
    mean_accuracies = df[existing_accuracy_cols].mean()
    std_accuracies = df[existing_accuracy_cols].std() # For error bars

    # Rename for plotting clarity
    plot_labels = [
        col.replace('_Prompt_Accuracy', '').replace('_', ' ') for col in existing_accuracy_cols
    ]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=plot_labels, y=mean_accuracies.values, palette='viridis')
    
    # Add error bars if multiple runs exist
    if df.shape[0] > 1:
        plt.errorbar(x=plot_labels, y=mean_accuracies.values, yerr=std_accuracies.values, 
                     fmt='none', capsize=5, color='black')

    plt.title('Average Prompt Accuracy Across Methods')
    plt.ylabel('Accuracy')
    plt.xlabel('Method')
    plt.ylim(0, 1.0) # Accuracy is between 0 and 1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'accuracy_comparison_chart{filename_suffix}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy comparison chart saved to {save_path}")


def plot_ablation_accuracy_vs_alpha(df: pd.DataFrame, output_dir: str, metric: str = 'Latent_Surgery_Prompt_Accuracy'):
    """
    Plots the specified metric (e.g., accuracy) against 'Surgery_Alpha' for ablation studies.
    Assumes 'Surgery_Alpha' and the specified metric column exist in the DataFrame.
    """
    print(f"\n--- Generating Ablation Chart: {metric} vs. Surgery Alpha ---")

    if 'Surgery_Alpha' not in df.columns or metric not in df.columns:
        print(f"Required columns ('Surgery_Alpha' or '{metric}') not found for ablation chart. Skipping.")
        return

    # Group by 'Surgery_Alpha' and calculate mean and std for the metric
    ablation_summary = df.groupby('Surgery_Alpha')[metric].agg(['mean', 'std']).reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=ablation_summary, x='Surgery_Alpha', y='mean', marker='o', 
                 label=f'Mean {metric.replace("_", " ")}', color='blue')
    
    # Add shaded error region if multiple runs exist per alpha
    if 'std' in ablation_summary.columns and df.shape[0] > df['Surgery_Alpha'].nunique():
        plt.fill_between(ablation_summary['Surgery_Alpha'], 
                         ablation_summary['mean'] - ablation_summary['std'], 
                         ablation_summary['mean'] + ablation_summary['std'], 
                         color='blue', alpha=0.2, label='Std Dev')

    plt.title(f'{metric.replace("_", " ")} vs. Surgery Alpha')
    plt.xlabel('Surgery Alpha (α)')
    plt.ylabel(metric.replace("_", " "))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'ablation_{metric.lower()}_vs_alpha.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Ablation chart saved to {save_path}")


def plot_accuracy_vs_efficiency(df: pd.DataFrame, output_dir: str, accuracy_metric: str = 'Latent_Surgery_Prompt_Accuracy'):
    """
    Plots accuracy against total run time to show efficiency trade-offs.
    Assumes 'Total_Run_Time_s' and the specified accuracy_metric exist.
    """
    print("\n--- Generating Accuracy vs. Efficiency Trade-off Chart ---")

    if 'Total_Run_Time_s' not in df.columns or accuracy_metric not in df.columns:
        print(f"Required columns ('Total_Run_Time_s' or '{accuracy_metric}') not found for efficiency chart. Skipping.")
        return
    
    # Use the mean values if multiple runs per ablation setting
    if 'Surgery_Alpha' in df.columns:
        plot_df = df.groupby('Surgery_Alpha')[[accuracy_metric, 'Total_Run_Time_s']].mean().reset_index()
    else:
        plot_df = df # Use the raw DataFrame if no ablation

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=plot_df, x='Total_Run_Time_s', y=accuracy_metric, hue='Surgery_Alpha' if 'Surgery_Alpha' in df.columns else None, 
                    s=100, alpha=0.8, palette='viridis')
    
    if 'Surgery_Alpha' in df.columns:
        # Add labels for each point (alpha value)
        for i, row in plot_df.iterrows():
            plt.text(row['Total_Run_Time_s'] + 0.05, row[accuracy_metric], f"α={row['Surgery_Alpha']}", fontsize=9)

    plt.title(f'{accuracy_metric.replace("_", " ")} vs. Total Run Time')
    plt.xlabel('Total Run Time (seconds)')
    plt.ylabel(accuracy_metric.replace("_", " "))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Surgery Alpha')
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'accuracy_vs_efficiency_{accuracy_metric.lower()}.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Accuracy vs. Efficiency chart saved to {save_path}")


def main():
    # Define the path to the multi-run results CSV
    output_dir = './outputs' # Assuming outputs are saved here by main.py
    results_csv_path = os.path.join(output_dir, 'multi_run_evaluation_results.csv')
    charts_output_dir = os.path.join(output_dir, 'charts')
    os.makedirs(charts_output_dir, exist_ok=True)

    if not os.path.exists(results_csv_path):
        print(f"Error: Results CSV not found at {results_csv_path}. Please run main.py first.")
        return

    df = pd.read_csv(results_csv_path)
    print(f"Loaded results from {results_csv_path}")
    print("DataFrame head:\n", df.head())
    print("DataFrame columns:\n", df.columns.tolist())

    # 1. Classifier Performance Charts (Accuracy Comparison Across Methods)
    plot_accuracy_comparison(df, charts_output_dir)

    # 2. Ablation Study Charts (e.g., Accuracy vs. Alpha)
    # This chart is only relevant if ablation was performed (i.e., multiple unique alpha values)
    if 'Surgery_Alpha' in df.columns and df['Surgery_Alpha'].nunique() > 1:
        plot_ablation_accuracy_vs_alpha(df, charts_output_dir, metric='Latent_Surgery_Prompt_Accuracy')
        # You can add more ablation metrics here if desired, e.g.:
        # plot_ablation_accuracy_vs_alpha(df, charts_output_dir, metric='Retrieval_mAP_Edited')
    else:
        print("Skipping ablation charts: 'Surgery_Alpha' column not found or only one unique alpha value.")


    # 3. Accuracy vs. Efficiency Trade-offs Charts
    # This chart is relevant if timing data is available
    if 'Total_Run_Time_s' in df.columns:
        plot_accuracy_vs_efficiency(df, charts_output_dir, accuracy_metric='Latent_Surgery_Prompt_Accuracy')
        # You can add more trade-off plots here if desired, e.g.:
        # plot_accuracy_vs_efficiency(df, charts_output_dir, accuracy_metric='Retrieval_mAP_Edited')
    else:
        print("Skipping accuracy vs. efficiency charts: 'Total_Run_Time_s' column not found.")

    print(f"\nAll charts generated and saved to {charts_output_dir}")

if __name__ == '__main__':
    main()