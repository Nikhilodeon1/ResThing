# Latent Surgery: Modular Codebase for CLIP Editing

This project introduces a novel approach to latent space editing within CLIP, enabling targeted semantic transformations of image representations without the need for retraining. Our method leverages a modular pipeline that facilitates reproducibility and scalability, demonstrating how modifying CLIP's latent space can achieve controlled semantic changes in images.

## Project Overview

We aim to show how "latent surgery" can improve semantic transformations by manipulating image representations directly within CLIP's latent space. The codebase is designed to be modular and scalable, allowing for experiments with various datasets and comparisons against baseline methods.

### Key Features:
- **Modular Design**: Clear separation of concerns for data handling, model interaction, latent space manipulation, and evaluation.
- **CLIP Integration**: Seamless embedding of images and text using a `CLIPModelWrapper`.
- **Semantic Direction Calculation**: Automated computation of direction vectors for specific semantic transformations (e.g., "smiling" to "not smiling").
- **Latent Space Editing**: Application of computed direction vectors to image embeddings to achieve desired semantic changes.
- **Baseline Comparison**: Includes baselines such as prompt tuning and random edits for comprehensive evaluation.
- **Quantitative & Qualitative Evaluation**: Metrics like accuracy, cosine similarity, and visualizations (t-SNE plots) to assess effectiveness.
- **Reproducibility**: Configuration-driven experiments with clear output logging.

## Project Phases

This project was developed in four distinct phases:

1.  **Phase 1: Data Preparation & Loading**
    -   Setup for dataset loading (CelebA, CUB-200, ImageNet).
    -   Data filtering based on attributes.
    -   Consistent image transformations.
    -   Configuration via `config.yaml`.
2.  **Phase 2: Model Setup & Latent Surgery Implementation**
    -   `CLIPModelWrapper` for image and text embedding.
    -   `direction_finder.py` for calculating semantic direction vectors.
    -   `edit_embedding.py` for applying latent transformations.
3.  **Phase 3: Baseline Comparison & Evaluation**
    -   Implementation of baseline models (`prompt_tuning.py`, `random_edit.py`).
    -   `metrics.py` for calculating evaluation metrics.
    -   `visualize.py` for generating t-SNE plots and other visualizations.
    -   Comparison of latent surgery against baselines and logging results.
4.  **Phase 4: Final Reporting, Reproducibility, and Documentation**
    -   Comprehensive code documentation.
    -   Ensuring reproducibility for public release.

## Setup Instructions

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/latent-surgery.git](https://github.com/your-username/latent-surgery.git)
    cd latent-surgery
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Datasets:**
    -   **CelebA**: Download the `img_align_celeba.zip` and `list_attr_celeba.csv` files. Extract `img_align_celeba` into `data/celeba/` and place `list_attr_celeba.csv` in `data/celeba/`.
        * Update `celeba_root_dir` in `config.yaml` to point to your CelebA dataset directory (e.g., `/content/drive/MyDrive/Paper2/sigmaPack121`).
    -   **CUB-200-2011**: Download and extract the dataset. Ensure `images/`, `images.txt`, `image_class_labels.txt`, and `classes.txt` are correctly placed.
        * Update `cub_root_dir` in `config.yaml`.
    -   **ImageNet**: Prepare your ImageNet subset or full dataset.
        * Update `imagenet_root_dir` in `config.yaml`.

## Configuration (`config.yaml`)

The `config.yaml` file controls all experiment parameters. Modify it to select your dataset, concepts, batch size, and output directories.

```yaml
# config.yaml (Example)
dataset: CelebA # Options: CelebA, CUB-200, ImageNet
concept:
  positive: Smiling
  negative: Not_Smiling
batch_size: 32
model_name: openai/clip-vit-base-patch32
surgery_alpha: 1.0 # Scaling factor for latent surgery
output_dir: ./outputs
save_tsne: True
run_prompt_baseline: True
run_linear_probe_baseline: True # If implemented
run_random_edit_baseline: True
celeba_root_dir: "/path/to/your/CelebA"
cub_root_dir: "/path/to/your/CUB_200_2011"
imagenet_root_dir: "/path/to/your/ImageNet"