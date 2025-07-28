# utils/config_loader.py
import yaml

def load_config(config_path="config.yaml"):
    """
    Loads and validates the YAML configuration file.
    """
    try:
        with open(config_path, 'r') as f:
            cfg = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        # Basic validation (can be expanded)
        required_keys = ['dataset', 'concept', 'batch_size', 'model_name', 'output_dir']
        for key in required_keys:
            if key not in cfg:
                raise ValueError(f"Missing required key in config.yaml: {key}")
        
        if cfg['dataset'] == 'CelebA' and 'celeba_root_dir' not in cfg:
            raise ValueError("For CelebA dataset, 'celeba_root_dir' must be specified in config.yaml")
        if cfg['dataset'] == 'CUB-200' and 'cub_root_dir' not in cfg:
            raise ValueError("For CUB-200 dataset, 'cub_root_dir' must be specified in config.yaml")
        if cfg['dataset'] == 'ImageNet' and 'imagenet_root_dir' not in cfg:
            raise ValueError("For ImageNet dataset, 'imagenet_root_dir' must be specified in config.yaml")

        return cfg
    except FileNotFoundError:
        print(f"Error: Config file not found at {config_path}")
        raise
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        raise
    except ValueError as exc:
        print(f"Configuration validation error: {exc}")
        raise

if __name__ == '__main__':
    try:
        config = load_config("../config.yaml")
        print("Config loaded successfully for testing:")
        print(config)
    except Exception as e:
        print(f"Error during config loading test: {e}")