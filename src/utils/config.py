import yaml
import os
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def save_config(config: Dict[str, Any], config_path: str):
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False, indent=2)


def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration parameters.
    
    Returns:
        Dictionary with default configuration
    """
    return {
        'environment': {
            'max_moves': 200,
            'reward_win': 1.0,
            'reward_draw': 0.0,
            'reward_loss': -1.0,
            'reward_material_factor': 0.1
        },
        'agent': {
            'type': 'dqn',
            'learning_rate': 0.001,
            'gamma': 0.99,
            'epsilon_start': 1.0,
            'epsilon_end': 0.01,
            'epsilon_decay': 0.995,
            'target_update_freq': 1000
        },
        'network': {
            'hidden_layers': [512, 256, 128],
            'activation': 'relu',
            'dropout': 0.1
        },
        'training': {
            'batch_size': 64,
            'buffer_size': 100000,
            'min_buffer_size': 1000,
            'update_freq': 4,
            'num_episodes': 10000,
            'eval_freq': 100,
            'save_freq': 500
        },
        'logging': {
            'log_dir': 'logs',
            'tensorboard': True,
            'save_models': True,
            'model_dir': 'models'
        }
    } 