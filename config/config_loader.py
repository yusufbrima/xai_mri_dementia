"""
YAML Configuration Loader for Deep Learning Experiments

This module handles loading YAML configurations with dynamic experiment directories.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Union, List
from string import Template
from datetime import datetime
import yaml

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)






def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load and process a YAML configuration file with dynamic experiment paths.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        The processed configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Load the YAML config
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    # Get current date in YYYYMMDD format
    current_date = datetime.now().strftime("%Y%m%d")

    # Process experiment name with format
    experiment_id = config['logging']['experiment_id']
    experiment_name = config['logging']['experiment_name'].format(
        experiment_id=experiment_id,
        date=current_date
    )
    config['logging']['experiment_name'] = experiment_name
    
    
    # Update output_base_dir with the formatted experiment name
    output_base_dir = Template(config['logging']['output_base_dir']).safe_substitute(
        experiment_name=experiment_name
    )
    config['logging']['output_base_dir'] = output_base_dir
    
    # Process all string values with ${} templates
    context = {
        'experiment_name': experiment_name,
        'output_base_dir': output_base_dir
    }
    process_templates(config, context)
    
    # Create output directories
    create_directories(config)
    
    return config


def process_templates(cfg: Union[Dict, List], context: Dict[str, str]) -> None:
    """
    Recursively process template strings in configuration.
    
    Args:
        cfg: Configuration section to process
        context: Template variables and their values
    """
    if isinstance(cfg, dict):
        for key, value in list(cfg.items()):
            if isinstance(value, str) and "${" in value:
                cfg[key] = Template(value).safe_substitute(context)
            elif isinstance(value, (dict, list)):
                process_templates(value, context)
    elif isinstance(cfg, list):
        for i, item in enumerate(cfg):
            if isinstance(item, str) and "${" in item:
                cfg[i] = Template(item).safe_substitute(context)
            elif isinstance(item, (dict, list)):
                process_templates(item, context)


def create_directories(config: Dict[str, Any]) -> None:
    """
    Create all output directories specified in configuration.
    
    Args:
        config: Configuration dictionary with directories
    """
    base_dir = Path(config['logging']['output_base_dir'])
    
    directories = [
        base_dir,
        base_dir / config['logging'].get('log_dir', 'logs'),
        base_dir / config['logging'].get('checkpoint_dir', 'checkpoints'),
        Path(config['explainability']['output_dir']),
        Path(config['output']['output_dir']),
        Path(config['misc']['figure_dir'])
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")
            raise


def get_experiment_paths(config: Dict[str, Any]) -> Dict[str, Path]:
    """
    Extract all experiment paths from processed config.
    
    Args:
        config: The processed configuration
        
    Returns:
        Dictionary containing all experiment directory paths
    """
    base_dir = Path(config['logging']['output_base_dir'])
    
    return {
        'base_dir': base_dir,
        'log_dir': base_dir / config['logging'].get('log_dir', 'logs'),
        'checkpoint_dir': base_dir / config['logging'].get('checkpoint_dir', 'checkpoints'),
        'explanations_dir': Path(config['explainability']['output_dir']),
        'predictions_dir': Path(config['output']['output_dir']),
        'figures_dir': Path(config['misc']['figure_dir'])
    }

if __name__ == "__main__":
    pass