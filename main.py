import torch
import numpy as np
import random
from data.nifti_loader import MedicalImageDatasetSplitter,MonaiDatasetCreator,MonaiDataLoaderManager
from config import config_loader
# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == "__main__":
    # Path to config
    config_path = "config/base_config.yaml"

    # Load & process config (creates directories once)
    config = config_loader.load_config(config_path)
    
    # Create dataset splitter
    dataset_splitter = MedicalImageDatasetSplitter(config)
    
    # Create dataset creator
    dataset_creator = MonaiDatasetCreator(dataset_splitter)
    
    # Create dataloader manager
    dataloader_manager = MonaiDataLoaderManager(dataset_creator, config)
    
    # Get all dataloaders
    dataloaders = dataloader_manager.get_dataloaders()
    
    # Access individual dataloaders
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']
    test_loader = dataloaders['test']
    
    # Get class information
    class_to_idx, idx_to_class = dataset_splitter.get_class_info()
    num_classes = dataset_splitter.get_num_classes()
    
    print(f"Number of classes: {num_classes}")

    

    