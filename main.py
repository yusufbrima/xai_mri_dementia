import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from data.nifti_loader import MedicalImageDatasetSplitter,MonaiDatasetCreator,MonaiDataLoaderManager
import matplotlib.pyplot as plt
from config import config_loader
import torch.nn as nn
from models.cnn_backbones import Small3DCNN,DenseNet3D,ResNet3D,BasicBlock3D
from training.trainer import train_model,test_model

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if __name__ == "__main__":
    # Path to config
    config_path = "config/base_config.yaml"

    # Load & process config (creates directories once)
    config = config_loader.load_config(config_path)


    config['data']['batch_size']      = 8
    config['data']['perform_slicing'] = False
    config['data']['image_size'] = [128, 128, 128]
    config['training']['epochs'] = 2
    
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
    NUM_CLASSES = dataset_splitter.get_num_classes()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Small3DCNN(num_classes=NUM_CLASSES).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)



    # Train the model on the available device (either GPU or CPU)
    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=config['training']['epochs'],
        device=device
    )

    # Accessing the training history
    print("Training History:")
    print(f"Train Loss: {history['train_loss']}")
    print(f"Train Accuracy: {history['train_accuracy']}")
    print(f"Validation Loss: {history['val_loss']}")
    print(f"Validation Accuracy: {history['val_accuracy']}")


    test_loss, test_acc, preds, labels = test_model(trained_model, test_loader, criterion)










    



    