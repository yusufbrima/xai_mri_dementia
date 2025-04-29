import torch
from tqdm import tqdm  # for progress bar

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, device=None):
    """
    Train the model, track train and val losses and accuracies, and return the trained model and history.

    Args:
    - model: The model to train (Simple3DCNN).
    - train_loader: DataLoader for training data.
    - val_loader: DataLoader for validation data.
    - criterion: Loss function (CrossEntropyLoss).
    - optimizer: Optimizer (Adam).
    - num_epochs: Number of epochs for training.
    - device: Device to train the model on (either 'cuda' or 'cpu'). If None, it will automatically choose based on availability.

    Returns:
    - model: Trained model.
    - history: Dictionary containing 'train_loss', 'val_loss', 'train_accuracy', 'val_accuracy'.
    """
    
    # Automatically choose device if None is provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to the selected device (GPU or CPU)
    model.to(device)
    
    # History dictionary to store losses and accuracies
    history = {'train_loss': [], 'val_loss': [], 'train_accuracy': [], 'val_accuracy': []}

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        # Training loop
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Training)"):
            inputs, labels = inputs['image'], labels['label']
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # Zero the gradients
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Update model parameters
            
            # Calculate training accuracy
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == labels).sum().item()
            total_train += labels.size(0)
            
            train_loss += loss.item()
        
        # Calculate training accuracy and loss
        train_accuracy = correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():  # No need to calculate gradients for validation
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} (Validation)"):
                inputs, labels = inputs['image'], labels['label']
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Calculate validation accuracy
                _, predicted = torch.max(outputs, 1)
                correct_val += (predicted == labels).sum().item()
                total_val += labels.size(0)
                
                val_loss += loss.item()
        
        # Calculate validation accuracy and loss
        val_accuracy = correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        # Print and record stats for this epoch
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Append to history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_accuracy'].append(train_accuracy)
        history['val_accuracy'].append(val_accuracy)
    
    return model, history

def save_model(model, save_path):
    """
    Save the model's state_dict to a file.

    Args:
    - model: The trained model.
    - save_path: Path where the model will be saved (including file name).
    """
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")



def load_model(model, load_path):
    """
    Load the model's state_dict from a file.

    Args:
    - model: The model to load the weights into.
    - load_path: Path to the saved model state_dict file.
    """
    model.load_state_dict(torch.load(load_path))
    model.eval()  # Set the model to evaluation mode
    print(f"Model loaded from {load_path}")
    return model