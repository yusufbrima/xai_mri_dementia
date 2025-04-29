import torch
import numpy as np

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

if __name__ == "__main__":
    # Example usage
    pass